import numpy as np
import torch
from tqdm import tqdm

import torch.nn.functional as F

from .metrics import score_ideas


def _logsumexp(x):
    """Numerically stable log-sum-exp for 1D numpy arrays."""
    if x.size == 0:
        return -np.inf
    m = np.max(x)
    return m + np.log(np.exp(x - m).sum())


def mean_pool(last_hidden_state, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    masked = last_hidden_state * mask
    summed = masked.sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

@torch.no_grad()
def embed_texts(model, tokenizer, texts, batch_size=512, max_tokens=256, prefix=None, device='cuda'):
    vecs = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        if prefix is not None:
            batch_texts = [prefix + text for text in batch_texts]
        encodings = tokenizer(batch_texts, truncation=True, padding=True, max_length=max_tokens, return_tensors='pt')
        encodings = {k: v.to(device) for k, v in encodings.items()} # Move encodings to device before passing to model
        outputs = model(**encodings)
        embeddings = mean_pool(outputs.last_hidden_state, encodings['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1).cpu().numpy()
        vecs.append(embeddings)
    return np.vstack(vecs)


# Very ineffiction, consider different method
def chunk_by_tokens(tokenizer, text, max_len=256, stride=32):
    tokens = tokenizer.encode(text)
    chunks = []
    for i in range(0, len(tokens), max_len - stride):
        chunk_tokens = tokens[i:i + max_len]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(chunk_text)
        if i + max_len >= len(tokens):
            break
    return chunks

# Because the episodes are long, we chunk them into smaller pieces, embed each piece,
# then aggregate the piece embeddings into episode embedding
# TODO: Play around with different aggregation methods

@torch.no_grad()
def embed_episodes(model, tokenizer, df, text_column='transcript', agg="max", max_len=256):
    episode_vecs = []
    for text in tqdm(df[text_column]):
        chunks = chunk_by_tokens(tokenizer, text, max_len=max_len)
        chunk_vecs = embed_texts(model, tokenizer, chunks, max_tokens=max_len)
        if agg == "max":
            episode_vec = np.max(chunk_vecs, axis=0)
        elif agg == "mean":
            episode_vec = np.mean(chunk_vecs, axis=0)
        else:
            raise ValueError(f"Unknown agg method: {agg}")
        episode_vecs.append(episode_vec)
    return np.vstack(episode_vecs)

# To fully utilize gpu, we gonna group all chunks across episodes and embed in large batches
@torch.no_grad()
def embed_episodes_batched(
    model,
    tokenizer,
    df,
    text_column='transcript',
    agg="mean",
    max_len=256,
    stride=32,
    batch_size=512,
    device='cuda'
):
    episode_counts = []
    all_chunks = []
    for text in tqdm(df[text_column]):
        chunks = chunk_by_tokens(tokenizer, text, max_len=max_len, stride=stride)
        episode_counts.append(len(chunks))
        all_chunks.extend(chunks)

    all_embs = []
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i+batch_size]
        embs = embed_texts(model, tokenizer, batch, max_tokens=max_len, batch_size=batch_size, device=device)
        all_embs.append(embs)
    all_embs = np.vstack(all_embs)

    episode_vecs = []
    start = 0
    for count in episode_counts:
        cur = all_embs[start:start+count]
        if agg == "max":
            episode_vecs.append(cur.max(axis=0))
        elif agg == "mean":
            episode_vecs.append(cur.mean(axis=0))
        else:
            raise ValueError(f"Unknown agg method: {agg}")
        start += count

    return np.vstack(episode_vecs)


@torch.no_grad()
def embed_episode_segments_batched(
    model,
    tokenizer,
    df,
    text_column='transcript',
    max_len=256,
    stride=32,
    batch_size=512,
    device='cuda'
):
    """
    Embed all transcript segments without aggregating to the episode level.

    Returns a list where each element is a numpy array of shape
    (num_segments_for_episode, embed_dim) corresponding to one episode.
    """
    episode_counts = []
    all_chunks = []
    for text in tqdm(df[text_column]):
        chunks = chunk_by_tokens(tokenizer, text, max_len=max_len, stride=stride)
        episode_counts.append(len(chunks))
        all_chunks.extend(chunks)

    all_embs = []
    for i in tqdm(range(0, len(all_chunks), batch_size)):
        batch = all_chunks[i:i+batch_size]
        embs = embed_texts(model, tokenizer, batch, max_tokens=max_len, batch_size=batch_size, device=device)
        all_embs.append(embs)
    all_embs = np.vstack(all_embs)

    episode_segments = []
    start = 0
    for count in episode_counts:
        episode_segments.append(all_embs[start:start+count])
        start += count

    return episode_segments


# TODO: impl
#def process_and_save_embeddings()

def score_df_by_ideas(
    df,
    vec_column,
    idea_vecs,
    score_columns=None,
    device='cuda',
    inplace=True # If false, return just the scores
):
    if score_columns is None:
        if len(idea_vecs) == 1:
            score_columns = ["idea_score"]
        else:
            score_columns = [f"idea_score_{i}" for i in range(len(idea_vecs))]
    else:
        assert len(score_columns) == len(idea_vecs), "Length of score_columns must match number of idea_vecs"

    scores = score_ideas(df[vec_column].tolist(), idea_vecs)

    if not inplace:
        return scores

    for i, col in enumerate(score_columns):
        df[col] = scores[:, i]

    return df


def _aggregate_similarity_values(
    sims,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
):
    """
    Aggregate similarity values for one episode/idea pair using different pooling rules.

    - topk_mean: mean of the top_k similarities
    - logsumexp: log(sum(exp(sim / tau)))
    - pct_above: fraction of sims above the threshold
    - mean: standard mean pooling (backwards compatible)
    """
    if sims.size == 0:
        return np.nan

    if agg == "topk_mean":
        k = min(top_k, sims.size)
        top_vals = np.partition(sims, -k)[-k:]
        return float(top_vals.mean())
    if agg == "logsumexp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for logsumexp pooling")
        return float(_logsumexp(sims / tau))
    if agg == "pct_above":
        return float((sims > threshold).mean())
    if agg == "mean":
        return float(sims.mean())

    raise ValueError(f"Unknown aggregation method: {agg}")


def score_episode_segments(
    segment_embs,
    idea_vecs,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
):
    """
    Compute aggregated similarity scores for one episode against many ideas.

    Parameters
    ----------
    segment_embs : np.ndarray
        Array of shape (num_segments, embed_dim) with normalized embeddings.
    idea_vecs : np.ndarray
        Array of shape (num_ideas, embed_dim) with normalized idea embeddings.
    agg : str
        Aggregation rule (topk_mean, logsumexp, pct_above, mean).

    Returns
    -------
    np.ndarray
        Array of shape (num_ideas,) with aggregated scores.
    """
    idea_vecs = np.atleast_2d(idea_vecs)

    if segment_embs.size == 0:
        return np.full((idea_vecs.shape[0],), np.nan, dtype=np.float32)

    sims = segment_embs @ idea_vecs.T
    scores = np.empty(sims.shape[1], dtype=np.float32)
    for j in range(sims.shape[1]):
        scores[j] = _aggregate_similarity_values(
            sims[:, j],
            agg=agg,
            top_k=top_k,
            tau=tau,
            threshold=threshold,
        )
    return scores


def score_segments_by_ideas(
    episode_segments,
    idea_vecs,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
):
    """
    Compute aggregated similarity scores for many episodes against many ideas.

    Parameters
    ----------
    episode_segments : List[np.ndarray]
        Each element contains embeddings for one episode's segments.
    idea_vecs : np.ndarray
        Array of shape (num_ideas, embed_dim) with normalized idea embeddings.

    Returns
    -------
    np.ndarray
        Shape (num_episodes, num_ideas) with aggregated scores.
    """
    idea_vecs = np.atleast_2d(idea_vecs)
    results = np.empty((len(episode_segments), idea_vecs.shape[0]), dtype=np.float32)
    for i, segs in enumerate(episode_segments):
        results[i] = score_episode_segments(
            segs,
            idea_vecs,
            agg=agg,
            top_k=top_k,
            tau=tau,
            threshold=threshold,
        )
    return results


def _softmax(x, tau):
    """Temperature-scaled softmax over a 1D numpy array."""
    z = x / tau
    z = z - np.max(z)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z)


def pool_segments_for_idea(
    segment_embs,
    idea_vec,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
    normalize=True,
):
    """
    Create an idea-aware pooled embedding for one episode.

    This keeps the pooling logic aligned with the similarity aggregation:
    - topk_mean: average the embeddings of the top_k similar segments
    - logsumexp: softmax-weighted sum of segment embeddings
    - pct_above: mean of embeddings above the similarity threshold
    - mean: uniform mean of all segments
    """
    if segment_embs.size == 0:
        return np.zeros_like(idea_vec)

    sims = segment_embs @ idea_vec

    if agg == "topk_mean":
        k = min(top_k, sims.size)
        top_idx = np.argpartition(sims, -k)[-k:]
        pooled = segment_embs[top_idx].mean(axis=0)
    elif agg == "logsumexp":
        if tau <= 0:
            raise ValueError("tau must be > 0 for logsumexp pooling")
        weights = _softmax(sims, tau)
        pooled = (segment_embs * weights[:, None]).sum(axis=0)
    elif agg == "pct_above":
        mask = sims > threshold
        if mask.any():
            pooled = segment_embs[mask].mean(axis=0)
        else:
            pooled = segment_embs.mean(axis=0)
    elif agg == "mean":
        pooled = segment_embs.mean(axis=0)
    else:
        raise ValueError(f"Unknown aggregation method: {agg}")

    if normalize:
        norm = np.linalg.norm(pooled)
        if norm > 0:
            pooled = pooled / norm

    return pooled


def pooled_episode_embeddings(
    episode_segments,
    idea_vec,
    agg="topk_mean",
    top_k=5,
    tau=0.07,
    threshold=0.2,
    normalize=True,
):
    """
    Pool embeddings for many episodes for a single idea.

    Returns a numpy array of shape (num_episodes, embed_dim).
    """
    pooled = []
    for segs in episode_segments:
        pooled.append(
            pool_segments_for_idea(
                segs,
                idea_vec,
                agg=agg,
                top_k=top_k,
                tau=tau,
                threshold=threshold,
                normalize=normalize,
            )
        )
    return np.vstack(pooled)
