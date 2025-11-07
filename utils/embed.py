import numpy as np
import torch
from tqdm import tqdm

import torch.nn.functional as F

from .metrics import score_ideas


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
    batch_size=512,
    device='cuda'
):
    episode_counts = []
    all_chunks = []
    for text in tqdm(df[text_column]):
        chunks = chunk_by_tokens(tokenizer, text, max_len=max_len)
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
