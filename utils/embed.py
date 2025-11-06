import numpy as np
import torch
from tqdm import tqdm

import torch.nn.functional as F

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
def embed_episodes_batched(
    model,
    tokenizer,
    df,
    text_column='transcript',
    agg="mean",
    max_len=256,
    batch_size=512,
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
        embs = embed_texts(model, tokenizer, batch, max_tokens=max_len, batch_size=batch_size)
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