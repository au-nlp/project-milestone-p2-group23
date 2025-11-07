# Features list and versioning meaning:

### episode_vecs:

Embedings for episodes meaning, used for comparisons(cosine similarity)

- v0.1.0:
    - Small **bug**: because we filtered out missing podTitles, this has to be always done to match the index of the episodes
    - Notebook used from commit `commit hash`
    - Model used: sentence-transformers/all-MiniLM-L6-v2
    - Processing time 1000 episodes/3 min on google colab T4 GPU => 5h/all episodes
    - Method: split episodes in chunks of `max_len=256` and `stride=32` tokens and aggregate with `mean`, 5000 embeds per file

- v0.1.1:
    - Same as v0.1 but fixing the **bug**, embeds should now align perfectly with the unmodified 

- v0.1.2:
    - The previous version was still not reliable
    - Changed to compressed format for better storage
    - Contains both `ids` and `vecs` so the load can be more consisten