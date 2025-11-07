import gzip
import pandas as pd
import itertools
import json

from huggingface_hub import hf_hub_download
from datasets import load_dataset


def load_episodes_to_df(data_dir, limit=100_000):
    episode_path = hf_hub_download(
        repo_id="blitt/SPoRC",
        filename="episodeLevelData.jsonl.gz",
        repo_type="dataset",
        local_dir=data_dir,
    )
    episodes = []
    with gzip.open(episode_path, 'rt') as f:
        for line in itertools.islice(f, limit):
            episodes.append(json.loads(line))
    return pd.DataFrame(episodes)


def load_speakers_to_df(data_dir, limit=600_000):
    speaker_path = hf_hub_download(
        repo_id="blitt/SPoRC",
        filename="speakerTurnData.jsonl.gz",
        repo_type="dataset",
        local_dir=data_dir,
    )
    speaker_ds = load_dataset("json", data_files=speaker_path, split="train", streaming=True)
    rows = []
    for i, sample in enumerate(speaker_ds):
        if i >= limit:
            break
        rows.append(sample)
    return pd.DataFrame(rows)