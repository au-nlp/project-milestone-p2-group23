from IPython.display import Image, Markdown, display

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
#from pathlib import Path

def view_image(img_path, width=600):
    display(Image(filename=img_path, width=width))

def view_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    display(Markdown(md_content))

def plot_transcript_length_distribution(episode_df):
    episode_df["transcriptLength"] = episode_df["transcript"].apply(lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(episode_df["transcriptLength"], bins=50, kde=True)
    plt.title("Transcript Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()

def plot_category_distribution(episode_df, top_n_categories=10):
    category_columns = [f"category{i}" for i in range(1, top_n_categories + 1) if f"category{i}" in episode_df.columns]
    category_counts = episode_df[category_columns].melt(value_name="category")["category"].value_counts().reset_index()
    category_counts
    sns.barplot(category_counts, x="category", y="count")

def plot_missing_values(episode_df, columns=None):
    if columns is None:
        columns = episode_df.columns
    missing_counts = episode_df[columns].isnull().sum()
    #plt.figure(figsize=(12, 6))
    sns.barplot(x=missing_counts.index, y=missing_counts.values)
    plt.title("Missing Values in Each Column")
    plt.xlabel("Columns")
    plt.ylabel("Number of Missing Values")
    plt.xticks(rotation=45)
    plt.show()

def plot_appearances(df, column, top_n=20):
    all_names = list(itertools.chain.from_iterable(df[column].dropna().tolist()))
    name_counts = pd.Series(all_names).value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=name_counts.values, y=name_counts.index)
    plt.title(f"Top {top_n} {column} appearances")
    plt.xlabel("Number of Appearances")
    plt.ylabel(column)
    plt.show()