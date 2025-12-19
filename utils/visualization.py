from IPython.display import HTML, Image, Markdown, display
from pyvis.network import Network
from glob import glob

import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import pandas as pd
import networkx as nx
# from pathlib import Path


def view_image(img_path, width=600):
    display(Image(filename=img_path, width=width))


def view_images(pattern, cols=6):
    display(HTML(
        f"<div style='display:grid;grid-template-columns:repeat({cols},1fr);gap:6px'>"
        + "".join(f"<img src='{p}' style='width:100%;aspect-ratio:1;object-fit:cover'>" for p in glob(pattern))
        + "</div>"
    ))


def view_md(md_path):
    with open(md_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
    display(Markdown(md_content))


def plot_transcript_length_distribution(episode_df):
    episode_df["transcriptLength"] = episode_df["transcript"].apply(
        lambda x: len(x.split()))
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(episode_df["transcriptLength"], bins=50, kde=True)
    plt.title("Transcript Length Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.show()


def plot_category_distribution(episode_df, top_n_categories=10, top_n=30):
    category_columns = [f"category{i}" for i in range(
        1, top_n_categories + 1) if f"category{i}" in episode_df.columns]
    category_counts = episode_df[category_columns].melt(
        value_name="category")["category"].value_counts().head(top_n).reset_index()
    sns.barplot(category_counts, y="category", x="count", orient="h")
    plt.xlabel("Count")
    plt.ylabel("Category")
    plt.title(f"Top {top_n} Categories Distribution")
    plt.tight_layout()
    plt.show()


def plot_missing_values(df, columns=None, ax=None, title="Missing Values in each column"):
    if columns is None:
        columns = df.columns
    missing_counts = df[columns].isnull().sum()

    if ax is None:
        ax = plt.gca()

    sns.barplot(x=missing_counts.index, y=missing_counts.values, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Columns")
    ax.set_ylabel("Number of Missing Values")
    ax.tick_params(axis='x', rotation=45)

    return ax


def plot_appearances(df, column, top_n=20):
    all_names = list(itertools.chain.from_iterable(
        df[column].dropna().tolist()))
    name_counts = pd.Series(all_names).value_counts().head(top_n)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=name_counts.values, y=name_counts.index)
    plt.title(f"Top {top_n} {column} appearances")
    plt.xlabel("Number of Appearances")
    plt.ylabel(column)
    plt.show()


def plot_animated_graph(G, save_path):
    net = Network(
        notebook=True,
        width="1600px",
        height="1000px",
        bgcolor="#222222",
        font_color="white"
    )

    net.barnes_hut(
        gravity=-20000,
        central_gravity=0.3,
        spring_length=200,
        spring_strength=0.05,
        damping=0.9
    )

    node_degree = dict(G.degree)
    nx.set_node_attributes(G, node_degree, "size")

    # You can color nodes
    # nx.set_node_attributes(G, communities, 'group')

    net.from_nx(G)

    # Interactive exploration settings
    net.repulsion(
        node_distance=200,
        spring_length=200,
        damping=0.85
    )

    net.show(save_path)
