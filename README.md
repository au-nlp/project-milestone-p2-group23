# Community Graph from Name Co-Mention Proximity

The Structured Podcast Research Corpus lends itself well to analyzing not only what is said in podcasts but also how communities form through discourse. The original SPoRC paper emphasizes networks built around shared guests across shows. While this captures connections at the production level, it overlooks communities that exist at the level of conversation. The idea of this project is to construct a new kind of network that we call the _Name Co-Mention Proximity Graph_. Here, each node represents a person's name mentioned in a podcast, and edges represent frequent co-mentions of those names within short textual windows or within a whole episode. This design captures who tends to be talked about together, even if those individuals never share the same guest seat.

## _TODO_

1. Write about the different ways to create the data for the graph (spacy, bert based model etc.)
2. Analyze the data cleaning pipeline and enhance it
3. For the entity recognition, we could further refine it through considering that a podcast may mention many entities: some are more central and critical, while others are only mentioned in passing. How to distinguish between them is an important question. We could use another model for that.
4.

Once the graph is constructed, community detection methods can reveal clusters of frequently co-mentioned names. These clusters may correspond to latent communities in discourse, such as groups of athletes, political figures, or cultural icons. To move beyond raw graph structure, I propose assigning pseudo-labels to clusters by extracting the most distinctive words and metadata associated with them. For example, if a cluster is dominated by "LeBron James," "NBA," and "playoffs," a plausible pseudo-label could be "Basketball." Building on this, a lightweight classifier or a prompting-based method could predict the "real-world" name of each community.

As an extension, I propose adding a similarity search in a reduced embedding space. Episodes can be represented as high-dimensional vectors from their transcripts and metadata, then projected into three dimensions using PCA. In this space, nearby points correspond to episodes with similar discourse patterns or overlapping communities. Nearest-neighbor search in 3D would make it easy to retrieve related episodes and to visualize clusters, highlighting both obvious and unexpected overlaps across categories.

[View the demo](drafts/community_graph.html)

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

<img width="1110" height="574" alt="grafik" src="https://github.com/user-attachments/assets/907ebf33-01e6-4e0c-bea3-1285cdd59f6c" />

