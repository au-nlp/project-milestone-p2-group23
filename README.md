# Community Graph from Name Co-Mention Proximity

The **Structured Podcast Research Corpus (SPoRC)** provides a rich foundation for analyzing not only _what_ is said in podcasts but also _how_ communities emerge through discourse.  
The original SPoRC paper focuses on networks of _shared guests_ across shows — a valuable production-level view, but one that overlooks how communities form at the conversational level.

This project proposes a new network representation called the **Name Co-Mention Proximity Graph**, which captures relationships between people who are _talked about together_ in podcast transcripts.  
Each node represents a person’s name mentioned in a podcast, and edges indicate frequent co-mentions within short textual windows or entire episodes. This structure reveals conversational connections, even among individuals who never appear together as guests.

## Graph Construction

The foundation is straightforward: extract named entities from transcripts and connect those that co-occur within defined contexts.  
The challenge — and the core research contribution — lies in constructing this graph _reliably and meaningfully_.

### 1. Text Preprocessing

Before entity recognition, we will experiment with several preprocessing strategies to assess their effect on detection quality.

- **Research Questions:**

  - Should stopwords be removed? While it can reduce noise, removing connectors like “and” may hinder detection of co-mentions such as _“Arne and Vlad.”_
  - How does sentence segmentation or punctuation normalization affect mention grouping?
  - Do conversational markers (“uh,” “you know,” etc.) harm or help context recognition?

- **Implementation Plan:**
  - Compare multiple preprocessing pipelines: raw text, stopword-removed text, and sentence-split text.
  - Use contextual sentence embeddings (e.g., _Sentence-BERT_) to preserve semantic proximity even if names don’t appear directly adjacent.
  - Measure downstream impact on entity recall and co-mention precision.

### 2. Entity Recognition and Normalization

This stage aims to detect personal names and unify their variants into canonical entities. The process integrates **entity recognition**, **coreference resolution**, and **semantic consolidation** into a single coherent pipeline.

- **Entity Detection**

  - Start with **spaCy’s NER** as a baseline.
  - Compare transformer-based models fine-tuned for informal or conversational text, such as:
    - _DistilBERT-NER_ or _Flair_ for lightweight contextual detection.
    - _LUKE (Language Understanding with Knowledge-based Embeddings)_ for entity-rich domains.
  - Incorporate **coreference resolution** (e.g., _SpanBERT_, _AllenNLP coref_) to link pronouns and repeated mentions.

- **Entity Linking and Normalization**

  - Use **entity linking** models (e.g., _BLINK_, _REL_) to map mentions to canonical entities in external knowledge bases when possible.
  - For in-domain consistency, cluster entity embeddings using cosine similarity to merge variations like _“LeBron”_, _“LeBron James”_, and nicknames.
  - Apply attention-based similarity or embedding-based salience scores to decide whether two mentions refer to the same individual.

- **Relevance and Salience**
  - Compute each entity’s contextual importance to filter out incidental mentions:
    - Aggregate attention weights across mentions.
    - Use _Salience-BERT_ or embedding-level contextual salience.
    - Apply weighted TF-IDF over mention contexts to downweight low-relevance names.

This unified approach yields a **clean, canonical set of nodes** — one per real-world individual — ready for co-mention graph construction.

### 3. Co-Mention Definition and Graph Building

Once entities are recognized and normalized, we define how and when two names are “co-mentioned.”

- **Windowing Strategies**

  - Test co-mentions within flexible text spans — sentence, paragraph, or speaker turn.
  - Use **sliding windows** with semantic thresholds (based on cosine similarity between segment embeddings) to capture implicit contextual co-occurrence.

- **Edge Weighting**

  - Construct a **weighted undirected graph**:
    - Nodes: canonical person entities.
    - Edges: weighted by co-mention frequency × contextual similarity.
  - Normalize weights by total mention frequency to avoid overemphasis on globally popular names.

- **Implementation**
  - Build and analyze graphs using libraries such as _NetworkX_ or _igraph_.
  - Store intermediate data (entity clusters, co-mention windows, weights) for reproducibility.

## Evaluation

Evaluating such graphs requires both structural and semantic perspectives.

1. **Graph Structural Validation**

   - Analyze degree distributions, clustering coefficients, and modularity to ensure realistic social structure.
   - Check stability across random subsets or episodes.

2. **External Validation**

   - Cross-reference detected communities with **Wikidata** or **Wikipedia categories**.
   - Measure alignment between discovered clusters and known domains (e.g., “NBA players,” “politicians”).

3. **Predictive Evaluation**

   - Learn graph embeddings (_node2vec_, _GraphSAGE_) and test them on downstream tasks, such as predicting podcast categories or guest topics.
   - Classification accuracy serves as an indirect quality metric.

4. **Qualitative Analysis**
   - Manually inspect representative clusters for interpretability.
   - Evaluate coherence of top names per cluster.

## Using the Graph

Once built, the **Name Co-Mention Proximity Graph** enables community detection and labeling.

### Community Detection

Apply algorithms such as **Louvain**, **Leiden**, or **Infomap** to discover clusters of frequently co-mentioned names. These clusters represent _latent discourse communities_ — e.g., athletes, political figures, or cultural icons.

### Cluster Interpretation and Labeling

To interpret communities, extract the most distinctive linguistic and contextual signals from each cluster.

- **Keyword and Topic Extraction**

  - Aggregate co-occurring words from transcript contexts.
  - Use **TF-IDF**, mutual information, or topic modeling methods (_LDA_, _BERTopic_) to summarize themes.
  - Example: a cluster dominated by “LeBron James,” “NBA,” and “playoffs” may be labeled _Basketball_.

- **Label Generation**
  - Use **prompt-based LLM labeling** to propose pseudo-labels automatically.
  - Optionally fine-tune a **lightweight classifier** on a few manually labeled clusters for scalable annotation.

## Expected Contributions

This project bridges **text semantics**, **entity normalization**, and **graph-based community detection** to reveal how individuals are discussed together in conversational media.

**Outcomes:**

1. A reproducible pipeline for constructing co-mention proximity graphs from podcast transcripts.
2. Comparative evaluation of entity recognition and normalization strategies for informal speech.
3. Graph evaluation combining structural, semantic, and predictive analyses.
4. A framework for automatic labeling and interpretation of discourse communities.

## Tools and Resources

- **Entity Recognition & Linking:** spaCy, LUKE, DistilBERT-NER, Flair, BLINK, REL
- **Coreference Resolution:** AllenNLP, SpanBERT
- **Graph Analysis:** NetworkX, igraph, node2vec, GraphSAGE
- **Community Detection:** Louvain, Leiden, Infomap
- **Topic Modeling:** LDA, BERTopic
- **Salience Modeling:** Salience-BERT

_This project aims to uncover how people are semantically connected through conversation, offering a novel way to study community formation in discourse._

### View some graphs
- [Community](drafts/graph/community_graph.html)
- [Community using bert and manual ner processing](drafts/graph/bert_graph.html)
- [Category corelation (log scale)](drafts/graph/category_graph.html)
- [Episode level coocurences of speakers](drafts/graph/metadata_persons_graph.html)



[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/hgNAtOO3)

<img width="1110" height="574" alt="grafik" src="https://github.com/user-attachments/assets/907ebf33-01e6-4e0c-bea3-1285cdd59f6c" />
