# Information flows, but can you catch it?

> Milestone P2 (proposal + proof of concept). This README states the project idea, feasibility, the current status and the plan. The final scope may be refined in P3.

## Abstract:
We want to explore how podcasts are linked by ideas and how ideas are linked by podcasts. As ideas propagate, their signals may or may not relate financial markets. Using podcast transcript, we embed episodes and user-defined(or t5-generated) "ideas" into the same semantic space. We are gonna be using "ideas" alot, so watch out! We then score each episode's idea relevance via cosine similarity and track that score over time within and across shows, hosts, and categories. This produces time series (idea intensity, or `idea_score` as in the code) which may or may not corelate with market data and graphs linking podcasts/hosts by shared ideas and linking ideas to each other when they co-occur.

 It's hard to travel in this much amount of data. So.. the end goal is a practical, map to accompany your journey. It contains visual insights of which podcasts pick up which ideas, how fast, how they connect. Now knowing better how the ideas are related, their dynamics and relations can anticipate market movements.

## Contributions
- Practical tooling: one-notebook pipeline, cached embeddings, interactive graphs, and concise summaries.
- Market linkage: align idea intensity with asset returns to test simple predictive structures.

## Proposed additional datasets
Our dataset is already very rich, containing 20gb compressed-info. We are only adding the market data from the May-June interval as offered by `yfinance` library (it may change, doesn't really matter).

## Methods:
> The methods are briefely discussed as the code shows the flow better.

First of all, we have to introduce a lot more work we have done:

### Alternatives we considered:
TODO: Make this "Alternatives we considered" section more compact, without removing any of the actual information

We started by trying to extract the entities from the text which seemed to be people, with different methods, with the spacy library extracting `.label_ == PERSON`. Removing stopwords. Aditionally, comparing 'Joe Biden' or 'Joseph Robinette Biden Jr.' is a problem in itself. We didn't want to invest any more time in this.

Using only the information form the dataset: `inferredHosts`, `inferredGuests`, gives us an easier time, plotting using the same setup. Looking at the graphs, they are not that connected so these columns are not that interesting this way.

VIsualize category of the podcasts, and ofcourse they are really well connected.

Because the categories come from different sources(distinct namings, sparse, etc.) (99 distinct categories), we trained our own category labeler, that could be used to smoothen the categories, by repredicting them. The categories are a simple deterministic way to filter things out. - `drafts/category_cls.ipynb`


### Back to business, current workflow:

- Data prep: Inspect column distributions, missing values, and apply basic filters.
- Semantic indexing: Compute and persist episode embeddings (avoid Colab session resets). Score episodes against topics/fields via cosine similarity, sort by relevance, store sorted indices, and keep only items above a threshold for a smaller, manageable dataset. (embeddings can now be computed on mutiple GPUs!!)
- Exploration & viz: Cluster and build graph-based views (interactive HTML + lightweight graphs). Use shortest-path (with chosen edge weights) to probe relationships.
- Aggregation: Group by podcast and host to derive podcast/host-level embeddings. Use these to identify high-relevance partners (e.g., for ads). A "view-count" column would further improve targeting.
- Summarization: A helper T5 model produces summaries/key ideas to quickly grasp selected episodes.
- Sentiment(optinal): For a chosen idea (e.g., “Pharmacies”), retrieve matching chunks and assess sentiment to understand stance (positive/negative/neutral).
- Idea-fit model (optional): Train a model taking (partial transcript, idea) → probability the idea fits that podcast. Weak labels: if the idea appears in other episodes of the same podcast, mark as positive.
- Use some reference events (for timelines/tests):
    - May 2020: COVID-19 continues; George Floyd protests (May 25); HK security law (May 21–28); SpaceX Crew Dragon (May 30).
    - June 2020: Ongoing BLM protests; COVID-19 surges; WHO: pandemic “accelerating” (Jun 21); India–China clash (Jun 15); global economic concerns.

### Outputs

- Query any idea → return relevant podcasts/episodes, visualized as graphs.
- Plot idea “intensity” (relevance scores) over time and compare with company stock series to explore correlations.

### Why it matters:
 - Scales beyond manual news/podcast monitoring and surfaces emerging trends directly from transcripts.

> It's impossible to follow all the news and listen to all podcast. Now as you can detect new trends from their transcript, you focus on what matter.

### Timeline:

- Play around with different aggregation methods for episode embedding. (15nov)
- Play around with different embedding models. (22nov)
- Exclude low information days from trading (22nov)
- Because the code is general enough, we can iterate over ideas generated by the t5 model and see which have more or less corelation with different traded instruments("AAPL", "BTC-USD" etc.). (Instead of searching with ideas by hand). Play around with bigger t5 models and with the prompt ofcourse. (29nov)
- Have better plots(different metrics) denoting the corelation and the "tradability" of the market according to our information. (6nov)
- Consider training a model to extract more impactful/recent information rather than content from boring discussion. (13dec)
- There are also some `TODO`s in the utils/ files, we plan on moving the number down. They usually involve code structure or experiments (allthetime)
- Just chill coding and imporving (until 19 dec)

### Organization within the team.
As you know, we lost our team mate 'Inaki' :(. We got a couple meetings with him showing how to use python, but we couldn't change the faith in the end. Anyways, the organization needs and will be much better from now on. Because the project was at the begining, it fastly developed and didn't allow all the members to contribute the same. Following the timeline, each week will have a meeting in the weekend were we show that we achieved the intermediate goals.

# Appendix

### Repo structure:

- main.ipynb - Well, the main jupiter notebook
- assets/ - Constains static sutff, like images/icons for a better main.ipynb preview
- checkpoints/ - Model weights/snapshots
- drafts/ - A lot of .ipynbs which do things separately, used for testing around, they make or may not make it to the final main.ipynb
- features/ - Contains preprocessed versioned features for stability and reusability.
- utils/ - Contains a bunch of .py files with helper functions to make the main.ipynb clean.
