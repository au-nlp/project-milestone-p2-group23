# Information flows, but can you catch it?

> Milestone P2 (proposal + proof of concept). This README states the project idea, feasibility, the current status and the plan. The final scope may be refined in P3.

## Abstract:
-----add
We want to explore how podcasts are linked by ideas and how ideas are linked by podcasts. As ideas propagate, their signals may or may not relate financial markets. Using podcast transcript, we embed episodes and user-defined(or t5 generated) "ideas" into the same semantic space. We are gonna be using "ideas" alot, so watch out! We then score each episode's idea relevance via cosine similarity and track that score over time within and across shows, hosts, and categories. This produces time series (idea intensity, or `idea_score` as in the code) which may or may not corelate with market data and graphs linking podcasts/hosts by shared ideas and linking ideas to each other when they co-occur.

Our proof-of-concept builds a pipeline for preprocessing, embedding, filtering and visualizing podcast-idea relations. We then align ideas with stocks(e.g. AAPL, BTC-USD) to explore correlations and lead-lag behaviour around key events (this is applied to the May-June interval, but the pipeline is time agnostic). It's hard to travel in this much amount of data. So.. the end goal is a practical, map to accompany your journey. It contains visual insights of which podcasts pick up which ideas, how fast, and how ideas connect communities. Now knowing better how the ideas are related, their dynamics and relations can anticipate market movements.

The interval of 2 months is good enough, as with all the information from one month, it is possible to predict certain movements from the upcomming month. You just need to analyze it well. 

## Contributions
- Practical tooling: one-notebook pipeline, cached embeddings, interactive graphs, and concise summaries.
- Market linkage: align idea intensity with asset returns to test simple predictive structures.

## Proposed additional datasets
Our dataset is already very very rich, containing 20gb of information. We are only adding the market data from the May-June interval as offered by `yfinance` library (it may change, doesn't really matter).

## Methods:
> The methods are briefely discussed as the code shows the flow better.

First of all, we have to introduce a lot more work we have done:

### Alternatives we considered:
TODO: Make this "Alternatives we considered" section more compact, without removing any of the actual information

We started by trying to extract the entities from the text which seemed to be people, with different methods, with the spacy library extracting `.label_ == PERSON`. Removing the stopwords definately helps, but merging longer length person seemd to be hard. Even with that fixed, you get a person like 'Joe Biden' or 'Joseph Robinette Biden Jr.'. Depending on the context, that reference may not be as obious, and that forms a problem in itself. We didn't want to invest any more time in this, as even with these extracted correctly, doing corelation based on how close they are to each other would involve some kind of euristic(such as being 3 sentences close) but this doesn't have any semantic meaning. So it would require some kind of model to do so, but this dataset doesn't seem to be born for this kind of task.

Trying to using more what exists in the dataset: inferredHosts, inferredGuests, gives us an easier time, plotting using the same setup. Inffered Guests and Hosts may not be that relevant from a global point of view, not many podcasters are that known anyways, and even with 100k episodes, it doesn't make the graph thaaat conex.
Some other easy things we could visualize is the category of the podcasts, and ofcourse they are really well connected.

Because the categories come from different sources(some may forgot to add important categories, or too many were applied, or really rare name used) (a total of 99 distinct categories), we trained our own category labeler, that could be used to smoothen the categories, by repredicting them. The categories are a simple deterministic way to filter things out, but currently having to many names is not that intuitive.


### Back to business

Analyzing the podcast(or the news).

First we check the trivial distribution of different columns, missing values etc.

Using this we can do some simple filtering.


Now, we use a semantic embeding to be able to compute once the meaning of the episodes(and store them on disk cuz google colab sucks and disconects us). Once we have these, we can compare and score them against fields of interests and topics of interest using cosine similarity.
Having these scores, we can sort the episodes descending for their relevance in that domain (and we can also store the sorted indexes as google colab sucks and disconects us :). This way we can restore the index. From these, we can pick only relevant episodes using a threshold. Now having these, we have a much smaller dataset which is much more managable.

To understand even more the data and visualize it, we can cluster or compute whatever graphs we want. Some nice interactive .html pages have been done, as well as some more resource efficient graph representaion. These are used to gain more intuition on the data, as the shortest path using any weight function can be used and ploted.

Because podcasts and hosts are not so different from episodes, we can aggregate grouping by these, and get some semantic embedding for podcasts and what a host talks in their podcast. If we try to push and advertisment, we can see who we have to talk with, who has the most relevance in that field. (a view count column in the dataset would've been very powerfull as well)

Also, currently only for better data viewing, we have a helper t5 model which can do summaries or extract the most important idea from an episode. Esentially, when extracting some important episodes/podcasts with whatever method, using this method can make us understanding what that episode is about.

To get a better representation of the now filtered dataset, a sentiment analyzer model can be used to detect whether or not an episode has a positive/negative intention wrt an idea. Maybe we chose to filter for "Pharmacies" and now we want to actually see if their opinion is good or not about them. To do some, we can get the chunks which match our specific idea "Should Pharmacies have more advertisement", and if any, get their sentiment score or classification.

For this specific task, would could train a model which gets "partial podcast transcript" and "idea" and gives a score on how good the idea would fit to be talked at that podcast or how probable would that be. To train such a model, we need data. For an episode or fraction of the episode where the "idea" doesn't appear (input) we can see if in other episodes that idea appears, if so, we have the true label being 1 (that means, yes, that idea would fit the podcast(because in the feature it will actually be used)).

Some key events we would like to follow:

**May 2020 – Key Events**

1. COVID-19 Pandemic Continues
2. George Floyd’s Death and Global Protests (May 25)
3. Hong Kong Security Law Announcement (May 21–28)
4. SpaceX Crew Dragon Launch (May 30)

**June 2020 – Key Events**

1. Worldwide BLM Protests Continue
2. COVID-19 Surges and Border Restrictions
3. WHO Warns Pandemic Is “Accelerating” (June 21)
4. India–China Border Clash (June 15)
5. Global Economic Concerns

Now we have the powerfull tool to query for a specific idea and get all related podcasts.
We analyze analysis can use the plotting of the intensity(score) of the idea over time to corelate with stocks of firms.
Using these, we can plot them and find corelations with stocks of firms.


It's impossible to follow all the news and listen to all podcast. Now as you can detect new trends from their transcript can make you focus on what matter.

### Timeline:

- Play around with different aggregation methods for episode embedding.
- Play around with different embedding models.
- Consider training a model to extract more impactful/recent information rather than content from boring discussion.
- Exclude low information days from trading
- Because the code is general enough, we can iterate over ideas generated by the t5 model and see which have more or less corelation with different traded instruments("AAPL", "BTC-USD" etc.). (Instead of searching with ideas by hand). Play around with bigger t5 models and with the prompt ofcourse. 
- Have better plots(different metrics) denoting the corelation and the "tradability" of the market according to our information.

### Organization within the team.
As you know, we lost our team mate 'Inaki' :(. We got a couple meetings with him showing how to use python, but we couldn't change the faith in the end. We even had 1-1 sessions. Anyways, the organization needs and will be much better from now on. Because the project was at the begining, it fastly developed and didn't allow all the members to contribute the same. Following the timeline, each week will have a meeting in the weekend were we show that we achieved the intermediate goals.

# Appendix

### Repo structure:

- main.ipynb - Well, the main jupiter notebook
- assets/ - Constains static sutff, like images/icons for a better main.ipynb preview
- checkpoints/ - Model weights/snapshots
- drafts/ - A lot of .ipynbs which do things separately, used for testing around, they make or may not make it to the final main.ipynb
- features/ - Contains preprocessed versioned features for stability and reusability.
- utils/ - Contains a bunch of .py files with helper functions to make the main.ipynb clean.
