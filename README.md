## Title: People connected by ideas and ideas connected by people.

>> NOTE! I will rewrite this whole README.md, as it was written in one go without language in mind.

I created the `main.ipynb` which definately needs more love in Data Visualization and Preprocessing:

When describing the relevant aspects of the data, and any other datasets you may intend to use, you should in particular show (non-exhaustive list):

    That you can handle the data in its size.
    That you understand what’s in the data (formats, distributions, missing values, correlations, etc.).
    That you considered ways to enrich, filter, transform the data according to your needs.
    That you have a reasonable plan and ideas for methods you’re going to use, giving their essential mathematical details in the notebook.
    That your plan for analysis and communication is reasonable and sound, potentially discussing alternatives to your choices that you considered but dropped.


Abstract:
We would like to really understand the connection in the podcast ecosystem, that is, how information makes different podcasts similar.
TODO: Complete me


### Alternatives we considered:

We started by trying to extract the entities from the text which seemed to be people, with different methods, with the spacy library extracting `.label_ == PERSON`. Removing the stopwords definately helps, but merging longer length person seemd to be hard. Even with that fixed, you get a person like 'Joe Biden' or 'Joseph Robinette Biden Jr.'. Depending on the context, that reference may not be as obious, and that forms a problem in itself. We didn't want to invest any more time in this, as even with these extracted correctly, doing corelation based on how close they are to each other would involve some kind of euristic(such as being 3 sentences close) but this doesn't have any semantic meaning. So it would require some kind of model to do so, but this dataset doesn't seem to be born for this kind of task.

Trying to using more what exists in the dataset: inferredHosts, inferredGuests, gives us an easier time, plotting using the same setup. Inffered Guests and Hosts may not be that relevant from a global point of view, not many podcasters are that known anyways, and even with 100k episodes, it doesn't make the graph thaaat conex.
Some other easy things we could visualize is the category of the podcasts, and ofcourse they are really well connected.

Because the categories come from different sources(some may forgot to add important categories, or too many were applied, or really rare name used), we trained our own category labeler, that could be used to smoothen the categories, by repredicting them. Even tho they may not be that accurate from time to time, it could be used as a good metric to link episodes/podcasts or even people.

The graphs can be viewed here:

- [Community](https://raw.githack.com/3lv/nlp_public_files/main/graph/community_graph.html)
- [Community using bert and manual ner processing](https://raw.githack.com/3lv/nlp_public_files/main/graph/bert_graph.html)
- [Category corelation (log scale)](https://raw.githack.com/3lv/nlp_public_files/main/graph/category_graph.html)
- [Episode level coocurences of speakers](https://raw.githack.com/3lv/nlp_public_files/main/graph/metadata_persons_graph.html)


### Back to business


Ideas discussed in podcasts is what brings listeners together. So we want to analyze how podcasts are grouped together by different ideas (not just topics/classes). How they react to new topics. Which one is the most up to date.
Using a latent space representation of the topics would give us nothing more than some words that define the entire podcast category. We want to see what is discussed in an episode specifically, not some general topics. To do so, we want to see how appealing an "idea" or "query" is to some podcast/person. By training an sentence embedder, we can use these to compare using cosine similarity the relevance of the idea with reference to a specific epsiode. Now, having this, for a given podcast/group of podcasts(grouped by guests, categories etc.), we can see how an ideas relevance changes through time. Because of the very small gap that the episodes were recorded(2 month) only very specific ideas will show an interest in the curve, as they can rise and fall.

The finality of this project is having a network connecting different podcats/categories of podcasts by ideas, and connecting ideas with podcast. With this research, it would enable to detect what ideas bring people together(categories) and which podcast are the most relevant for connecting two categoties.

Less abstract:
Create a graph that has podcasts as nodes and some kind of inverse of the content similarity as the weight of the edges. Now you can find the shortest path from one podcast to a different podcast. Now, as you know the path, you also know what contents are on the path, and these are the ones you need to aproach more if you were to shift your audience to the dirrection of a different podcast.
Adding to the content similarity, because we have a global view, we can scale it according to how fast it got adopted, or for how long relative to the others.




### Retake on flow of the main.ipynb and the project itself:


Analyzing the podcast(or the news).

First we check the dumb distribution of different columns, missing values etc.

Using this we can do some simple filtering.


Now, we use a semantic embeding to be able to compute once the meaning of the episodes(and store them on disk cuz google colab sucks and disconects us). Once we have these, we can compare and score them against fields of interests and topics of interest using cosine similarity.
Having these scores, we can sort the episodes descending for their relevance in that domain (and we can also store the sorted indexes as google colab sucks and disconects us :). This way we can restore the index. From these, we can pick only relevant episodes using a threshold. Now having these, we have a much smaller dataset which is much more managable.

To understand even more the data and visualize it, we can cluster or compute whatever graphs we want. Some nice interactive .html pages have been done, as well as some more resource efficient graph representaion. These are used to gain more intuition on the data, as the shortest path using any weight function can be used and ploted.

Because podcasts and hosts are not so different from episodes, we can aggregate grouping by these, and get some semantic embedding for podcasts and what a host talks in their podcast. If we try to push and advertisment, we can see who we have to talk with, who has the most relevance in that field. (a view count column in the dataset would've been very powerfull as well)

Also, currently only for better data viewing, we have a helper t5 model which can do summaries or extract the most important idea from an episode. Esentially, when extracting some important episodes/podcasts with whatever method, using this method can make us understanding what that episode is about.

`cardiffnlp/twitter-roberta-base-sentiment-latest`
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
Continuation analysis can use the pllotting of the intensity(score) of the idea over time to corelate with stocks of firms.
Using these, we can plot them and find corelations with stocks of firms.


It's impossible to follow all the news and listen to all podcast. Now as you can detect new trends from their transcript can make you focus on what matter.


### Repo structure:

- main.ipynb - Well, the main jupiter notebook
- assets/ - Constains statis sutff, like images/icons for a better main.ipynb preview
- checkpoints/ - Model weights/snapshots
- drafts/ - A lot of .ipynbs which do things separately, used for testing around, they make or may not make it to the final main.ipynb
- features/ - Contains preprocessed versioned features for stability and reusability.
- utils/ - Contains a bunch of .py files with helper functions to make the main.ipynb clean.

TODO: Write the timeline