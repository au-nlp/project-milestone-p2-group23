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