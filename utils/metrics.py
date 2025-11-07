import numpy as np

def score_ideas(episode_vecs, idea_vecs):
    scores = np.dot(episode_vecs, idea_vecs.T)
    return scores

def get_daily_scores(df, score_column, agg):
    return df.groupby('date')[score_column].agg(agg).reset_index()