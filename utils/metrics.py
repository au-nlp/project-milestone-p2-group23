import numpy as np

def score_ideas(episode_vecs, idea_vecs):
    scores = np.dot(episode_vecs, idea_vecs.T)
    return scores