import numpy as np
import spacy

class MeanEmbeddingVectorizer(object):
    def __init__(self, model):
        self.nlp = model
        #!!change this to something else later
        self.dim = len(self.nlp('apple').vector)

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            self.nlp(sent).vector for sent in X
        ])