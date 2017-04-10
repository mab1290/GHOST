from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction import DictVectorizer
from analysis.ghost_features import *

class HandCraftedFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass
        self.vect = DictVectorizer()

    def get_features(self, text):
        """Helper code to compute average word length of a name"""
        features = process_features(text)
        return features

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        dicts = list(map(self.get_features, texts))
        return self.vect.fit_transform(dicts).toarray()


    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self