from sklearn.base import BaseEstimator, TransformerMixin
from ghost_features import *

class HandCraftedFeatureExtractor(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def get_features(self, text):
        """Helper code to compute average word length of a name"""
        features = process_features(text)
        return list(features.values())

    def transform(self, texts, y=None):
        """The workhorse of this feature extractor"""
        return list(map(self.get_features, texts))

    def fit(self, df, y=None):
        """Returns `self` unless something different happens in train and test"""
        return self