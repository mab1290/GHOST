#GHOST: Genre, Happening, or Sentiment Tagger
#Michael Berezny, Pat Putnam, Sushant Kafle

import numpy as np
import pandas as pd
import nltk, string, time, csv, string
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from pprint import pprint
from collections import Counter
from sklearn.pipeline import Pipeline
from models.HandCraftedFeatureExtractor import HandCraftedFeatureExtractor

DATA_SRC = "PS3_training_data.txt"
DATA_HEADERS = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']
TASKS = ['GENRE', 'SENTIMENT', 'CATEGORY']

#DATA LOAD
dataframe = pd.read_csv(DATA_SRC, sep='\t', index_col=False, header=None, names=DATA_HEADERS)
print ("Number of data points: %d" % len(dataframe))
print ("Average length of the text (tentative): %f" % np.mean([len(x.split()) for x in dataframe['TEXT']]))
print ("(Sanity check) Sample output:", dataframe.iloc[2]['TEXT'])


#TRAIN-TEST SPLIT
train_data, test_data = model_selection.train_test_split(dataframe, test_size = 0.1)

print (train_data.shape)
vectorizer = HandCraftedFeatureExtractor()
X = vectorizer.fit_transform(train_data['TEXT'])
print (np.shape(X))

#training
kfold = model_selection.KFold(n_splits=10)
results = model_selection.cross_val_score(LinearSVC(), X, train_data['SENTIMENT'], cv=kfold)
mean_acc = results.mean()
print (mean_acc)

'''

pipeline = Pipeline ([
        ('features', FeatureUnion([
            ('tfidf', TfidfTransformer(norm='l2', use_idf = True)),
            ('hand', HandCraftedFeatureExtractor())
        ]),
        ('chi2', SelectKBest(chi2, k=1000)),
        ('clf', VotingClassifier(estimators=selected_classifiers[:], voting='hard', n_jobs=-1))
    ])

pipeline.fit(train_data['X'], train['Y'])'''



