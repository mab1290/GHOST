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
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from utils import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from models.HandCraftedFeatureExtractor import HandCraftedFeatureExtractor
from collections import Counter


'''
Hyperparameter tuning for selected models
=========================================
Here we are looking for best set of parameters for each of the top model that we picked
for each of the task.
'''


DATA_SRC = "../PS3_training_data.txt"
DATA_HEADERS = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']
TASKS = ['GENRE', 'SENTIMENT', 'CATEGORY']

#DATA LOAD
dataframe = pd.read_csv(DATA_SRC, sep='\t', index_col=False, header=None, names=DATA_HEADERS)
print ("Number of data points: %d" % len(dataframe))
print ("Average length of the text (tentative): %f" % np.mean([len(x.split()) for x in dataframe['TEXT']]))
print ("(Sanity check) Sample output:", dataframe.iloc[2]['SENTIMENT'])

TASK = 'SENTIMENT'
#TRAIN-TEST SPLIT
train_data, test_data = model_selection.train_test_split(dataframe, test_size = 0.1)

#FOR CATEGORY LABELING TASK, TRAIN TEST DATA HAS TO BE SPECIAL
#train_data, test_data = model_selection.train_test_split(dataframe.loc[dataframe['GENRE'] == 'GENRE_A'], test_size = 0.1)
print ("Test data: %d" % len(test_data))
print ("Train data: %d" % len(train_data))

pipeline = Pipeline ([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('chi2', SelectKBest(chi2)),
    ('clf', RidgeClassifier())
])

#THIS NEEDS TO CHANGE OF EACH CLASSIFIER
'''
#SVM HYPERPARAMETERS
'clf__C': (1, 0.1, 0.01, 0.001)

#RIDGE 
'clf__tol': (0.1, 0.01, 0.001, 0.0001)

#NAIVE BAYES 
'clf__alpha': (0.1, 0.01, 0.001, 0.0001)

#SGD 
'clf__alpha': (0.00001, 0.000001),
'clf__penalty': ('l2', 'elasticnet'),
'clf__n_iter': (10, 50, 80),


vect__max_df': (0.5, 0.75),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'chi2__k': (1000, 2000, 3000),
'''

parameters = {
    'vect__max_df': (0.5, 0.75),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'tfidf__use_idf': (True, False),
    'tfidf__norm': ('l1', 'l2'),
    'chi2__k': (1000, 2000, 3000),
    'clf__tol': (0.1, 0.01, 0.001, 0.0001)
}

grid_search = GridSearchCV(pipeline, parameters, verbose=1)

print("Performing grid search for task (%s) ..." % TASK)
print("pipeline:", [name for name, _ in pipeline.steps])
print("parameters:")
pprint(parameters)
t0 = time.time()

grid_search.fit(list(map(clean_text, dataframe['TEXT'])), dataframe[TASK])
#grid_search.fit(dataframe['TEXT'], dataframe[TASK])
print("done in %0.3fs" % (time.time() - t0))
print()

print("Best score: %0.3f" % grid_search.best_score_)
print("Best parameters set:")
best_parameters = grid_search.best_estimator_.get_params()
for param_name in sorted(parameters.keys()):
    print("\t%s: %r" % (param_name, best_parameters[param_name]))
