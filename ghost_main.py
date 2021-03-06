#GHOST: Genre, Happening, or Sentiment Tagger
#Michael Berezny, Pat Putnam, Sushant Kafle

import numpy as np
import pandas as pd
import nltk, string, time, csv, string, os, sys
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, RidgeClassifier, LogisticRegression, Perceptron
from sklearn.svm import LinearSVC
from pprint import pprint
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.externals import joblib
from utils.utils import *

DATA_SRC = "PS3_training_data.txt"
DATA_HEADERS = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']
TASKS = ['GENRE', 'SENTIMENT', 'CATEGORY']
CLASSIFIERS = {'GENRE': 'classifiers/genre.clf.pkl', 'SENTIMENT': 'classifiers/sentiment.clf.pkl',
            'CATEGORY': 'classifiers/category.clf.pkl'}


#TRAINING DATA LOAD
dataframe = pd.read_csv(DATA_SRC, sep='\t', index_col=False, header=None, names=DATA_HEADERS)
cdataframe = dataframe.loc[dataframe['GENRE'] == 'GENRE_A']

#TEST DATA SRC
assert len(sys.argv) > 1 , "Error\npython ghost_main.py <test data src> <output filename>"
TEST_DATA_SRC = sys.argv[1]
OUTFILE_SRC = sys.argv[2]


#PREPARE/LOAD THE MODEL
genre_clf = None
sentiment_clf = None
category_clf = None

if os.path.isfile(CLASSIFIERS['GENRE']):
    print ("Loading the genre classifier")
    genre_clf = joblib.load(CLASSIFIERS['GENRE']) 
    print ("Loading complete!")
else:
    print ("Preparing the genre classifier")
    Xs = clean_texts(dataframe['TEXT'])
    Ys = dataframe['GENRE']

    selected_classifiers = [("Multinomial Naive Bayes Classifier", MultinomialNB(alpha=.1)),\
                   ("Ridge Classifier", RidgeClassifier(tol=1e-1, solver="sag")),\
                   ("SVM", LinearSVC())]

    goto_pipeline = Pipeline ([
        ('vect', CountVectorizer(max_df=0.5, ngram_range=(1,2))),
        ('tfidf', TfidfTransformer(norm='l2', use_idf = True)),
        ('chi2', SelectKBest(chi2, k=3000)),
        ('clf', VotingClassifier(estimators=selected_classifiers[:], voting='hard', n_jobs=-1))
    ])

    goto_pipeline.fit(Xs, Ys)
    joblib.dump(goto_pipeline, CLASSIFIERS['GENRE']) 
    print("Classifier saved at %s" % CLASSIFIERS['GENRE'])

    print ("Loading the genre classifier")
    genre_clf = joblib.load(CLASSIFIERS['GENRE']) 
    print ("Loading complete!")


if os.path.isfile(CLASSIFIERS['SENTIMENT']):
    print ("Loading the sentiment classifier")
    sentiment_clf = joblib.load(CLASSIFIERS['SENTIMENT']) 
    print ("Loading complete!")
else:
    print ("Preparing the sentiment classifier")
    Xs = clean_texts(dataframe['TEXT'])
    Ys = dataframe['SENTIMENT']

    selected_classifiers = [("Multinomial Naive Bayes Classifier", MultinomialNB(alpha=.1)),\
                   ("Ridge Classifier", RidgeClassifier(tol=0.01, solver="sag")),\
                   ("SVM", LinearSVC(C=1))]

    goto_pipeline = Pipeline ([
        ('vect', CountVectorizer(max_df=0.75, ngram_range=(1,1))),
        ('tfidf', TfidfTransformer(norm='l2', use_idf = True)),
        ('chi2', SelectKBest(chi2, k=3000)),
        ('clf', VotingClassifier(estimators=selected_classifiers[:], voting='hard', n_jobs=-1))
    ])

    goto_pipeline.fit(Xs, Ys)
    joblib.dump(goto_pipeline, CLASSIFIERS['SENTIMENT']) 
    print("Classifier saved at %s" % CLASSIFIERS['SENTIMENT'])

    print ("Loading the sentiment classifier")
    sentiment_clf = joblib.load(CLASSIFIERS['SENTIMENT']) 
    print ("Loading complete!")


if os.path.isfile(CLASSIFIERS['CATEGORY']):
    print ("Loading the category classifier")
    category_clf = joblib.load(CLASSIFIERS['CATEGORY']) 
    print ("Loading complete!")
else:
    print ("Preparing the category classifier")
    Xs = clean_texts(cdataframe['TEXT'])
    Ys = cdataframe['CATEGORY']

    selected_classifiers = [('Stocastic Gradient Descent Classifier', SGDClassifier(penalty="elasticnet", n_iter=10, alpha=0.000001)),\
                   ("Ridge Classifier", RidgeClassifier(solver="sag", tol=0.1)),\
                   ("SVM", LinearSVC(C=1))]

    goto_pipeline = Pipeline ([
        ('vect', CountVectorizer(max_df=0.5, ngram_range=(1,1))),
        ('tfidf', TfidfTransformer(norm='l2', use_idf = True)),
        ('chi2', SelectKBest(chi2, k=1000)),
        ('clf', VotingClassifier(estimators=selected_classifiers[:], voting='hard', n_jobs=-1))
    ])

    goto_pipeline.fit(Xs, Ys)
    joblib.dump(goto_pipeline, CLASSIFIERS['CATEGORY']) 
    print("Classifier saved at %s" % CLASSIFIERS['CATEGORY'])

    print ("Loading the category classifier")
    category_clf = joblib.load(CLASSIFIERS['CATEGORY']) 
    print ("Loading complete!")


#Run the classifiers on the test data and save it.
#Loading the test data into the dataframe
test_dataframe = pd.read_csv(TEST_DATA_SRC, sep='\t', index_col=False, header=None, names=DATA_HEADERS)
test_cdataframe = test_dataframe.loc[test_dataframe['GENRE'] == 'GENRE_A']

# Evalation 
print ("Calculating the training accuracy of the models")
pred_genre = genre_clf.predict(test_dataframe['TEXT'])
pred_sentiment = sentiment_clf.predict(test_dataframe['TEXT'])
pred_category = category_clf.predict(test_cdataframe['TEXT'])

test_eval_genre = sum(1 for x,y in zip(pred_genre, test_dataframe['GENRE']) if x.strip() == y.strip()) / len(pred_genre)
test_eval_sentiment = sum(1 for x,y in zip(pred_sentiment, test_dataframe['SENTIMENT']) if x.strip() == y.strip()) / len(pred_sentiment)
test_eval_category = sum(1 for x,y in zip(pred_category, test_cdataframe['CATEGORY']) if x.strip() == y.strip()) / len(pred_category)
print ("Genre: %f" % test_eval_genre)
print ("Sentiment: %f" % test_eval_sentiment)
print ("Category: %f" % test_eval_category)


# Output
output_dataframe = test_dataframe[['ID', 'TEXT', 'GENRE']].copy()
output_dataframe.rename(columns={'GENRE':'GIVEN_GENRE'}, inplace=True)

# Add 'NONE' for items in GENRE_B
for index, row in output_dataframe.iterrows():
    if row['GIVEN_GENRE'] == 'GENRE_B':
        pred_category = np.insert(pred_category, index, 'NONE')

output_dataframe[DATA_HEADERS[2]] = pred_sentiment
output_dataframe[DATA_HEADERS[3]] = pred_category
output_dataframe[DATA_HEADERS[4]] = pred_genre

output_dataframe.drop(['GIVEN_GENRE'], axis=1, inplace=True)

output_dataframe.to_csv(OUTFILE_SRC, sep='\t', header=False, index=False, encoding='utf-8')
