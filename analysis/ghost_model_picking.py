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

DATA_SRC = "../PS3_training_data.txt"
DATA_HEADERS = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']
TASKS = ['GENRE', 'SENTIMENT', 'CATEGORY']

#DATA LOAD
dataframe = pd.read_csv(DATA_SRC, sep='\t', index_col=False, header=None, names=DATA_HEADERS)
print ("Number of data points: %d" % len(dataframe))
print ("Average length of the text (tentative): %f" % np.mean([len(x.split()) for x in dataframe['TEXT']]))
print ("(Sanity check) Sample output:", dataframe.iloc[2]['TEXT'])


#TRAIN-TEST SPLIT
train_data, test_data = model_selection.train_test_split(dataframe, test_size = 0.1)
#FOR CATEGORY LABELING TASK, TRAIN TEST DATA HAS TO BE SPECIAL
train_data_cat, test_data_cat = model_selection.train_test_split(dataframe.loc[dataframe['GENRE'] == 'GENRE_A'], test_size = 0.1)

vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

for task in TASKS:
    print ("Working on %s task" % task)

    if task == 'CATEGORY':
        y_train = train_data_cat[task]
        y_test = test_data_cat[task]

        X_train = vectorizer.fit_transform(train_data_cat['TEXT'])
        X_test = vectorizer.transform(test_data_cat['TEXT'])

    else:            
        y_train = train_data[task]
        y_test = test_data[task]

        X_train = vectorizer.fit_transform(train_data['TEXT'])
        X_test = vectorizer.transform(test_data['TEXT'])

    #FEATURE REDUCTION
    ##Since we are working with words, and their counts (tfdif) the feature vector
    #will be quite sparse (a lot of zeros). It is helpful to reduce the feature vector
    #to something more condensed. We can use chi-squared test to filter out the feature
    #vectors. This reduces model completexity and also increases model performance. If
    #we want to be fancy, we can also try LDA based feature reduction technique.
    #Notice, the num_features variable - it defines the number of features we want to use.
    #We might want to spend some time figuring out the right number of features.
    num_features = 1500
    chi_squared = SelectKBest(chi2, k=num_features)
    X_train = chi_squared.fit_transform(X_train, y_train)
    X_test = chi_squared.transform(X_test)

    feature_info = vectorizer.get_feature_names()
    feature_names = [feature_info[i] for i in chi_squared.get_support(indices=True)]

    #This is the seed we use for randomization during cross validation.
    #It can basically be any number. But, what this will mean is that, no two results
    #might be same.
    seed = 10

    #LIST OF MODEL THAT WE WANT TO USE
    #We have all the model that we want to use listed in the 'classifiers' variable
    #For now we are just looking that their mean cross-validation accuracy. We can easily look into more
    #informative metrics like F1-score, confusing metrics etc. But, cv-accuracy should be enough, I think!
    #Here, I have commented some to save some time. You can try adding more and un-commmenting these codes.
    #NOTE: You might need to import the models.
    classifiers = [("Multinomial Naive Bayes Classifier", MultinomialNB(alpha=.01)),\
                   #("Bagging Classifier (Decision Tree)", BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed)),\
                   ("SVM", SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=5, random_state=42)),\
                   ("Random Forest Classifier", RandomForestClassifier(n_estimators=100, max_features=10)),\
                   ("Ridge Classifier", RidgeClassifier(tol=1e-2, solver="sag")),\
                   ("Logistic Regression", LogisticRegression(solver="sag", multi_class="multinomial")),\
                   ("Linear SVC", LinearSVC(random_state=seed)),\
                   ("Perceptron", Perceptron(random_state=seed))]


    #MAJORITY VOTING CLASSIFIER
    #This model is our "ultimate" model ;) This looks into predictions from all the classifiers
    #and outputs the majority class.
    #NOTICE "estimators=classifiers[:]", we should be careful here. "classifiers[:]" is used to aviod
    #the infinite recursion. This does the hard copy of lists. In python, copy of list is "reference copy"
    #by default.
    MajorityVotingClassifier = VotingClassifier(estimators=classifiers[:], voting='hard', n_jobs=-1)
    classifiers.append(("Majority Voting Classifier", MajorityVotingClassifier))

    #X_train_dense = X_train.toarray()
    #MODELS TRAINING AND EVALUATION
    best_classifier = None
    best_accuracy = -1
    for name, classifier in classifiers:
        print ("Working on %s" % name)
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        results = model_selection.cross_val_score(classifier, X_train, y_train, cv=kfold)
        mean_acc = results.mean()
        print("K-fold cross-validation accuracy (mean): %f" % mean_acc)
        if mean_acc > best_accuracy:
            best_classifier = classifier
            best_accuracy = mean_acc






