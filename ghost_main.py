#GHOST: Genre, Happening, or Sentiment Tagger
#Michael Berezny, Pat Putnam, Sushant Kafle

import numpy as np
import pandas as pd
import nltk, string, time, csv, string


#This might give some of you an import error
#Try updating the scikit-learn module (pip install -U scikit-learn)
#to fix the issue
#If that doesn't work, try:
#from sklearn.cross_validation import train_test_split
#but, don't push it version of the code to the repo.
from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import MultinomialNB

from collections import Counter

DATA_SRC = "PS3_training_data.txt"
DATA_HEADERS = ['ID', 'TEXT', 'SENTIMENT', 'CATEGORY', 'GENRE']

#DATA LOAD
###This comment can be removed once read!
#The data is loaded into a pandas dataframe. It's a neat way to store the data
#You can query the dataframe pretty easily, use the column names above (DATA_HEADERS)
#to access the columns. You operate on the result like any other python list.
dataframe = pd.read_csv(DATA_SRC, sep='\t', index_col=0, header=None, names=DATA_HEADERS)

print ("Number of data points: %d" % len(dataframe))
print ("Average length of the text (tentative): %f" % np.mean([len(x.split()) for x in dataframe['TEXT']]))
print ("(Sanity check) Sample output:", dataframe.iloc[2]['TEXT'])


#TRAIN-TEST SPLIT
##The test_data will not be touched during any part of training and tunning.
#It will be used in the end to get our final evaluation.
#Meanwhile, the train_data can futher be split up into training and validation dataset,
#if we need.
#The purpose of validation dataset would be to tune the model on some extra data
#that the model hasn't seen before.
train_data, test_data = model_selection.train_test_split(dataframe, test_size = 0.1)


#DATA PREPROCESSING
##We should be careful to run this preprocessing module, because if we are to extract some
#features like NER, Dependency Parsing etc. we will need proper sentence structure.
#I have thought of using following criterions:
#1. No symbols except symbols with semantic meaning like apostrope (') (not single quotes), dollar ($),
# & sign, % sign, - (hyphen), ? mark. If possible, normalize some exceptions such as replacing '&' with 'and' etc. 
#2. Replace numbers with "N" tag.
#3. Lowercase text
#4. 

'''
DATA PREPROCESSING CODE HERE
#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

"""Read Training Data"""
data=[]
with open("PS3_training_data.txt", "r") as f:
    reader=csv.reader(f,delimiter='\t')
    for case in reader:
        d={"ID":case[0], "Sentence":case[1], "Sentiment":case[2], "Event":case[3], "Genre":case[4]}
        data.append(d)

voc=[]
for d in data:
    for word in d["Sentence"].translate(punc_strip).split(" "):
        voc.append(word)
vocabulary= set(voc)

#Takes sentence string, returns dicitonary of features
def get_features(sentence):
    features = {}
    bag_of_words=sentence.translate(punc_strip).split(" ")
    unique_words=set(word.lower() for word in bag_of_words)
    features["Number of Tokens"] = len(bag_of_words)
    features["Number of Types"] = len(unique_words)
    for word in vocabulary:
        features['Has (%s)' % word] = (word in bag_of_words)
    return features

print(get_features(data[0]["Sentence"]))
'''


#FEATURE EXTRACTION
##This process could later be changed to word2vec transform.
#Lets work with Tfidf scores for now.
#Notice the TfidfVectorizer ignores the list of stop words in english
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
X_train = vectorizer.fit_transform(train_data['TEXT'])
X_test = vectorizer.transform(test_data['TEXT'])


###SOMETHING TO DECIDE ON:
#From this point, the code extract has to be "TASK" specific
#We need to decide on how we want to structure this.
#For now, I have just selected our "TASK" to be GENRE

TASK = 'GENRE'
y_train = train_data[TASK]
y_test = test_data[TASK]

#FEATURE REDUCTION
##Since we are working with words, and their counts (tfdif) the feature vector
#will be quite sparse (a lot of zeros). It is helpful to reduce the feature vector
#to something more condensed. We can use chi-squared test to filter out the feature
#vectors. This reduces model completexity and also increases model performance. If 
#we want to be fancy, we can also try LDA based feature reduction technique.
#Notice, the num_features variable - it defines the number of features we want to use.
#We might want to spend some time figuring out the right number of features.
num_features = 100
chi_squared = SelectKBest(chi2, k=num_features)
X_train = chi_squared.fit_transform(X_train, y_train)
X_test = chi_squared.transform(X_test)

feature_info = vectorizer.get_feature_names()
feature_names = [feature_info[i] for i in chi_squared.get_support(indices=True)]

print ("Selected Features: ")
print (feature_names)

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
			   ("Bagging Classifier (Decision Tree)", BaggingClassifier(base_estimator=DecisionTreeClassifier(), n_estimators=100, random_state=seed))]
			   #("Random Forest Classifier", RandomForestClassifier(n_estimators=100, max_features=10)),\
			   #("Gradient Boosting Classifier", GradientBoostingClassifier(n_estimators=100, random_state=seed)),\
			   #("Ridge Classifier", RidgeClassifier(tol=1e-2, solver="lsqr"))]


#MAJORITY VOTING CLASSIFIER
#This model is our "ultimate" model ;) This looks into predictions from all the classifiers
#and outputs the majority class.
#NOTICE "estimators=classifiers[:]", we should be careful here. "classifiers[:]" is used to aviod
#the infinite recursion. This does the hard copy of lists. In python, copy of list is "reference copy"
#by default.
MajorityVotingClassifier = VotingClassifier(estimators=classifiers[:], voting='hard')
classifiers.append(("Majority Voting Classifier", MajorityVotingClassifier))

#MODELS TRAINING AND EVALUATION
for name, classifier in classifiers:
	print ("Working on %s" % name)
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	results = model_selection.cross_val_score(classifier, X_train, y_train, cv=kfold)
	print("K-fold cross-validation accuracy (mean): %f" % results.mean())





