'''GHOST: Genre, Happening, or Sentiment Tagger
Michael Berezny, Pat Putnam, Sushant Kafle'''

"""Import"""
import csv
import string

"""Read Training Data"""
data=[]
with open("PS3_training_data.txt", "r") as f:
    reader=csv.reader(f,delimiter='\t')
    for case in reader:
        d={"ID":case[0], "Sentence":case[1], "Sentiment":case[2], "Event":case[3], "Genre":case[4]}
        data.append(d)

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Takes sentence string, returns dicitonary of features
def get_features(sentence):
    features = {}
    bag_of_words=sentence.translate(punc_strip).split(" ")
    unique_words=set(word.lower() for word in bag_of_words)
    features["Number of Tokens"] = len(bag_of_words)
    features["Number of Types"] = len(unique_words)
    

    return features
