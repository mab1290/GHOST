'''GHOST: Genre, Happening, or Sentiment Tagger
Michael Berezny, Pat Putnam, Sushant Kafle'''

"""Import"""
import csv
import string

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
