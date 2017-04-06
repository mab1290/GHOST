#List of funcitons which return binary or numerical values given a sentence
'''Imports'''
import string
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
import csv

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Reads in list of positive and negative words
positive=[p.replace("\n", "") for p in open('positive-words.txt').readlines()[35:]]
negative=[n.replace("\n", "") for n in open('negative-words.txt').readlines()[35:]]

#Checks if sentence has any positive words  
def has_pos(s):
    words = s.translate(punc_strip).split(" ")
    if not set(words).isdisjoint(positive):
        return 1
    else:
        return 0

#Checks if sentence has any negative words
def has_neg(s):
    words = s.translate(punc_strip).split(" ")
    if not set(words).isdisjoint(negative):
        return 1
    else:
        return 0

#Returns percentage of positive words    
def percent_pos(s):
    words = s.translate(punc_strip).split(" ")
    return len([p for p in words if p in positive])/len(words)

#Returns percentage of negative words
def percent_neg(s):
    words = s.translate(punc_strip).split(" ")
    return len([n for n in words if n in negative])/len(words)

#Returns percentage of neutral words
def percent_neu(s):
    words = s.translate(punc_strip).split(" ")
    return (len(words)-len([p for p in words if p in positive])-len([n for n in words if n in negative]))/len(words)

#Returns Sentence length
def length(s):
    words = s.translate(punc_strip).split(" ")
    return len(words)

#Returns the number of "!" in a sentence
def count_exc(s):
    return len([c for c in s if c=="!"])

def has_dollar(s):
    if "$" in s:
        return 1
    else:
        return 0
    
def has_money(s):
    if "money" in s:
        return 1
    else:
        return 0

#Returns number of Named Entities
#http://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
def name_entities(s):
     chunked = ne_chunk(pos_tag(word_tokenize(s)))
     continuous_chunk = []
     current_chunk = []
     for i in chunked:
             if type(i) == Tree:
                     current_chunk.append(" ".join([token for token, pos in i.leaves()]))
             elif current_chunk:
                     named_entity = " ".join(current_chunk)
                     if named_entity not in continuous_chunk:
                             continuous_chunk.append(named_entity)
                             current_chunk = []
             else:
                     continue
     return len(continuous_chunk)
 

'''
#This is just here so I can test functions
data=[]
with open("PS3_training_data.txt", "r") as f:
    reader=csv.reader(f,delimiter='\t')
    for case in reader:
        d={"ID":case[0], "Sentence":case[1], "Sentiment":case[2], "Event":case[3], "Genre":case[4]}
        data.append(d)
'''