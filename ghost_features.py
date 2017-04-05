#List of funcitons which return binary or numerical values given a sentence
'''Imports'''
import string

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Reads in list of positive and negative words
positive=[p.replace("\n", "") for p in open('positive-words.txt').readlines()[35:]]
for p in positive:
    p.replace('\n', '')

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
    
def percent_pos(s):
    words = s.translate(punc_strip).split(" ")
    return len([p for p in words if p in positive])/len(words)

def percent_neg(s):
    words = s.translate(punc_strip).split(" ")
    return len([n for n in words if n in negative])/len(words)

def percent_neu(s):
    words = s.translate(punc_strip).split(" ")
    return (len(words)-len([p for p in words if p in positive])-len([n for n in words if n in negative]))/len(words)
    