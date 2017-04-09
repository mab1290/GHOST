#List of funcitons which return binary or numerical values given a sentence
'''Imports'''
import string, csv, re, os, sys
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag import StanfordNERTagger

parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, 'vendor')

sys.path.append(vendor_dir)

from textblob import TextBlob as TB
from textblob import Word

st = StanfordNERTagger(
    './vendor/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz',
    './vendor/stanford-ner-2016-10-31/stanford-ner.jar')

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Reads in list of positive and negative words
positive=[p.replace("\n", "") for p in open('positive-words.txt').readlines()[35:]]
negative=[n.replace("\n", "") for n in open('negative-words.txt').readlines()[35:]]

#NOT A FEATURE FUNCTION
#Used by other functions. Converts a Penn corpus tag into a Wordnet tag.
def _penn_to_wordnet(tag):
    if tag in ("NN", "NNS", "NNP", "NNPS"):
        return "n"
    if tag in ("JJ", "JJR", "JJS"):
        return "a"
    if tag in ("VB", "VBD", "VBG", "VBN", "VBP", "VBZ"):
        return "v"
    if tag in ("RB", "RBR", "RBS"):
        return "r"
    return None

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

#Checks if sentence has '$'
def has_dollar(s):
    if "$" in s:
        return 1
    else:
        return 0

#Checks if sentence has "money"
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

#Checks whether sentence contains a location
def has_location(s):
    loc = [blah for blah in st.tag(s.split()) if blah[1]=='LOCATION']
    if len(loc)>0:
        return 1
    else:
        return 0


#Checks whether sentece contains a quotatons
def has_quote(s):
    comp=re.compile(r'".+"')
    if comp.search(s):
        return 1
    else:
        return 0

#Returns number of characters
def chars(s):
    return len(s)

#Returns the percentagenumber of capital words
def cap_words(s):
    words = s.translate(punc_strip).split(" ")
    return len([word for word in words if word.isupper()])/len(words)

#Returns the percentage of capital letters
def cap_chars(s):
    return len([char for char in s if char.isupper()])/len(s)

#Checks for the "go" lemma, should recognize 'going', 'went', etc.
def has_go(s):
    lemmas=[]
    for wt in TB(s).tags:
        lemmas.append(Word(wt[0]).lemmatize(_penn_to_wordnet(wt[1])))
    if 'go' in lemmas:
        return 1
    else:
        return 0
    
#Checks whether sentece contains more than two adjacent repeated letters
#Doesn't really occur in English, could indicate exaggerated emotions e.g. "booooooring"
def has_repeats(s):
    comp=re.compile(r"(\w)\1{2,}")
    if comp.search(s):
        return 1
    else:
        return 0

#Returns the number of nouns in a sentences    
def count_nouns(s):
    return len([w for w in TB(s).tags if w[1]=="NN" or "NNS"])

#Returns the number of proper nouns in a sentences    
def count_Pnouns(s):
    return len([w for w in TB(s).tags if w[1]=="NNP" or "NNPS"])

#Returns the number of adjectives in a sentences    
def count_adj(s):
    return len([w for w in TB(s).tags if w[1]=="JJ" or "JJR"])

#Returns the number of superlatives in a sentences    
def count_super(s):
    return len([w for w in TB(s).tags if w[1]=="JJS"])


#Returns the number of verbs in a sentences    
def count_verb(s):
    return len([w for w in TB(s).tags if w[1]=="VB" or "VBZ" or "VBP" or "VBD" or "VBN" or "VBG"])

#Returns the number of personal pronouns in a sentences    
def count_pro(s):
    return len([w for w in TB(s).tags if w[1]=="PRP"])

#Returns the TextBlob sentiment polarity between -1 and 1 where -1 is the msot negative
def blob_sent(s):
    return TB(s).sentiment.polarity

'''
#This is just here so I can test functions
data=[]
with open("PS3_training_data.txt", "r") as f:
    reader=csv.reader(f,delimiter='\t')
    for case in reader:
        d={"ID":case[0], "Sentence":case[1], "Sentiment":case[2], "Event":case[3], "Genre":case[4]}
        data.append(d)
'''