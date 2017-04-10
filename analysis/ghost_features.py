#List of funcitons which return binary or numerical values given a sentence
'''Imports'''
import string, re, os, sys
from nltk import ne_chunk, pos_tag, word_tokenize
from nltk.tree import Tree
from nltk.tag import StanfordNERTagger

parent_dir = os.path.abspath(os.path.dirname(__file__))
vendor_dir = os.path.join(parent_dir, '../vendor')

sys.path.append(vendor_dir)

from textblob import TextBlob as TB
from textblob import Word

st = StanfordNERTagger(
    './vendor/stanford-ner-2016-10-31/classifiers/english.all.3class.distsim.crf.ser.gz',
    './vendor/stanford-ner-2016-10-31/stanford-ner.jar')

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Reads in list of positive and negative words
positive=[p.replace("\n", "") for p in open('../vendor/positive-words.txt').readlines()[35:]]
negative=[n.replace("\n", "") for n in open('../vendor/negative-words.txt').readlines()[35:]]

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
def _has_pos(s):
    words = s.translate(punc_strip).split(" ")
    if not set(words).isdisjoint(positive):
        return 1
    else:
        return 0

#Checks if sentence has any negative words
def _has_neg(s):
    words = s.translate(punc_strip).split(" ")
    if not set(words).isdisjoint(negative):
        return 1
    else:
        return 0

#Returns category based on the percentage of positive words    
def _percent_pos(s):
    words = s.translate(punc_strip).split(" ")
    pp = len([p for p in words if p in positive])/len(words)
    if pp==0:
        return "No Positive"
    elif pp<0.25:
        return "Some Positive"
    else:
        return "Much Positive"

#Returns category based on the percentage of negative words
def _percent_neg(s):
    words = s.translate(punc_strip).split(" ")
    pn=len([n for n in words if n in negative])/len(words)
    if pn==0:
        return "No Negative"
    elif pn<0.25:
        return "Some Negative"
    else:
        return "Much Negative"

#Returns category based on the percentage of neutral words
def _percent_neu(s):
    words = s.translate(punc_strip).split(" ")
    pneu=(len(words)-len([p for p in words if p in positive])-len([n for n in words if n in negative]))/len(words)
    if pneu==1:
        return "All Neutral"
    elif pneu>0.5:
        return "Mostly Neutral"
    else:
        return "Less than Mostly Neutral"

#Returns Sentence length
def _length(s):
    words = s.translate(punc_strip).split(" ")
    return len(words)

#Returns the number of "!" in a sentence
def _count_exc(s):
    return len([c for c in s if c=="!"])

#Checks if sentence has '$'
def _has_dollar(s):
    if "$" in s:
        return 1
    else:
        return 0

#Checks if sentence has "money"
def _has_money(s):
    if "money" in s:
        return 1
    else:
        return 0

#Returns number of Named Entities
#http://stackoverflow.com/questions/31836058/nltk-named-entity-recognition-to-a-python-list
def _name_entities(s):
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
def _has_location(s):
    loc = [blah for blah in st.tag(s.split()) if blah[1]=='LOCATION']
    if len(loc)>0:
        return 1
    else:
        return 0


#Checks whether sentece contains a quotatons
def _has_quote(s):
    comp=re.compile(r'".+"')
    if comp.search(s):
        return 1
    else:
        return 0

#Returns number of characters
def _chars(s):
    return len(s)

#Returns the number of capital words
def _cap_words(s):
    words = s.translate(punc_strip).split(" ")
    return len([word for word in words if word.isupper()])

#Returns the number of capital letters
def _cap_chars(s):
    return len([char for char in s if char.isupper()])

#Checks for the "go" lemma, should recognize 'going', 'went', etc.
def _has_go(s):
    lemmas=[]
    for wt in TB(s).tags:
        lemmas.append(Word(wt[0]).lemmatize(_penn_to_wordnet(wt[1])))
    if 'go' in lemmas:
        return 1
    else:
        return 0
    
#Checks whether sentece contains more than two adjacent repeated letters
#Doesn't really occur in English, could indicate exaggerated emotions e.g. "booooooring"
def _has_repeats(s):
    comp=re.compile(r"(\w)\1{2,}")
    if comp.search(s):
        return 1
    else:
        return 0

#Returns the number of nouns in a sentences    
def _count_nouns(s):
    return len([w for w in TB(s).tags if w[1]=="NN" or "NNS"])

#Returns the number of proper nouns in a sentences    
def _count_Pnouns(s):
    return len([w for w in TB(s).tags if w[1]=="NNP" or "NNPS"])

#Returns the number of adjectives in a sentences    
def _count_adj(s):
    return len([w for w in TB(s).tags if w[1]=="JJ" or "JJR"])

#Returns the number of superlatives in a sentences    
def _count_super(s):
    return len([w for w in TB(s).tags if w[1]=="JJS"])

#Returns the number of verbs in a sentences    
def _count_verb(s):
    return len([w for w in TB(s).tags if w[1]=="VB" or "VBZ" or "VBP" or "VBD" or "VBN" or "VBG"])

#Returns the number of personal pronouns in a sentences    
def _count_pro(s):
    return len([w for w in TB(s).tags if w[1]=="PRP"])

#Returns the TextBlob sentiment polarity between -1 and 1 where -1 is the msot negative
def _blob_sent(s):
    sent=TB(s).sentiment.polarity
    if sent>0.5:
        return "Very Positive"
    elif sent>0.0:
        return "Positive"
    elif sent==0.0:
        return "Neutral"
    elif sent>-0.5:
        return "Negative"
    else:
        return "Very Negative"


def process_features(string):
    features = {}
    features['has_pos_word'] = _has_pos(string)
    features['has_neg_word'] = _has_neg(string)
    features['percent_pos_words'] = _percent_pos(string)
    features['percent_neg_words'] = _percent_neg(string)
    features['percent_neu_words'] = _percent_neu(string)
    features['length'] = _length(string)
    features['count_exclamation_mark'] = _count_exc(string)
    features['has_dollar_sign'] = _has_dollar(string)
    features['contains_word_money'] = _has_money(string)
    #features['count_of_named_entities'] = _name_entities(string)
    features['has_location'] = _has_location(string)
    features['has_quotations'] = _has_quote(string)
    features['count_of_characters'] = _chars(string)
    features['percent_upper_case_words'] = _cap_words(string)
    features['percent_upper_case_chars'] = _cap_chars(string)
    features['has_go_lemma'] = _has_go(string)
    features['has_repeated_letters'] = _has_repeats(string)
    features['count_of_nouns'] = _count_nouns(string)
    features['count_of_proper_nouns'] = _count_Pnouns(string)
    features['count_of_adjectives'] = _count_adj(string)
    features['count_of_superlatives'] = _count_super(string)
    features['count_of_verbs'] = _count_verb(string)
    features['count_of_pronouns'] = _count_pro(string)
    #features['textblob_sentiment_polarity'] = _blob_sent(string)
    return features