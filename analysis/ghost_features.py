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

#For removing punctuation
punc_strip = str.maketrans('', '', string.punctuation)

#Reads in list of positive and negative words
positive=[p.replace("\n", "") for p in open('../vendor/positive-words.txt').readlines()[35:]]
negative=[n.replace("\n", "") for n in open('../vendor/negative-words.txt').readlines()[35:]]

QUOTES_REGEX = re.compile(r'".+"')
REPEATS_REGEX = re.compile(r"(\w)\1{2,}")

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
def _has_pos(words):
    if not set(words).isdisjoint(positive):
        return 1
    else:
        return 0

#Checks if sentence has any negative words
def _has_neg(words):
    if not set(words).isdisjoint(negative):
        return 1
    else:
        return 0

#Returns category based on the percentage of positive words    
def _percent_pos(words, len_of_words):
    pp = len([p for p in words if p in positive])/len_of_words
    if pp==0:
        return "percent_pos_no_pos"
    elif pp<0.25:
        return "percent_pos_some_pos"
    else:
        return "percent_pos_much_pos"

#Returns category based on the percentage of negative words
def _percent_neg(words, len_of_words):
    pn=len([n for n in words if n in negative])/len_of_words
    if pn==0:
        return "percent_neg_no_neg"
    elif pn<0.25:
        return "percent_neg_some_neg"
    else:
        return "percent_neg_much_neg"

#Returns category based on the percentage of neutral words
def _percent_neu(words, len_of_words):
    pneu=(len(words)-len([p for p in words if p in positive])-len([n for n in words if n in negative]))/len_of_words
    if pneu==1:
        return "percent_neu_all_neu"
    elif pneu>0.5:
        return "percent_neu_mostly_neu"
    else:
        return "percent_neu_less_neu"

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


#Checks whether sentece contains a quotatons
def _has_quote(s):
    if QUOTES_REGEX.search(s):
        return 1
    else:
        return 0

#Returns number of characters
def _chars(s):
    return len(s)

#Returns the number of capital words
def _cap_words(words):
    return len([word for word in words if word.isupper()])

#Returns the number of capital letters
def _cap_chars(s):
    return len([char for char in s if char.isupper()])

#Checks for the "go" lemma, should recognize 'going', 'went', etc.
def _has_go(tags):
    lemmas=[]
    for wt in tags:
        lemmas.append(Word(wt[0]).lemmatize(_penn_to_wordnet(wt[1])))

    if 'go' in lemmas:
        return 1
    else:
        return 0
    
#Checks whether sentece contains more than two adjacent repeated letters
#Doesn't really occur in English, could indicate exaggerated emotions e.g. "booooooring"
def _has_repeats(s):
    if REPEATS_REGEX.search(s):
        return 1
    else:
        return 0

#Returns the number of nouns in a sentences    
def _count_nouns(tags):
    return len([w for w in tags if w[1]=="NN" or "NNS"])

#Returns the number of proper nouns in a sentences    
def _count_Pnouns(tags):
    return len([w for w in tags if w[1]=="NNP" or "NNPS"])

#Returns the number of adjectives in a sentences    
def _count_adj(tags):
    return len([w for w in tags if w[1]=="JJ" or "JJR"])

#Returns the number of superlatives in a sentences    
def _count_super(tags):
    return len([w for w in tags if w[1]=="JJS"])

#Returns the number of verbs in a sentences    
def _count_verb(tags):
    return len([w for w in tags if w[1]=="VB" or "VBZ" or "VBP" or "VBD" or "VBN" or "VBG"])

#Returns the number of personal pronouns in a sentences    
def _count_pro(tags):
    return len([w for w in tags if w[1]=="PRP"])

#Returns the TextBlob sentiment polarity between -1 and 1 where -1 is the msot negative
def _blob_sent(s):
    sent=TB(s).sentiment.polarity
    if sent>0.5:
        return "blob_sent_mostly_pos"
    elif sent>0.0:
        return "blob_sent_pos"
    elif sent==0.0:
        return "blob_sent_neu"
    elif sent>-0.5:
        return "blob_sent_neg"
    else:
        return "blob_sent_mostly_neg"


def process_features(string):
    tags = TB(string).tags
    words = string.translate(punc_strip).split(" ")
    len_of_words = len(words)

    features = {}

    # numerical
    features['has_pos_word'] = _has_pos(words)
    features['has_neg_word'] = _has_neg(words)
    features['length'] = len_of_words
    features['count_exclamation_mark'] = _count_exc(string)
    features['has_dollar_sign'] = _has_dollar(string)
    features['contains_word_money'] = _has_money(string)
    features['count_of_named_entities'] = _name_entities(string)
    features['has_quotations'] = _has_quote(string)
    features['count_of_characters'] = _chars(string)
    features['percent_upper_case_words'] = _cap_words(words)
    features['percent_upper_case_chars'] = _cap_chars(string)
    features['has_go_lemma'] = _has_go(tags)
    features['has_repeated_letters'] = _has_repeats(string)
    features['count_of_nouns'] = _count_nouns(tags)
    features['count_of_proper_nouns'] = _count_Pnouns(tags)
    features['count_of_adjectives'] = _count_adj(tags)
    features['count_of_superlatives'] = _count_super(tags)
    features['count_of_verbs'] = _count_verb(tags)
    features['count_of_pronouns'] = _count_pro(tags)

    # categorical
    features[_blob_sent(string)] = 1
    features[_percent_pos(words, len_of_words)] = 1
    features[_percent_neg(words, len_of_words)] = 1
    features[_percent_neu(words, len_of_words)] = 1
    
    return features
