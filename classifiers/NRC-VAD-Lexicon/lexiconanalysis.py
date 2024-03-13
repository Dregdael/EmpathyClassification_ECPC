import numpy as np
import torch
import pandas as pd
import re
import nltk


#To obtain lemmatize words
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
#To eliminate stop words
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
#To expand contractions
#from pycontractions import Contractions
#To get keywords

def clean_string(string2clean):
    string2clean = re.sub("_comma_", ',', string2clean)
    string2clean = re.sub("[^0-9a-zA-Z']+", ' ', string2clean)
    return string2clean.lower()

def string2array(string2convert):
    arr = string2convert.split()
    for i in arr:
        if i in stop_words:
            arr.remove(i)
        if type(i) == 'int':
            arr.remove(i)
    return arr

def setup_lexicon():
    #df = pd.read_csv("NRC-VAD-Lexicon.txt", sep='\t', header=None) #use 0 to 1 scale
    lexicon_df = pd.read_csv("BipolarScale/NRC-VAD-Lexicon.txt", sep='\t', header=None) #use -1 to 1 scale
    lexicon_df = lexicon_df.rename(columns={0:'word',1:'valence', 2:'arousal',3:'dominance'})
    return lexicon_df

def get_avg_vad(wrd_lst,lexicon_df,lmtzr):
    valence = 0 
    arousal = 0
    dominance = 0
    words_in_lex = 0 
    for word in wrd_lst:
        lemma = lmtzr.lemmatize(word)
        if len(lexicon_df.loc[lexicon_df['word'] == lemma,])>0:
            #print(df.loc[df['word'] == lemma,].index[0])
            index_of_word = lexicon_df.loc[lexicon_df['word'] == lemma,].index[0]
            #print(f'{word} was found in the lexicon!')
            arousal += lexicon_df.loc[index_of_word,'arousal']
            valence += lexicon_df.loc[index_of_word,'valence']
            dominance += lexicon_df.loc[index_of_word,'dominance']
            words_in_lex += 1
    if words_in_lex == 0:
        #no words were found in the lexicon, so setup neutral value
        #avg_vad = [0.5,0.5,0.5] #0 to 1 scale
        avg_vad = [0,0,0] #-1 to 1 scale
    else:
        #get average vad vectors from the words in the lexicon
        avg_vad = [arousal/words_in_lex,valence/words_in_lex,dominance/words_in_lex]
    return avg_vad

test_str = "Oh_comma_ that sounds awful! Some people have zero respect for others belonging and even less for things we all need to share."
test_str = clean_string(test_str)
word_lst = string2array(test_str)
#print(f'word list: {wrd_lst}')
df = setup_lexicon()
vad = get_avg_vad(word_lst,df,wnl)
print(vad)






#print(df.loc[df['word'] == 'admissibility',])
#print(df.loc[1020:1040])