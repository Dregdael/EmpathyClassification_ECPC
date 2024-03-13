import numpy as np
import torch
import pandas as pd

import nltk
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

from nltk.corpus import wordnet


df = pd.read_csv("NRC-VAD-Lexicon.txt", sep=' ')
print(df)