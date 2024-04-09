import pickle
import pandas as pd
import torch
import os
import contractions


# contracted text
text = '''Ill be there within 5 min. Shouldn't you be there too? 
          I'd love to see u there my dear. It's awesome to meet new friends.
          We've been waiting for this day for so long.'''
 
# creating an empty list
expanded_words = []    
for word in text.split():
  # using contractions.fix to expand the shortened words
  expanded_words.append(contractions.fix(word))   
   
print(expanded_words)
expanded_text = ' '.join(expanded_words)
print('Original text: ' + text)
print('Expanded_text: ' + expanded_text)