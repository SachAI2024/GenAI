import gensim
import numpy as np
import nltk 

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

nltk.download('punkt')

text = """ Who would have thought that computer programs would be analyzing 
human sentiments and human emotions like human brain. As technology advances, we can expect more 
sophisticated AI systems that can perform increasingly complex tasks.
Artificial intelligence will change the way we think, \
operate, and communicate. We believe that Artificial General \
Intelligence, refered to as AGI, would be reached in the \
next 5 to 7 years.
"""

#convert the text to lowercase. 

lower_case_text = text.lower()
tokenized_text = word_tokenize(lower_case_text)

model = Word2Vec([tokenized_text], min_count=1, vector_size=50, workers=9, sg=1, 
                 negative=1, epochs=40)

#sg = 1 means skip-gram is training algorithm, negative = 1 means negative sampling

train = model.train([tokenized_text], 
                    total_examples=model.corpus_count,
                    epochs=model.epochs)

print(model.wv['programs'])
print('-'*80) # printing '-' 80 times.

print(sorted(model.wv.most_similar('programs',topn=3),reverse=True))


