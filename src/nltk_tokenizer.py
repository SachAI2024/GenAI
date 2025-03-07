


# import nltk
# from nltk.tokenize import word_tokenize
# nltk.download('punkt')

# text = """Artificial intelligence will change the way we think, \
# operate, and communicate. We believe that Artificial General \
# Intelligence, refered to as AGI, would be reached in the \
# next 5 to 7 years."""


# # start by converting the test to lower case.

# text = text.lower()
# print('Lowercase test:')
# print('-'*80)
# print(text)

# #tokenize text
# tokens = word_tokenize(text)
# #output the tokens

# print('-'*80)
# print('Tokenized text:')
# print('-'*80)
# print(tokens)


# Import libraries
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')

# User input text
# text = """Artificial intelligence will change the way we think, \
# operate, and communicate. We believe that Artificial General \
# Intelligence, refered to as AGI, would be reached in the \
# next 5 to 7 years."""

text = """ Shakespeare's poetic English is often characterized by a rich tapestry \
    of complex words, including archaic terms, vivid imagery, and deliberate \
    ambiguity, all woven together with a rhythmic structure that enhances the \
    emotional impact of his verse"""

# Start by converting the text to a lower case
text = text.lower()
print('Lowercase text:')
print('-'*80)
print(text)

# Tokenize text
tokens = word_tokenize(text)
# Output the tokens
print('-'*80)
print('Tokenized text:')
print('-'*80)
print(tokens)
# the challenges I had to face while building this was to install nltk 
    # I did this $pip3 install nltk and then I had to download the punkt package
    # I did this by running the code nltk.download('punkt') in the console.
    # I ran 3 commands in the console to get the nltk to work.
    # python3 
    # import nltk
    # nltk.download()
    # if the above commands didn't work then I had to run the following command
    # import nltk
    # nltk.download('punkt')

# then the nltk.download() didn't work for some reasone.