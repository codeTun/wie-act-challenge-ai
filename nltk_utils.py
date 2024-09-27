import nltk
from nltk.stem.porter import PorterStemmer
import numpy as np

nltk.download('punkt')
nltk.download('punkt_tab')


#First step 
def tokenize(sentence):
    return nltk.word_tokenize(sentence) # split sentence into words

#Second step
stemmer = PorterStemmer()
def stem(word):
    return stemmer.stem(word.lower()) # convert word to lower case and find the root word

#Third step
def bag_of_words(tokenized_sentence, words):
    
    sentence_words = [stem(word) for word in tokenized_sentence]
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1.0
    return bag



