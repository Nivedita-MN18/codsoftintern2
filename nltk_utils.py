import nltk
import numpy as np
from nltk.stem.porter import PorterStemmer

stem = PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def bag_of_words(tokenized_sentence, all_words):
    tokenized_sentence = [stem.stem(word.lower()) for word in tokenized_sentence]
    bag = np.zeros(len(all_words), dtype=np.float32)
    for idx, word in enumerate(all_words):
        if word in tokenized_sentence:
            bag[idx] = 1
    return bag
