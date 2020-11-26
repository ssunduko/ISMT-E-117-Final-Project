import csv
import nltk
import spacy
import sklearn
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from pylab import *
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from spacy import displacy

import string
from spacy.lang.en.stop_words import STOP_WORDS

def list_of_clean_lemmas(sentence):
    clean_lemmas = []
    tokenizer = spacy.lang.en.English()
    token_list = tokenizer(sentence)

    for word in token_list:
        clean_lemmas.append(word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_)
    clean_list = [word for word in clean_lemmas if word not in STOP_WORDS
                  and word not in string.punctuation]

    print(clean_list)

def pos_tagging_and_display(sentence):

    tokenizer = spacy.load("en_core_web_sm")
    token_list = tokenizer(sentence)
    for token in token_list:
        print(token, token.tag_, token.pos_, spacy.explain(token.tag_))

    displacy.serve(token_list, style="dep")

def ner_labeling_and_display(sentence):

    tokenizer = spacy.load("en_core_web_sm")
    token_list = tokenizer(sentence)
    for token in token_list.ents:
        print(token.text, '->', token.label_)

    displacy.serve(token_list, style = "ent")


def data_preprocessing():
    # Read and cleanup data
    print ("Data Preprocessing ...")
    sentence = ("My very photogenic mother died in a freak accident (picnic, lightning) "
                               "when I was three, and, save for a pocket of warmth in the darkest past, "
                               "nothing of her subsists within the hollows and dells of memory, over "
                               "which, if you can still stand my style (I am writing under observation), "
                               "the sun of my infancy had set: surely, you all know those redolent "
                               "remnants of day suspended, with the midges, about some hedge in bloom "
                               "or suddenly entered and traversed by the rambler, at the bottom of a "
                               "hill, in the summer dusk; a furry warmth, golden midges.")
    list_of_clean_lemmas(sentence)
    pos_tagging_and_display(sentence)
    ner_labeling_and_display(sentence)


def final_project(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def sergey_nlp():
    print ("Sergey's Work")

def morgan_nlp():
    print ("Morgan's Work")

def norberto_nlp():
    print ("Norberto's Work")

def freeman_nlp():
    print ("Freeman's Work")

def rekha_nlp():
    print ("Rekha's Work")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    final_project('Welcome to Final Project')
    data_preprocessing()
    sergey_nlp()
    morgan_nlp()
    norberto_nlp()
    freeman_nlp()
    rekha_nlp()