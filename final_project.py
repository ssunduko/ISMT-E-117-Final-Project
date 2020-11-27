import csv
import nltk
import spacy
from collections import Counter
import seaborn as sns
import sklearn
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

from pylab import *
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans
from spacy import displacy

import re
import io
from wordcloud import WordCloud
from PIL import Image
import PIL.ImageOps
from wordcloud import ImageColorGenerator
import string
import unicodedata
from nltk.corpus import stopwords
import nltk
import spacy
import random
from os import path, getcwd
from PIL import Image

import string
from spacy.lang.en.stop_words import STOP_WORDS

def cosine_similarity_display(first_sentence, second_sentence):

    clean_lemmas = []
    first_lemmas = ' '.join(map(str, list_of_clean_lemmas(first_sentence)))
    second_lemmas = ' '.join(map(str, list_of_clean_lemmas(second_sentence)))

    clean_lemmas.append(first_lemmas)
    clean_lemmas.append(second_lemmas)

    clean_lemmas_array = TfidfVectorizer().fit_transform(clean_lemmas).toarray()

    first_lemmas_vector = clean_lemmas_array[0].reshape(1,-1)
    second_lemmas_vector = clean_lemmas_array[1].reshape(1, -1)

    print(cosine_similarity(first_lemmas_vector,second_lemmas_vector)[0][0])

    return cosine_similarity(first_lemmas_vector,second_lemmas_vector)[0][0]

def list_of_clean_lemmas(sentence):
    clean_lemmas = []
    tokenizer = spacy.lang.en.English()
    token_list = tokenizer(sentence)

    for word in token_list:
        clean_lemmas.append(word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_)
    clean_list = [word for word in clean_lemmas if word not in STOP_WORDS
                  and word not in string.punctuation]

    print(clean_list)

    return clean_list

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

def most_common_words_display(sentence):

    freq = Counter(' '.join(map(str, list_of_clean_lemmas(sentence))).split(" "))
    sns.set_style("darkgrid")
    words = [word[0] for word in freq.most_common(10)]
    count = [word[1] for word in freq.most_common(10)]

    plt.figure(figsize=(10, 10))

    sns_bar = sns.barplot(x=words, y=count)
    sns_bar.set_xticklabels(words, rotation=90)
    plt.title('Most Common Words')
    plt.show()

def word_cloud_display(sentence):

    d = getcwd()
    mask = np.array(Image.open(path.join(d, "resources/family-gathering.png")))
    image_colors = ImageColorGenerator(mask)


    wc = WordCloud(background_color="white", max_words=200, width=400, height=400,
                 mask=mask, random_state=1).generate(' '.join(map(str, list_of_clean_lemmas(sentence))))

    plt.figure(figsize=[7, 7])
    plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.savefig(path.join(d, 'resources/wordcloud.jpg'), dpi=200)
    plt.show()

def data_preprocessing():
    print ("Data Preprocessing ...")

    list_of_clean_lemmas(first_sentence)


def final_project(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def sergey_nlp():
    print("Sergey's Work")

    pos_tagging_and_display(first_sentence)
    ner_labeling_and_display(first_sentence)
    most_common_words_display(first_sentence)
    cosine_similarity_display(first_sentence, second_sentence)
    word_cloud_display(first_sentence)

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
    first_sentence = ("My very photogenic mother died in a freak accident (picnic, lightning) "
                      "when I was three, and, save for a pocket of warmth in the darkest past, "
                      "nothing of her subsists within the hollows and dells of memory, over "
                      "which, if you can still stand my style (I am writing under observation), "
                      "the sun of my infancy had set: surely, you all know those redolent "
                      "remnants of day suspended, with the midges, about some hedge in bloom "
                      "or suddenly entered and traversed by the rambler, at the bottom of a "
                      "hill, in the summer dusk; a furry warmth, golden midges.")
    second_sentence = ("Kind of similar but not really surely, you all know those redolent "
                       "remnants of day suspended, with the midges, about some hedge in bloom "
                       "or suddenly entered and traversed by the rambler, at the bottom ")

    data_preprocessing()
    sergey_nlp()
    morgan_nlp()
    norberto_nlp()
    freeman_nlp()
    rekha_nlp()