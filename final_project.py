import csv
from itertools import islice

import nltk
import spacy
from collections import Counter
import seaborn as sns
import sklearn
import re
from csv import DictReader

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from sklearn.ensemble import RandomForestClassifier
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

def classification_display(list_of_sentences):
    print("In Classification")

def sentiment_display(list_of_sentences):
    print("In Sentiment")

def reason_display(list_of_sentences):
    print("In Reason")

def adjectives_display(list_of_sentences):
    print("In Adjectives")

def intention_vs_action_display(list_of_sentences):
    print("In Intention")

def important_factors_display(list_of_sentences):

    d = getcwd()

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(list_of_sentences)

    lsa = TruncatedSVD(25, algorithm='randomized')
    dtm_lsa = lsa.fit_transform(tfidf)
    dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    sing_vecs = lsa.components_[0]
    index = np.argsort(sing_vecs).tolist()
    index.reverse()

    terms = [tfidf_vectorizer.get_feature_names()[weightIndex] for weightIndex in index[0:10]]
    weights = [sing_vecs[weightIndex] for weightIndex in index[0:10]]

    terms.reverse()
    weights.reverse()

    plt.barh(terms, weights, align="center")
    plt.savefig(path.join(d, 'resources/word_weight.jpg'), dpi=200)
    plt.show()

def cosine_similarity_display(document_corpus, sentence):

    document_corpus.append(sentence)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    sparse_matrix = tfidf_vectorizer.fit_transform(document_corpus)

    cosine = cosine_similarity(sparse_matrix[0:1], sparse_matrix)

    circle1 = plt.Circle((0, 0), 1, alpha=.5)
    plt.ylim([-1.1, 1.1])
    fig = plt.gcf()
    fig.gca().add_artist(circle1)

    for i in range(len(cosine)):
        for j in range(len(cosine[i])):
            d = 2 * 1 * (1 - cosine[0][j])
            circle2 = plt.Circle((d, 0), 1, alpha=.5)
            plt.xlim([-1.1, 1.1 + d])
            fig.gca().add_artist(circle2)

    plt.savefig('ISMT-E-117-Final-Project/resources/cosine_overlap.jpg')
    plt.show()

def list_of_clean_tokens(sentence):

    tokenizer = spacy.lang.en.English()
    token_list = tokenizer(sentence)

    token_list = [word.lower_.strip() for word in token_list if word.is_alpha and not word.is_stop]

    return token_list


def list_of_clean_lemmas(sentence):

    tokenizer = spacy.load('en_core_web_sm')
    token_list = tokenizer(' '.join(map(str, list_of_clean_tokens(sentence))))

    token_list = [word.lemma_ for word in token_list if word.lemma_ != "-PRON-"]

    return token_list

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

    d = getcwd()
    freq = Counter(sentence.split(" "))
    sns.set_style("darkgrid")
    words = [word[0] for word in freq.most_common(10)]
    count = [word[1] for word in freq.most_common(10)]

    plt.figure(figsize=(10, 10))

    sns_bar = sns.barplot(x=words, y=count)
    sns_bar.set_xticklabels(words, rotation=90)
    plt.title('Most Common Words')
    plt.savefig(path.join(d, 'resources/word_count.jpg'), dpi=200)
    plt.show()

def word_cloud_display(sentence):

    d = getcwd()
    mask = np.array(Image.open(path.join(d, "resources/family-gathering.png")))
    image_colors = ImageColorGenerator(mask)


    wc = WordCloud(background_color="white", max_words=200, width=400, height=400,
                 mask=mask, random_state=1).generate(sentence)

    plt.figure(figsize=[7, 7])
    plt.imshow(wc.recolor(color_func=image_colors), interpolation="bilinear")
    plt.axis("off")
    plt.imshow(wc.recolor(color_func=image_colors))
    plt.savefig(path.join(d, 'resources/word_cloud.jpg'), dpi=200)
    plt.show()

def data_preprocessing():

    print ("Data Preprocessing ...")
    tokenizer = spacy.load("en_core_web_sm")

    with open('data/data_set.csv', 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in islice(csv_dict_reader, 50):
            tokens = ' '.join(map(str, list_of_clean_tokens(row['tweet'])))
            lemmas = ' '.join(map(str, list_of_clean_lemmas(row['tweet'])))
            clean_token_sentences.append(tokens)
            clean_lemma_sentences.append(lemmas)
            clean_tokens.append(tokenizer(tokens))
            dirty_sentences.append(row['tweet'])


def final_project(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def sergey_nlp():
    print("Sergey's Work")

    data_preprocessing()
    clean_tokens_strings = ' '.join(map(str, clean_tokens))

    most_common_words_display(clean_tokens_strings)
    cosine_similarity_display(clean_lemma_sentences,' '.join(map(str, list_of_clean_lemmas("I want to kill myself"))))
    word_cloud_display(clean_tokens_strings)
    important_factors_display(clean_token_sentences)
    pos_tagging_and_display(clean_tokens_strings)
    ner_labeling_and_display(clean_tokens_strings)

def morgan_nlp():
    print ("Morgan's Work")

def norberto_nlp():
    print("Norberto's Work")

    classification_display(clean_lemma_sentences)


def freeman_nlp():
    print ("Freeman's Work")

    reason_display(clean_lemma_sentences)
    adjectives_display(clean_lemma_sentences)
    intention_vs_action_display(clean_lemma_sentences)

def rekha_nlp():
    print ("Rekha's Work")
    sentiment_display(clean_lemma_sentences)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    final_project('Welcome to Final Project')

    dirty_sentences = []
    clean_token_sentences = []
    clean_lemma_sentences = []
    clean_tokens = []
    clena_lemmas = []

    sergey_nlp()
    morgan_nlp()
    norberto_nlp()
    freeman_nlp()
    rekha_nlp()