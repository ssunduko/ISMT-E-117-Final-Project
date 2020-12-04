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

from textblob import TextBlob
from textblob.sentiments import NaiveBayesAnalyzer

from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split


def classification_display(list_of_sentences):
    print("In Classification")

def sentiment_display():

    suicide_dataset = read_dataset()[3000:4000]
    emotions_dataset = read_emotion_dataset()

    tfidf_vect, tfidf = tfidf_vectorizer(emotions_dataset['cleaned'])
    X_features = pd.concat([pd.DataFrame(tfidf.toarray())], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_features, emotions_dataset['emotion_label'], test_size=0.2)
    nb_tfidf = naive_bayes_model(X_train, X_val, y_train, y_val)
    sgd_tfidf = sgd_classifier_model(X_train, X_val, y_train, y_val)
    logreg_tfidf = logistic_regression_model(X_train, X_val, y_train, y_val)
    rf_tfidf = random_forest_model(X_train, X_val, y_train, y_val)

    count_vect, count = count_vectorizer(emotions_dataset['cleaned'])
    X_features = pd.concat([pd.DataFrame(count.toarray())], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_features, emotions_dataset['emotion_label'], test_size=0.2)
    nb_count = naive_bayes_model(X_train, X_val, y_train, y_val)
    sgd_count = sgd_classifier_model(X_train, X_val, y_train, y_val)
    logreg_count = logistic_regression_model(X_train, X_val, y_train, y_val)
    rf_count = random_forest_model(X_train, X_val, y_train, y_val)

    suicide_dataset['nb_tfidf'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(nb_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['sgd_tfidf'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(sgd_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['logreg_tfidf'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(logreg_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['rf_tfidf'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(rf_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['nb_count'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(nb_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['sgd_count'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(sgd_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['logreg_count'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(logreg_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['rf_count'] = suicide_dataset['cleaned'].apply(lambda x: value_to_emotions(rf_count.predict(count_vect.transform([x]))[0]))

    suicide_dataset.to_csv('resources/testset_emotions.csv',sep=',')
    print ("Test Set data is available in resources/testset_emotions.csv")

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

def random_forest_similarity_display(sentence):

    suicide_dataset = read_dataset()[3000:4000]

    tfidf_vect, tfidf = tfidf_vectorizer(suicide_dataset['tweet'])
    X_features = pd.concat([pd.DataFrame(tfidf.toarray())], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_features, suicide_dataset['label'], test_size=0.2)

    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_train, y_train)

    label = {0: 'Rejected', 1: 'Confirmed'}
    X = tfidf_vect.transform([sentence])

    print('Prediction: %s\nAccuracy: %.2f%%'
          % (label[rf.predict(X)[0]], np.max(rf.predict_proba(X)) * 100))

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

    plt.savefig('resources/cosine_overlap.jpg')
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

def read_dataset():
    data = pd.read_csv('data/data_set.csv')
    data['cleaned'] = data['tweet'].apply(lambda x: clean_sentences(x))
    data['sentiments'] = data['cleaned'].apply(lambda x: sentiment_analysis(x))
    return data

def read_emotion_dataset():
    data = pd.read_csv('data/dataset_emotions.csv')
    data['cleaned'] = data['tweet'].apply(lambda x: clean_sentences(x))
    data['sentiments'] = data['cleaned'].apply(lambda x: sentiment_analysis(x))
    data['emotion_label'] = data['emotion'].apply(lambda x: emotions_to_index(x))
    return data

def emotions_to_index(label):
    generic_emotions = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6, 'unknown': 7}
    if (label in generic_emotions.keys()):
        return (generic_emotions[label])
    else: return (generic_emotions['unknown'])

def value_to_emotions(val):
    generic_emotions = {'anger': 0, 'disgust': 1, 'fear': 2, 'joy': 3, 'neutral': 4, 'sadness': 5, 'surprise': 6, 'unknown': 7}
    for key, value in generic_emotions.items():
        if val == value:
            return key
    return 'unknown'

def clean_sentences(sentence):
    noNumbers = ''.join([item for item in sentence if not item.isnumeric()])
    noPunct = ''.join([item for item in noNumbers if item not in string.punctuation])
    lowerWords = noPunct.lower()
    return (lowerWords)

def tfidf_vectorizer(data):
    print ("TF-IDF Vectorizer")
    tfidf_vect = TfidfVectorizer()
    X_train_tfidf = tfidf_vect.fit_transform(data)
    return tfidf_vect, X_train_tfidf

def count_vectorizer(data):
    print ("Count Vectorizer")
    count_vect = CountVectorizer()
    X_train_count = count_vect.fit_transform(data)
    return count_vect, X_train_count

def print_predictions(yval, y_pred, algorithm):
    print('%s accuracy %s'.format(algorithm, accuracy_score(y_pred, yval)))
    precision, recall, fscore, support = score(yval, y_pred, average='weighted')
    print ("Precision: {} Recall: {} Accuracy: {}".format(round(precision,2),
                                                            round(recall,2),
                                                            round((y_pred==yval).sum()/len(y_pred), 2)))

def naive_bayes_model(xtrain, xval, ytrain, yval):
    nb = MultinomialNB()
    nb.fit(xtrain, ytrain)
    y_pred = nb.predict(xval)
    print_predictions(yval, y_pred, 'Naive Bayes Model')
    return nb

def sgd_classifier_model(xtrain, xval, ytrain, yval):
    lsvm = SGDClassifier(alpha=0.001, random_state=5, max_iter=15, tol=None)
    lsvm.fit(xtrain, ytrain)
    y_pred = lsvm.predict(xval)
    print_predictions(yval, y_pred, 'SGD Classifier')
    return lsvm

def logistic_regression_model(xtrain, xval, ytrain, yval):
    logreg = LogisticRegression(C=1)
    logreg.fit(xtrain, ytrain)
    y_pred = logreg.predict(xval)
    print_predictions(yval, y_pred, 'Logistic Regression')
    return logreg

def random_forest_model(xtrain, xval, ytrain, yval):
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(xtrain, ytrain)
    y_pred = rf.predict(xval)
    print_predictions(yval, y_pred, 'Random Forest Classifier')
    return rf

def sentiment_analysis(review):
    # 0 - 'Negative', 1 - 'Neutral', 3 - 'Positive']
    blob = TextBlob(review)
    sentiment = round((blob.sentiment.polarity + 1) * 3) % 3
    return sentiment

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
    random_forest_similarity_display("I am thinking about walking in the park")

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
    sentiment_display()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    final_project('Welcome to Final Project')

    dirty_sentences = []
    clean_token_sentences = []
    clean_lemma_sentences = []
    clean_tokens = []
    clena_lemmas = []

    sergey_nlp()
    #morgan_nlp()
    #norberto_nlp()
    #freeman_nlp()
    #rekha_nlp()