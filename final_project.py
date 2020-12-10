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
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer

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
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from gensim.summarization.summarizer import summarize

from nltk.stem import PorterStemmer
ps = PorterStemmer()

def classification_display(list_of_sentences):
    print("In Classification")

def sentiment_display():
    suicide_dataset = read_dataset()[0:1000]
    emotions_dataset = read_emotion_dataset('data/best_dataset_emotions.csv')

    tfidf_vect, tfidf = tfidf_vectorizer(emotions_dataset['cleaned'])
    X_features = pd.concat([pd.DataFrame(tfidf.toarray())], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_features, emotions_dataset['emotion_label'], test_size=0.2)
    nb_tfidf = naive_bayes_model(X_train, X_val, y_train, y_val)
    sgd_tfidf = sgd_classifier_model(X_train, X_val, y_train, y_val)
    logreg_tfidf = logistic_regression_model(X_train, X_val, y_train, y_val)
    rf_tfidf = random_forest_model(X_train, X_val, y_train, y_val)
    dt_tfidf = decision_tree_classifier(X_train, X_val, y_train, y_val)

    count_vect, count = count_vectorizer(emotions_dataset['cleaned'])
    X_features = pd.concat([pd.DataFrame(count.toarray())], axis=1)
    X_train, X_val, y_train, y_val = train_test_split(X_features, emotions_dataset['emotion_label'], test_size=0.2)
    nb_count = naive_bayes_model(X_train, X_val, y_train, y_val)
    sgd_count = sgd_classifier_model(X_train, X_val, y_train, y_val)
    logreg_count = logistic_regression_model(X_train, X_val, y_train, y_val)
    rf_count = random_forest_model(X_train, X_val, y_train, y_val)
    dt_count = decision_tree_classifier(X_train, X_val, y_train, y_val)

    def print_sentence_emotion(emotion, sentence):
        print ('Input sentence               : ', sentence)
        print ('Expected Emotion             : ', emotion)
        print ('TF-IDF + Naive Bayes         : ', value_to_emotions(nb_tfidf.predict(tfidf_vect.transform([clean_sentences(sentence)]))[0]))
        print ('TF-IDF + SGD Classifier      : ', value_to_emotions(sgd_tfidf.predict(tfidf_vect.transform([clean_sentences(sentence)]))[0]))
        print ('TF-IDF + Logistic Regression : ', value_to_emotions(logreg_tfidf.predict(tfidf_vect.transform([clean_sentences(sentence)]))[0]))
        print ('TF-IDF + Random Forest       : ', value_to_emotions(rf_tfidf.predict(tfidf_vect.transform([clean_sentences(sentence)]))[0]))
        print ('TF-IDF + Decision Tree       : ', value_to_emotions(dt_tfidf.predict(tfidf_vect.transform([clean_sentences(sentence)]))[0]))
        print ('Count  + Naive Bayes         : ', value_to_emotions(nb_count.predict(count_vect.transform([clean_sentences(sentence)]))[0]))
        print ('Count  + SGD Classifier      : ', value_to_emotions(sgd_count.predict(count_vect.transform([clean_sentences(sentence)]))[0]))
        print ('Count  + Logistic Regression : ', value_to_emotions(logreg_count.predict(count_vect.transform([clean_sentences(sentence)]))[0]))
        print ('Count  + Random Forest       : ', value_to_emotions(rf_count.predict(count_vect.transform([clean_sentences(sentence)]))[0]))
        print ('Count  + Decision Tree       : ', value_to_emotions(dt_count.predict(count_vect.transform([clean_sentences(sentence)]))[0]))
        print ('*' * 50)
        print ('\n')

    sentences = {'anger': 'i am feeling outraged it shows everywhere',
                 'sadness': 'i have longed for a little affection',
                 'fear': 'i pay attention it deepens into a feeling of being invaded and helpless',
                 'neutral': 'i am okay with what i have'}
    for key, value in sentences.items():
        print_sentence_emotion(key, value)

    suicide_dataset['nb_tfidf'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(nb_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['sgd_tfidf'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(sgd_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['logreg_tfidf'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(logreg_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['rf_tfidf'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(rf_tfidf.predict(tfidf_vect.transform([x]))[0]))
    suicide_dataset['dt_tfidf'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(dt_tfidf.predict(tfidf_vect.transform([x]))[0]))

    suicide_dataset['nb_count'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(nb_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['sgd_count'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(sgd_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['logreg_count'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(logreg_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['rf_count'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(rf_count.predict(count_vect.transform([x]))[0]))
    suicide_dataset['dt_count'] = suicide_dataset['summarized'].apply(lambda x: value_to_emotions(dt_count.predict(count_vect.transform([x]))[0]))

    suicide_dataset.to_csv('resources/testset_emotions.csv',sep=',')
    print ("Test Set data is available in resources/testset_emotions.csv")

def unsupervised_clustering():
    emotions_dataset = read_emotion_dataset('data/best_dataset_emotions.csv')
    texts = emotions_dataset.cleaned
    vectorizer = TfidfVectorizer(max_df=0.45, min_df=2, stop_words='english', use_idf=True)
    X = vectorizer.fit_transform(texts)
    number_of_clusters = 5
    model = KMeans(n_clusters=number_of_clusters, init='k-means++', max_iter=100, n_init=1)
    model.fit(X)
    centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(number_of_clusters):
        print("Cluster %d:" % i)
        for index in centroids[i, :10]:
            print(' %s' % terms[index])

    sentences = {'anger': 'i am feeling outraged it shows everywhere',
                 'sadness': 'i have longed for a little affection',
                 'fear': 'i pay attention it deepens into a feeling of being invaded and helpless',
                 'neutral': 'i am okay with what i have'}
    for key, value in sentences.items():
        print ('Input sentence               : ', value)
        print ('Expected Emotion             : ', key)
        X = vectorizer.transform([clean_sentences(value)])
        cluster = model.predict(X)[0]
        print("Text belongs to cluster number {0}".format(cluster))

def reason_display(list_of_sentences):
    print("In Reason")
    d = getcwd()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(list_of_sentences)

    lsa = TruncatedSVD(2, algorithm='randomized')
    dtm_lsa = lsa.fit_transform(tfidf)
    #dtm_lsa = Normalizer(copy=False).fit_transform(dtm_lsa)

    sigma = lsa.singular_values_
    VT = lsa.components_.T



    sing_vecs = lsa.components_[0]
    index = np.argsort(sing_vecs).tolist()
    index.reverse()
    terms = [tfidf_vectorizer.get_feature_names()[weightIndex] for weightIndex in index]


    #factors evaluated for suicide as described by https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1449832/
    fam_list = []
    school_list = []
    ethnicity_list = []
    religion_list = []
    community_list = []
    friends_list = []
    sex_list = []
    alcohol_list = []
    depression_list = []
    appearance_list = []


    syns_family = wordnet.synsets("family")
    syns_school = wordnet.synsets("school")
    syns_ethnicity = wordnet.synsets("ethnicity")
    syns_religion = wordnet.synsets("religion")
    syns_community = wordnet.synsets("community")
    syns_friends = wordnet.synsets("friends")
    syns_sex = wordnet.synsets("sex")
    syns_alcohol = wordnet.synsets("alcohol")
    syns_depression = wordnet.synsets("depression")
    syns_appearance = wordnet.synsets("appearance")





    for t in terms:
        word = wordnet.synsets(t)
        if wordnet.synsets(t) != []:
            fam_sim = (word[0].path_similarity(syns_family[0]))
            if fam_sim != None:
                fam_list.append(fam_sim)

            school_sim = (word[0].path_similarity(syns_school[0]))
            if school_sim != None:
                school_list.append(school_sim)

            ethnicity_sim = (word[0].path_similarity(syns_ethnicity[0]))
            if ethnicity_sim != None:
                ethnicity_list.append(ethnicity_sim)

            religion_sim = (word[0].path_similarity(syns_religion[0]))
            if religion_sim != None:
                religion_list.append(religion_sim)

            community_sim = (word[0].path_similarity(syns_community[0]))
            if community_sim != None:
                community_list.append(community_sim)

            friends_sim = (word[0].path_similarity(syns_friends[0]))
            if friends_sim != None:
                friends_list.append(friends_sim)

            sex_sim = (word[0].path_similarity(syns_sex[0]))
            if sex_sim != None:
                sex_list.append(sex_sim)

            alcohol_sim = (word[0].path_similarity(syns_alcohol[0]))
            if alcohol_sim != None:
                alcohol_list.append(alcohol_sim)

            depression_sim = (word[0].path_similarity(syns_depression[0]))
            if depression_sim != None:
                depression_list.append(depression_sim)

            appearance_sim = (word[0].path_similarity(syns_appearance[0]))
            if appearance_sim != None:
                appearance_list.append(appearance_sim)


    avg_sim = []

    avg_sim.append(mean(fam_list))
    avg_sim.append(mean(school_list))
    avg_sim.append(mean(ethnicity_list))
    avg_sim.append(mean(religion_list))
    avg_sim.append(mean(community_list))
    avg_sim.append(mean(friends_list))
    avg_sim.append(mean(sex_list))
    avg_sim.append(mean(alcohol_list))
    avg_sim.append(mean(depression_list))
    avg_sim.append(mean(appearance_list))

    factors = ['Family', 'School', 'Ethnicity', 'Religion', 'Community', 'Friends', 'Sex', 'Alcohol', 'Depression', 'Appearance']
    i = 0
    for f in factors:

        print('Average Path Similarity of the factor', f, 'to the terms in the data was :', avg_sim[i] )
        i += 1

    colors = ['green', 'teal', 'red', 'blue', 'purple', 'brown', 'orange', 'cyan', 'magenta', 'black']

    plt.figure(figsize = [15, 10])
    plt.bar(factors, avg_sim, color = colors)
    plt.title("Reasons for Suicide")
    plt.xlabel("Suicide Factors", labelpad = 10)
    plt.ylabel("Relevance in Tweets")
    plt.grid(True)
    plt.savefig(path.join(d, 'resources/reasons_suicide.jpg'), dpi=200)

    plt.show()


def adjectives_display(list_of_sentences):
    print("In Adjectives")

    d = getcwd()



    alive_dataset = read_dataset()[4000:4700]
    suicide_dataset = read_dataset()[1000:1050]
    sui_str = suicide_dataset.to_string()
    alive_str = alive_dataset.to_string()


    tokenizer = spacy.load("en_core_web_sm")
    adjectives_sui = []

    adjectives_alive = []


    token_list_sui = tokenizer(sui_str)
    token_list_alive = tokenizer(alive_str)


    print('number of tokens in suicide data list', len(token_list_sui))


    print('number of tokens in non suicide data list', len(token_list_alive))

    for token in token_list_sui:
        if token.pos_ == "ADJ":
            #print('sui', token)
            #print(token, token.tag_, token.pos_, spacy.explain(token.tag_))
            adjectives_sui.append(token)

    for token in token_list_alive:
        if token.pos_ == "ADJ":
            #print('alive', token)
            #print(token, token.tag_, token.pos_, spacy.explain(token.tag_))
            adjectives_alive.append(token)


    num_adj_sui = len(adjectives_sui)
    num_adj_alive = len(adjectives_alive)
    print('number of adj in suicide data list', num_adj_sui)
    print('number of adj in non suicide data list', num_adj_alive)

    non_adj_sui = len(token_list_sui) - num_adj_sui
    non_adj_alive = len(token_list_alive) - num_adj_alive

    label_sui = ['Adjectives' , 'Non Adjectives']
    label_alive = ['Adjectives' , 'Non Adjectives']

    plt.rc('font', size = 8)
    breakdown_sui = [num_adj_sui, non_adj_sui]
    breakdown_alive = [num_adj_alive, non_adj_alive]
    fig, axs = plt.subplots(2)
    fig.suptitle('Use of Adjectives vs. other POS in Suicidal Tweets')
    explode = (0, .2)
    axs[0].pie(breakdown_sui, explode = explode, labels = label_sui, autopct = '%1.1f%%', shadow = True)
    axs[0].set_title('Tweets that Led to Suicide')
    axs[1].pie(breakdown_alive, explode = explode, labels = label_alive, autopct = '%1.1f%%', shadow = True)
    axs[1].set_title('Did not Lead to Suicide')
    plt.savefig(path.join(d, 'resources/adjectives.jpg'), dpi=200)
    plt.show()


def intention_vs_action_display(list_of_sentences):
    print("In Intention")

def important_factors_display(list_of_sentences):

    d = getcwd()

    tfidf_vectorizer = TfidfVectorizer(max_df=0.45, min_df=2, stop_words='english', use_idf=True)
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

    suicide_dataset = read_dataset()

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

    document_corpus.insert(0, sentence)

    tfidf_vectorizer = TfidfVectorizer(max_df=0.45, min_df=2, stop_words='english', use_idf=True)
    sparse_matrix = tfidf_vectorizer.fit_transform(document_corpus)

    cosine = cosine_similarity(sparse_matrix[0:1], sparse_matrix)

    fig = plt.gcf()

    for i in range(len(cosine)):
        for j in range(len(cosine[i])):
            d = 2 * 1 * (1 - cosine[0][j])
            if(j == 0):
                circle = plt.Circle((d, 0), 1, alpha=.5, color='g')
            else:
                circle = plt.Circle((d, 0), 1, alpha=.5, color='b')
            plt.xlim([-1.1, 1.1 + d])
            fig.gca().add_artist(circle)

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
        if (spacy.explain(token.tag_) == 'adjective'):
            adj.append(token)

    #displacy.serve(token_list, style="dep")

def ner_labeling_and_display(sentence):

    tokenizer = spacy.load("en_core_web_sm")
    token_list = tokenizer(sentence)
    for token in token_list.ents:
        print(token.text, '->', token.label_)

    #displacy.serve(token_list, style = "ent")

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

def most_common_adj_display(sentence):

    pos_tagging_and_display(sentence)

    adjectives = ' '.join(map(str, adj))

    d = getcwd()
    freq = Counter(adjectives.split(" "))
    sns.set_style("darkgrid")
    words = [word[0] for word in freq.most_common(10)]
    count = [word[1] for word in freq.most_common(10)]

    plt.figure(figsize=(10, 10))

    sns_bar = sns.barplot(x=words, y=count)
    sns_bar.set_xticklabels(words, rotation=90)
    plt.title('Most Used Adjectives')
    plt.savefig(path.join(d, 'resources/adj_count.jpg'), dpi=200)
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
            clean_lemmas.append(tokenizer(lemmas))
            dirty_sentences.append(row['tweet'])


def read_dataset(filename='data/data_set.csv'):
    data = pd.read_csv(filename)
    data['cleaned'] = data['tweet'].apply(lambda x: clean_sentences(x))
    data['summarized'] = data['tweet'].apply(lambda x: text_summarization(x))
    data['sentiments'] = data['cleaned'].apply(lambda x: sentiment_analysis(x))
    return data

def read_emotion_dataset(filename='data/dataset_emotions.csv'):
    data = pd.read_csv(filename)
    data['cleaned'] = data['tweet'].apply(lambda x: clean_sentences(x))
    data['sentiments'] = data['cleaned'].apply(lambda x: sentiment_analysis(x))
    data['emotion_label'] = data['emotion'].apply(lambda x: emotions_to_index(x))
    return data

def emotions_to_index(label):
    generic_emotions = {'anger': 0, 'shock': 1, 'fear': 2, 'neutral': 3, 'sadness': 4, 'disgust': 5, 'unknown': 6}
    if (label in generic_emotions.keys()):
        return (generic_emotions[label])
    else: return (generic_emotions['unknown'])

def value_to_emotions(val):
    generic_emotions = {'anger': 0, 'shock': 1, 'fear': 2, 'neutral': 3, 'sadness': 4, 'disgust': 5, 'unknown': 6}
    for key, value in generic_emotions.items():
        if val == value:
            return key
    return 'unknown'

def clean_sentences(sentence):
    noNumbers = ''.join([item for item in sentence if not item.isnumeric()])
    noPunct = ''.join([item for item in noNumbers if item not in string.punctuation])
    lowerWords = noPunct.lower()
    stemmedWords = ''.join([ps.stem(item) for item in lowerWords])
    return (stemmedWords)

def tfidf_vectorizer(data):
    print ("TF-IDF Vectorizer")
    tfidf_vect = TfidfVectorizer(max_df=0.45, min_df=2, stop_words='english', use_idf=True)
    X_train_tfidf = tfidf_vect.fit_transform(data)
    return tfidf_vect, X_train_tfidf

def hashing_vectorizer(data):
    print ("Hashing Vectorizer")
    hash_vect = HashingVectorizer()
    X_train_hash = hash_vect.fit_transform(data)
    return hash_vect, X_train_hash

def count_vectorizer(data):
    print ("Count Vectorizer")
    count_vect = CountVectorizer()
    X_train_count = count_vect.fit_transform(data)
    return count_vect, X_train_count

def print_predictions(yval, y_pred, algorithm):
    print('{}'.format(algorithm))
    precision, recall, fscore, support = score(yval, y_pred, average='weighted', labels=np.unique(y_pred))
    print ("Precision: {} Recall: {} Accuracy: {} FScore: {}".format(round(precision,2),
                                                                     round(recall,2),
                                                                     round((y_pred==yval).sum()/len(y_pred), 2),
                                                                     round(fscore, 2)))

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

def decision_tree_classifier(xtrain, xval, ytrain, yval):
    dt = DecisionTreeClassifier()
    dt.fit(xtrain, ytrain)
    y_pred = dt.predict(xval)
    print_predictions(yval, y_pred, 'Decision Tree Classifier')
    return dt

def sentiment_analysis(review):
    # 0 - 'Negative', 1 - 'Neutral', 3 - 'Positive']
    blob = TextBlob(review)
    sentiment = round((blob.sentiment.polarity + 1) * 3) % 3
    return sentiment

def text_summarization(data):
    try:
        summarized_text = summarize(data, word_count = 300)
        return summarized_text
    except:
        return data

def final_project(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


def sergey_nlp():
    print("Sergey's Work")

    #data_preprocessing()
    #clean_tokens_strings = ' '.join(map(str, clean_tokens))
    #clean_lemma_strings = ' '.join(map(str, clean_lemmas))

    #most_common_words_display(clean_lemma_strings)
    #word_cloud_display(clean_lemma_strings)
    #cosine_similarity_display(dirty_sentences,"I want to walk in the park")
    #cosine_similarity_display(dirty_sentences, "I want to kill myself")
    #word_cloud_display(clean_tokens_strings)
    #important_factors_display(clean_lemma_sentences)
    #pos_tagging_and_display(clean_tokens_strings)
    #most_common_adj_display(clean_tokens_strings)
    #ner_labeling_and_display(clean_tokens_strings)
    #random_forest_similarity_display("I am thinking about walking in the park")
    random_forest_similarity_display("I want to kill myself")

def morgan_nlp():
    print ("Morgan's Work")

def norberto_nlp():
    print("Norberto's Work")

    classification_display(clean_lemma_sentences)


def freeman_nlp():
    print ("Freeman's Work")

    adjectives_display(clean_lemma_sentences)
    reason_display(clean_lemma_sentences)
    #intention_vs_action_display(clean_lemma_sentences)

def rekha_nlp():
    print ("Rekha's Work")
    sentiment_display()
    unsupervised_clustering()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    final_project('Welcome to Final Project')

    dirty_sentences = []
    clean_token_sentences = []
    clean_lemma_sentences = []
    clean_tokens = []
    clean_lemmas = []
    adj = []

    # sergey_nlp()
    #morgan_nlp()
    #norberto_nlp()
    #freeman_nlp()
    rekha_nlp()