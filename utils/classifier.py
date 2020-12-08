import utils
import random
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from utils import ingestor
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


#TODO: We should optimize this by saving the model to disk so that the classifier doesn't have to be trained every time it's loaded
def get_trained_naive_bayes_classifier(custom_stop_words, training_set):
    tfidf_vectorizer, transformed_training_set, training_set_labels = get_trained_tfidf_vectorizer(custom_stop_words, training_set)

    gnb = GaussianNB()
    gnb.fit(transformed_training_set.toarray(), training_set_labels)

    return tfidf_vectorizer, gnb


def get_trained_multinb_classifier(custom_stop_words, training_set):
    tfidf_vectorizer, transformed_training_set, training_set_labels = get_trained_tfidf_vectorizer(custom_stop_words, training_set)

    gnb = MultinomialNB()
    gnb.fit(transformed_training_set.toarray(), training_set_labels)

    return tfidf_vectorizer, gnb


def get_trained_svm_classifier(custom_stop_words, training_set):
    tfidf_vectorizer, transformed_training_set, training_set_labels = get_trained_tfidf_vectorizer(custom_stop_words, training_set)

    #Create a svm Classifier
    clf = svm.SVC(kernel='linear', probability=True, C=0.5) # Linear Kernel

    #Train the model using the training sets
    clf.fit(transformed_training_set, training_set_labels)

    return tfidf_vectorizer, clf

def get_trained_random_forest_classifier(custom_stop_words, training_set):
    tfidf_vectorizer, transformed_training_set, training_set_labels = get_trained_tfidf_vectorizer(custom_stop_words, training_set)

    clf = RandomForestClassifier(n_estimators=500)

    #Train the model using the training sets
    clf.fit(transformed_training_set, training_set_labels)

    return tfidf_vectorizer, clf


def get_trained_tfidf_vectorizer(custom_stop_words, training_set):
    # Prepare training data
    random.shuffle(training_set)
    training_set_data = [text[1] for text in training_set]
    training_set_labels = [text[0] for text in training_set]

    # Apply TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=1, min_df=1, stop_words=custom_stop_words, use_idf=True)
    transformed_training_set = tfidf_vectorizer.fit_transform(training_set_data)

    return tfidf_vectorizer, transformed_training_set, training_set_labels

def categorize_using_trained_classifier(fitted_vectorizer, trained_gnb, input_sentences):
    transformed_input = fitted_vectorizer.transform(input_sentences).toarray()
    predicted_category = trained_gnb.predict(transformed_input)
    predicted_probability = trained_gnb.predict_proba(transformed_input)

    return predicted_category, predicted_probability