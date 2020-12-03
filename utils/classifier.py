import utils
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from utils import ingestor

TOTAL_ROWS_IN_TRAINING_SET = 9120


#TODO: We should optimize this by saving the model to disk so that the classifier doesn't have to be trained every time it's loaded
def get_trained_naive_bayes_classifier():
    # Prepare set
    training_set = ingestor.read_training_file(TOTAL_ROWS_IN_TRAINING_SET)
    random.shuffle(training_set)
    training_set_data = [text[1] for text in training_set]
    training_set_labels = [text[0] for text in training_set]

    # Apply TF-IDF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=True)
    transformed_training_set = tfidf_vectorizer.fit_transform(training_set_data)

    gnb = GaussianNB()
    gnb.fit(transformed_training_set.toarray(), training_set_labels)

    return tfidf_vectorizer, gnb


def categorize_using_naive_bayes(fitted_vectorizer, trained_gnb, input_sentence):
    transformed_input = fitted_vectorizer.transform([input_sentence]).toarray()
    predicted_category = trained_gnb.predict(transformed_input)
    predicted_probability = trained_gnb.predict_proba(transformed_input)

    return predicted_category, predicted_probability










