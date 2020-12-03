import unittest
from utils import classifier

class Tests(unittest.TestCase):

    def test_naive_bayes_classifier(self):
        fitted_vectorizer, trained_gnb = classifier.get_trained_naive_bayes_classifier()
        input_sentence = "my life is meaningless"
        predicted_category, predicted_proba = classifier.categorize_using_naive_bayes(fitted_vectorizer, trained_gnb, input_sentence)
        self.assertEqual("1", predicted_category)
        input_sentence = "that s a beautiful quote"
        predicted_category, predicted_proba = classifier.categorize_using_naive_bayes(fitted_vectorizer, trained_gnb, input_sentence)
        self.assertEqual("0", predicted_category)
        