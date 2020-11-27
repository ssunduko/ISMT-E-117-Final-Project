import unittest
from preprocessing import preprocessor

class Tests(unittest.TestCase):

    def test_fix_contractions(self):
        self.assertEqual("I am", preprocessor.fix_contractions("I'm"))


    def test_remove_numbers_text(self):
        self.assertEqual("I am  years old.", preprocessor.remove_numbers("I am 33 years old."))


    def test_remove_numbers_list(self):
        input_list = [("I", 'XX'), ("am", "VB"), ("33", "NU"), ("years", "XX"), ("old", "XX")]
        expected_list = [("I", 'XX'), ("am", "VB"), ("years", "XX"), ("old", "XX")] 
        self.assertEqual(expected_list, preprocessor.remove_numbers(input_list))


    def test_remove_puntuation_marks_text(self):
        self.assertEqual("this is a test", preprocessor.remove_puntuation_marks("this.is.a.test"))


    def test_remove_puntuation_marks_list(self):
        input_list = [("This.","XX"), (".","XX"), (".is","XX"), ("a","XX"), (",","XX"), ("Test","XX")]
        expected_list = [("This","XX"), ("is","XX"), ("a","XX"), ("Test","XX")]
        self.assertEqual(expected_list, preprocessor.remove_puntuation_marks(input_list))


    def test_remove_stop_words_text(self):
        tokenized_text = ["This", "is", "not", "a", "test"]
        expected_text = ["test"]
        self.assertEqual(expected_text, preprocessor.remove_stop_words(tokenized_text))


    def test_remove_stop_words_text_updates_stop_words(self):
        tokenized_text = ["This", "is", "not", "a", "test"]
        expected_text = ["is", "not"]
        words_to_include = ["test"]
        words_to_exclude = ["is", "not"]
        self.assertEqual(expected_text, preprocessor.remove_stop_words(tokenized_text, words_to_include=words_to_include, words_to_exclude=words_to_exclude))


    def test_remove_stop_words_tokenized_text(self):
        tokenized_text = [("This", "XX"), ("is", "XX"), ("not", "XX"), ("a", "XX"), ("test", "XX")]
        expected_text = [("test", "XX")]
        self.assertEqual(expected_text, preprocessor.remove_stop_words(tokenized_text, True))


    def test_remove_stop_words_tokenized_text_updates_stop_words(self):
        tokenized_text = [("This", "XX"), ("is", "XX"), ("not", "XX"), ("a", "XX"), ("test", "XX")]
        expected_text = [("is", "XX"), ("not", "XX")]
        words_to_include = ["test"]
        words_to_exclude = ["is", "not"]
        self.assertEqual(expected_text, preprocessor.remove_stop_words(tokenized_text, True, words_to_include, words_to_exclude))


if __name__ == "__main__":
    unittest.main()