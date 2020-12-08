import unittest
from sklearn import metrics
from sklearn.metrics import precision_recall_fscore_support as score
from utils import classifier, preprocessor, ingestor
import textwrap

TOTAL_ROWS_IN_TRAINING_SET = 9120

class Tests(unittest.TestCase):

    def test_simple_classification(self):
        custom_stop_words = preprocessor.get_stop_words(words_to_exclude=["not", "myself", "to", "don't"])
        all_training_set = ingestor.read_file(ingestor.BEST_TRAINING_FILE)

        fitted_vectorizer, trained_gnb = classifier.get_trained_multinb_classifier(custom_stop_words, all_training_set)

        input_sentences = ["I am considering committing suicide"]
        expected = ['suicide']
        # test_set = ingestor.read_test_file(TOTAL_ROWS_IN_TRAINING_SET)
        # expected = [record[0] for record in test_set]
        # input_sentences = [record[1] for record in test_set]

        predicted_category, predicted_proba = classifier.categorize_using_trained_classifier(fitted_vectorizer, trained_gnb, input_sentences)

        print("Test_simple_classification: Category:" + str(predicted_category) + " proba: " + str(predicted_proba))


    def test_all_classifiers(self):
        with open("resources\classifiers_performance.csv", "w", encoding="utf-8") as output_file:
            output_file.write("Classifier Name,Training Set,Test Set,Precision Suicide, Precision Non-suicide\n")

            sample_sentences = ["I want to commit suicide", 
            "I want to kill myself", 
            "I am thinking about walking in the park", 
            "I really love burgers"]
        
            sample_expected = ['1', '1', '0', '0']

            custom_stop_words = preprocessor.get_stop_words(words_to_exclude=["not", "myself", "to", "don't"])
            all_training_set = ingestor.read_file(ingestor.TRAINING_FILE)
            best_training_set = ingestor.read_file(ingestor.BEST_TRAINING_FILE)
            test_set = ingestor.read_file(ingestor.TEST_FILE)
            test_expected = [record[0] for record in test_set]
            test_sentences = [record[1] for record in test_set]

            #### NAIVE BAYES CLASSIFIER #### 
            #----- ALL TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_naive_bayes_classifier(custom_stop_words, all_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Naive Bayes", output_file, "all")

            #----- BEST TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_naive_bayes_classifier(custom_stop_words, best_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Naive Bayes", output_file, "best")

            # #### MULTINOMIAL CLASSIFIER #### 
            #----- ALL TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_multinb_classifier(custom_stop_words, all_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Multinomial", output_file, "all")

            #----- BEST TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_multinb_classifier(custom_stop_words, best_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Multinomial", output_file, "best")

            #### RANDOM FOREST CLASSIFIER #### 
            #----- ALL TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_random_forest_classifier(custom_stop_words, all_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Random Forest", output_file, "all")

            #----- BEST TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_random_forest_classifier(custom_stop_words, best_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "Random Forest", output_file, "best")


            #### SVM FOREST CLASSIFIER #### 
            #----- ALL TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_svm_classifier(custom_stop_words, all_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "SVM", output_file, "all")

            #----- BEST TRAINING SET -----#
            fitted_vectorizer, trained_classifier = classifier.get_trained_svm_classifier(custom_stop_words, best_training_set)

            classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_expected, test_sentences, test_expected, "SVM", output_file, "best")

    def test_brute_force(self):
        return

        custom_stop_words = preprocessor.get_stop_words(words_to_exclude=["not", "myself", "to", "don't"])
        all_training_set = ingestor.read_file(ingestor.TRAINING_FILE)

        all_suicide_set = []
        all_non_suicide_set = []

        for record in all_training_set:
            if record[0] == '1':
                all_suicide_set.append(record)
            else:
                all_non_suicide_set.append(record)

        ##################################################
        words = set([])
        index_suicide = 0
        index_non_suicide = 0
        counter = 0
        best_classifier = None
        best_vect = None

        best_training_set = []
        test_set = []

        with open("classifier-brute-force.csv", "w", encoding="utf-8") as output_file:
            output_file.write("id, suicide count, non suicide count, NB accuracy, 0 expected NB, 0 predicted NB, 0 match NB, 0 prob NB, 1 expected NB, 1 predicted NB, 1 match NB, 1 prob NB, 2 expected NB, 2 predicted NB, 2 match NB, 2 prob NB, 3 expected NB, 3 predicted NB, 3 match NB, 3 prob NB, MN accuracy, 0 expected MN, 0 predicted MN, 0 match MN, 0 prob MN, 1 expected MN, 1 predicted MN, 1 match MN, 1 prob MN, 2 expected MN, 2 predicted MN, 2 match MN, 2 prob MN, 3 expected MN, 3 predicted MN, 3 match MN, 3 prob MN, new words\n")
            training_set = []
            max_precision = 0

            # best_training_set_mult = []
            # max_precision_mult = 0

            while index_suicide <  len(all_suicide_set) or index_non_suicide < len(all_non_suicide_set):
                new_words = set([])
                processing_id = ''
                training_set = best_training_set.copy()
                sentence_to_process = []


                #TODO PREPROCESS AND CLEAN                
                if index_suicide < len(all_suicide_set): #and counter % 2 == 0:
                    suicide_sentence = all_suicide_set[index_suicide]
                    training_set.append((suicide_sentence[0], suicide_sentence[1]))
                    sentence_to_process_suicide = (suicide_sentence[0], suicide_sentence[1])
                    processing_id = suicide_sentence[2]                    

                    for word in suicide_sentence[1].split():
                        if word not in words:
                            words.add(word)
                            new_words.add(word)

                    index_suicide += 1

                if index_non_suicide < len(all_non_suicide_set): # and counter %2 == 1:
                    non_suicide_sentence = all_non_suicide_set[index_non_suicide]
                    training_set.append((non_suicide_sentence[0], non_suicide_sentence[1]))
                    sentence_to_process_non_suicide = (non_suicide_sentence[0], non_suicide_sentence[1])
                    processing_id = non_suicide_sentence[2]

                    for word in non_suicide_sentence[1].split():
                        if word not in words:
                            words.add(word)
                            new_words.add(word)

                    index_non_suicide +=1

                input_sentences = ["i wish i got to watch it with you i miss you and how was the premiere",
                                    "I want to kill myself",
                                    "i just want to end my life",
                                    "I had fun dancing tonight"]

                #### NAIVE BAYES ####
                fitted_vectorizer_gnb, trained_gnb = classifier.get_trained_naive_bayes_classifier(custom_stop_words, training_set)
                predicted_categories, predicted_proba = classifier.categorize_using_naive_bayes(fitted_vectorizer_gnb, trained_gnb, input_sentences)

                acc = 0
                predicted_0_match = "non-suicide"== predicted_categories[0]
                if predicted_0_match:
                    acc+=1
                predicted_1_match = "suicide"==predicted_categories[1]
                if predicted_1_match:
                    acc+=1
                predicted_2_match = "suicide"==predicted_categories[2]
                if predicted_2_match:
                    acc+=1                
                predicted_3_match = "non-suicide"==predicted_categories[3]
                if predicted_3_match:
                    acc+=1
                acc/=4

                output_file.write(f"{processing_id},{index_suicide + 1},{index_non_suicide + 1},{acc},")
                output_file.write(f"non-suicide,{predicted_categories[0]},{predicted_0_match},{predicted_proba[0][0]},")
                output_file.write(f"suicide,{predicted_categories[1]},{predicted_1_match},{predicted_proba[1][0]},")
                output_file.write(f"suicide,{predicted_categories[2]},{predicted_2_match},{predicted_proba[2][0]},")
                output_file.write(f"non-suicide,{predicted_categories[3]},{predicted_3_match},{predicted_proba[3][0]},")

                #### MULT ####
                fitted_vectorizer_mult, trained_mult = classifier.get_trained_multinb_classifier(custom_stop_words, training_set)
                predicted_categories, predicted_proba = classifier.categorize_using_naive_bayes(fitted_vectorizer_mult, trained_mult, input_sentences)

                acc = 0
                predicted_0_match = "non-suicide"== predicted_categories[0]
                if predicted_0_match:
                    acc+=1
                predicted_1_match = "suicide"==predicted_categories[1]
                if predicted_1_match:
                    acc+=1
                predicted_2_match = "suicide"==predicted_categories[2]
                if predicted_2_match:
                    acc+=1                
                predicted_3_match = "non-suicide"==predicted_categories[3]
                if predicted_3_match:
                    acc+=1
                acc/=4

                if acc >= max_precision:
                    best_training_set.append(sentence_to_process_suicide)
                    best_training_set.append(sentence_to_process_non_suicide)
                    max_precision = acc
                else:
                    test_set.append(sentence_to_process_suicide)
                    test_set.append(sentence_to_process_non_suicide)

                output_file.write(f"{acc},")
                output_file.write(f"non-suicide,{predicted_categories[0]},{predicted_0_match},{predicted_proba[0][0]},")
                output_file.write(f"suicide,{predicted_categories[1]},{predicted_1_match},{predicted_proba[1][0]},")
                output_file.write(f"suicide,{predicted_categories[2]},{predicted_2_match},{predicted_proba[2][0]},")
                output_file.write(f"non-suicide,{predicted_categories[3]},{predicted_3_match},{predicted_proba[3][0]},{'-'.join(new_words)},\n")

                counter+=1

                # if processing_id == "226":
                #     input_sentences = ["Cat"]
                #     predicted_categories, predicted_proba = classifier.categorize_using_naive_bayes(fitted_vectorizer_mult, trained_mult, input_sentences)
                #     print(f"Calc cat:{str(predicted_categories)}, proba {predicted_proba}")
                #     break

        with open("best_training_set.csv", "w", encoding="utf-8") as output_file:
            for record in best_training_set:
                output_file.write(f"{record[0]},{record[1]}\n")

        with open("test_set.csv", "w", encoding="utf-8") as output_file:
            for record in test_set:
                output_file.write(f"{record[0]},{record[1]}\n")


def classify(fitted_vectorizer, trained_classifier, sample_sentences, sample_ground_truth, test_sentences, test_ground_truth, classifier_name, output_file, training_set_name):
    print_classifier_result(fitted_vectorizer, trained_classifier, sample_sentences, sample_ground_truth, classifier_name, output_file, training_set_name, "sample_sentences")

    print_classifier_result(fitted_vectorizer, trained_classifier, test_sentences, test_ground_truth, classifier_name, output_file, training_set_name, "test_set")

def print_classifier_result(fitted_vectorizer, trained_classifier, input_sentences, ground_truth, classifier_name, output_file, training_set_name, test_set):
    predicted_categories, predicted_probas = classifier.categorize_using_trained_classifier(fitted_vectorizer, trained_classifier, input_sentences)
    metrics_report = metrics.classification_report(ground_truth, predicted_categories, output_dict=True)
    suicide_precision = str(round(metrics_report['1']['precision'], 2))
    non_suicide_precision = str(round(metrics_report['0']['precision'], 2))
    output_file.write(f"{classifier_name},{training_set_name},{test_set},{suicide_precision},{non_suicide_precision}\n")
    print(f"{classifier_name},{training_set_name},{test_set},{suicide_precision},{non_suicide_precision},{str(predicted_categories)}")