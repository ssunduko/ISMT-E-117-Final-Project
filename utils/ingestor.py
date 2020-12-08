import csv
from itertools import islice
from csv import DictReader

TRAINING_FILE = 'data/data_set.csv'
BEST_TRAINING_FILE = 'data/best_training_set.csv'
TEST_FILE = 'data/test_set.csv'

def read_file(file):
    result = []

    with open(file, 'r', encoding='utf-8') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            result.append((row["label"], row["tweet"]))

    return result