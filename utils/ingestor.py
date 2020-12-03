import csv
from itertools import islice
from csv import DictReader

TRAINING_FILE = 'data/data_set.csv'

#TODO: This should return a class, not a list of tuples
#TODO: Return all records
def read_training_file(limit):
    result = []

    with open(TRAINING_FILE, 'r', encoding='utf-8') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in islice(csv_dict_reader, limit):
            result.append((row["label"], row["tweet"]))

    return result
