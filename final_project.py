import csv
import nltk
import sklearn
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from importlib import reload

from pylab import *
from nltk.corpus import stopwords
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.cluster import KMeans, MiniBatchKMeans

# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def final_project(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def sergey_nlp():
    print ("Hello From Sergey")

def morgan_nlp():
    print ("Hello From Morgan")

def norberto_nlp():
    print ("Hello From Norberto")

def freeman_nlp():
    print ("Hello From Freeman")

def rekha_nlp():
    print ("Hello From Rekha")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    final_project('Welcome to Final Project')
    sergey_nlp()
    morgan_nlp()
    norberto_nlp()
    freeman_nlp()
    rekha_nlp()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/