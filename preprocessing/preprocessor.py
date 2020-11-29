import contractions
import nltk
import re
from nltk.corpus import stopwords

def fix_contractions(text):
    """
    Expands contractions in a given text
    """
    return contractions.fix(text)

def remove_numbers(text):
    """
    Removes all the numbers in the input text. Accepts as input a string or a list.
    """
    if type(text) == list:
        result = []
        for token in text:
            new_text = actually_remove_numbers(token[0])
            if new_text != '':
                result.append((new_text, token[1]))
        return result
    else:
        return actually_remove_numbers(text)


def actually_remove_numbers(text):
    return ''.join([i for i in text if not i.isdigit()])


def remove_puntuation_marks(text):
    """
    Removes punctuations from the input text. Accepts as input a string or a list.
    """
    if type(text) == list:
        result = []
        for token in text:
            no_punctuation = re.sub(r'[^\w\s]', '', token[0])
            result.append((no_punctuation, token[1]))
        return [token for token in result if token[0] != '']
    else:
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(text)
        result = ' '.join(tokens)
        return result


def remove_stop_words(tokenized_text, already_tagged=False, words_to_include=[], words_to_exclude=[]):
    """
    Removes the stop words from a tokenized. The input could be already tagged (e.g. POS tagged, NER tagged)
    """
    stop_words = get_stop_words(words_to_include, words_to_exclude)
    result = []

    for token in tokenized_text:
        if already_tagged and token[0].lower() not in stop_words:
            result.append((token[0], token[1]))
        elif not already_tagged and token.lower() not in stop_words:
            result.append(token)

    return result

def get_stop_words(words_to_include=[], words_to_exclude=[]):
    """
    Gets the list of stop words which can be expanded or reduced, allowing for experimentation
    """
    stop_words = set(stopwords.words('english'))
    for word_to_include in words_to_include:
        stop_words.add(word_to_include)

    for word_to_exclude in words_to_exclude:
        stop_words.remove(word_to_exclude)
    return stop_words