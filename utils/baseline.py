from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

import collections
import matplotlib.pyplot as plt

import nltk as n
from nltk.corpus import wordnet as wn
from sklearn.model_selection import learning_curve
from nltk.corpus import brown
from nltk.corpus import cess_esp as cess
import numpy as np
from sklearn.model_selection import ShuffleSplit

import string
import pyphen

# n.download('cess_esp')
# s = ''.join(brown.words())
# s.capitalize()
# for n in string.punctuation:
#     if n in s:
#         print(n)
#         s.replace(n, '')
# print(s)

class Baseline(object):

    def __init__(self, language):
        self.language = language
        # from 'Multilingual and Cross-Lingual Complex Word Identification' (Yimam et. al, 2017)
        if language == 'english':
            self.avg_word_length = 5.3
            self.dic = pyphen.Pyphen(lang='en')
            self.alphabet = {
                'a': 8.167, 'b': 1.492, 'c': 2.782, 'd': 4.253, 'e': 12.702, 'f': 2.228, 'g': 2.015,
                'h': 6.094, 'i': 6.966, 'j': 0.153, 'k': 0.772, 'l': 4.025,  'm': 2.406, 'n': 6.749,
                'o': 7.507, 'p': 1.929, 'q': 0.095, 'r': 5.987, 's': 6.327,  't': 9.056, 'u': 2.758,
                'v': 0.978, 'w': 2.360, 'x': 0.150, 'y': 1.974, 'z': 0.074,
            }
            self.corpus_dict = dict(collections.Counter(brown.words()))
        else:  # spanish
            self.avg_word_length = 6.2
            self.dic = pyphen.Pyphen(lang='es')
            self.alphabet = {
                'a': 12.525, 'b': 2.215, 'c': 4.139, 'd': 5.860, 'e': 13.681, 'f': 0.692, 'g': 1.768,
                'h': 0.703,  'i': 6.247, 'j': 0.443, 'k': 0.011, 'l': 4.967,  'm': 3.157, 'n': 6.71,
                'o': 8.683,  'p': 2.510, 'q': 0.877, 'r': 6.871, 's': 7.977,  't': 4.632, 'u': 3.927,
                'v': 1.138,  'w': 0.017, 'x': 0.215, 'y': 1.008, 'z': 0.517,  'á': 0.502, 'é': 0.433,
                'í': 0.725,  'ñ': 0.311, 'ó': 0.827, 'ú': 0.168, 'ü': 0.012,
            }
            self.corpus_dict = dict(collections.Counter(cess.words()))



        # self.model = LogisticRegression()
        self.model = SVC()

    def extract_features(self, word):
        len_chars = len(word) / self.avg_word_length
        len_tokens = len(word.split(' '))
        len_syn = len(wn.synsets(word))  # The number of synonyms
        len_syll = len(self.dic.inserted(word).split('-')) / len_tokens
        tf = 0
        for i in range(len_tokens):
            if word.split(' ')[i] in self.corpus_dict.keys():
                tf += self.corpus_dict[word.split(' ')[i]]
            else:
                tf += 1
        word_letter_weight = 0
        for i in word.lower():
            if i in self.alphabet.keys():
                word_letter_weight += self.alphabet[i]
        return [len_chars, len_tokens, len_syll, len_syn, tf, word_letter_weight]

    def train(self, trainset):
        X = []
        y = []
        for sent in trainset:
            X.append(self.extract_features(sent['target_word']))
            y.append(sent['gold_label'])

        # start point :: this part of code is refer to the Plotting Learning Curves of scikit-learn
        #                Url: http://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html
        train_sizes, train_scores, test_scores = learning_curve(
            SVC(), X, y, cv=ShuffleSplit(n_splits=10, test_size=0.2, random_state=0))
        plt.figure()
        plt.title("Learning Curves (SVM, RBF kernel)")
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
        plt.show()
        # end point
        self.model.fit(X, y)


    def test(self, testset):
        X = []
        for sent in testset:
            X.append(self.extract_features(sent['target_word']))

        return self.model.predict(X)
