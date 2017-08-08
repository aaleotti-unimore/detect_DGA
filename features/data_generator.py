# coding=utf-8
from __future__ import division

import logging

import enchant
import numpy as np
from nltk import ngrams
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcr_dict = enchant.Dict("en_US")
ns_dict = open("features/top10000en.txt").readlines()

path_good = 'DGA-master/top100k.txt'
path_bad = 'DGA-master/bad100k.txt'


def generate_good(X_len):
    """
    Generates a np.array matrix containing the feature vector for all the good samples and saves it into a good.pkl file
    :return:
    """
    with open(path_good, 'rb') as f:
        ar = []
        i = 0
        for line in f:
            if i == int(X_len / 2):
                break
            ar.append(__generate_features(line.split(",")[2]))
            logger.debug("added %s %s" % (line.split(",")[2], str(ar[-1:])))
            i += 1
        X = np.array(ar)
        joblib.dump(X, "datas/good_%s" % int(X_len / 2), compress=3)

    logger.info("good dataset saved: %s " % X_len)


def generate_bad(X_len):
    """
    Generates a np.array matrix containing the feature vector for all the DGA samples and saves it into a bad.pkl file
    :return:
    """
    with open(path_bad, 'rb') as f:
        ar = []
        i = 0
        for line in f:
            if i == int(X_len / 2):
                break
            ar.append(__generate_features(line.split(" ")[0]))
            logger.debug("added %s %s" % (line.split(" ")[0], str(ar[-1:])))
            i += 1
        X = np.array(ar)
        joblib.dump(X, "datas/bad_%s" % int(X_len / 2), compress=3)

    logger.info("bad dataset saved: %s " % X_len)


def __generate_features(string):
    """
    generates the features vector for a single string
    :param string: domain name
    :return: list of features
    """
    ratio = __get_mcr(string)
    nscore = []
    for i in [1, 2, 3]:
        nscore.append(__get_ns(string, i))
    logger.debug(string + " mcr:%.3f nscores: %.3f %.3f %.3f" % (ratio, nscore[0], nscore[1], nscore[2]))
    return [ratio] + nscore


def __get_mcr(domain_name):
    """
    Meaningful Characters Ratio. Models the ratio of characters of the string p that comprise a meaningful word. Low values indicate automatic algorithms. Specifically, we split p into n meaningful subwords wi of at least 3 symbols: |wi| ≥ 3, leaving out as few symbols as possible: R(d) = R(p) = max((sum from i=1 to n) |wi|)/|p|. If p = facebook, R(p) = (|face| + |book|)/8 = 1, the prefix is fully composed of meaningful words, whereas p = pub03str, R(p) = (|pub|)/8 = 0.375.

    :param domain_name: string
    :return: ratio. float value
    """
    min_subtr = 3
    maxl = 0
    for i in range(min_subtr, len(domain_name)):
        tuples = ngrams(domain_name, i)
        # tuples = zip(*[domain_name[j::i] for j in range(i)]) #alternative way to split the string. chunks are not overlapping
        split = [''.join(t) for t in tuples]
        tmpsum = 0
        tmps = []
        for s in split:
            if mcr_dict.check(s):
                tmpsum += len(s)
                tmps.append(s)
        if tmps:
            logger.debug("      %s , sum(w_i)=%s" % (tmps, tmpsum))
        if tmpsum > maxl:
            maxl = tmpsum
    return maxl / int(len(domain_name))


def __get_ns(domain_name, n):
    """
    This class of features captures the pronounceability of a domain name. The more permissible the combinations of phonemes, the more pronounceable a word is. Domains with a low number of such combinations are likely DGA-generated.
    We calculate this class of features by extracting the n-grams of p, which are the substrings of p of length n {1, 2, 3}, and counting their occurrences in the (English) language dictionary.
    The features are thus parametric to n: Sn(d) = Sn(p) := ((sum of n-gram t in p) count(t))/(|p|−n+1), where count(t) are the occurrences of the n-gram t in the dictionary
    :param domain_name: string
    :param n: size of the n-grams. int
    :return: score of the n-grams. float
    """
    tuples = ngrams(domain_name, n)
    myngrams = (''.join(t) for t in tuples)
    scoresum = 0
    for s in myngrams:
        counter = 0
        for words in ns_dict:
            if s in words:
                counter += 1
        logger.debug(" sub:%s count:%s" % (s, counter))
        scoresum += counter
    return scoresum / (len(domain_name) - n + 1)


def create_target(X_len):
    """
    generates target vector
    :param X_len: Size of the X vector. int
    """
    y_0 = np.zeros(int(X_len / 2))  # good samples
    y_1 = np.ones(int(X_len / 2))  # bad samples: DGA
    y = np.concatenate((y_0, y_1))
    joblib.dump(y, "datas/y_%s" % X_len, compress=3)
    logger.info("target vector saved %s" % len(y))
