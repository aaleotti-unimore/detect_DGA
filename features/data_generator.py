from __future__ import division

import logging

import enchant
import numpy as np
from nltk import ngrams
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
logger.level = logging.INFO

mcr_dict = enchant.Dict("en_US")
ns_dict = open("features/top10000en.txt").readlines()

good = 'DGA-master/top10kaa.txt'
bad = 'DGA-master/bad10kaa.txt'
splitvalue = 2000


def generate_good():
    """
    Generates a np.array matrix containing the feature vector for all the good samples and saves it into a good.pkl file
    :return:
    """
    with open(good, 'rb') as f:
        ar = []
        i = 0
        for line in f:
            ar.append(generate_features(line.split(",")[2]))
            logger.info("added %s %s" % (line.split(",")[2], str(ar[-1:])))
            if i == splitvalue:
                break
            i += 1
        X = np.array(ar)
        joblib.dump(X, "good.pkl")


def generate_bad():
    """
    Generates a np.array matrix containing the feature vector for all the DGA samples and saves it into a bad.pkl file
    :return:
    """
    with open(bad, 'rb') as f:
        ar = []
        i = 0
        for line in f:
            ar.append(generate_features(line.split(" ")[0]))
            logger.debug("added %s %s" % (line.split(" ")[0], str(ar[-1:])))
            if i == splitvalue:
                break
            i += 1
        X = np.array(ar)
        joblib.dump(X, "bad.pkl")


def generate_features(string):
    """
    generates the features vector for a single string
    :param string: domain name
    :return: list of features
    """
    ratio = get_mcr(string)
    nscore = []
    for i in [1, 2, 3]:
        nscore.append(get_ns(string, i))
    logger.debug(string + " mcr:%.3f nscores: %.3f %.3f %.3f" % (ratio, nscore[0], nscore[1], nscore[2]))
    return [ratio] + nscore


def get_mcr(domain):
    """
    Generates an mcr value for a single domain
    :param domain:
    :return:
    """
    min_subtr = 3
    maxl = 0
    for i in range(min_subtr, len(domain)):
        tuples = ngrams(domain, i)
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
    return (maxl / int(len(domain)))


def get_ns(domain, n):
    """
    generate a tuple containing the 3 ns values for a single domain
    :param domain:
    :param n:
    :return:
    """
    tuples = ngrams(domain, n)
    myngrams = (''.join(t) for t in tuples)
    scoresum = 0
    for s in myngrams:
        counter = 0
        for words in ns_dict:
            if s in words:
                counter += 1
        logger.debug(" sub:%s count:%s" % (s, counter))
        scoresum += counter
    return scoresum / (len(domain) - n + 1)
