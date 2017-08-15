# coding=utf-8
from __future__ import division

import logging

import enchant
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
logger = logging.getLogger(__name__)

mcr_dict = enchant.Dict("en_US")
ns_dict = open("features/top10000en.txt").readlines()


class MCRExtractor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.logger = logger
        self.logger.setLevel(logging.INFO)

    def __get_mcr(self, domain_name):
        """
        Meaningful Characters Ratio. Models the ratio of characters of the string p that comprise a meaningful word. Low values indicate automatic algorithms. Specifically, we split p into n meaningful subwords wi of at least 3 symbols: |wi| ≥ 3, leaving out as few symbols as possible: R(d) = R(p) = max((sum from i=1 to n) |wi|)/|p|. If p = facebook, R(p) = (|face| + |book|)/8 = 1, the prefix is fully composed of meaningful words, whereas p = pub03str, R(p) = (|pub|)/8 = 0.375.

        :param domain_name: string
        :return: ratio. float value
        """
        min_subtr = 3
        maxl = 0
        for i in range(min_subtr, len(domain_name)):
            # tuples = ngrams(domain_name, i) #example: facebook=fa+ac+ce+eb+bo+oo+ok
            tuples = zip(
                *[domain_name[j::i] for j in range(
                    i)])  # alternative way to split the string. in this case, text chunks are not overlapping. facebook=fa+ce+bo+ok
            split = [''.join(t) for t in tuples]
            tmpsum = 0
            tmps = []
            for s in split:
                if mcr_dict.check(s):
                    tmpsum += len(s)
                    tmps.append(s)
            if tmps:
                self.logger.debug("      %s , sum(w_i)=%s" % (tmps, tmpsum))
            if tmpsum > maxl:
                maxl = tmpsum
        return maxl / int(len(domain_name))

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        f = np.vectorize(self.__get_mcr)
        return f(df)

    def fit(self, X, y=None):
        return self  # does nothing


class NormalityScoreExtractor(BaseEstimator, TransformerMixin):
    def __init__(self, n):
        self.logger = logger
        self.logger.setLevel(logging.INFO)
        self.n = n

    def __get_ns(self, domain_name):
        """
        n-gram Normality Score.
        This class of features captures the pronounceability of a domain name. The more permissible the combinations of phonemes, the more pronounceable a word is. Domains with a low number of such combinations are likely DGA-generated.
        We calculate this class of features by extracting the n-grams of p, which are the substrings of p of length n {1, 2, 3}, and counting their occurrences in the (English) language dictionary.
        The features are thus parametric to n: Sn(d) = Sn(p) := ((sum of n-gram t in p) count(t))/(|p|−n+1), where count(t) are the occurrences of the n-gram t in the dictionary
        :param domain_name: string
        :param self.n: size of the n-grams. int
        :return: score of the n-grams. float
        """
        tuples = ngrams(domain_name, self.n)
        myngrams = (''.join(t) for t in tuples)
        scoresum = 0
        for s in myngrams:
            counter = 0
            for words in ns_dict:
                if s in words:
                    counter += 1
            self.logger.debug(" sub:%s count:%s" % (s, counter))
            scoresum += counter
        return scoresum / (len(domain_name) - self.n + 1)

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        f = np.vectorize(self.__get_ns)
        return f(df)

    def fit(self, X, y=None):
        return self  # does nothing
