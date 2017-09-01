# coding=utf-8
"""
Module for feature extraction. used in a pipeline.
"""
from __future__ import division

import logging
import tldextract
import numpy as np
from nltk import ngrams
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

import os

dir = os.path.dirname(__file__)
dictfile = os.path.join(dir, "top10000en.txt")
eng_dict = open(dictfile).readlines()

from collections import Counter


# TODO possibili features trovate sul paper di DeepDGA di Anderson Woodbridge Filar: lunghezza del nome di dominio, entropia della distrubuzione dei caratteri, rapporto vocali/consonanti, contare la frequenza degli n-grammi 3,4,5 all'interno del dataset Alexa 1Million. Inoltre cambiare i valori del Normality Score Extractor da 1,2,3 a 3,4,5

class MCRExtractor(BaseEstimator, TransformerMixin):
    """
    Meaningful Characters Ratio. Models the ratio of characters of the string p that comprise a meaningful word.
    Low values indicate automatic algorithms. Specifically, we split p into n meaningful subwords wi of at least 3 symbols: |wi| ≥ 3, leaving out as few symbols as possible: R(d) = R(p) = max((sum from i=1 to n) |wi|)/|p|.
    If p = facebook, R(p) = (|face| + |book|)/8 = 1, the prefix is fully composed of meaningful words, whereas p = pub03str, R(p) = (|pub|)/8 = 0.375.
    """

    def __init__(self, mode=0):
        self.mode = mode

    def __get_mcr(self, domain_name):
        logger.debug("domain name: %s" % domain_name)
        # if len(str(domain_name)) == 0:
        #     return 0

        min_subtr = 3
        maxl = 0
        for i in range(min_subtr, len(str(domain_name))):
            if self.mode == 1:
                tuples = ngrams(str(domain_name), i)  # overlapping chunks. example: facebook=fa+ac+ce+eb+bo+oo+ok
            else:
                # alternative way to split the string. in this case, text chunks are not overlapping. eg: facebook=fa+ce+bo+ok
                tuples = zip(
                    *[str(domain_name)[j::i] for j in range(
                        i)])
            split = [''.join(t) for t in tuples]
            tmpsum = 0
            tmps = []
            for s in split:
                if s in eng_dict:
                    tmpsum += len(s)
                    tmps.append(s)

            if tmpsum > maxl:
                maxl = tmpsum
        return maxl / int(len(str(domain_name)))

    def transform(self, df, y=None):
        f = np.vectorize(self.__get_mcr)
        return f(df)

    def fit(self, X, y=None):
        return self  # does nothing

    def __getstate__(self):
        d = dict(self.__dict__)
        # del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class NormalityScoreExtractor(BaseEstimator, TransformerMixin):
    """
     n-gram Normality Score.
        This class of features captures the pronounceability of a domain name. The more permissible the combinations of phonemes, the more pronounceable a word is. Domains with a low number of such combinations are likely DGA-generated.
        We calculate this class of features by extracting the n-grams of p, which are the substrings of p of length n {1, 2, 3}, and counting their occurrences in the (English) language dictionary.
        The features are thus parametric to n: Sn(d) = Sn(p) := ((sum of n-gram t in p) count(t))/(|p|−n+1), where count(t) are the occurrences of the n-gram t in the dictionary
    """

    def __init__(self, n):
        self.n = n

    def __get_ns(self, domain_name):
        logger.debug("domain name: %s" % domain_name)

        tuples = ngrams(str(domain_name), self.n)
        myngrams = (''.join(t) for t in tuples)
        scoresum = 0
        for s in myngrams:
            counter = 0
            for words in eng_dict:
                if s in words:
                    counter += 1
            scoresum += counter
        return scoresum / (len(str(domain_name)) - self.n + 1)

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        f = np.vectorize(self.__get_ns)
        return f(df)

    def fit(self, X, y=None):
        return self  # does nothing

    def __getstate__(self):
        d = dict(self.__dict__)
        # del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class NumCharRatio(BaseEstimator, TransformerMixin):
    """
    % of numerical characters feature
    """

    def __init__(self):
        pass

    def __get_ncr(self, domain_name):
        logger.debug("domain name: %s" % domain_name)
        counter = Counter(domain_name)
        ncr = 0
        for key, value in counter.iteritems():
            if key.isdigit():
                ncr += value

        return ncr / len(domain_name)

    def transform(self, df, y=None):
        """The workhorse of this feature extractor"""
        f = np.vectorize(self.__get_ncr)
        return f(df)

    def fit(self, X, y=None):
        return self  # does nothing

    def __getstate__(self):
        d = dict(self.__dict__)
        # del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class ItemSelector(BaseEstimator, TransformerMixin):
    """For data grouped by feature, select subset of data at a provided key.

    The data is expected to be stored in a 2D data structure, where the first
    index is over features and the second is over samples.  i.e.

    >> len(data[key]) == n_samples

    Please note that this is the opposite convention to scikit-learn feature
    matrixes (where the first index corresponds to sample).

    ItemSelector only requires that the collection implement getitem
    (data[key]).  Examples include: a dict of lists, 2D numpy array, Pandas
    DataFrame, numpy record array, etc.

    >> data = {'a': [1, 5, 2, 5, 2, 8],
               'b': [9, 4, 1, 4, 1, 3]}
    >> ds = ItemSelector(key='a')
    >> data['a'] == ds.transform(data)

    ItemSelector is not designed to handle data grouped by sample.  (e.g. a
    list of dicts).  If your data is structured this way, consider a
    transformer along the lines of `sklearn.feature_extraction.DictVectorizer`.

    Parameters
    ----------
    key : hashable, required
        The key corresponding to the desired value in a mappable.
    """

    def __init__(self, key):
        self.key = key

    def fit(self, x, y=None):
        return self

    def transform(self, data_dict):
        X = (data_dict[self.key].values).reshape(-1, 1)
        logger.debug(X)
        return X

    def __getstate__(self):
        d = dict(self.__dict__)
        # del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)


class DomainExtractor(BaseEstimator, TransformerMixin):
    """
    Extracts the domain name frome an url
    """

    def __init__(self):
        pass

    def fit(self, x, y=None):
        return self

    def __get_domain(self, domain):
        if domain is u'':
            return "string"
        else:
            extractor = tldextract.TLDExtract(suffix_list_urls=None)
            ext = extractor(domain)
            logger.debug("%s domain: %s" % (domain, ext.domain))
            if ext.domain:
                return ext.domain
            else:
                return "string"

    def transform(self, X):
        f = np.vectorize(self.__get_domain)
        return f(X)

    def __getstate__(self):
        d = dict(self.__dict__)
        # del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
