# coding=utf-8
"""
n-gram Normality Score. This class of features captures the pronounceability of a domain name. This is a well-studied problem in linguistics, and can be reduced to quantifying the extent to which a string adheres to the phonotactics of the (English) language. The more permissible the combinations of phonemes [4, 18], the more pronounceable a word is. Domains with a low number of such combinations are likely DGA-generated. We calculate this class of features by extracting the n-grams of p, which are the substrings of p of length n  {1, 2, 3}, and counting their occurrences in the (English) language dictionary3
. If needed, the dictionary can be extended to include known benign, yet DGA-looking names. The features are thus parametric to n: Sn(d) = Sn(p) := (Pn-gram t in pcount(t))/(|p|−n+1), where count(t) are the occurrences of the n-gram t in the dictionary. For example, S2(facebook) = fa109 +ac343 + ce438 + eb29 + bo118 + oo114 + ok45 = 170.8 seems a non-automatically generated, whereas S2(aawrqv) = aa4 + aw45 + wr17 + rq0 + qv0 = 13.2 seems automatically generated.
"""
from __future__ import division

import logging

from nltk import ngrams

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

dictionary = open("features/top10000en.txt").readlines()

def get_score(domain, n):
    tuples = ngrams(domain, n)
    myngrams = (''.join(t) for t in tuples)
    scoresum = 0
    for s in myngrams:
        counter = 0
        for words in dictionary:
            if s in words:
                counter += 1
        logger.debug(" sub:%s count:%s" % (s, counter))
        scoresum += counter
    return scoresum / (len(domain) - n + 1)
