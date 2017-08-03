"""
Meaningful Characters Ratio.
Models the ratio of characters of the string p in the domain d that comprise a meaningful word. Low values indicate automatic algorithms.
Specifically, we split p into n meaningful subwords w_i of at least 3 symbols: |w_i| >=3, leaving out as few symbols as possible: R(d) = R(p) = max(sum |w_i|, i=1 to n)/|p|. 
If p = facebook, R(p) = (|face| + |book|)/8 = 1, the prefix is fully composed of
meaningful words, whereas p = pub03str, R(p) = (|pub|)/8 = 0.375.
"""

from __future__ import division

import logging

import enchant

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


d = enchant.Dict("en_US")


def get_ratio(domain):
    res = 0
    min_subtr = 3
    maxl = 0
    for i in range(min_subtr, len(domain)):
        split = split(domain, i)
        tmpsum = 0
        for s in split:
            if d.check(s):
                tmpsum += len(s)
        logger.debug("%s , sum(w_i)=%s" % (split,tmpsum))
        if tmpsum > maxl:
            maxl = tmpsum
    return (maxl / int(len(domain)))


def split(s, chunk_size):
    """
    splits a string into chunks
    :param s: string
    :param chunk_size: size of the chunk
    :return: list of strings
    """
    a = zip(*[s[i::chunk_size] for i in range(chunk_size)])
    return [''.join(t) for t in a]
