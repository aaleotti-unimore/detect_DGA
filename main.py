import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

h = .02  # step size in the mesh

from features import mcr, ns

string = "pub03str"
mcr = mcr.get_ratio(string)
logger.info("Meaningful character ratio = %s" % mcr )
nscore=[]
for i in [1,2,3]:
    nscore.append(ns.get_score(string,i))
    logger.info("Normality score for %d = %f "%(i,nscore[i-1]))

f_t =  [mcr] + nscore
logger.info(f_t)

#
# from nltk import ngrams, bigrams
# sentence = 'facebook'
# n = 2
# sixgrams = ngrams(sentence, n)
# # bigram = bigrams(sentence)
# # for bi in bigram:
# #     logger.info(bi)
# for grams in sixgrams:
#   logger.info(grams)
