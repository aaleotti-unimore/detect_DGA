# coding=utf-8
from __future__ import division

import logging
import enchant
import numpy as np
import pandas as pd
from nltk import ngrams
from sklearn.externals import joblib
from sklearn.datasets import load_files

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcr_dict = enchant.Dict("en_US")
ns_dict = open("features/top10000en.txt").readlines()

path_good = 'DGA-master/benign/majestic_million.csv'
path_bad = 'DGA-master/DGA/all_dga.txt'


def generate_dataset(sample):
    good_df = pd.read_csv(path_good, usecols=['Domain'])
    bad_df = pd.read_csv(path_bad, sep=' ', header=None, names=['Domain', 'Type'], usecols=['Domain'])
    good_df['Target'] = 0
    bad_df['Target'] = 1
    df = pd.DataFrame(pd.concat((good_df, bad_df))).sample(sample)
    joblib.dump(df, "datas/dataframe_%s.pkl" % sample, compress=5)
    return df


def load_dataset(sample):
    try:
        return joblib.load("dataframe_%s.pkl" % sample)
    except IOError as e:
        logger.error(e)
        return None
