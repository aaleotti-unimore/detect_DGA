# coding=utf-8
from __future__ import division

import logging
import enchant
import numpy as np
import pandas as pd
import json
from pprint import pprint
from nltk import ngrams
from sklearn.externals import joblib
from sklearn.datasets import load_files
from pandas import read_json

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

mcr_dict = enchant.Dict("en_US")
ns_dict = open("features/top10000en.txt").readlines()

path_good = 'DGA-master/benign/majestic_million.csv'
path_bad = 'DGA-master/DGA/all_dga.txt'

path_json = [
    'DGA-master/062000_00.json',
    # 'DGA-master/062001_00.json',
    # 'DGA-master/062002_00.json',
    # 'DGA-master/062003_00.json',
]

bad_df = pd.read_csv(path_bad, sep=' ', header=None, names=['Domain', 'Type'], usecols=['Domain'])


## OLD TXT DATASET
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
        return joblib.load("datas/dataframe_%s.pkl" % sample)
    except IOError as e:
        logger.error(e)
        return None

def check_dga(domain):
    matched = [c for c in bad_df if domain in c]
    if matched:
        return True  # malicious
    return False


def load_json(sample=1):
    # pd.concat([df.drop(['b'], axis=1), df['b'].apply(pd.Series)], axis=1)
    df = pd.DataFrame()
    for file in path_json:
        tmp = read_json(file, lines=True, orient='record')
        df = pd.concat([df, tmp], axis=0)


    df = df.sample(n=sample, random_state=42, replace=True)
    df = pd.concat([df.drop(['dns'], axis=1), df['dns'].apply(pd.Series)], axis=1)
    df = df[(df.rrname != u'') & (df.rrname > 3) & (df.rrname != u'?')]

    #stampa tutti i nomi di dominio con risposta NXDOMAIN su un file
    # df = df[(df.rcode == u'NXDOMAIN') & (u'unimo' not in df.rrname) & (df.rrname != u'') & (u'sophos' not in df.rrname) ]
    # df['rrname'].to_csv("DGA-master/NX.txt")
    ############


    ##labeling
    for i, row in df.iterrows():
        target_val = 0
        domain = df.get_value(index=i, col='rrname')
        logger.debug("domain %s" % domain)
        if check_dga(domain):
            target_val = 1
        df.set_value(i, 'Target', target_val)

    return df
