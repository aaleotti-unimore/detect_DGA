# coding=utf-8
from __future__ import division

import logging

import pandas as pd
from pandas import read_json
from sklearn.externals import joblib

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

path_good = 'datasets/majestic_million.csv'
path_bad = 'datasets/all_dga.txt'

legitdga_domains = "datasets/legit-dga_domains.csv"  # max lines = 133929


def generate_dataset(n_samples):
    logger.info("Generating new dataset with %s samples" % n_samples)
    df = pd.DataFrame(
        pd.read_csv(legitdga_domains, sep=",", usecols=['domain', 'class'])
    )
    # joblib.dump(df, "datas/dataframe_%s.pkl" % n_samples)
    # logger.info("dataframe saved to datas/dataframe_%s.pkl" % n_samples)

    return df.sample(n_samples, random_state=42)


def load_dataset(samples):
    try:
        ld = joblib.load("datas/dataframe_%s.pkl" % samples)
        logger.info("dataframe dataframe_%s.pkl loaded" % samples)
        return ld
    except IOError as e:
        logger.warning(e)
        return generate_dataset(samples)


def load_balboni(sample):
    """
    unlabeled data from balbonee
    :param sample: size of sample
    :return:
    """
    path_json = [
        'datasets/062000_00.json',
        # 'datasets/062001_00.json',
        # 'datasets/062002_00.json',
        # 'datasets/062003_00.json',
    ]

    df = pd.DataFrame()
    for file in path_json:
        tmp = read_json(file, lines=True, orient='record')
        logger.debug("json %s opened" % file)
        df = pd.concat([df, tmp], axis=0)

    df = df.sample(n=sample, random_state=42, replace=True)
    df = pd.concat([df.drop(['dns'], axis=1), df['dns'].apply(pd.Series)], axis=1)

    #TODO ripulire le righe del dataset che hanno rrname vuoto o con un numero o con '?'

    ###########
    # stampa tutti i nomi di dominio con risposta NXDOMAIN su un file
    # df = df[(df.rcode == u'NXDOMAIN') & (u'unimo' not in df.rrname) & (df.rrname != u'') & (u'sophos' not in df.rrname) ]
    # df['rrname'].to_csv("DGA-master/NX.txt")
    ############

    logger.debug(df)
    return df
