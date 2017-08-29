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

path_json = [
    'datasets/062000_00.json',
    # 'datasets/062001_00.json',
    # 'datasets/062002_00.json',
    # 'datasets/062003_00.json',
]

legitdga_domains = "datasets/legit-dga_domains.csv"  # max lines = 133929


def generate_dataset(n_samples):
    logger.info("Generating new dataset with %s samples" % n_samples )
    df = pd.DataFrame(
            pd.read_csv(legitdga_domains, sep=",", usecols=['domain', 'class'])
        )
    joblib.dump(df, "datas/dataframe_%s.pkl" % n_samples, compress=5)
    logger.info("dataframe saved to datas/dataframe_%s.pkl" % n_samples)

    return df.sample(n_samples, random_state=42)


def load_dataset(samples):
    try:
        ld = joblib.load("datas/dataframe_%s.pkl" % samples)
        logger.info("dataframe dataframe_%s.pkl loaded" % samples)
        return ld
    except IOError as e:
        logger.warning(e)
        return generate_dataset(samples)


def load_json(sample=1):
    """
    unlabeled data from balbonee
    :param sample:
    :return:
    """
    # pd.concat([df.drop(['b'], axis=1), df['b'].apply(pd.Series)], axis=1)

    df = pd.DataFrame()
    for file in path_json:
        tmp = read_json(file, lines=True, orient='record')
        logger.debug("json %s opened" % file)
        df = pd.concat([df, tmp], axis=0)

    df = df.sample(n=sample, random_state=42, replace=True)
    df = pd.concat([df.drop(['dns'], axis=1), df['dns'].apply(pd.Series)], axis=1)
    df = df[df['rrname'] > 0]

    # stampa tutti i nomi di dominio con risposta NXDOMAIN su un file
    # df = df[(df.rcode == u'NXDOMAIN') & (u'unimo' not in df.rrname) & (df.rrname != u'') & (u'sophos' not in df.rrname) ]
    # df['rrname'].to_csv("DGA-master/NX.txt")
    ############


    # ##labeling
    # for i, row in df.iterrows():
    #     target_val = 0
    #     domain = df.get_value(index=i, col='rrname')
    #     logger.debug("domain %s" % domain)
    #     if check_dga(domain):
    #         target_val = 1
    #     df.set_value(i, 'Target', target_val)
    logger.debug(df)
    return df
