# coding=utf-8
from __future__ import division

import logging
import os

import pandas as pd
from numpy.random import RandomState
from pandas import read_json
from sklearn.externals import joblib
from sklearn.pipeline import FeatureUnion
from features_extractors import *
import detect_DGA
from sklearn import preprocessing

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', None)
pd.options.display.float_format = '{:.2f}'.format
np.set_printoptions(precision=3, suppress=True)

basedir = os.path.dirname(__file__)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

lb = preprocessing.LabelBinarizer()

# vecchio dataset
path_good = os.path.join(basedir, '../datasets/majestic_million.csv')
path_bad = os.path.join(basedir, '../datasets/all_dga.txt')

# nuovo dataset
domains_dataset = os.path.join(basedir, "../datasets/legit-dga_domains.csv")  # max lines = 133929
features_dataset = os.path.join(basedir, "../datas/features_dataset.csv")


def generate_dataset(n_samples):
    logger.info("Generating new dataset with %s samples" % n_samples)
    df = pd.DataFrame(
        pd.read_csv(domains_dataset, sep=",", usecols=['domain', 'class'])
    )
    # joblib.dump(df, "datas/dataframe_%s.pkl" % n_samples)
    # logger.info("dataframe saved to datas/dataframe_%s.pkl" % n_samples)

    return df.sample(n=n_samples, random_state=RandomState())


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

    # TODO ripulire le righe del dataset che hanno rrname vuoto o con un numero o con '?'

    ###########
    # stampa tutti i nomi di dominio con risposta NXDOMAIN su un file
    # df = df[(df.rcode == u'NXDOMAIN') & (u'unimo' not in df.rrname) & (df.rrname != u'') & (u'sophos' not in df.rrname) ]
    # df['rrname'].to_csv("DGA-master/NX.txt")
    ############

    logger.debug(df)
    return df


def save_features_dataset(n_samples=None):
    n_jobs = 1
    if detect_DGA.kula:
        logger.debug("detected kula settings")
        logger.setLevel(logging.INFO)
        n_jobs = 9

    ft = FeatureUnion(
        transformer_list=[
            ('mcr', MCRExtractor()),
            ('ns1', NormalityScoreExtractor(1)),
            ('ns2', NormalityScoreExtractor(2)),
            ('ns3', NormalityScoreExtractor(3)),
            ('ns4', NormalityScoreExtractor(4)),
            ('ns5', NormalityScoreExtractor(5)),
            ('len', DomainNameLength()),
            ('vcr', VowelConsonantRatio()),
            ('ncr', NumCharRatio()),
        ],
        n_jobs=n_jobs
    )

    logger.debug("\n%s" % ft.get_params())

    xy = pd.DataFrame(
        pd.read_csv(domains_dataset, sep=",", usecols=['domain', 'class'])
    )

    if n_samples:
        logger.info("sample size: %s" % n_samples)
        xy = xy.sample(n=n_samples, random_state=RandomState())
    else:
        logger.info("Converting all domains")

    X = xy['domain'].values.reshape(-1, 1)

    df = pd.DataFrame(np.c_[xy, ft.transform(X)],
                      columns=['domain', 'class', 'mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr'])

    df.to_csv(os.path.join(basedir, "../datas/features_dataset.csv"), index=False)
    logger.setLevel(logging.DEBUG)
    return True


def load_features_dataset():
    logger.setLevel(logging.INFO)
    df = pd.DataFrame(pd.read_csv(features_dataset, sep=","))
    X = df[['mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr']].values
    y = np.ravel(lb.fit_transform(df['class'].values))
    return X, y
