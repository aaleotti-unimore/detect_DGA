# coding=utf-8
from __future__ import division
import sys
import random
from sklearn.utils import shuffle
import pandas as pd
from numpy.random import RandomState
from pandas import read_json
from sklearn import preprocessing
from sklearn.pipeline import FeatureUnion

# from detect_DGA import isKULA
from features_extractors import *

import os

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
path_good = ('../datasets/majestic_million.csv')
path_bad = ('../datasets/all_dga.txt')

# nuovo dataset
domains_dataset = os.path.join(basedir, "../datasets/legit-dga_domains.csv")  # max lines = 133929
features_dataset = os.path.join(basedir, "../datas/features_dataset.csv")

suppobox = os.path.join(basedir, "../datas/suppobox_dataset.csv")


def generate_domain_dataset(n_samples=None):
    logger.info("Generating new dataset with %s samples" % n_samples)
    df = pd.DataFrame(
        pd.read_csv(domains_dataset, sep=",", usecols=['domain', 'class'])
    )
    if n_samples:
        return df.sample(n=n_samples, random_state=RandomState())
    return df


def load_balboni(n_samples=None):
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
    if n_samples:
        df = df.sample(n=n_samples, random_state=42, replace=True)
    df = pd.concat([df.drop(['dns'], axis=1), df['dns'].apply(pd.Series)], axis=1)

    # TODO ripulire le righe del dataset che hanno rrname vuoto o con un numero o con '?'

    ###########
    # stampa tutti i nomi di dominio con risposta NXDOMAIN su un file
    # df = df[(df.rcode == u'NXDOMAIN') & (u'unimo' not in df.rrname) & (df.rrname != u'') & (u'sophos' not in df.rrname) ]
    # df['rrname'].to_csv("DGA-master/NX.txt")
    ############

    logger.debug(df)
    return df


# def save_features_dataset(n_samples=None):
#     n_jobs = 1
#     if detect_DGA.isKULA:
#         logger.debug("detected kula settings")
#         logger.setLevel(logging.INFO)
#         n_jobs = 9
#
#     ft = FeatureUnion(
#         transformer_list=[
#             ('mcr', MCRExtractor()),
#             ('ns1', NormalityScoreExtractor(1)),
#             ('ns2', NormalityScoreExtractor(2)),
#             ('ns3', NormalityScoreExtractor(3)),
#             ('ns4', NormalityScoreExtractor(4)),
#             ('ns5', NormalityScoreExtractor(5)),
#             ('len', DomainNameLength()),
#             ('vcr', VowelConsonantRatio()),
#             ('ncr', NumCharRatio()),
#         ],
#         n_jobs=n_jobs
#     )
#
#     logger.debug("\n%s" % ft.get_params())
#
#     xy = pd.DataFrame(
#         pd.read_csv(domains_dataset, sep=",", usecols=['domain', 'class'])
#     )
#
#     if n_samples:
#         logger.info("sample size: %s" % n_samples)
#         xy = xy.sample(n=n_samples, random_state=RandomState())
#     else:
#         logger.info("Converting all domains")
#
#     X = xy['domain'].values.reshape(-1, 1)
#
#     df = pd.DataFrame(np.c_[xy, ft.transform(X)],
#                       columns=['domain', 'class', 'mcr', 'ns1',
#                                'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr'])
#
#     df.to_csv(("../datas/features_dataset.csv"), index=False)
#     logger.info("features_dataset.csv saved to disk")
#     return True

def delete_column(df, column_name):
    # TODO debug
    columns = list(df.columns)
    columns.remove(column_name)
    return df[columns]
    pass


# # TODO debug : OK (ricontrolla cosa succede quando passi una lista di filename)
# def load_and_concat_dataset(df_filenames, usecols=None):
#     if type(df_filenames) == type(''):
#         result = pd.read_csv(df_filenames, usecols=usecols)
#         pass
#     elif type(df_filenames) == type([]):
#         result = None
#         for filename in df_filenames:
#             partial_df = pd.read_csv(filename, usecols=usecols)
#             if result is not None:
#                 result = pd.concat([result, partial_df])
#             else:
#                 result = partial_df
#             pass
#         pass
#     # else:
#     #     raise TypeError('df_filenames must be a string or a list of strings')
#     return result
#     pass


# TODO debug : OK
def extract_features(df, n_jobs=1):
    # FEATURES EXTRACTOR
    ft = get_feature_union(n_jobs=n_jobs)

    logger.debug("\n%s" % ft.get_params())



    X = df['domain'].values.reshape(-1, 1)

    out_df = pd.DataFrame(np.c_[df[['domain', 'class']], ft.transform(X)],
                          columns=['domain', 'class', 'mcr', 'ns1',
                                   'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr'])

    # out_df.to_csv((out_file), index=False)
    # logger.info("features_dataset.csv saved to disk")
    return out_df


def load_features_dataset(n_samples=None, dataset=features_dataset):
    if dataset == "suppobox":
        dataset = suppobox
    df = pd.DataFrame(pd.read_csv(dataset, sep=","))
    if n_samples:
        df = df.sample(n=n_samples, random_state=RandomState())
    X = df[['mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr']].values
    y = np.ravel(lb.fit_transform(df['class'].values))
    return X, y


def __random_line(afile):
    """
    gets a random line from a file
    :param afile: filex
    :return: line
    """
    lines = open(afile).read().splitlines()
    return random.choice(lines)


def __generate_suppobox_dataset(n_samples=None):
    suppodictdict = os.path.join(basedir, "../datasets/suppobox/suppodict.txt")
    li = []
    for i in range(0, n_samples):
        w1 = __random_line(suppodictdict)
        w1 += __random_line(suppodictdict)
        li.append(w1)

    X = pd.DataFrame(li)
    y = np.chararray([len(li), 1], itemsize=3)
    y[:] = "dga"
    print(y)
    return X, y


def __save_suppobox_dataset(n_samples=None):
    n_jobs = 1
    import socket
    if socket == "classificatoredga":
        logger.debug("detected kula settings")
        logger.setLevel(logging.INFO)
        n_jobs = 9

    ft = get_feature_union()

    logger.debug("\n%s" % ft.get_params())

    X, y = __generate_suppobox_dataset(n_samples=n_samples)

    if n_samples:
        logger.info("sample size: %s" % n_samples)
        X = X.sample(n=n_samples, random_state=RandomState())
    else:
        logger.info("Converting all domains")

    df = pd.DataFrame(np.c_[X, y, ft.transform(X)],
                      columns=['domain', 'class', 'mcr', 'ns1', 'ns2', 'ns3', 'ns4', 'ns5', 'len', 'vcr', 'ncr'])

    df.to_csv(os.path.join(basedir, "../datas/suppobox_dataset.csv"), index=False)
    logger.info("features_dataset.csv saved to disk")
    return True


def load_both_datasets(n_samples=None, verbose=False):
    X1, y1 = load_features_dataset()
    X2, y2 = load_features_dataset(
        dataset="suppobox")
    X = np.concatenate((X1, X2), axis=0).astype(float)
    y = np.concatenate((y1, y2), axis=0).astype(int)
    if verbose:
        # logger.debug("X shape %s" % (X.shape))
        # logger.debug("y shape %s" % (y.shape))
        get_balance(y)
    if n_samples:
        return shuffle(X, y, random_state=RandomState(), n_samples=n_samples)
    return shuffle(X, y, random_state=RandomState())


def get_balance(y):
    unique, counts = np.unique(y, return_counts=True)
    di = dict(zip(unique, counts))
    logger.debug("Dataset balance")
    for key, value in di.iteritems():
        logger.debug("%s %s" % (key, value / np.shape(y)[0] * 100))


if __name__ == '__main__':
    dir = '../datasets/total/'
    for dataset_filename in os.listdir(dir):
        dataset = load_and_concat_dataset(dir + dataset_filename)
        feat = extract_features(dataset, n_jobs=1)
        feat.to_csv(dir+'../feat/' + dataset_filename + '.feat', index=False)


