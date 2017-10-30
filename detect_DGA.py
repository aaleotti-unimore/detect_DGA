# coding=utf-8
import logging
import os
import socket
import numpy as np
import json

import pandas as pd

import dask.bag as db
import dask.multiprocessing

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from features.data_generator import load_both_datasets, load_features_dataset
from features.features_extractors import *

from myclassifier import MyClassifier

from utils import *

basedir = os.path.dirname(__file__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
clf_n_jobs = 8


# TODO debug : OK (ricontrolla cosa succede quando passi una lista di filename)
def load_and_concat_dataset(df_filenames, usecols=None):
    if type(df_filenames) == type(''):
        result = pd.read_csv(df_filenames, usecols=usecols)
        pass
    elif type(df_filenames) == type([]):
        result = None
        for filename in df_filenames:
            partial_df = pd.read_csv(filename, usecols=usecols)
            if result is not None:
                result = pd.concat([result, partial_df])
            else:
                result = partial_df
            pass
        pass
    else:
        return None
    return result
    pass

# TODO debug
def has_rrname_filter(record):
    return record['dns'].has_key('rrname')
    pass

def predict(estimator, domains):
    from sklearn.pipeline import Pipeline
    from features.features_extractors import get_feature_union
    if len(domains) == 0:
        raise ValueError("Empty array")
    if len(domains) == 1:
        domains = np.array(domains).reshape(1, -1)
    else:
        domains = np.array(domains).reshape(-1, 1)

    dom_ext = DomainExtractor()

    pip = Pipeline(steps=[('domain extractor', dom_ext), ('feats', get_feature_union()), ('clf', estimator)])

    pred = pip.predict(domains)
    # for index, domain in enumerate(domains):
    #     print("%s -> %s" % (domain, ("legit" if pred[index] == 0 else "DGA")))
    return pred

# TODO debug
def test_on_balboni_set(estimator, in_file):
    dataset = db.read_text(in_file, compression='gzip').map(json.loads)
    dataset = dataset.filter(lambda record: True if record['dns'].has_key('rrname') else False)
    dataset = dataset.filter(lambda record: True if record['dns']['rrname'] != '' else False)
    dataset = dataset.filter(lambda record: True if '.' in record['dns']['rrname'] else False)

    def map_labels(x):
        x['label'] = str(predict(estimator, x['dns']['rrname'])[0])
        return x
        pass

    dataset = dataset.map(map_labels)
    dataset = dataset.map(json.dumps)
    dataset.to_textfiles('dns_requests_dataset_with_labels/*.json')
    pass




def train_all_dataset():
    dir = 'datasets/feat/'
    filenames = os.listdir(dir)
    paths = []

    for filename in filenames:
        paths.append(dir + filename)
        pass

    df = load_and_concat_dataset(paths)
    x, y = get_x_y(df, 'class')

    x = delete_column(x, 'domain')
    x = x.values

    y = y.map(lambda label: 0 if label == 'legit' else 1)
    y = y.values

    rf = RandomForestClassifier(n_jobs=8)
    clf = MyClassifier(rf)

    results = clf.cross_validate(x,y)
    clf.save_results(results)
    clf.save_clf()

    return rf.fit(X=x,y=y)
    pass

if __name__ == "__main__":

    try:
        dask.set_options(get=dask.multiprocessing.get)
        dask.set_options(optimize_graph=True)
        dask.set_options(num_workers=8)

        clf = train_all_dataset()
        test_on_balboni_set(clf, '../06/*/*/*')

        logger.info("Exiting...")
    except Exception as e:
        f_err = open('error.txt', 'w')
        f_err.write(str(e) + ' ' + str(e.args))
        f_err.close()
        pass
