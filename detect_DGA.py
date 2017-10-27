# coding=utf-8
import logging
import os
import socket
import numpy as np

import pandas as pd

from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

from features.data_generator import load_both_datasets, load_features_dataset
#from myclassifier import MyClassifier

from utils import *

basedir = os.path.dirname(__file__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
clf_n_jobs = 1


def test_balboni_dataset():
    # X, y = load_balboni(n_samples=n_samples)
    # #### test del dataset
    # model = trained_clfs['RandomForest']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=RandomState())
    # # model.set_params(features_extractors__n_jobs=2)
    # logger.debug(model)
    # y_pred = model.predict(X_test)
    #
    # logger.info("\n%s" % classification_report(y_true=y_test, y_pred=y_pred, target_names=['DGA', 'Legit']))

    # y_pred = lb.inverse_transform(y_pred)
    # y_test = lb.inverse_transform(y_test)
    # pd.options.display.max_rows = 99999999
    # logger.info(pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=['DOMAIN', 'TEST', 'PREDICTION']))
    ########

    # print(data_generator.load_balboni(20))
    pass


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

def main():
    # TODO unificare i database per il training: sia legit-dga_domains.csv che i due all_legit.txt e
    # all_dga.txt presenti su https://github.com/andrewaeva/DGA .
    # questi due file txt vanno prima pre-processati con features_extractor.DomainExtractor
    #  in modo da ottenre solo il dominio di secondo livello.

    # TODO testare i classificatori con i json di balboni,
    #  prendendo dalla colonna rrname solo quelli ripuliti.
    #  fare riferimento alla funzione data_generator.load_balboni
    #  già implementata a metà. I paper che ho letto finora usano solo
    #  i pachetti NXDOMAIN per fare detection, meglio filtrare quella colonna e usare solo quelli.

    # TODO NB: la pipeline prende in pasto un vettore di stringhe.
    # la funzione get_feature_union ritorna i vari features_extractors
    # generano le features in parallelo a partire da questo dataset.
    pass

def test_all_dataset():
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

    rf = RandomForestClassifier()

    print cross_val_score(rf, x, y, scoring='f1', cv=20)
    pass

if __name__ == "__main__":
    # model_training()
    # X, y = load_features_dataset()
    # X_test2, y_test2 = load_features_dataset(dataset="suppobox")
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    # X_test = np.concatenate((X_test, X_test2))
    # y_test = np.concatenate((y_test, y_test2))
    # X_test, y_test = shuffle(X_test, y_test, random_state=42)
    # print(X_test)
    # print(X_test.shape)
    # print(y_test)
    # print(y_test.shape)
    #
    # print(X_train.shape)

    #
    # _clf = RandomForestClassifier(random_state=True, max_features="auto", n_estimators=100,
    #                               min_samples_leaf=50, n_jobs=-1, oob_score=True)
    # # myc = MyClassifier(clf=_clf)
    # # myc.fit(X_train, y_train)
    # # myc.classification_report(X_test, y_test)
    # # myc.cross_validate(X_train, y_train)
    # # myc.plot_AUC(X_test, y_test)
    #
    # nosup = MyClassifier(clf=_clf)
    # nosup.fit(X_train, y_train)
    # nosup.classification_report(X_test, y_test, plot=True)
    # nosup.cross_validate(X_train, y_train)
    # nosup.plot_AUC(X_test, y_test)
    # from numpy import reshape
    # rndf = MyClassifier(directory="models/RandomForest tra:sup tst:sup")

    #domains = np.array(["facebook"]).reshape(-1,1)
    # .reshape(1, -1)
    #rndf.predict(domains)
    # print("PREDICT: %s " % rndf.predict(domains))

    test_all_dataset()

    logger.info("Exiting...")
