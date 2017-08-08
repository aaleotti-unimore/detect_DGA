import logging

import numpy as np
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.externals import joblib
from sklearn.metrics import *
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict, train_test_split
import time
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.level = logging.DEBUG


def tenfold_SVC(n_splits, X_len):
    X_g = joblib.load('datas/good_%s' % int(X_len / 2))
    X_b = joblib.load('datas/bad_%s' % int(X_len / 2))
    X = np.concatenate((X_g, X_b))
    y = joblib.load('datas/y_%s' % X_len)
    clf = joblib.load('models/linear_SVC_(%s).pkl' % X_len)

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    predict = cross_val_predict(clf, X, y, cv=cv)
    # logger.debug("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # joblib.dump(scores, "models/%sfold_SVC_(%s)_scores.pkl" % (n_splits, X_len))
    joblib.dump(cv, "models/%sfold_SVC_(%s)_cv.pkl" % (n_splits, X_len))

    fpr, tpr, thresholds = roc_curve(y, predict)
    prec = precision_score(y, predict)
    recall = recall_score(y, predict)
    f1score = f1_score(y, predict)
    _auc = auc(fpr, tpr)

    logger.info("\tPrecision: %1.3f" % prec)
    logger.info("\tRecall: %1.3f" % recall)
    logger.info("\tF1: %1.3f" % f1score)
    logger.info("\tAUC %1.3f\n" % _auc)


def generate_dataset(X_len):
    from features import data_generator
    data_generator.generate_good(X_len=X_len)
    data_generator.generate_bad(X_len=X_len)
    data_generator.create_target(X_len=X_len)


def train_svc(X_len, kernel, max_iter=-1, C=1):
    X_g = joblib.load('datas/good_%s' % int(X_len / 2))
    X_b = joblib.load('datas/bad_%s' % int(X_len / 2))
    X = np.concatenate((X_g, X_b))
    y = joblib.load('datas/y_%s' % X_len)
    X, y = shuffle(X, y, random_state=0)
    n_estimators = 10
    start = time.time()

    clf = SVC(kernel='linear', C=C, max_iter=max_iter, verbose=False)
    # clf = OneVsRestClassifier(
    #     BaggingClassifier(
    #         SVC(kernel=kernel, probability=True, class_weight='balanced', max_iter=max_iter, C=C),
    #         max_samples=1.0 / n_estimators,
    #         n_estimators=n_estimators))
    clf.fit(X, y)

    end = time.time()
    score = clf.score(X, y)
    # logger.debug("Bagging SVC time:%s score:%s " % (end - start, score))
    joblib.dump(clf, "models/%s_SVC_(%s).pkl" % (kernel, X_len))
    return score


X_len = 10000
# generate_dataset(X_len)
lst = [0]
# for i in range(1, 1000):
#     lst.append(train_svc(X_len, "linear", i))
#
# print("index best score: %s" % lst.index(max(lst)))
train_svc(X_len, "linear", 1)
tenfold_SVC(10, X_len)
logger.info("exiting...")
