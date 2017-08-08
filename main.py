import logging

import numpy as np
from sklearn.utils import shuffle
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import (precision_score, recall_score,
                             f1_score, auc, roc_curve)
from sklearn.model_selection import KFold, cross_val_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.level = logging.INFO


def tenfold_SVC(n_splits, X_len, max_iter=-1):

    X_g = joblib.load('datas/good_%s' % int(X_len / 2))
    X_b = joblib.load('datas/bad_%s' % int(X_len / 2))
    X = np.concatenate((X_g, X_b))
    y = joblib.load('datas/y_%s' % X_len)
    X, y = shuffle(X, y, random_state=0)

    logger.debug("%s" % X)
    logger.debug("%s" % y)

    cv = KFold(n_splits=n_splits, shuffle=False)
    clf = svm.SVC(kernel='linear', C=5, max_iter=max_iter)
    clf.fit(X, y)
    scores = cross_val_score(clf, X, y, cv=cv)
    logger.debug("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    joblib.dump(clf, "models/(%s)fold_model_(%s).pkl" % (n_splits, X_len))
    joblib.dump(scores, "models/(%s)fold_model_(%s)_scores.pkl" % (n_splits, X_len))
    joblib.dump(cv, "models/(%s)fold_model_(%s)_cv.pkl" % (n_splits, X_len))

    for train, test in cv.split(X):
        X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]
        y_pred = clf.predict(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        print("\tPrecision: %1.3f" % precision_score(y_test, y_pred))
        print("\tRecall: %1.3f" % recall_score(y_test, y_pred))
        print("\tF1: %1.3f\n" % f1_score(y_test, y_pred))
        print("\tAUC %1.3f" % auc(fpr, tpr))


def generate_dataset(X_len):
    from features import data_generator
    data_generator.generate_good(X_len=X_len)
    data_generator.generate_bad(X_len=X_len)
    data_generator.create_target(X_len=X_len)

X_len=10000
# generate_dataset(X_len)
tenfold_SVC(10, X_len, max_iter=7)
logger.info("exiting...")
