import logging

import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from sklearn.model_selection import KFold, cross_val_score

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.level = logging.INFO


def tenfold_SVC():
    X_g = joblib.load('good.pkl')
    X_b = joblib.load('bad.pkl')
    X = np.concatenate((X_g, X_b))
    y = joblib.load('y.pkl')
    # X, y = shuffle(X, y, random_state=0)
    logger.debug("%s" % X)
    logger.debug("%s" % y)

    cv = KFold(n_splits=10, shuffle=True, random_state=0)
    clf = svm.SVC(kernel='linear', C=5)
    scores = cross_val_score(clf, X, y, cv=cv)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    joblib.dump(clf, "models/10fold_model.pkl")


def create_target(limit):
    y_0 = np.zeros(limit)
    y_1 = np.ones(limit)
    y = np.concatenate((y_0, y_1))
    joblib.dump(y, "y.pkl")

def get_metrics(clf):
    from sklearn.metrics import f1_score


# create_target(2001)
# tenfold_SVC()
get_metrics(joblib.load(clf, "models/10fold_model.pkl"))
logger.info("exiting...")
