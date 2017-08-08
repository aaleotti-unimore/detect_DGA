"""
Statistical Linguistic Filter.
"""

import logging

import numpy as np

logger = logging.getLogger(__name__)
logger.level = logging.DEBUG
from sklearn.externals import joblib


def calc_mean(X):
    """
    calculates mean of the training set, saving it into a mean.pkl file
    :param X: Training set, numpy array
    :return: numpy array
    """
    u = np.mean(X, axis=0)
    logger.debug(X)
    logger.debug(" mean %s" % u)
    joblib.dump(u, "mean.pkl")
    return u


def calc_cov(X):
    """
    calculates covariance matrix of the training set.
    :param X: training set, numpy array
    :return: numpy array
    """
    cov = np.cov(X)
    logger.debug(" cov %s" % cov)
    joblib.dump(cov, "cov.pkl")
    return cov
