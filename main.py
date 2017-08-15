import logging
import time

import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import precision_score, recall_score, f1_score, auc, roc_curve, classification_report
from sklearn.model_selection import KFold, cross_val_predict, train_test_split
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline, FeatureUnion
from features.features_extractors import MCRExtractor, NormalityScoreExtractor
from time import time
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from pprint import pprint

from features import data_generator

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

logger.level = logging.DEBUG


def tenfold_SVC(n_splits, X_len):
    X_g = joblib.load('datas/good_%s' % int(X_len / 2))
    X_b = joblib.load('datas/bad_%s' % int(X_len / 2))
    X = np.concatenate((X_g, X_b))
    y = joblib.load('datas/y_%s' % X_len)
    clf = joblib.load('models/linear_SVC_(%s).pkl' % X_len)


    # logger.debug("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    # joblib.dump(scores, "models/%sfold_SVC_(%s)_scores.pkl" % (n_splits, X_len))
    joblib.dump(cv, "models/%sfold_SVC_(%s)_cv.pkl" % (n_splits, X_len))

    fpr, tpr, thresholds = roc_curve(y, predict)
    prec = precision_score(y, predict)
    recall = recall_score(y, predict)
    f1score = f1_score(y, predict)
    _auc = auc(fpr, tpr)

    # logger.info("\tPrecision: %1.3f" % prec)
    # logger.info("\tRecall: %1.3f" % recall)
    # logger.info("\tF1: %1.3f" % f1score)
    logger.info("\tAUC %1.3f\n" % _auc)

    logger.info("\n" + classification_report(y, predict))


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


n_samples=5000
n_splits=10

df = data_generator.load_dataset(n_samples)
if df is None:
    df = data_generator.generate_dataset(n_samples)

X, y = df['Domain'].values, df['Target'].values
X = X.reshape(-1, 1)
y = np.ravel(y)

estimators = [
    ('mcr', MCRExtractor()),
    ('ns1', NormalityScoreExtractor(1)),
    ('ns2', NormalityScoreExtractor(2)),
    ('ns3', NormalityScoreExtractor(3))
]

combined = FeatureUnion(estimators,
                        n_jobs=1)

pipeline = Pipeline(steps=[
    ('features_extractors', combined),
    ('clf', SVC(kernel='linear'))
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    cv = KFold(n_splits=n_splits, shuffle=True, random_state=0)

    t0 = time()
    model = pipeline.fit(X_train, y_train)
    logger.info("fitting done in %0.3fs" % (time() - t0))
    print()
    t0 = time()
    y_pred = model.predict(X_test)
    logger.info("prediction done in %0.3fs" % (time() - t0))
    logger.info("\n%s" % classification_report(y_test, y_pred, target_names=['Benign', 'DGA'], ))
