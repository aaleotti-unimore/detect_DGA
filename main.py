import logging

import numpy as np
from tempfile import mkdtemp
from shutil import rmtree
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.model_selection import cross_val_predict
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from time import time
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib

from sklearn.model_selection import train_test_split
from features import data_generator
from features.features_extractors import MCRExtractor, NormalityScoreExtractor

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

## Dataset Loading/Generation
n_samples = 1000
df = data_generator.load_dataset(n_samples)
if df is None:
    df = data_generator.generate_dataset(n_samples)
    logger.debug("generated dataset %s" % n_samples)
else:
    logger.debug("loaded dataset %s" % n_samples)

## X, y defininition
X, y = df['Domain'].values, df['Target'].values
X = X.reshape(-1, 1)
y = np.ravel(y)

## Pipeline Definition
cachedir = mkdtemp()
memory = joblib.Memory(cachedir=cachedir, verbose=0)
pipeline = Pipeline(memory=memory, steps=[
    ('features_extractors',
     FeatureUnion([
         ('mcr', MCRExtractor(mode=1)),
         ('ns1', NormalityScoreExtractor(1)),
         ('ns2', NormalityScoreExtractor(2)),
         ('ns3', NormalityScoreExtractor(3))
     ],
         n_jobs=2
     )
     ),
    ('clf', SVC(kernel='linear')
     )
])

##### MODEL TRAIN #####

clfs = {
    # "RandomForest": RandomForestClassifier(random_state=True),
    "SVC": SVC(kernel='linear', C=.9999999999999995e-07, max_iter=50),
    # "GaussianNB": GaussianNB()
}

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
for clf_name, clf_value in clfs.iteritems():
    pipeline.set_params(**{'clf': clf_value})
    model = pipeline.fit(X_train, y_train)
    # joblib.dump(model, "models/%s.pkl" % clf_name)
    y_pred = model.predict(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)
    _auc = auc(fpr, tpr)
    logger.info("### MODEL %s Performance ###" % clf_name)
    logger.info("\n\n%s" % classification_report(y_test, y_pred, target_names=['Benign', 'DGA'], ))
    logger.info("\tAUC %1.3f\n" % _auc)

#### TENFOLD #####
# for clf_name, clf_value in clfs.iteritems():
#     pipeline.set_params(**{'clf': clf_value})
#
#     y_pred = cross_val_predict(
#         pipeline, X, y,
#         cv=10,
#         verbose=1,
#         n_jobs=2
#     )
#
#     fpr, tpr, thresholds = roc_curve(y, y_pred)
#     _auc = auc(fpr, tpr)
#     logger.info("### %s Performance ###" % clf_name)
#     logger.info("\n\n%s" % classification_report(y, y_pred, target_names=['Benign', 'DGA'], ))
#     logger.info("\tAUC %1.3f\n" % _auc)

# ##### GRID SEARCH #####
# parameters = {
#     'mcr__mode': (0, 1)
#     # 'clf__C': np.logspace(-6, -1, 10),
#     # 'clf__max_iter': (10, 50, 80),
# }
# print("Grid Search")
# t0 = time()
# grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
# grid_search.fit(X, y)
# logger.info("search done in %0.3fs for %s samples" % ((time() - t0), n_samples))
# logger.info("Best score: %0.3f" % grid_search.best_score_)
# logger.info("Best Params:")
# best_params = grid_search.best_estimator_.get_params()
# for param_name in sorted(parameters.keys()):
#     print("\t%s: %r" % (param_name, best_params[param_name]))

logger.info("Exiting...")
rmtree(cachedir)  # clearing pipeline cache
