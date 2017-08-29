import logging
from shutil import rmtree
from tempfile import mkdtemp
from time import time

import numpy as np
from scipy import interp
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import data_generator
from features.features_extractors import *
from plot_module import *

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

lb = preprocessing.LabelBinarizer()

n_samples = 2000
n_jobs = 2

#### Dataset Loading/Generation
logger.info("samples %s" % n_samples)
df = data_generator.load_dataset(n_samples)
X = df['domain'].values.reshape(-1, 1)
y = np.ravel(lb.fit_transform(df['class'].values))

## Pipeline Definition
cachedir = mkdtemp()

memory = joblib.Memory(cachedir=cachedir, verbose=0)

pipeline = Pipeline(
    memory=memory,
    steps=[
        ('features_extractors',
         FeatureUnion(
             transformer_list=[
                 ('mcr', MCRExtractor()),
                 ('ns1', NormalityScoreExtractor(1)),
                 ('ns2', NormalityScoreExtractor(2)),
                 ('ns3', NormalityScoreExtractor(3)),
                 ('ncr', NumCharRatio()),
             ],
             n_jobs=n_jobs
         )),
        ('clf', SVC(kernel='linear', probability=True))
    ])

clfs = {
    "RandomForest": RandomForestClassifier(random_state=True),
    "SVC": SVC(kernel='linear', C=.9999999999999995e-07, max_iter=50, probability=True),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
}

##already trained CLFS
trained_clfs = {
    "RandomForest": joblib.load("models/10Fold/model_RandomForest_50000.pkl"),
    # "SVC": joblib.load("models/10Fold/model_SVC_50000.pkl"),
    # "GaussianNB": joblib.load("models/10Fold/model_GaussianNB_50000.pkl")
}


def normal_training():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    reports = {}
    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        logger.info("testing: %s" % clf_name)
        pipeline.set_params(**{'clf': clf_value})
        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info("### MODEL %s Performance ###" % clf_name)
        reports[clf_name] = classification_report(y_test, y_pred, target_names=['Benign', 'DGA'], )
        logger.info("\n\n%s" % reports[clf_name])
        joblib.dump(model, "models/model_%s_%s.pkl" % (clf_name, n_samples), compress=5)

    return reports


def roc_comparison():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = KFold(n_splits=10)
    graphic_datas = {}

    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        # for each clf in the pipepline
        pipeline.set_params(**{'clf': clf_value})
        logger.info("testing: %s" % clf_name)
        for train, test in cv.split(X, y):
            model = pipeline.fit(X[train], y[train])
            probas_ = model.predict_proba(X[test])
            # Compute ROC curve and area the curve
            fpr, tpr, thresholds = roc_curve(y[test], probas_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)

        # tpr,fpr,auc plot
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # std dev plot
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)

        graphic_datas[clf_name] = [mean_tpr, mean_fpr, tprs_lower, tprs_upper, mean_auc, std_auc]
        joblib.dump(pipeline, "models/10Fold/model_%s_%s.pkl" % (clf_name, n_samples), compress=5)
        logger.info("models/10Fold/model_%s_%s.pkl saved to disk" % (clf_name, n_samples))

    joblib.dump(graphic_datas, "models/graph/graphic_datas_%s.pkl" % (n_samples))
    logger.info("models/graph/graphic_datas_%s.pkl saved to disk" % (n_samples))

    return graphic_datas,


def grid_search():
    """
    performing grid search based on parameters
    :return:
    """

    parameters = {
        'mcr__mode': (0, 1)
        # 'clf__C': np.logspace(-6, -1, 10),
        # 'clf__max_iter': (10, 50, 80),
    }
    print("Grid Search")
    t0 = time()
    grid_search = GridSearchCV(pipeline, parameters, n_jobs=1, verbose=1)
    grid_search.fit(X, y)
    logger.info("search done in %0.3fs for %s samples" % ((time() - t0), n_samples))
    logger.info("Best score: %0.3f" % grid_search.best_score_)
    logger.info("Best Params:")
    best_params = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_params[param_name]))


def main():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=42)
    model = trained_clfs['RandomForest']
    logger.debug(model)
    logger.debug(X_test)
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, target_names=['DGA', 'Legit'])
    print report
    plot_classification_report(report, 'RandomForest', n_samples=n_samples)
    pass


if __name__ == "__main__":
    main()
    logger.info("Exiting...")
    rmtree(cachedir)  # clearing pipeline cache
