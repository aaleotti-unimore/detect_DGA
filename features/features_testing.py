import os
import numpy as np
from tempfile import mkdtemp

from numpy import interp
from numpy.random.mtrand import RandomState
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline, FeatureUnion

from features.data_generator import *
from features.features_extractors import get_feature_union
from plot_module import plot_classification_report, plot_AUC
import socket
import logging
from shutil import rmtree

basedir = os.path.dirname(__file__)

logger = logging.getLogger(__name__)
# Impostazioni per KULA
if socket == "classificatoredga":
    n_samples = -1  # tutto il dataset
    isKULA = True
    n_jobs_pipeline = 8
    clf_n_jobs = -1
    # impostazioni per stampare gli output del logger su results.log
    hdlr = logging.FileHandler('results.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
else:
    # impostazioni di testing sulla propria macchina
    n_samples = 1000
    isKULA = False
    clf_n_jobs = 1
    n_jobs_pipeline = 2

cachedir = mkdtemp()
memory = joblib.Memory(cachedir=cachedir, verbose=0)
pipeline = Pipeline(
    memory=memory,
    steps=[
        ('features_extractors', get_feature_union(n_jobs_pipeline)),
        ('clf', RandomForestClassifier(random_state=True, max_features="auto", n_estimators=100,
                                       min_samples_leaf=50, n_jobs=clf_n_jobs, oob_score=True))
    ])

clfs = {
    "RandomForest": RandomForestClassifier(random_state=True, max_features="auto", n_estimators=100,
                                           min_samples_leaf=50, n_jobs=clf_n_jobs, oob_score=True),
    # "SVC": SVC(kernel='linear', C=.9999999999999995e-07, max_iter=50, probability=True),
    # "GaussianNB": GaussianNB(),
    # "DecisionTree": DecisionTreeClassifier(),
}


def pipeline_training(n_samples=-1):
    """
    performs training on the classifiers of the pipeline in the clfs dictionary
    :return:
    """
    X, y = generate_domain_dataset(n_samples)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=RandomState())

    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        logger.info("testing: %s" % clf_name)
        pipeline.set_params(**{'clf': clf_value})
        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info("### MODEL %s Performance ###" % clf_name)
        report = classification_report(y_test, y_pred, target_names=['Benign', 'DGA'])
        plot_classification_report(report,
                                   title=clf_name)
        joblib.dump(model, os.path.join(basedir, "models/model_%s_%s.pkl" % (clf_name, n_samples)), compress=5)
        logger.info("model %s saved to models/model_%s_%s.pkl" % (clf_name, clf_name, n_samples))

    rmtree(cachedir)  # clearing pipeline cache


def roc_comparison(clfs=clfs, n_samples=-1):
    """
    train and calculates the mean ROC curve of all the classifier in the clfs dictionary
    :return: dictionary of plot datas needed by plot_module.plot_AUC()
    """
    X, y = load_features_dataset(n_samples=n_samples)

    logger.debug("X: %s" % str(X.shape))
    logger.debug("y: %s" % str(y.shape))

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = KFold(n_splits=10)
    plot_datas = {}
    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        # for each clf in the pipepline
        pipeline.set_params(**{'clf': clf_value})
        logger.info("testing: %s" % clf_name)
        for train, test in cv.split(X, y):
            model = clf_value.fit(X[train], y[train])
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

        plot_datas[clf_name] = [mean_tpr, mean_fpr, tprs_lower, tprs_upper, mean_auc, std_auc]
        joblib.dump(pipeline,
                    os.path.join(basedir, "models/10Fold/model_%s_%s.pkl" % (clf_name, n_samples)),
                    compress=5)
        logger.info("models/10Fold/model_%s_%s.pkl saved to disk" % (clf_name, n_samples))

    plot_AUC(plot_datas,
             n_samples=n_samples)


def pipeline_grid_search(n_samples=-1):
    X, y = generate_domain_dataset(n_samples=n_samples)

    # Split the dataset in two equal parts
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=RandomState())

    # Set the parameters by cross-validation
    # tuned_parameters = [{'clf__kernel': ['rbf'], 'clf__gamma': [1e-3, 1e-4],
    #                      'clf__C': [1, 10, 100, 1000]},
    #                     {'clf__kernel': ['linear'], 'clf__C': [1, 10, 100, 1000]}]
    tuned_parameters = [
        {'features_extractors__ns1__n': [1, 2, 3], 'features_extractors__ns2__n': [2, 3, 4],
         'features_extractors__ns3__n': [3, 4, 5]},
    ]
    scores = ['precision', 'recall']

    for score in scores:
        logger.debug("# Tuning hyper-parameters for %s" % score)
        logger.debug("")

        clf = GridSearchCV(pipeline, tuned_parameters, cv=5,
                           scoring='%s_macro' % score, n_jobs=1)
        clf.fit(X_train, y_train)

        logger.debug("Best parameters set found on development set:")
        logger.debug("")
        logger.debug(clf.best_params_)
        logger.debug("")
        logger.debug("Grid scores on development set:")
        logger.debug("")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            logger.debug("%0.3f (+/-%0.03f) for %r"
                         % (mean, std * 2, params))
        logger.debug("")

        logger.debug("Detailed classification report:")
        logger.debug("")
        logger.debug("The model is trained on the full development set.")
        logger.debug("The scores are computed on the full evaluation set.")
        logger.debug("")
        y_true, y_pred = y_test, clf.predict(X_test)
        logger.debug("\n%s" % classification_report(y_true, y_pred, target_names=['DGA', 'Legit']))
        logger.debug("")
