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
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, KFold
from scipy import interp
from sklearn import tree

from sklearn.model_selection import train_test_split
from features import data_generator
from features.features_extractors import MCRExtractor, NormalityScoreExtractor, ItemSelector, NumCharRatio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dataset Loading/Generation
n_samples = 20000
df = data_generator.load_dataset(n_samples)


# df = data_generator.load_json(100)

## X, y defininition
X, y = df['Domain'].values, df['Target'].values
X = X.reshape(-1, 1)
y = np.ravel(y)

## Pipeline Definition
cachedir = mkdtemp()

memory = joblib.Memory(cachedir=cachedir, verbose=0)

pipeline = Pipeline(
    memory=memory,
    steps=[
        # ('selector', ItemSelector(key='Domain')),
        ('features_extractors',
         FeatureUnion(
             transformer_list=[

                 #  # Pipeline for pulling features from the post's subject line
                 #  ('mcr_pip', Pipeline([
                 #      ('selector', ItemSelector(key='rrname')),
                 #      ('mcr', MCRExtractor()),
                 #  ])),
                 #
                 #  ('subject', Pipeline([
                 #      ('selector', ItemSelector(key='rrname')),
                 #      ('ns1', NormalityScoreExtractor(1)),
                 #  ])),
                 #
                 #  ('ns2_pip', Pipeline([
                 #      ('selector', ItemSelector(key='rrname')),
                 #      ('ns2', NormalityScoreExtractor(2)),
                 #  ])),
                 #
                 #  ('ns3_pip', Pipeline([
                 #      ('selector', ItemSelector(key='rrname')),
                 #      ('ns3', NormalityScoreExtractor(3)),
                 #  ])),
                 #
                 # ('ncr_pip', Pipeline([
                 #      ('selector', ItemSelector(key='rrname')),
                 #      ('ncr', NumCharRatio()),
                 #  ])),

                 ('mcr', MCRExtractor()),
                 ('ns1', NormalityScoreExtractor(1)),
                 ('ns2', NormalityScoreExtractor(2)),
                 ('ns3', NormalityScoreExtractor(3)),
                 ('ncr', NumCharRatio()),
             ],
             n_jobs=2
         )),

        ('clf', SVC(kernel='linear', probability=True))

    ])

clfs = {
    "RandomForest": RandomForestClassifier(random_state=True),
    # "SVC": SVC(kernel='linear', C=.9999999999999995e-07, max_iter=50, probability=True),
    # "GaussianNB": GaussianNB()
    # "DecisionTree": tree.DecisionTreeClassifier(),
}


def normal_training():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        pipeline.set_params(**{'clf': clf_value})
        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info("### MODEL %s Performance ###" % clf_name)
        logger.info("\n\n%s" % classification_report(y_test, y_pred, target_names=['Benign', 'DGA'], ))


def roc_comparison():
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(1)
    cv = KFold(n_splits=10)

    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        ##for each clf in the pipepline
        pipeline.set_params(**{'clf': clf_value})

        for train, test in cv.split(X, y):
            probas_ = pipeline.fit(X[train], y[train]).predict_proba(X[test])
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
        colors = ['r', 'g', 'b']
        plt.plot(mean_fpr, mean_tpr, color=colors[index],
                 label=r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (clf_name, mean_auc, std_auc),
                 lw=2, alpha=.8)
        # std dev plot
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color=colors[index], alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

    # plot definition
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.savefig("plot.png")
    # plt.show()


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


# normal_training()
roc_comparison()
# data_generator.load_json(20)

logger.info("Exiting...")
rmtree(cachedir)  # clearing pipeline cache
