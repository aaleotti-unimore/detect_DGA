import logging
from shutil import rmtree
from tempfile import mkdtemp
from time import time

import matplotlib.pyplot as plt
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
from features.features_extractors import MCRExtractor, NormalityScoreExtractor, NumCharRatio

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

lb = preprocessing.LabelBinarizer()

#### Dataset Loading/Generation
n_samples = 2000
logger.info("samples %s" % n_samples)
df = data_generator.load_dataset(n_samples, mode=0)

# ## OLD X, y defininition
# X, y = df['Domain'].values, df['Target'].values
# X = X.reshape(-1, 1)
# y = np.ravel(y)


X = df['domain'].values.reshape(-1, 1)
y = np.ravel(lb.fit_transform(df['class'].values))

logger.debug(X)
logger.debug(y)

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
             n_jobs=8
         )),
        ('clf', SVC(kernel='linear', probability=True))
    ])

clfs = {
    "RandomForest": RandomForestClassifier(random_state=True),
    "SVC": SVC(kernel='linear', C=.9999999999999995e-07, max_iter=50, probability=True),
    "GaussianNB": GaussianNB(),
    "DecisionTree": DecisionTreeClassifier(),
}


def normal_training():
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        logger.info("testing: %s" % clf_name)
        pipeline.set_params(**{'clf': clf_value})
        model = pipeline.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        logger.info("### MODEL %s Performance ###" % clf_name)
        logger.info("\n\n%s" % classification_report(y_test, y_pred, target_names=['Benign', 'DGA'], ))
        joblib.dump(model, "models/model_%s_%s.pkl" % (clf_name, n_samples), compress=5)


def roc_comparison(graphic=True):
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

    if graphic:
        # show data, save image
        plot_data(graphic_datas=graphic_datas, n=n_samples)
    else:
        # save data
        joblib.dump(graphic_datas, "models/graph/graphic_datas_%s.pkl" % (n_samples))
        logger.info("models/graph/graphic_datas_%s.pkl saved to disk" % (n_samples))

    if graphic:
        plt.show()


def plot_data(graphic_datas, n):
    plt.figure(1)
    colors = ['r', 'g', 'b', 'c', 'm', 'b']

    for index, (clf_name, clf_graph) in enumerate(graphic_datas.iteritems()):
        [mean_tpr, mean_fpr, tprs_lower, tprs_upper, mean_auc, std_auc] = clf_graph
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper,
                         color=colors[index],
                         alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
        plt.plot(mean_fpr, mean_tpr,
                 color=colors[index],
                 label=r'%s Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (clf_name, mean_auc, std_auc),
                 lw=1,
                 alpha=.8)

    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
             label='Luck', alpha=.8)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig("models/graph/plot_comparison_%s.png" % n)
    plt.show()


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


def plot_trained_model(clfs):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    cv = KFold(n_splits=10)
    graphic_datas = {}

    for index, (clf_name, clf_value) in enumerate(clfs.iteritems()):
        for train, test in cv.split(X, y):
            probas_ = clf_value.predict_proba(X[test])
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
    plot_data(graphic_datas, n_samples)


##
# normal_training()
# roc_comparison(graphic=False)
# data_generator.load_json(20)
# plot_data(joblib.load("graphic_datas.pkl"), n_samples)


# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.10, random_state=42)
# y_pred = model.predict(X_test)
# logger.info("\n\n%s" % classification_report(y_test, y_pred, target_names=['DGA', 'Legit'], ))
# trained_clfs = {
#     "RandomForest" : joblib.load("models/10Fold/model_RandomForest_50000.pkl"),
#     "SVC" : joblib.load("models/10Fold/model_SVC_50000.pkl"),
#     "GaussianNB" : joblib.load("models/10Fold/model_GaussianNB_50000.pkl")
# }
#
# plot_trained_model(trained_clfs)
from features.features_extractors import ItemSelector, DomainExtractor

df = ItemSelector(key="rrname").transform(data_generator.load_json(20))
X = DomainExtractor().transform(df)

model = joblib.load("models/10Fold/model_RandomForest_50000.pkl")
y = model.predict(X)
print(y)

logger.info("Exiting...")
rmtree(cachedir)  # clearing pipeline cache
