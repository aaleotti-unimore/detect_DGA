# coding=utf-8
from pprint import pprint
from shutil import rmtree
from tempfile import mkdtemp

from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.model_selection import GridSearchCV, KFold, train_test_split, cross_validate, ShuffleSplit
from sklearn.pipeline import Pipeline
from numpy.random import RandomState
from features.data_generator import *
from features.features_extractors import *
from plot_module import *

basedir = os.path.dirname(__file__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

n_samples = -1
kula = True

if kula:
    n_jobs_pipeline = 8
    clf_n_jobs = -1
    hdlr = logging.FileHandler('results.log')
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
else:
    clf_n_jobs = 1
    n_jobs_pipeline = 2

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
                 ('ns4', NormalityScoreExtractor(4)),
                 ('ns5', NormalityScoreExtractor(5)),
                 ('len', DomainNameLength()),
                 ('vcr', VowelConsonantRatio()),
                 ('ncr', NumCharRatio()),
             ],
             n_jobs=n_jobs_pipeline
         )),
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

##already trained CLFS
trained_clfs = {
    "RandomForest": joblib.load(os.path.join(basedir, "models/model_RandomForest.pkl")),
    "RandomForest_suppo": joblib.load(os.path.join(basedir, "models/model_RandomForest_suppo.pkl")),
}


def pipeline_training():
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


def roc_comparison(clfs=clfs, n_samples=n_samples):
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


def pipeline_grid_search():
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


def test_balboni_dataset():
    X, y = load_balboni(n_samples=n_samples)
    #### test del dataset
    model = trained_clfs['RandomForest']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=RandomState())
    # model.set_params(features_extractors__n_jobs=2)
    logger.debug(model)
    y_pred = model.predict(X_test)

    logger.info("\n%s" % classification_report(y_true=y_test, y_pred=y_pred, target_names=['DGA', 'Legit']))

    # y_pred = lb.inverse_transform(y_pred)
    # y_test = lb.inverse_transform(y_test)
    # pd.options.display.max_rows = 99999999
    # logger.info(pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=['DOMAIN', 'TEST', 'PREDICTION']))
    ########

    # print(data_generator.load_balboni(20))


def detect(domain):
    # model = joblib.load(os.path.join(basedir, "models/model_RandomForest_1000.pkl"))
    pipeline.set_params(**{'clf': joblib.load(os.path.join(basedir, "models/model_RandomForest.pkl"))})
    return pipeline.predict(pd.DataFrame(domain).values.reshape(-1, 1))


def model_training():
    logger.info("Training")
    cv = KFold(n_splits=10)

    X1, y1 = load_features_dataset()
    X2, y2 = load_features_dataset(
        dataset=os.path.join(basedir, "datas/suppobox_dataset.csv"))
    X = np.concatenate((X1, X2), axis=0)
    y = np.concatenate((y1, y2), axis=0)

    from sklearn.utils import shuffle
    X, y = shuffle(X, y, random_state=RandomState())

    logger.debug("X: %s" % str(X.shape))
    logger.debug("y: %s" % str(y.shape))

    scoring = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
    clf = RandomForestClassifier(random_state=True, max_features="auto", n_estimators=100,
                                 min_samples_leaf=50, n_jobs=clf_n_jobs, oob_score=True)
    clf.fit(X, y)
    logger.info("clf fitted")
    scores = cross_validate(clf, X, y, scoring=scoring,
                            cv=10, return_train_score=False, n_jobs=-1, verbose=1)
    joblib.dump(clf, os.path.join(basedir, "models/model_RandomForest_2.pkl"), compress=3)
    #
    logger.info("scores")
    logger.info(scores)
    # title = "Learning Curves Random Forest"
    # # Cross validation with 100 iterations to get smoother mean test and train
    # # score curves, each time with 20% data randomly selected as a validation set.
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # logger.info("plotting learning curve")
    # plot_learning_curve(clf, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.50, random_state=RandomState())
    y_pred = clf.predict(X_test)
    logger.info("plotting classification report")
    plot_classification_report(
        classification_report(y_true=y_test, y_pred=y_pred, target_names=['dga', 'legit']),
        n_samples=n_samples
    )


def keras():
    pass


def main():
    # TODO unificare i database per il training: sia legit-dga_domains.csv che i due all_legit.txt e all_dga.txt presenti su https://github.com/andrewaeva/DGA . questi due file txt vanno prima pre-processati con features_extractor.DomainExtractor in modo da ottenre solo il dominio di secondo livello.

    # TODO testare i classificatori con i json di balboni, prendendo dalla colonna rrname solo quelli ripuliti. fare riferimento alla funzione data_generator.load_balboni già implementata a metà. I paper che ho letto finora usano solo i pachetti NXDOMAIN per fare detection, meglio filtrare quella colonna e usare solo quelli.

    # TODO NB: la pipeline prende in pasto un vettore di stringhe. i vari features_extractors generano le features a partire da questo dataset.
    pass


if __name__ == "__main__":
    model_training()
    logger.info("Exiting...")
    rmtree(cachedir)  # clearing pipeline cache
