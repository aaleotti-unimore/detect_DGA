# coding=utf-8
from shutil import rmtree

from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
from sklearn.metrics import classification_report
from sklearn.model_selection import KFold, train_test_split, cross_validate, ShuffleSplit

from features.data_generator import *
from features.features_testing import *
from features.features_extractors import *
from plot_module import *

basedir = os.path.dirname(__file__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

n_samples=-1
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

##already trained CLFS
trained_clfs = {
    "RandomForest": joblib.load(os.path.join(basedir, "models/model_RandomForest.pkl")),
    "RandomForest_suppo": joblib.load(os.path.join(basedir, "models/model_RandomForest_suppo.pkl")),
}


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
    pipeline.set_params(**{'clf': joblib.load(os.path.join(basedir, "models/model_RandomForest_2.pkl"))})
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




def main():
    # TODO unificare i database per il training: sia legit-dga_domains.csv che i due all_legit.txt e all_dga.txt presenti su https://github.com/andrewaeva/DGA . questi due file txt vanno prima pre-processati con features_extractor.DomainExtractor in modo da ottenre solo il dominio di secondo livello.

    # TODO testare i classificatori con i json di balboni, prendendo dalla colonna rrname solo quelli ripuliti. fare riferimento alla funzione data_generator.load_balboni già implementata a metà. I paper che ho letto finora usano solo i pachetti NXDOMAIN per fare detection, meglio filtrare quella colonna e usare solo quelli.

    # TODO NB: la pipeline prende in pasto un vettore di stringhe. i vari features_extractors generano le features a partire da questo dataset.
    pass


if __name__ == "__main__":
    model_training()
    logger.info("Exiting...")
    rmtree(cachedir)  # clearing pipeline cache
