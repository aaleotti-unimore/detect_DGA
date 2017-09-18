# coding=utf-8
import logging
import os
import socket

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features.data_generator import load_both_datasets, load_features_dataset
from myclassifier import MyClassifier

basedir = os.path.dirname(__file__)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
clf_n_jobs = 1

def test_balboni_dataset():
    # X, y = load_balboni(n_samples=n_samples)
    # #### test del dataset
    # model = trained_clfs['RandomForest']
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.90, random_state=RandomState())
    # # model.set_params(features_extractors__n_jobs=2)
    # logger.debug(model)
    # y_pred = model.predict(X_test)
    #
    # logger.info("\n%s" % classification_report(y_true=y_test, y_pred=y_pred, target_names=['DGA', 'Legit']))

    # y_pred = lb.inverse_transform(y_pred)
    # y_test = lb.inverse_transform(y_test)
    # pd.options.display.max_rows = 99999999
    # logger.info(pd.DataFrame(np.c_[X_test, y_test, y_pred], columns=['DOMAIN', 'TEST', 'PREDICTION']))
    ########

    # print(data_generator.load_balboni(20))
    pass


def main():
    # TODO unificare i database per il training: sia legit-dga_domains.csv che i due all_legit.txt e all_dga.txt presenti su https://github.com/andrewaeva/DGA . questi due file txt vanno prima pre-processati con features_extractor.DomainExtractor in modo da ottenre solo il dominio di secondo livello.

    # TODO testare i classificatori con i json di balboni, prendendo dalla colonna rrname solo quelli ripuliti. fare riferimento alla funzione data_generator.load_balboni già implementata a metà. I paper che ho letto finora usano solo i pachetti NXDOMAIN per fare detection, meglio filtrare quella colonna e usare solo quelli.

    # TODO NB: la pipeline prende in pasto un vettore di stringhe. la funzione get_feature_union ritorna i vari features_extractors generano le features in parallelo a partire da questo dataset.
    pass


if __name__ == "__main__":
    # model_training()
    X, y = load_features_dataset()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    #
    _clf = RandomForestClassifier(random_state=True, max_features="auto", n_estimators=100,
                                  min_samples_leaf=50, n_jobs=-1, oob_score=True)
    myc = MyClassifier(clf=_clf)
    myc.fit(X_train, y_train)
    myc.classification_report(X_test, y_test)
    myc.cross_validate(X_train, y_train)
    myc.plot_AUC(X_test, y_test)
    logger.info("Exiting...")
