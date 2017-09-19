import matplotlib as mpl

mpl.use('Agg')
import matplotlib.pyplot as plt
import logging
import os
import socket
import time
import numpy as np

from sklearn.externals import joblib
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import cross_validate

from plot_module import plot_classification_report


class MyClassifier():
    def __init__(self, clf=None, directory=None):
        self.now = time.strftime("%Y-%m-%d %H:%M")
        self.clf = clf
        if directory:
            self.directory = directory
            self.clf = self.__load_clf()
        else:
            self.directory = self.__make_exp_dir()

        self.formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        self.logger = logging.getLogger(__name__)
        self.hdlr = logging.FileHandler(os.path.join(self.directory, 'results.log'))
        self.hdlr.setFormatter(self.formatter)
        self.logger.addHandler(self.hdlr)

    def __make_exp_dir(self):
        directory = "models/" + self.now
        if socket.gethostname() == "classificatoredga":
            directory += " kula"
        if not os.path.exists(directory):
            os.makedirs(directory)

        return directory

    def save_clf(self):
        import json
        # saving clf
        dirmod = os.path.join(self.directory, 'clf.pkl')
        joblib.dump(self.clf, dirmod, compress=3)
        self.logger.info("clf saved to %s" % dirmod)
        dirpar = os.path.join(self.directory, 'params.json')
        with open(dirpar, 'w') as fp:
            json.dump(self.clf.get_params(), fp, sort_keys=True, indent=True)

    def __load_clf(self):
        dirmod = os.path.join(self.directory, 'clf.pkl')
        return joblib.load(dirmod)

    def get_clf(self):
        return self.clf

    def classification_report(self, X_test, y_test, plot=True):
        y_pred = self.clf.predict(X_test)
        repo = classification_report(y_true=y_test,
                                     y_pred=y_pred,
                                     target_names=['dga', 'legit'])
        if plot:
            plot_classification_report(repo,
                                       directory=self.directory
                                       )
        self.logger.info("\n%s" % repo)
        return repo

    def fit(self, X, y, save=True):
        self.clf.fit(X, y)
        if save:
            self.save_clf()

    def save_results(self, results):
        import json
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))

        _res = {k: v.tolist() for k, v in results.items()}
        with open(os.path.join(self.directory, 'cv_results.json'), 'w') as fp:
            try:
                json.dump(_res, fp, sort_keys=True, indent=4)
            except BaseException as e:
                self.logger.error(e)

    def load_results(self):
        import json
        with open(os.path.join(self.directory, "cv_results.json"), 'rb') as fd:
            results = json.load(fd)

        results = {k: np.asfarray(v) for k, v in results.iteritems()}
        for key, value in sorted(results.iteritems()):
            if not "time" in key:
                self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
            else:
                self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))
        return results

    def plot_AUC(self, X_test, y_test):
        y_score = self.clf.predict_proba(X_test)
        fpr, tpr, _ = roc_curve(y_true=y_test, y_score=y_score[:, 1])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', label='ROC curve (area = %0.2f)' % roc_auc, lw=1.5,
                 alpha=.8)
        plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',
                 label='Luck', alpha=.8)
        plt.xlim([0, 1.00])
        plt.ylim([0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')
        plt.legend(loc="lower right")
        dirplt = os.path.join(self.directory, 'roc_plot.png')
        plt.savefig(dirplt, format="png")

    def cross_validate(self, X_train, y_train, scoring=None, save=True):
        if scoring is None:
            scoring = ['f1', 'precision', 'recall', 'accuracy', 'roc_auc']
        results = cross_validate(self.clf, X_train, y_train, cv=10, scoring=scoring)
        if save:
            self.save_results(results)
        else:
            for key, value in sorted(results.iteritems()):
                if not "time" in key:
                    self.logger.info("%s: %.2f%% (%.2f%%)" % (key, value.mean() * 100, value.std() * 100))
                else:
                    self.logger.info("%s: %.2fs (%.2f)s" % (key, value.mean(), value.std()))
        return results
