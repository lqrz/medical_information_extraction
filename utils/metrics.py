__author__ = 'root'
from sklearn.metrics import metrics
from nltk import FreqDist
import numpy as np

class Metrics:

    @staticmethod
    def compute_accuracy_score(y_true, y_pred, **kwargs):
        return metrics.accuracy_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_f1_score(y_true, y_pred, **kwargs):
        return metrics.f1_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_classification_report(y_true, y_pred, labels, **kwargs):
        return metrics.classification_report(y_true, y_pred, labels, **kwargs)

    @staticmethod
    def compute_precision_score(y_true, y_pred, **kwargs):
        return metrics.precision_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_recall_score(y_true, y_pred, **kwargs):
        return metrics.recall_score(y_true, y_pred, **kwargs)

    @staticmethod
    def compute_all_metrics(y_true, y_pred, **kwargs):
        results = dict()

        results['accuracy'] = Metrics.compute_recall_score(y_true, y_pred, **kwargs)
        results['precision'] = Metrics.compute_precision_score(y_true, y_pred, **kwargs)
        results['recall'] = Metrics.compute_recall_score(y_true, y_pred, **kwargs)
        results['f1_score'] = Metrics.compute_f1_score(y_true, y_pred, **kwargs)

        return results

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, labels):
        fd = FreqDist(y_true)
        totals = map(lambda x: fd[x], labels)

        cm = metrics.confusion_matrix(y_true, y_pred, labels)
        cm_normalized = np.nan_to_num(np.divide(cm, np.array(totals).astype(dtype='float')[:, np.newaxis]))
        return cm_normalized