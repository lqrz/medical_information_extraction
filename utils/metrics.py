__author__ = 'root'
from sklearn.metrics import metrics
from collections import Counter
import numpy as np

class Metrics:

    @staticmethod
    def compute_accuracy_score(y_true, y_pred, **kwargs):
        return metrics.accuracy_score(y_true, y_pred)

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

        results['accuracy'] = Metrics.compute_accuracy_score(y_true, y_pred, **kwargs)
        results['precision'] = Metrics.compute_precision_score(y_true, y_pred, **kwargs)
        results['recall'] = Metrics.compute_recall_score(y_true, y_pred, **kwargs)
        results['f1_score'] = Metrics.compute_f1_score(y_true, y_pred, **kwargs)

        return results

    @staticmethod
    def compute_confusion_matrix(y_true, y_pred, labels):
        fd = Counter(y_true)
        totals = map(lambda x: fd[x], labels)

        cm = metrics.confusion_matrix(y_true, y_pred, labels)
        cm_normalized = np.nan_to_num(np.divide(cm, np.array(totals).astype(dtype='float')[:, np.newaxis]))

        return cm_normalized

    @staticmethod
    def compute_classification_stats(y_true, y_pred, labels):
        results = dict()
        fd = Counter(y_true)
        cm = metrics.confusion_matrix(y_true, y_pred, labels)

        for ix, lab in enumerate(labels):
            tp = np.sum(cm[ix, ix])
            fp = np.sum(cm[:, ix]) - tp
            fn = fd[lab] - tp #np.sum(cm[ix,:]) - tp
            tn = y_true.__len__() - tp - fp - fn

            results[lab] = [tp, tn, fp, fn]

        return results

    @staticmethod
    def compute_averaged_scores(y_true, y_pred, labels):
        results = dict()
        stats = Metrics.compute_classification_stats(y_true=y_true, y_pred=y_pred, labels=labels)

        acc_tp = 0.
        acc_fp = 0.
        acc_fn = 0.
        precisions = []
        recalls = []
        f1s = []
        na_precision = 0.
        na_recall = 0.
        na_f1 = 0.
        for label in labels:
            tp = float(stats[label][0])
            fp = float(stats[label][2])
            fn = float(stats[label][3])

            if label == 'NA':
                na_precision = tp / (tp + fp)
                na_recall = tp / (tp + fn)
                na_f1 = 2 * na_precision * na_recall / (na_precision + na_recall)
            else:

                acc_tp += tp
                acc_fp += fp
                acc_fn += fn

                try:
                    precision = tp / (tp + fp)
                except ZeroDivisionError:
                    precision = 0.
                try:
                    recall = tp / (tp + fn)
                except ZeroDivisionError:
                    recall = 0.
                try:
                    f1 = 2 * precision * recall / (precision + recall)
                except ZeroDivisionError:
                    f1 = 0.

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        # MICRO (without NA)
        micro_precision = acc_tp / (acc_tp + acc_fp)
        micro_recall = acc_tp / (acc_tp + acc_fn)
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall)

        # MACRO (without NA)
        macro_precision = np.mean(precisions)
        macro_recall = np.mean(recalls)
        macro_f1 = np.mean(f1s)

        results['micro_precision'] = micro_precision
        results['micro_recall'] = micro_recall
        results['micro_f1'] = micro_f1
        results['macro_precision'] = macro_precision
        results['macro_recall'] = macro_recall
        results['macro_f1'] = macro_f1
        results['na_precision'] = na_precision
        results['na_recall'] = na_recall
        results['na_f1'] = na_f1

        return results