__author__ = 'root'
from sklearn.metrics import metrics
from collections import Counter
import numpy as np
import time
import pandas as pd
import cPickle
import sys

from data import get_training_classification_report_labels
from data import get_validation_classification_report_labels
from data import get_testing_classification_report_labels
from data import get_aggregated_classification_report_labels
from data import get_all_classification_report_labels
from plot_confusion_matrix import plot_confusion_matrix

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
                na_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
                na_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
                na_f1 = 2 * na_precision * na_recall / (na_precision + na_recall) if (na_precision + na_recall) > 0 else 0.
            else:

                acc_tp += tp
                acc_fp += fp
                acc_fn += fn

                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.
                f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.

                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)

        # MICRO (without NA)
        micro_precision = acc_tp / (acc_tp + acc_fp) if (acc_tp + acc_fp) > 0 else 0.
        micro_recall = acc_tp / (acc_tp + acc_fn) if (acc_tp + acc_fn) > 0 else 0.
        micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.

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

    @staticmethod
    def print_metric_results(train_y_true, train_y_pred, valid_y_true, valid_y_pred, test_y_true, test_y_pred,
                             metatags,
                             get_output_path,
                             additional_labels=[],
                             logger=None):

        if logger is None:
            import logging
            logging.basicConfig(stream=sys.stdout, format='%(asctime)s : %(levelname)s : %(message)s',
                                level=logging.DEBUG)
            logger = logging.getLogger(__name__)

        actual_time = time.time()

        if train_y_pred:

            assert train_y_pred is not None
            assert train_y_true.__len__() == train_y_pred.__len__()

            train_results_micro = Metrics.compute_all_metrics(y_true=train_y_true, y_pred=train_y_pred, average='micro')
            train_results_macro = Metrics.compute_all_metrics(y_true=train_y_true, y_pred=train_y_pred, average='macro')

            if metatags:
                train_labels_list = get_aggregated_classification_report_labels()
            else:
                train_labels_list = get_training_classification_report_labels()

            assert train_labels_list is not None

            train_averaged = Metrics.compute_averaged_scores(y_true=train_y_true, y_pred=train_y_pred,
                                                             labels=train_labels_list)
            logger.info('Train MICRO results')
            logger.info(train_results_micro)

            logger.info('Train MACRO results')
            logger.info(train_results_macro)

            logger.info('Train AVERAGED results')
            logger.info(train_averaged)

        if valid_y_pred:
            assert valid_y_pred is not None
            assert valid_y_true.__len__() == valid_y_pred.__len__()

            valid_results_micro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='micro')
            valid_results_macro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='macro')

            if metatags:
                valid_labels_list = get_aggregated_classification_report_labels()
            else:
                valid_labels_list = get_validation_classification_report_labels()

            assert valid_labels_list is not None

            valid_averaged = Metrics.compute_averaged_scores(y_true=valid_y_true, y_pred=valid_y_pred,
                                                             labels=valid_labels_list)
            logger.info('Valid MICRO results')
            logger.info(valid_results_micro)

            logger.info('Valid MACRO results')
            logger.info(valid_results_macro)

            logger.info('Valid AVERAGED results')
            logger.info(valid_averaged)

        if test_y_pred:

            assert test_y_true is not None
            assert test_y_true.__len__() == test_y_pred.__len__()

            test_results_micro = Metrics.compute_all_metrics(y_true=test_y_true, y_pred=test_y_pred, average='micro')
            test_results_macro = Metrics.compute_all_metrics(y_true=test_y_true, y_pred=test_y_pred, average='macro')

            if metatags:
                test_labels_list = get_aggregated_classification_report_labels()
            else:
                test_labels_list = get_testing_classification_report_labels()

            assert test_labels_list is not None

            test_averaged = Metrics.compute_averaged_scores(y_true=test_y_true, y_pred=test_y_pred,
                                                            labels=test_labels_list)

            logger.info('Test MICRO results')
            logger.info(test_results_micro)

            logger.info('Test MACRO results')
            logger.info(test_results_macro)

            logger.info('Test AVERAGED results')
            logger.info(test_averaged)

            all_labels = get_all_classification_report_labels() + additional_labels

            # no-average results are computed against all report labels, although it can only have training labels.
            results_noaverage = Metrics.compute_all_metrics(test_y_true, test_y_pred,
                                                            labels=all_labels, average=None)

            logger.info('...Saving no-averaged results to CSV file')
            df = pd.DataFrame(results_noaverage, index=all_labels)
            df.to_csv(get_output_path('no_average_results_' + str(actual_time) + '.csv'))

            logger.info('...Ploting confusion matrix')
            cm = Metrics.compute_confusion_matrix(test_y_true, test_y_pred, labels=test_labels_list+additional_labels)
            # plot_confusion_matrix(cm, labels=test_labels_list+additional_labels,
            #                       output_filename=get_output_path('confusion_matrix_' + str(actual_time) + '.png'))

            cPickle.dump(cm, file=open(get_output_path('confusion_matrix.p'), 'wb'))
            plot_confusion_matrix(cm, labels=test_labels_list+additional_labels,
                                  output_filename=get_output_path('confusion_matrix_' + str(actual_time) + '.png'))

            logger.info('...Computing classification stats')
            stats = Metrics.compute_classification_stats(test_y_true, test_y_pred, all_labels)
            df = pd.DataFrame(stats, index=['tp', 'tn', 'fp', 'fn'], columns=all_labels).transpose()
            df.to_csv(get_output_path('classification_stats_' + str(actual_time) + '.csv'))

        def format_averaged_values(results_dict):
            return ' & '.join(map(lambda x: "%0.3f" % x, [results_dict['micro_precision'],
                                                          results_dict['micro_recall'],
                                                          results_dict['micro_f1'],
                                                          results_dict['macro_precision'],
                                                          results_dict['macro_recall'],
                                                          results_dict['macro_f1'],
                                                          results_dict['na_precision'],
                                                          results_dict['na_recall'],
                                                          results_dict['na_f1']]))

        logger.info('\\todo{Add Caption and Label.}')
        logger.info('\\begin{table}[h]')
        logger.info('\\centering')
        logger.info('\\begin{adjustbox}{width=1\\textwidth}')
        logger.info('\\begin{tabular}{l|c|c|c|c|c|c|c|c|c}')
        logger.info('\\multirow{2}{*}{\\textbf{Dataset}} &  \\multicolumn{3}{|c|}{\\textbf{Micro}} &  \\multicolumn{3}{|c|}{\\textbf{Macro}} &  \\multicolumn{3}{|c}{\\textbf{NA}} \\\\')
        logger.info(' & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\')
        logger.info('\\hline')

        if train_y_pred:
            logger.info('Training & ' + format_averaged_values(train_averaged) + '\\\\')

        if valid_y_pred:
            logger.info('Validation & ' + format_averaged_values(valid_averaged) + '\\\\')

        if test_y_pred:
            logger.info('Testing & ' + format_averaged_values(test_averaged) + '\\\\')

        logger.info('\\end{tabular}')
        logger.info('\\end{adjustbox}')
        logger.info('\\caption{Caption}')
        logger.info('\\label{tab:label}')
        logger.info('\\end{table}')

        return True