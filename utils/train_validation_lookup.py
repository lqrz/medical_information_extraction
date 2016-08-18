__author__='lqrz'

from itertools import chain
from collections import Counter
import numpy as np
import argparse
import time
import pandas as pd

from plot_confusion_matrix import plot_confusion_matrix
from data.dataset import Dataset
from metrics import Metrics
from empirical_distribution import Empirical_distribution
from trained_models import get_lookup_path
from data import get_training_classification_report_labels
from data import get_validation_classification_report_labels
from data import get_testing_classification_report_labels
from data import get_all_classification_report_labels

def construct_word_tags_counts(training_words, training_true):
    d = dict()
    # training_words_lower = [w.lower() for w in training_words]
    unique_training_words = set(training_words)
    for word in unique_training_words:
        idxs = np.where(np.array(training_words) == word)[0]
        word_tags = np.array(training_true)[idxs]
        d[word] = Counter(word_tags)

    return d

def predict(dataset_words, word_tag_counts, default_tag=None):
    predictions = []
    # validation_words_lower = [w.lower() for w in dataset_words]
    for word in dataset_words:
        try:
            prediction = word_tag_counts[word].most_common(n=1)[0][0]
        except KeyError:
            if default_tag:
                prediction = default_tag
            else:
                #sample from empirical distribution
                prediction = Empirical_distribution.Instance().sample_from_training_empirical_distribution()

        predictions.append(prediction)

    return predictions

def get_training_data():
    training_data, _, training_document_sentence_words, training_document_sentence_tags = \
        Dataset.get_clef_training_dataset(lowercase=True)

    training_words = list(chain(*chain(*training_document_sentence_words.values())))
    training_tags = list(chain(*chain(*training_document_sentence_tags.values())))

    return training_words, training_tags

def get_validation_data():
    _, _, validation_document_sentence_words, validation_document_sentence_tags = \
        Dataset.get_clef_validation_dataset(lowercase=True)

    validation_words = list(chain(*chain(*validation_document_sentence_words.values())))
    validation_tags = list(chain(*chain(*validation_document_sentence_tags.values())))

    return validation_words, validation_tags

def get_testing_data():
    _, _, testing_document_sentence_words, testing_document_sentence_tags = \
        Dataset.get_clef_testing_dataset(lowercase=True)

    testing_words = list(chain(*chain(*testing_document_sentence_words.values())))
    testing_tags = list(chain(*chain(*testing_document_sentence_tags.values())))

    return testing_words, testing_tags

def parse_arguments():
    parser = argparse.ArgumentParser(description='Lookup baseline')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--sampling", action="store_true",
                       help="If word not found in training data, sample a tag from training empirical distribution.")
    group.add_argument("--nosampling", action="store_true",
                       help="If word not found in training data, use an unseen token as tag.")

    #parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['with_sampling'] = arguments.sampling

    return args

if __name__ == '__main__':
    actual_time = time.time()

    args = parse_arguments()

    print '...Getting data'
    training_words, training_true = get_training_data()
    validation_words, valid_true = get_validation_data()
    testing_words, testing_true = get_testing_data()

    print '...Constructing word-tag-count dictionary'
    word_tag_counts = construct_word_tags_counts(training_words, training_true)

    if args['with_sampling']:
        print '...Predicting (sampling from empirical dist)'
        default_tag = None
    else:
        print '...Predicting (NOT sampling from empirical dist)'
        default_tag = '#IDK#'   #if i dont have it in the training data, then: i dont know

    train_pred = predict(training_words, word_tag_counts, default_tag)
    valid_pred = predict(validation_words, word_tag_counts, default_tag)
    test_pred = predict(testing_words, word_tag_counts, default_tag)

    Metrics.print_metric_results(train_y_true=training_true, train_y_pred=train_pred,
                                 valid_y_true=valid_true, valid_y_pred=valid_pred,
                                 test_y_true=testing_true, test_y_pred=test_pred,
                                 metatags=False,
                                 get_output_path=get_lookup_path,
                                 additional_labels=[default_tag])

    # train_averaged = Metrics.compute_averaged_scores(y_true=training_true, y_pred=train_pred,
    #                                                  labels=get_all_classification_report_labels())
    #
    # valid_averaged = Metrics.compute_averaged_scores(y_true=valid_true, y_pred=valid_pred,
    #                                                  labels=get_all_classification_report_labels())
    #
    # test_averaged = Metrics.compute_averaged_scores(y_true=testing_true, y_pred=test_pred,
    #                                                 labels=get_all_classification_report_labels())
    #
    # results_noaverage = Metrics.compute_all_metrics(testing_true, test_pred,
    #                                                 labels=get_all_classification_report_labels(),
    #                                                 average=None)
    #
    # print '...Saving no-averaged results to CSV file'
    # df = pd.DataFrame(results_noaverage, index=get_all_classification_report_labels())
    # df.to_csv(get_lookup_path('no_average_results_' + str(actual_time) + '.csv'))
    #
    # print '...Ploting confusion matrix'
    # cm = Metrics.compute_confusion_matrix(testing_true, test_pred, labels=get_all_classification_report_labels())
    # plot_confusion_matrix(cm, labels=get_all_classification_report_labels(),
    #                       output_filename=get_lookup_path('confusion_matrix_' + str(actual_time) + '.png'))
    #
    #
    # def format_averaged_values(results_dict):
    #     return ' & '.join(map(lambda x: "%0.3f" % x, [results_dict['micro_precision'],
    #                                                   results_dict['micro_recall'],
    #                                                   results_dict['micro_f1'],
    #                                                   results_dict['macro_precision'],
    #                                                   results_dict['macro_recall'],
    #                                                   results_dict['macro_f1'],
    #                                                   results_dict['na_precision'],
    #                                                   results_dict['na_recall'],
    #                                                   results_dict['na_f1']]))
    #
    # print '\\todo{Add Caption and Label.}'
    # print '\\begin{table}[h]'
    # print '\\centering'
    # print '\\begin{adjustbox}{width=1\\textwidth}'
    # print '\\begin{tabular}{l|c|c|c|c|c|c|c|c|c}'
    # print '\\multirow{2}{*}{\\textbf{Dataset}} &  \\multicolumn{3}{|c|}{\\textbf{Micro}} &  \\multicolumn{3}{|c|}{\\textbf{Macro}} &  \\multicolumn{3}{|c}{\\textbf{NA}} \\\\'
    # print ' & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} & \\textbf{Precision} & \\textbf{Recall} & \\textbf{F1} \\\\'
    # print '\\hline'
    # print 'Training & ' + format_averaged_values(train_averaged) + '\\\\'
    # print 'Validation & ' + format_averaged_values(valid_averaged) + '\\\\'
    # print 'Testing & ' + format_averaged_values(test_averaged) + '\\\\'
    # print '\\end{tabular}'
    # print '\\end{adjustbox}'
    # print '\\caption{Caption}'
    # print '\\label{tab:label}'
    # print '\\end{table}'

    # assert validation_tags.__len__() == predictions.__len__()
    #
    # print '...Computing metrics'
    # results_micro = Metrics.compute_all_metrics(y_true=validation_tags, y_pred=predictions, average='micro')
    # results_macro = Metrics.compute_all_metrics(y_true=validation_tags, y_pred=predictions, average='macro')
    #
    # labels = get_classification_report_labels()
    # labels = labels + [default_tag]
    # results_noaverage = Metrics.compute_all_metrics(y_true=validation_tags, y_pred=predictions, average=None,
    #                                                 labels=labels)
    #
    # cm = Metrics.compute_confusion_matrix(y_true=validation_tags, y_pred=predictions, labels=labels)
    # df = pd.DataFrame(cm, index=labels, columns=labels)
    # cm_filename = get_lookup_path(''.join(['lookup_confusion_matrix',
    #                                        '_sampling' if args['with_sampling'] else '_no_sampling']))
    # plot_confusion_matrix(cm, labels, output_filename=cm_filename, title='Lookup confusion matrix')
    #
    # results_df = pd.DataFrame(results_noaverage, index=labels)
    # results_filename = get_lookup_path(''.join(['lookup_results',
    #                                            '_sampling' if args['with_sampling'] else '_no_sampling'
    #                                            ,'.csv']))
    # results_df.to_csv(results_filename)
    #
    # correct = np.where(map(lambda (x,y): x==y, zip(validation_tags, predictions)))[0].__len__()
    # errors = validation_tags.__len__() - correct
    #
    # print '##ERRORS'
    # print errors
    #
    # print '##MICRO average'
    # print results_micro
    #
    # print '##MACRO average'
    # print results_macro
    #
    # print '##No average'
    # print results_noaverage

    print '...End'