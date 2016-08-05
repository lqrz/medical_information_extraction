__author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import pycrfsuite
import logging
import numpy as np
import cPickle
from itertools import chain
import argparse
import time
import pandas as pd

from data.dataset import Dataset
from trained_models import get_pycrf_neuralcrf_folder
from utils.metrics import Metrics
from data import get_classification_report_labels
from utils.plot_confusion_matrix import plot_confusion_matrix
from data import get_aggregated_classification_report_labels

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(1234)

class CRF(object):

    def __init__(self, hidden_activation_folder, output_model_filename, verbose, metatags,
                 sentence_level, document_level,
                 **kwargs):

        self.sentence_level = sentence_level
        self.document_level = document_level

        self.x_train, self.x_valid = self.load_x_datasets(hidden_activation_folder)
        self.output_model_filename = output_model_filename
        self.y_train, self.y_valid = self.load_true_labels()
        self.verbose = verbose
        self.metatags = metatags

    def get_sentence_level_features(self, document_items, activations):

        x = []
        acc = 0
        for doc in document_items.values():
            for sent in doc:
                sent_reps = []
                for rep in activations[acc: acc+sent.__len__()]:
                    sent_reps.append(dict(zip(map(str, range(rep.__len__())), rep)))
                    # x_train.append(training_hidden_activations[acc: acc+sent.__len__()])
                acc += sent.__len__()
                x.append(sent_reps)

        assert acc == list(chain(*chain(*document_items.values()))).__len__()

        return x

    def get_document_level_features(self, document_items, activations):

        x = []
        acc = 0
        for doc in document_items.values():
            doc_reps = []
            n_doc_words = list(chain(*doc)).__len__()
            for rep in activations[acc: acc + n_doc_words]:
                doc_reps.append(dict(zip(map(str, range(rep.__len__())), rep)))
                # x_train.append(training_hidden_activations[acc: acc+sent.__len__()])
            acc += n_doc_words
            x.append(doc_reps)

        assert acc == list(chain(*chain(*document_items.values()))).__len__()

        return x

    def load_x_datasets(self, hidden_activation_folder):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        assert os.path.exists(root+'/'+hidden_activation_folder+'/training_hidden_activations.p')
        assert os.path.exists(root+'/'+hidden_activation_folder+'/validation_hidden_activations.p')

        training_hidden_activations = cPickle.load(open(root+'/'+hidden_activation_folder+'/training_hidden_activations.p','rb'))
        validation_hidden_activations = cPickle.load(open(root+'/'+hidden_activation_folder+'/validation_hidden_activations.p','rb'))

        _, _, _, training_labels = Dataset.get_clef_training_dataset(lowercase=True)
        _, _, _, validation_labels = Dataset.get_clef_validation_dataset(lowercase=True)

        if self.sentence_level:
            logger.info('Using sentence level')
            x_train = self.get_sentence_level_features(training_labels, training_hidden_activations)
            x_valid = self.get_sentence_level_features(validation_labels, validation_hidden_activations)
        elif self.document_level:
            logger.info('Using document level')
            x_train = self.get_document_level_features(training_labels, training_hidden_activations)
            x_valid = self.get_document_level_features(validation_labels, validation_hidden_activations)
        else:
            raise Exception()

        return x_train, x_valid

    def load_true_labels(self):
        _, _, _, training_labels = Dataset.get_clef_training_dataset(lowercase=True)
        _, _, _, validation_labels = Dataset.get_clef_validation_dataset(lowercase=True)

        if self.sentence_level:
            y_train = list(chain(*training_labels.values()))
            y_valid = list(chain(*validation_labels.values()))
        elif self.document_level:
            y_train = []
            y_valid = []
            for doc in training_labels.values():
                y_train.append(list(chain(*doc)))
            for doc in validation_labels.values():
                y_valid.append(list(chain(*doc)))
        else:
            raise Exception()

        return y_train, y_valid

    def train(self, x_train, y_train, verbose=False):
        crf_trainer = pycrfsuite.Trainer(verbose=verbose)

        for xseq, yseq in zip(x_train, y_train):
            crf_trainer.append(xseq, yseq)

        crf_trainer.select(algorithm='lbfgs')  # l2sgd or lbfgs

        crf_trainer.set_params({
            # 'c1': 0,  # not passing it equals to setting it to zero.
            'feature.minfreq': 1,
            'max_linesearch': 20,
            'linesearch': 'MoreThuente',  # Backtracking StrongBacktracking MoreThuente
            'epsilon': 1e-5,
            'c2': 1e-3,
            'max_iterations': None,  # set it to None for 'epsilon' to take over, otherwise self.crf_iters
            'feature.possible_transitions': False,
            'feature.possible_states': False
        })

        print crf_trainer.get_params()

        crf_trainer.train(self.output_model_filename)

        log = zip(range(len(crf_trainer.logparser.iterations)), crf_trainer.logparser.iterations)

        return log

    def predict(self):

        predictions = []

        # get testing features
        x_valid = self.x_valid
        y_valid = self.y_valid

        assert x_valid.__len__() == y_valid.__len__()

        if self.metatags:
            y_valid = convert_metatags(y_valid)

        tagger = pycrfsuite.Tagger()
        tagger.open(self.output_model_filename)

        for sent in x_valid:
            # print tagger.tag(sent)
            predictions.extend(tagger.tag(sent))

        y_test_flat = list(chain(*y_valid))

        return y_test_flat, predictions

def convert_metatags(y_dataset):
    pass

def use_testing_dataset(crf_model, predict_function, **kwargs):

    x_train = crf_model.x_train
    y_train = crf_model.y_train

    assert x_train.__len__() == y_train.__len__()

    if crf_model.metatags:
        y_train = convert_metatags(y_train)

    logger.info('Training the model')
    log = crf_model.train(x_train, y_train, verbose=crf_model.verbose)

    logger.info('Predicting')
    y_test_flat, predicted_tags = predict_function()
    # predicted_tags = predict_function(x_test)

    return log, predicted_tags, y_test_flat

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural CRF')
    parser.add_argument('--folder', action='store', type=str, required=True)
    parser.add_argument('--metatags', action='store_true', default=False)
    parser.add_argument('--verbose', action='store_true', default=False)
    level_group = parser.add_mutually_exclusive_group(required=True)
    level_group.add_argument('--sentence', action='store_true', default=False)
    level_group.add_argument('--document', action='store_true', default=False)

    arguments = parser.parse_args()

    args = dict()
    args['hidden_activation_folder'] = arguments.folder
    args['metatags'] = arguments.metatags
    args['verbose'] = arguments.verbose
    args['sentence_level'] = arguments.sentence
    args['document_level'] = arguments.document

    return args

if __name__ == '__main__':
    actual_time = time.time()

    args = parse_arguments()

    get_output_path = get_pycrf_neuralcrf_folder

    crf_model = CRF(output_model_filename=get_output_path('pycrfsuite.model'), **args)

    train_log, valid_y_pred, valid_y_true = use_testing_dataset(crf_model, crf_model.predict, **args)

    assert valid_y_pred is not None
    assert valid_y_true.__len__() == valid_y_pred.__len__()
    results_macro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='macro')
    results_micro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='micro')

    print 'MICRO results'
    print results_micro

    print 'MACRO results'
    print results_macro

    if args['metatags']:
        labels_list = get_aggregated_classification_report_labels()
    else:
        labels_list = get_classification_report_labels()
    assert labels_list is not None

    results_noaverage = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, labels=labels_list, average=None)

    print '...Saving no-averaged results to CSV file'
    df = pd.DataFrame(results_noaverage, index=labels_list)
    df.to_csv(get_output_path('no_average_results_' + str(actual_time) + '.csv'))

    print '...Ploting confusion matrix'
    cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
    plot_confusion_matrix(cm, labels=labels_list,
                          output_filename=get_output_path('confusion_matrix_' + str(actual_time) + '.png'))

    print '...Computing classification stats'
    stats = Metrics.compute_classification_stats(valid_y_true, valid_y_pred, labels_list)
    df = pd.DataFrame(stats, index=['tp', 'tn', 'fp', 'fn'], columns=labels_list).transpose()
    df.to_csv(get_output_path('classification_stats_' + str(actual_time) + '.csv'))