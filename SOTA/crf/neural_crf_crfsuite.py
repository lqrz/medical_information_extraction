__author__ = 'lqrz'

'''
This neural CRF is meant to be trained using Cokkan's CRFsuite.
I am following the word embedding integration as scaling factors proposed by Turian.

Original paper (doesn't say anything about implementation details):
http://aclweb.org/anthology//P/P10/P10-1040.pdf

Code for the paper (look at scripts/to_crfsuite.py):
https://github.com/turian/crfchunking-with-wordrepresentations

Check how scaling factors are used:
http://www.chokkan.org/software/crfsuite/manual.html
'''

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import logging
import numpy as np
import cPickle
from itertools import chain
import argparse
import time
import pandas as pd
import datetime
from utils.turian.stats import stats

from data.dataset import Dataset
from trained_models import get_pycrf_neuralcrf_crfsuite_folder
from utils.metrics import Metrics
from data import get_classification_report_labels
from utils.plot_confusion_matrix import plot_confusion_matrix
from data import get_aggregated_classification_report_labels
from Tools import get_crfsuite_base_call

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

np.random.seed(1234)


def run(cmd):
    '''
    This function is used to call command lines.
    '''

    print >> sys.stderr, cmd
    print >> sys.stderr, stats()
    os.system(cmd)
    print >> sys.stderr, stats()

    return True


class CRF(object):

    def __init__(self, pickle_folder,
                 output_model_filename,
                 output_training_filename,
                 output_validation_filename,
                 output_testing_filename,
                 output_predictions_filename,
                 verbose, metatags,
                 sentence_level, document_level,
                 hidden_layer, output_layer,
                 scale_factor,
                 **kwargs):

        self.sentence_level = sentence_level
        self.document_level = document_level

        self.x_train, self.x_valid, self.x_test = self.load_x_datasets(pickle_folder, hidden_layer, output_layer)

        self.output_model_filename = output_model_filename
        self.output_training_filename = output_training_filename
        self.output_validation_filename = output_validation_filename
        self.output_testing_filename = output_testing_filename
        self.output_predictions_filename = output_predictions_filename

        self.y_train, self.y_valid, self.y_test = self.load_true_labels()
        self.verbose = verbose
        self.metatags = metatags

        self.scale_factor = scale_factor

    def get_sentence_level_features(self, document_items, activations):

        x = []
        acc = 0
        for doc in document_items.values():
            for sent in doc:
                x.append(activations[acc: acc+sent.__len__()])
                acc += sent.__len__()

        assert acc == list(chain(*chain(*document_items.values()))).__len__()

        return x

    def get_document_level_features(self, document_items, activations):

        x = []
        acc = 0
        for doc in document_items.values():
            n_doc_words = list(chain(*doc)).__len__()
            x.append(activations[acc: acc + n_doc_words])
            acc += n_doc_words

        assert acc == list(chain(*chain(*document_items.values()))).__len__()

        return x

    def load_x_datasets(self, pickle_folder, hidden_layer, output_layer):
        root = os.path.dirname(os.path.dirname(os.path.dirname(__file__))) + '/' + pickle_folder

        print root

        if hidden_layer:
            assert os.path.exists(root+'/training_hidden_activations.p')
            assert os.path.exists(root+'/validation_hidden_activations.p')
            assert os.path.exists(root+'/testing_hidden_activations.p')

            training_activations = cPickle.load(open(root+'/training_hidden_activations.p','rb'))
            validation_activations = cPickle.load(open(root+'/validation_hidden_activations.p','rb'))
            testing_activations = cPickle.load(open(root+'/testing_hidden_activations.p','rb'))
        elif output_layer:
            assert os.path.exists(root+'/training_output_logits.p')
            assert os.path.exists(root+'/validation_output_logits.p')
            assert os.path.exists(root+'/testing_output_logits.p')

            training_activations = cPickle.load(open(root+'/training_output_logits.p','rb'))
            validation_activations = cPickle.load(open(root+'/validation_output_logits.p','rb'))
            testing_activations = cPickle.load(open(root+'/testing_output_logits.p','rb'))
        else:
            raise Exception()

        _, _, _, training_labels = Dataset.get_clef_training_dataset(lowercase=True)
        _, _, _, validation_labels = Dataset.get_clef_validation_dataset(lowercase=True)
        _, _, _, testing_labels = Dataset.get_clef_testing_dataset(lowercase=True)

        if self.sentence_level:
            logger.info('Using sentence level')
            x_train = self.get_sentence_level_features(training_labels, training_activations)
            x_valid = self.get_sentence_level_features(validation_labels, validation_activations)
            x_test = self.get_sentence_level_features(testing_labels, testing_activations)
        elif self.document_level:
            logger.info('Using document level')
            x_train = self.get_document_level_features(training_labels, training_activations)
            x_valid = self.get_document_level_features(validation_labels, validation_activations)
            x_test = self.get_document_level_features(testing_labels, testing_activations)
        else:
            raise Exception()

        return x_train, x_valid, x_test

    def load_true_labels(self):
        _, _, _, training_labels = Dataset.get_clef_training_dataset(lowercase=True)
        _, _, _, validation_labels = Dataset.get_clef_validation_dataset(lowercase=True)
        _, _, _, testing_labels = Dataset.get_clef_testing_dataset(lowercase=True)

        if self.sentence_level:
            y_train = list(chain(*training_labels.values()))
            y_valid = list(chain(*validation_labels.values()))
            y_test = list(chain(*testing_labels.values()))
        elif self.document_level:
            y_train = []
            y_valid = []
            y_test = []
            for doc in training_labels.values():
                y_train.append(list(chain(*doc)))
            for doc in validation_labels.values():
                y_valid.append(list(chain(*doc)))
            for doc in testing_labels.values():
                y_test.append(list(chain(*doc)))
        else:
            raise Exception()

        return y_train, y_valid, y_test

    def generate_file(self, x_dataset, y_dataset, fout):

        # TODO: should i add the words?
        # fs.append('U00=%s' % seq[i-2][0])
        # fs.append('U01=%s' % seq[i-1][0])
        # fs.append('U02=%s' % seq[i][0])
        # fs.append('U03=%s' % seq[i+1][0])
        # fs.append('U04=%s' % seq[i+2][0])
        # fs.append('U05=%s/%s' % (seq[i-1][0], seq[i][0]))
        # fs.append('U06=%s/%s' % (seq[i][0], seq[i+1][0]))

        for x_seq, y_seq in zip(x_dataset, y_dataset):

            assert x_seq.shape[0] == np.array(y_seq).shape[0]

            for x_item, y_item in zip(x_seq, y_seq):
                fs = map(lambda x: "%semb-%d=1:%g" % ('U00', x[0], float(x[1])*self.scale_factor), zip(range(x_item.shape[0]), x_item))
                fout.write('%s\t%s\n' % (y_item, '\t'.join(fs)))

            fout.write('\n')  # it might be the end of a sentence or the end of a document.

        fout.close()

        return True

    def generate_training_file(self):
        if not os.path.exists(self.output_training_filename):
            fout = open(self.output_training_filename, 'wb')

            self.generate_file(self.x_train, self.y_train, fout)

        return True

    def generate_validation_file(self):
        if not os.path.exists(self.output_validation_filename):
            fout = open(self.output_validation_filename, 'wb')

            self.generate_file(self.x_valid, self.y_valid, fout)

        return True

    def generate_testing_file(self):
        if not os.path.exists(self.output_testing_filename):
            fout = open(self.output_testing_filename, 'wb')

            self.generate_file(self.x_test, self.y_test, fout)

        return True

    def generate_dataset_files(self):

        self.generate_training_file()
        self.generate_validation_file()
        self.generate_testing_file()

        return True

    def train(self):

        # set params
        min_freq = 1
        l2 = 1e-3

        # generate the train.txt file
        logger.info('Using scaling factor %f' % self.scale_factor)
        self.generate_dataset_files()

        # run the cmd
        cmd = get_crfsuite_base_call()
        cmd += " learn -p feature.minfreq=%s -a lbfgs -p feature.possible_transitions=0 " \
              "-p feature.possible_states=0 -p epsilon=1e-5 -p linesearch=MoreThuente -p max_linesearch=20 " \
              "-p c1=0 -p c2=%s -e 2 -l -L training.log -m %s %s %s 2>&1 | tee training.err" % \
              (min_freq, l2, self.output_model_filename, self.output_training_filename, self.output_validation_filename)

        run(cmd)

        # crf_trainer.select(algorithm='lbfgs')  # l2sgd or lbfgs
        # crf_trainer.set_params({
        #     # 'c1': 0,  # not passing it equals to setting it to zero.
        #     'feature.minfreq': 1,
        #     'max_linesearch': 20,
        #     'linesearch': 'MoreThuente',  # Backtracking StrongBacktracking MoreThuente
        #     'epsilon': 1e-5,
        #     'c2': 1e-3,
        #     'max_iterations': None,  # set it to None for 'epsilon' to take over, otherwise self.crf_iters
        #     'feature.possible_transitions': False,
        #     'feature.possible_states': False
        # })

        return True

    def predict(self):

        # will check if all CRF .txt files were created.
        self.generate_dataset_files()

        logger.info('Predicting on training set')
        train_y_pred = self.predict_dataset(self.output_training_filename)
        train_y_true = list(chain(*self.y_train))

        logger.info('Predicting on validation set')
        valid_y_pred = self.predict_dataset(self.output_validation_filename)
        valid_y_true = list(chain(*self.y_valid))

        logger.info('Predicting on testing set')
        test_y_pred = self.predict_dataset(self.output_testing_filename)
        test_y_true = list(chain(*self.y_test))

        assert train_y_true.__len__() == train_y_pred.__len__()
        assert valid_y_true.__len__() == valid_y_pred.__len__()
        assert test_y_true.__len__() == test_y_pred.__len__()

        return train_y_true, train_y_pred, valid_y_true, valid_y_pred, test_y_true, test_y_pred

    def predict_dataset(self, dataset_filename):

        predictions = []

        # run the cmd
        cmd = get_crfsuite_base_call()
        cmd += " tag -m %s %s > %s" % \
               (self.output_model_filename, dataset_filename, self.output_predictions_filename)

        run(cmd)

        assert os.path.exists(dataset_filename)
        assert os.path.exists(self.output_predictions_filename)

        fin = open(self.output_predictions_filename, 'rb')

        for line in fin:
            tag = line.strip()
            if tag != '':
                predictions.append(line.strip())

        return predictions

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
    layer_group = parser.add_mutually_exclusive_group(required=True)
    layer_group.add_argument('--hidden', action='store_true', default=False)
    layer_group.add_argument('--output', action='store_true', default=False)
    parser.add_argument('--scalefactor', action='store', type=float, required=True)

    arguments = parser.parse_args()

    args = dict()
    args['pickle_folder'] = arguments.folder
    args['metatags'] = arguments.metatags
    args['verbose'] = arguments.verbose
    args['sentence_level'] = arguments.sentence
    args['document_level'] = arguments.document
    args['hidden_layer'] = arguments.hidden
    args['output_layer'] = arguments.output
    args['scale_factor'] = arguments.scalefactor

    return args

if __name__ == '__main__':
    actual_time = time.time()

    args = parse_arguments()

    get_output_path = get_pycrf_neuralcrf_crfsuite_folder

    crf_model = CRF(output_model_filename=get_output_path('crfsuite.model'),
                    output_training_filename=get_output_path('train.txt'),
                    output_validation_filename=get_output_path('validation.txt'),
                    output_testing_filename=get_output_path('testing.txt'),
                    output_predictions_filename=get_output_path('predictions.txt'),
                    **args)

    crf_model.train()

    train_y_true, train_y_pred, valid_y_true, valid_y_pred, test_y_true, test_y_pred = crf_model.predict()

    Metrics.print_metric_results(train_y_true=train_y_true, train_y_pred=train_y_pred,
                                 valid_y_true=valid_y_true, valid_y_pred=valid_y_pred,
                                 test_y_true=test_y_true, test_y_pred=test_y_pred,
                                 metatags=False,
                                 get_output_path=get_output_path,
                                 additional_labels=[],
                                 logger=logger)

    # # train_log, valid_y_pred, valid_y_true = use_testing_dataset(crf_model, crf_model.predict, **args)
    #
    # assert valid_y_pred is not None
    # assert valid_y_true.__len__() == valid_y_pred.__len__()
    # results_macro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='macro')
    # results_micro = Metrics.compute_all_metrics(y_true=valid_y_true, y_pred=valid_y_pred, average='micro')
    #
    # print 'MICRO results'
    # print results_micro
    #
    # print 'MACRO results'
    # print results_macro
    #
    # if args['metatags']:
    #     labels_list = get_aggregated_classification_report_labels()
    # else:
    #     labels_list = get_classification_report_labels()
    # assert labels_list is not None
    #
    # results_noaverage = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, labels=labels_list, average=None)
    #
    # print '...Saving no-averaged results to CSV file'
    # df = pd.DataFrame(results_noaverage, index=labels_list)
    # df.to_csv(get_output_path('no_average_results_' + str(actual_time) + '.csv'))
    #
    # print '...Ploting confusion matrix'
    # cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
    # plot_confusion_matrix(cm, labels=labels_list,
    #                       output_filename=get_output_path('confusion_matrix_' + str(actual_time) + '.png'))
    #
    # print '...Computing classification stats'
    # stats = Metrics.compute_classification_stats(valid_y_true, valid_y_pred, labels_list)
    # df = pd.DataFrame(stats, index=['tp', 'tn', 'fp', 'fn'], columns=labels_list).transpose()
    # df.to_csv(get_output_path('classification_stats_' + str(actual_time) + '.csv'))