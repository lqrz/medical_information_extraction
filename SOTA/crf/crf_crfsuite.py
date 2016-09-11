__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import logging
from itertools import chain
import argparse

from SOTA.crf.crf_sklearn_crfsuite import CRF
from data.dataset import Dataset
from trained_models import get_crf_crfsuite_folder
from Tools import get_crfsuite_base_call
from utils.turian.stats import stats
from utils.metrics import Metrics

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CRF_Crfsuite(CRF):

    def __init__(self,
                 output_training_filename,
                 output_validation_filename,
                 output_testing_filename,
                 output_predictions_filename,
                 scale_factor,
                 **kwargs):

        CRF.__init__(self, **kwargs)

        self.output_model_filename = kwargs['output_model_filename']
        self.output_training_filename = output_training_filename
        self.output_validation_filename = output_validation_filename
        self.output_testing_filename = output_testing_filename
        self.output_predictions_filename = output_predictions_filename
        self.scale_factor = scale_factor

    def train(self, x_train, y_train, verbose=True):
        # set params
        min_freq = 1
        l2 = 1e-3

        # generate the train.txt file
        logger.info('Using scaling factor %f' % self.scale_factor)

        # run the cmd
        cmd = get_crfsuite_base_call()
        cmd += " learn -p feature.minfreq=%s -a lbfgs -p feature.possible_transitions=0 " \
               "-p feature.possible_states=0 -p epsilon=1e-5 -p linesearch=MoreThuente -p max_linesearch=20 " \
               "-p c1=0 -p c2=%s -e 2 -l -L training.log -m %s %s %s 2>&1 | tee training.err" % \
               (
               min_freq, l2, self.output_model_filename, self.output_training_filename, self.output_validation_filename)

        run(cmd)

        return True

    def predict(self, filename):

        predictions = []

        # run the cmd
        cmd = get_crfsuite_base_call()
        cmd += " tag -m %s %s > %s" % \
               (self.output_model_filename, filename, self.output_predictions_filename)

        run(cmd)

        assert os.path.exists(filename)

        fin = open(self.output_predictions_filename, 'rb')

        for line in fin:
            tag = line.strip()
            if tag != '':
                predictions.append(line.strip())

        return predictions

def run(cmd):
    '''
    This function is used to call command lines.
    '''

    print >> sys.stderr, cmd
    print >> sys.stderr, stats()
    os.system(cmd)
    print >> sys.stderr, stats()

    return True

def generate_file(x_dataset, y_dataset, output_filename, scale_factor):
    fout = open(output_filename, 'wb')

    n_feats = None

    for x_seq, y_seq in zip(x_dataset, y_dataset):
        assert x_seq.__len__() == y_seq.__len__()

        if n_feats is not None:
            assert set(map(len, x_seq)).__len__() == 1 and n_feats == set(map(len, x_seq)).pop()
        else:
            assert set(map(len, x_seq)).__len__() == 1
            n_feats = set(map(len, x_seq)).pop()

        for x_item, y_item in zip(x_seq, y_seq):
            fs = []
            for feat_name, value in x_item.iteritems():
                if isinstance(value, float):
                    pass
                else:
                    fs.append('%s=%s' % (feat_name, value))
            fout.write('%s\t%s\n' % (y_item, '\t'.join(fs)))

        fout.write('\n')  # it might be the end of a sentence or the end of a document.

    fout.close()

    return True

def use_testing_dataset(crf_model, feature_function, scale_factor,
                        output_training_filename, output_validation_filename, output_testing_filename):

    # get training features
    x = crf_model.get_features_from_crf_training_data(crf_model.training_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.training_data)

    x_train = list(chain(*x.values()))
    y_train = list(chain(*y.values()))

    # get testing features
    x = crf_model.get_features_from_crf_training_data(crf_model.validation_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.validation_data)

    x_valid = list(chain(*x.values()))
    y_valid = list(chain(*y.values()))

    # get testing features
    x = crf_model.get_features_from_crf_training_data(crf_model.testing_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.testing_data)

    x_test = list(chain(*x.values()))
    y_test = list(chain(*y.values()))

    logger.info('Generate files')
    generate_file(x_train, y_train, output_training_filename, scale_factor)
    generate_file(x_valid, y_valid, output_validation_filename, scale_factor)
    generate_file(x_test, y_test, output_testing_filename, scale_factor)

    logger.info('Training the model')
    crf_model.train(x_train, y_train, verbose=True)

    logger.info('Predicting on training set')
    train_predictions = crf_model.predict(output_training_filename)
    y_train_flat = list(chain(*y_train))

    assert train_predictions.__len__() == y_train_flat.__len__()

    logger.info('Predicting on validation set')
    valid_predictions = crf_model.predict(output_validation_filename)
    y_valid_flat = list(chain(*y_valid))

    assert valid_predictions.__len__() == y_valid_flat.__len__()

    logger.info('Predicting on testing set')
    test_predictions = crf_model.predict(output_testing_filename)
    y_test_flat = list(chain(*y_test))

    assert test_predictions.__len__() == y_test_flat.__len__()

    # flat_pred = [tag for tag in chain(*predicted_tags)]

    return y_train_flat, train_predictions, y_valid_flat, valid_predictions, y_test_flat, test_predictions

def determine_output_path(**kwargs):

    return get_crf_crfsuite_folder

def check_arguments_consistency(args):
    #TODO: i shouldnt load the model if its for using kmeans or vector-features and a cache-dict is provided
    if (args['w2v_similar_words'] or args['kmeans_features'] or args['w2v_vector_features']) \
            and not (args['w2v_model_file'] or args['w2v_vectors_cache']):
        logger.error('Provide a word2vec model or vector dictionary for vector extraction.')
        exit()

    if args['kmeans_features'] and not args['kmeans_model_name']:
        logger.error('Provide a Kmeans model when using Kmeans-features')
        exit()

    if args['use_custom_features'] and args['incl_metamap']:
        logger.error('Metamap features are not included in custom features')
        exit()

    return True

def parse_arguments():

    parser = argparse.ArgumentParser(description='CRF Sklearn')

    feat_function_group = parser.add_mutually_exclusive_group(required=True)
    feat_function_group.add_argument('--originalfeatures', action='store_true', default=False)
    feat_function_group.add_argument('--customfeatures', action='store_true', default=False)

    parser.add_argument('--w2vsimwords', action='store_true', default=False)
    parser.add_argument('--w2vvectors', action='store_true', default=False)
    parser.add_argument('--w2vmodel', action='store', type=str, default=None)
    parser.add_argument('--w2vvectorscache', action='store', type=str, default=None)
    parser.add_argument('--kmeans', action='store_true', default=False)
    parser.add_argument('--kmeansmodel', action='store', type=str, default=None)
    parser.add_argument('--lda', action='store_true', default=False)
    parser.add_argument('--unkscore', action='store_true', default=False)
    parser.add_argument('--metamap', action='store_true', default=False)
    parser.add_argument('--zipfeatures', action='store_true', default=False)
    parser.add_argument('--outputaddid', default=None, type=str, help='Output folder for the model and logs')
    parser.add_argument('--leaveoneout', action='store_true', default=False)
    parser.add_argument('--cviters', action='store', type=int, default=0)
    parser.add_argument('--crfiters', action='store', type=int, default=50)
    parser.add_argument('--knowledgegraph', action='store', type=str, default=False)
    parser.add_argument('--scalefactor', action='store', type=float, required=True)

    arguments = parser.parse_args()

    # output_model_filename = arguments.outputfolder+ '/' + 'crf_trained.model'

    args = dict()
    args['use_original_paper_features'] = arguments.originalfeatures
    args['use_custom_features'] = arguments.customfeatures
    args['w2v_similar_words'] = arguments.w2vsimwords
    args['w2v_vector_features'] = arguments.w2vvectors
    args['w2v_model_file'] = arguments.w2vmodel
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['kmeans_features'] = arguments.kmeans
    args['kmeans_model_name'] = arguments.kmeansmodel
    args['lda_features'] = arguments.lda
    args['incl_unk_score'] = arguments.unkscore
    args['incl_metamap'] = arguments.metamap
    args['zip_features'] = arguments.zipfeatures
    args['max_cv_iters'] = arguments.cviters
    args['outputaddid'] = arguments.outputaddid
    args['use_leave_one_out'] = arguments.leaveoneout
    args['crf_iters'] = arguments.crfiters
    args['knowledge_graph'] = arguments.knowledgegraph
    args['scale_factor'] = arguments.scalefactor

    return args

if __name__ == '__main__':
    args = parse_arguments()

    check_arguments_consistency(args)

    get_output_path = determine_output_path(**args)

    training_data, _, _, _ = Dataset.get_clef_training_dataset(lowercase=False)
    validation_data, _, _, validation_tags = Dataset.get_clef_validation_dataset(lowercase=False)
    testing_data, _, _, testing_tags = Dataset.get_clef_testing_dataset(lowercase=False)

    output_training_filename = get_crf_crfsuite_folder('train.txt')
    output_validation_filename = get_crf_crfsuite_folder('valid.txt')
    output_testing_filename = get_crf_crfsuite_folder('test.txt')

    crf_model = CRF_Crfsuite(training_data=training_data,
                             validation_data=validation_data,
                             testing_data=testing_data,
                             output_model_filename=get_output_path('crfsuite.model'),
                             output_training_filename=output_training_filename,
                             output_validation_filename=output_validation_filename,
                             output_testing_filename=output_testing_filename,
                             output_predictions_filename=get_output_path('predictions.txt'),
                             **args)

    if args['use_original_paper_features']:
        feature_function = crf_model.get_original_paper_word_features
    elif args['use_custom_features']:
        feature_function = crf_model.get_custom_word_features

    logger.info('Extracting features with: ' + feature_function.__str__())

    train_y_true, train_y_pred, valid_y_true, valid_y_pred, test_y_true, test_y_pred = \
        use_testing_dataset(crf_model, feature_function, args['scale_factor'],
                           output_training_filename=output_training_filename,
                           output_validation_filename=output_validation_filename,
                           output_testing_filename=output_testing_filename
                           )

    Metrics.print_metric_results(train_y_true=train_y_true, train_y_pred=train_y_pred,
                                 valid_y_true=valid_y_true, valid_y_pred=valid_y_pred,
                                 test_y_true=test_y_true, test_y_pred=test_y_pred,
                                 metatags=False,
                                 get_output_path=get_output_path,
                                 additional_labels=[],
                                 logger=logger)