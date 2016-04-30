__author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import argparse
import logging
import cPickle
from itertools import chain

from data import get_w2v_model
from data import get_w2v_directory
import utils
from data.dataset import Dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_w2v_model(w2v_model_filename, **kwargs):
    logger.info('Loading W2V model')
    W2V_PRETRAINED_FILENAME = w2v_model_filename
    w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))

    return w2v_model

def parse_arguments():
    parser = argparse.ArgumentParser(description='Get vectors from w2v model')
    parser.add_argument('--w2vmodel', type=str, action='store', required=True)
    parser.add_argument('--output', type=str, action='store')
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)

    # parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['w2v_model_filename'] = arguments.w2vmodel
    args['output_filename'] = arguments.output
    args['use_train'] = arguments.train
    args['use_valid'] = arguments.valid
    args['use_test'] = arguments.test

    return args

def get_words(training=False, validation=False, testing=False):
    words = []

    if training:
        _, sentences, _, _ = Dataset.get_clef_training_dataset()
        words.extend(list(chain(*sentences.values())))

    if validation:
        _, sentences, _, _ = Dataset.get_clef_validation_dataset()
        words.extend(list(chain(*sentences.values())))

    if testing:
        _, sentences, _, _ = Dataset.get_clef_testing_dataset()
        words.extend(list(chain(*sentences.values())))

    return set(words)

def get_representations(w2v_model, words):
    representations = dict()

    for word in words:
        word = word.lower()
        try:
            rep = w2v_model[word]
            representations[word] = rep
        except KeyError:
            pass

    return representations

def check_args_consistency(args):
    if not args['use_train'] and not args['use_valid'] and not args['use_test']:
        logger.error('Use at least one dataset.')
        exit()

if __name__ == '__main__':

    logger.info('...Parsing arguments')
    args = parse_arguments()

    check_args_consistency(args)

    logger.info('...Loading w2v model')
    w2v_model = load_w2v_model(**args)

    logger.info('...Getting unique words')
    words = get_words(training=args['use_train'], validation=args['use_valid'], testing=args['use_test'])

    logger.info('...Getting representations from w2v model')
    representations = get_representations(w2v_model, words)

    logger.info('...Pickling file')
    output_filename = args['output_filename']
    if not output_filename:
        output_filename = '_'.join([
            'googlenews',
            'representations',
            'train',
            str(args['use_train']),
            'valid',
            str(args['use_valid']),
            'test',
            str(args['use_test'])
        ])+'.p'

    cPickle.dump(representations, open(get_w2v_directory(output_filename), 'wb'))

    logger.info('End')