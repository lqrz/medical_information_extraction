__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.cluster import KMeans
import logging
from joblib import dump
from nltk import word_tokenize
import numpy as np
import argparse
import cPickle

from data import get_w2v_training_data_vectors
from trained_models import get_kmeans_path
from get_vector_representations import load_w2v_model
from get_vector_representations import get_words
from get_vector_representations import get_representations

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def construct_sample_matrix(vectors):
    n_dims = vectors[vectors.keys()[0]].__len__()
    n_samples = vectors.__len__()
    matrix = np.zeros((n_samples, n_dims))

    logger.info('#Samples: %d #Dimensions: %d' % (n_samples, n_dims))

    for i,vec in enumerate(vectors.values()):
        matrix[i] = vec

    return matrix

def get_vector_representations(args):
    w2v_vectors = None

    training_vectors_filename = get_w2v_training_data_vectors(args['w2v_vectors_cache'])

    if os.path.exists(training_vectors_filename):
        logger.info('...Loading W2V vectors from pickle file')
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
    else:
        logger.info('...Loading w2v model')
        w2v_model = load_w2v_model(**args)

        logger.info('...Getting unique words')
        words = get_words(training=args['use_train'], validation=args['use_valid'], testing=args['use_test'])

        logger.info('...Getting representations from w2v model')
        w2v_vectors = get_representations(w2v_model, words)

    return w2v_vectors

def parse_arguments():
    parser = argparse.ArgumentParser(description='Kmeans trainer')
    parser.add_argument('--w2vmodel', default=None, type=str)
    parser.add_argument('--w2vvectorscache', default=None, type=str)
    parser.add_argument('--outputname', default=None, type=str)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--valid', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--nclusters', action='store', type=int, required=True)
    parser.add_argument('--njobs', action='store', type=int, default=-1)

    arguments = parser.parse_args()
    args = dict()
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['w2v_model_filename'] = arguments.w2vmodel
    args['use_train'] = arguments.train
    args['use_valid'] = arguments.valid
    args['use_test'] = arguments.test
    args['output_filename'] = arguments.outputname
    args['n_clusters'] = arguments.nclusters
    args['n_jobs'] = arguments.njobs

    return args

def check_args_consistency(args):
    if not args['use_train'] and not args['use_valid'] and not args['use_test']:
        logger.error('Use at least one dataset.')
        exit()

if __name__ == '__main__':
    args = parse_arguments()

    check_args_consistency(args)

    logger.info('...Getting samples representations')
    w2v_vectors = get_vector_representations(args)

    logger.info('...Initializing KMeans model with '+ str(args['n_clusters']) + ' clusters.')
    kmeans_model = KMeans(n_clusters=args['n_clusters'], init='k-means++', n_jobs=args['n_jobs'])

    matrix = construct_sample_matrix(w2v_vectors)

    logger.info('...Training KMeans model')
    kmeans_model.fit(matrix)

    logger.info('...Persisting KMeans model')
    output_filename = args['output_filename']
    if not output_filename:
        output_filename = '_'.join([
            'kmeans',
            'train',
            str(args['use_train']),
            'valid',
            str(args['use_valid']),
            'test',
            str(args['use_test'])
        ])+'.model'

    dump(kmeans_model, get_kmeans_path(output_filename))
