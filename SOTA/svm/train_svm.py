__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import os
import cPickle
from sklearn.svm import SVC
import numpy as np
import argparse

from SOTA.neural_network.A_neural_network import A_neural_network
from SOTA.neural_network.multi_feature_type_hidden_layer_context_window_net import Multi_Feature_Type_Hidden_Layer_Context_Window_Net
from utils.utils import Others
from utils.utils import NeuralNetwork
from utils.utils import Word2Vec
from data import get_param, get_w2v_training_data_vectors, get_w2v_model
from utils.metrics import Metrics

def load_w2v_model_and_vectors_cache(args):
    w2v_vectors = None
    w2v_model = None
    w2v_dims = None

    training_vectors_filename = get_w2v_training_data_vectors(args['w2v_vectors_cache'])

    if os.path.exists(training_vectors_filename):
        print('Loading W2V vectors from pickle file: '+args['w2v_vectors_cache'])
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
        w2v_dims = len(w2v_vectors.values()[0])
    else:
        print('Loading W2V model')
        W2V_PRETRAINED_FILENAME = args['w2v_model_name']
        w2v_model = Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
        w2v_dims = w2v_model.syn0.shape[0]

    return w2v_vectors, w2v_model, w2v_dims

def parse_arguments():
    parser = argparse.ArgumentParser(description='An SVM baseline')

    group_w2v = parser.add_mutually_exclusive_group(required=True)
    group_w2v.add_argument('--w2vvectorscache', action='store', type=str)
    group_w2v.add_argument('--w2vmodel', action='store', type=str, default=None)

    parser.add_argument('--normalizesamples', action='store_true', default=False)
    parser.add_argument('--window', action='store', type=int, required=True)
    parser.add_argument('--tags', action='store', type=str, default=None)

    parser.add_argument('--multifeats', action='store', type=str, nargs='*', default=[],
                           choices=Multi_Feature_Type_Hidden_Layer_Context_Window_Net.FEATURE_MAPPING.keys())

    arguments = parser.parse_args()
    args = dict()

    args['window_size'] = arguments.window
    args['norm_samples'] = arguments.normalizesamples
    args['multi_features'] = arguments.multifeats
    args['tags'] = arguments.tags
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['w2v_model_name'] = arguments.w2vmodel

    return args

if __name__ == '__main__':
    add_tags = []
    add_feats = []
    add_words = ['<PAD>']
    feat_positions = []

    args = parse_arguments()

    n_window = args['window_size']
    multi_feats = args['multi_features']
    normalize_samples = args['norm_samples']
    tags = args['tags']

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    feat_positions = Multi_Feature_Type_Hidden_Layer_Context_Window_Net.get_features_crf_position(multi_feats)

    x_train_idxs, y_train, x_train_feats, \
    x_valid_idxs, y_valid, x_valid_feats, \
    x_test_idxs, y_test, x_test_feats, \
    word2index, index2word, \
    label2index, index2label, \
    features_indexes = \
        A_neural_network.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                          add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=n_window,
                          feat_positions=feat_positions)

    if normalize_samples:
        print('Normalizing number of samples')
        x_train_idxs, y_train_idxs = NeuralNetwork.perform_sample_normalization(x_train_idxs, y_train)

    x_train_sent_nr_feats = None
    x_valid_sent_nr_feats = None
    x_test_sent_nr_feats = None
    if any(map(lambda x: str(x).startswith('sent_nr'), multi_feats)):
        x_train_sent_nr_feats, x_valid_sent_nr_feats, x_test_sent_nr_feats = \
            A_neural_network.get_word_sentence_number_features(clef_training=True, clef_validation=True, clef_testing=True)

    x_train_tense_feats = None
    x_valid_tense_feats = None
    x_test_tense_feats = None
    tense_probs = None
    if any(map(lambda x: str(x).startswith('tense'), multi_feats)):
        x_train_tense_feats, x_valid_tense_feats, x_test_tense_feats, tense_probs = \
            A_neural_network.get_tenses_features(clef_training=True, clef_validation=True, clef_testing=True)

    unique_words = word2index.keys()

    pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    x_train_idxs = np.array(x_train_idxs)
    x_valid_idxs = np.array(x_valid_idxs)

    if tags:
        tags = get_param(tags)
        y_train, y_valid, label2index, index2label = \
            Others.filter_tags_to_predict(y_train, y_valid, index2label, tags)

    x_train = []
    for sample_idxs in x_train_idxs:
        x_train.append(pretrained_embeddings[sample_idxs].reshape(-1,))

    x_valid = []
    for sample_idxs in x_valid_idxs:
        x_valid.append(pretrained_embeddings[sample_idxs].reshape(-1,))

    svm_model = SVC()

    print '...Training the model'
    svm_model.fit(np.array(x_train), np.array(y_train))

    predictions = svm_model.predict(np.array(x_valid))

    assert predictions.__len__() == y_valid.__len__()

    results_macro = Metrics.compute_all_metrics(y_valid, y_pred=predictions, average='macro')
    results_micro = Metrics.compute_all_metrics(y_valid, y_pred=predictions, average='micro')

    print '##MACRO results'
    print results_macro

    print '##MICRO results'
    print results_micro

    print '...End'