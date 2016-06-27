__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.cross_validation import LeaveOneOut
from utils import utils
import cPickle
import logging
import argparse
import numpy as np
import time
from itertools import chain
import pandas as pd
import ConfigParser
from collections import defaultdict

from utils.metrics import Metrics
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from trained_models import get_vector_tag_path
from trained_models import get_last_tag_path
from trained_models import get_rnn_path
from trained_models import get_single_mlp_path
from trained_models import get_cw_rnn_path
from trained_models import get_cwnn_path
from trained_models import get_multi_hidden_cw_path
from trained_models import get_two_cwnn_path
from trained_models import get_tf_cwnn_path
from trained_models import get_tf_cnn_path
from trained_models import get_POS_nnet_path
from data import get_param
from utils.plot_confusion_matrix import plot_confusion_matrix
from data.dataset import Dataset
from data import get_classification_report_labels
from data import get_hierarchical_mapping
from data import get_aggregated_classification_report_labels
from data import get_config_path
from utils.get_word_tenses import get_training_set_tenses, get_validation_set_tenses, get_testing_set_tenses
from utils.utils import Others
from utils.utils import NeuralNetwork
from SOTA.neural_network.A_neural_network import A_neural_network

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

np.random.seed(1234)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_cw','hidden_cw','vector_tag','last_tag','rnn', 'cw_rnn', 'multi_hidden_cw',
                                 'two_hidden_cw',
                                 'tf_hidden_cw', 'tf_cnn'],
                        help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')
    parser.add_argument('--regularization', action='store_true', default=False)
    parser.add_argument('--minibatch', action='store', type=int, default=False)
    parser.add_argument('--tagdim', action='store', type=int, default=None)
    parser.add_argument('--cviters', type=int, action='store', default=0,
                        help='Nr of cross-validation iterations.')
    parser.add_argument('--leaveoneout', action='store_true', default=False)
    parser.add_argument('--gradmeans', action='store_true', default=False)

    group_w2v = parser.add_mutually_exclusive_group(required=False)
    group_w2v.add_argument('--w2vvectorscache', action='store', type=str)
    group_w2v.add_argument('--w2vmodel', action='store', type=str, default=None)
    group_w2v.add_argument('--w2vrandomdim', action='store', type=int, default=None)

    group_rnn = parser.add_argument_group(title='rnn', description='Recurrent neural net specifics.')
    group_rnn.add_argument('--bidirectional', action='store_true', default=False)
    group_rnn.add_argument('--sharedparams', action='store_true', default=False)

    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--tags', action='store', type=str, default=None)

    group_cnn = parser.add_argument_group(title='cnn', description='Convolutional neural net specifics.')
    group_cnn.add_argument('--cnnfilters', action='store', type=int, default=None)
    group_cnn.add_argument('--static', action='store_true', default=False)
    group_cnn.add_argument('--maxpool', action='store_true', default=False)
    group_cnn.add_argument('--regionsizes', action='store', type=int, nargs='*')
    group_cnn.add_argument('--multifeats', action='store', type=str, nargs='*', default=[],
                           choices=A_neural_network.FEATURE_MAPPING.keys())

    parser.add_argument('--autoencoded', action='store_true', default=False)
    parser.add_argument('--aggregated', action='store_true', default=False)
    parser.add_argument('--normalizesamples', action='store_true', default=False)
    parser.add_argument('--negativesampling', action='store_true', default=False)
    parser.add_argument('--lrtrain', action='store', type=float, default=.1)
    parser.add_argument('--lrtune', action='store', type=float, default=.001)
    parser.add_argument('--lrdecay', action='store_true', default=False)
    parser.add_argument('--hidden', action='store', type=int, default=False)
    parser.add_argument('--logreg', action='store_true', default=False) #tensorflow. use logistic regression architecture.

    parser.add_argument('--configini', action='store', type=str, default=None)

    #parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['max_cv_iters'] = arguments.cviters
    args['nn_name'] = arguments.net
    args['window_size'] = arguments.window
    args['max_epochs'] = arguments.epochs
    args['regularization'] = arguments.regularization
    args['tagdim'] = arguments.tagdim
    args['use_leave_one_out'] = arguments.leaveoneout
    args['minibatch_size'] = arguments.minibatch
    args['use_grad_means'] = arguments.gradmeans
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['w2v_model_name'] = arguments.w2vmodel
    args['bidirectional'] = arguments.bidirectional
    args['shared_params'] = arguments.sharedparams
    args['plot'] = arguments.plot
    args['tags'] = arguments.tags
    args['n_filters'] = arguments.cnnfilters
    args['static'] = arguments.static
    args['max_pool'] = arguments.maxpool
    args['region_sizes'] = arguments.regionsizes
    args['multi_features'] = arguments.multifeats
    args['use_autoencoded_weight'] = arguments.autoencoded
    args['meta_tags'] = arguments.aggregated
    args['norm_samples'] = arguments.normalizesamples
    args['negative_sampling'] = arguments.negativesampling
    args['learning_rate_train'] = arguments.lrtrain
    args['learning_rate_tune'] = arguments.lrtune
    args['n_hidden'] = arguments.hidden
    args['w2v_random_dim'] = arguments.w2vrandomdim
    args['lr_decay'] = arguments.lrdecay
    args['log_reg'] = arguments.logreg
    args['config_ini_filename'] = arguments.configini

    return args

def check_arguments_consistency(args):
    if not args['w2v_vectors_cache'] and not args['w2v_model_name'] and not args['w2v_random_dim'] \
            and args['nn_name'] != 'tf_cnn':
        logger.error('Provide either a w2vmodel or a w2v vectors cache')
        exit()

    if (args['nn_name'] == 'vector_tag' or args['nn_name'] == 'last_tag') and not args['tagdim']:
        logger.error('Provide the tag dimensionality for tag embeddings')
        exit()

    if args['nn_name'] == 'multi_hidden_cw' and not args['multi_features']:
        logger.error('Provide features for the nnet')
        exit()

    if args['nn_name'] == 'multi_hidden_cw':
        if np.max(args['region_sizes']) > args['window_size']:
            logger.error('Region size higher than window size.')
            exit()

    if args['nn_name'] == 'two_hidden_cw' and not args['n_hidden']:
        logger.error('Must specify a hidden layer size for a two hidden layer nnet.')
        exit()

def perform_leave_one_out(nn_class,
                          hidden_f,
                          out_f,
                          n_window,
                          w2v_model=None,
                          w2v_vectors=None,
                          w2v_dims=None,
                          minibatch_size=None,
                          max_cv_iters=None,
                          max_epochs=None,
                          **kwargs):

    logger.info('Loading CRF training data')
    words_per_document, tags_per_document, document_sentences_words, document_sentences_tags, n_docs, unique_words, unique_labels, index2word, word2index, index2label, label2index, n_unique_words, n_out = nn_class.get_data(crf_training_data_filename)

    #store crossvalidation results
    accuracy_results = []
    f1_score_results = []

    prediction_results = dict()

    loo = LeaveOneOut(n_docs)
    for cross_idx, (x_idx, y_idx) in enumerate(loo):

        if (max_cv_iters > 0) and ((cross_idx+1) > max_cv_iters):
            break

        logger.info('Cross-validation %d' % (cross_idx))

        logger.info('Instantiating Neural network')

        params = {
            'hidden_activation_f': hidden_f,
            'out_activation_f': out_f,
            'n_window': n_window,
            'words_per_document': words_per_document,
            'tags_per_document': tags_per_document,
            'document_sentences_words': document_sentences_words,
            'document_sentences_tags': document_sentences_tags,
            'unique_words': unique_words,
            'unique_labels': unique_labels,
            'index2word': index2word,
            'word2index': word2index,
            'index2label': index2label,
            'label2index': label2index,
            'n_unique_words': n_unique_words,
            'n_out': n_out
        }

        nn_trainer = nn_class(**params)

        nn_trainer.initialize_w(w2v_dims, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

        nn_trainer.get_partitioned_data(x_idx,y_idx)

        logger.info('Training Neural network')
        nn_trainer.train(learning_rate=.01, batch_size=minibatch_size, max_epochs=max_epochs, save_params=False)

        logger.info('Predicting')
        flat_true, flat_predictions = nn_trainer.predict()

        accuracy = utils.metrics.accuracy_score(flat_true, flat_predictions)
        accuracy_results.append(accuracy)

        f1_score = utils.metrics.f1_score(flat_true, flat_predictions)
        f1_score_results.append(f1_score)

        prediction_results[cross_idx] = (flat_true, flat_predictions)

        cPickle.dump(prediction_results,open('w2v_vector_tag_predictions'+str(cross_idx)+'.p', 'wb'))

    return prediction_results

def load_w2v_model_and_vectors_cache(args):
    w2v_vectors = None
    w2v_model = None
    w2v_dims = None

    training_vectors_filename = get_w2v_training_data_vectors(args['w2v_vectors_cache'])

    if os.path.exists(training_vectors_filename):
        logger.info('Loading W2V vectors from pickle file: '+args['w2v_vectors_cache'])
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
        w2v_dims = len(w2v_vectors.values()[0])
    else:
        logger.info('Loading W2V model')
        W2V_PRETRAINED_FILENAME = args['w2v_model_name']
        w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
        w2v_dims = w2v_model.syn0.shape[0]

    return w2v_vectors, w2v_model, w2v_dims

def determine_nnclass_and_parameters(args):

    hidden_f = utils.NeuralNetwork.tanh_activation_function
    out_f = utils.NeuralNetwork.softmax_activation_function
    add_words = []
    add_tags = []
    add_feats = []
    tag_dim = None
    n_window = args['window_size']
    nn_class = None
    get_output_path = None
    multi_feats = []
    normalize_samples = False

    if args['nn_name'] == 'single_cw':
        from single_Layer_Context_Window_Net import Single_Layer_Context_Window_Net
        nn_class = Single_Layer_Context_Window_Net
        hidden_f = None #no hidden layer in the single MLP.
        get_output_path = get_single_mlp_path
        add_words = ['<PAD>']
    if args['nn_name'] == 'hidden_cw':

        from hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Hidden_Layer_Context_Window_Net
        get_output_path = get_cwnn_path
        add_words = ['<PAD>']
        multi_feats = args['multi_features']
        normalize_samples = args['norm_samples']
        add_feats = ['<PAD>']
    elif args['nn_name'] == 'vector_tag':

        from vector_Tag_Contex_Window_Net import Vector_Tag_Contex_Window_Net
        nn_class = Vector_Tag_Contex_Window_Net
        get_output_path = get_vector_tag_path
        tag_dim = args['tagdim']
        add_tags = ['<PAD>', '<UNK>']
        add_words = ['<PAD>']
    elif args['nn_name'] == 'last_tag':
        from last_tag_neural_network import Last_tag_neural_network_trainer
        nn_class = Last_tag_neural_network_trainer
        get_output_path = get_last_tag_path
        tag_dim = args['tagdim']
        add_words = ['<PAD>']
        add_tags = ['<PAD>']
    elif args['nn_name'] == 'rnn':
        from recurrent_net import Recurrent_net
        #the RNN init function overwrites the n_window param and sets it to 1.
        nn_class = Recurrent_net
        get_output_path = get_rnn_path
        n_window = 1
    elif args['nn_name'] == 'cw_rnn':

        from recurrent_Context_Window_net import Recurrent_Context_Window_net
        nn_class = Recurrent_Context_Window_net
        get_output_path = get_cw_rnn_path
        add_words = ['<PAD>']
    elif args['nn_name'] == 'multi_hidden_cw':
        from multi_feature_type_hidden_layer_context_window_net import Multi_Feature_Type_Hidden_Layer_Context_Window_Net
        nn_class = Multi_Feature_Type_Hidden_Layer_Context_Window_Net
        get_output_path = get_multi_hidden_cw_path
        add_words = ['<PAD>']
        add_feats = ['<PAD>']
        multi_feats = args['multi_features']
    elif args['nn_name'] == 'two_hidden_cw':

        from two_hidden_Layer_Context_Window_Net import Two_Hidden_Layer_Context_Window_Net
        nn_class = Two_Hidden_Layer_Context_Window_Net
        get_output_path = get_two_cwnn_path
        add_words = ['<PAD>']

    elif args['nn_name'] == 'tf_hidden_cw':
        from tensor_flow.feed_forward_mlp_net import Neural_Net
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Neural_Net
        get_output_path = get_tf_cwnn_path
        add_words = ['<PAD>']
        multi_feats = args['multi_features']
        normalize_samples = args['norm_samples']
        add_feats = ['<PAD>']
    elif args['nn_name'] == 'tf_cnn':
        from tensor_flow.convolutional_net import Convolutional_Neural_Net
        nn_class = Convolutional_Neural_Net
        get_output_path = get_tf_cnn_path
        add_words = ['<PAD>']
        multi_feats = args['multi_features']
        normalize_samples = args['norm_samples']
        add_feats = ['<PAD>']

    return nn_class, hidden_f, out_f, add_words, add_tags, add_feats, tag_dim, n_window, get_output_path, multi_feats, \
           normalize_samples

def determine_key_indexes(label2index, word2index):
    pad_tag = None
    unk_tag = None
    pad_word = None
    try:
        pad_tag = label2index['<PAD>']
    except KeyError:
        pass
    try:
        unk_tag = label2index['<UNK>']
    except KeyError:
        pass
    try:
        pad_word = word2index['<PAD>']
    except KeyError:
        pass

    return pad_tag, unk_tag, pad_word


def get_aggregated_tags(y_train, y_valid, mapping, index2label):

    y_original_train = map(lambda x: index2label[x], y_train)
    y_aggregated_train_labels = map(lambda x: mapping[x], y_original_train)

    y_original_valid = map(lambda x: index2label[x], y_valid)
    y_aggregated_valid_labels = map(lambda x: mapping[x], y_original_valid)

    used_aggegrated_labels = set(y_aggregated_train_labels).union(set(y_aggregated_valid_labels))

    aggregated_label2index = dict(zip(used_aggegrated_labels, range(used_aggegrated_labels.__len__())))
    aggregated_index2label = dict(zip(range(used_aggegrated_labels.__len__()), used_aggegrated_labels))

    y_aggregated_train_indexes = map(lambda x: aggregated_label2index[x], y_aggregated_train_labels)
    y_aggregated_valid_indexes = map(lambda x: aggregated_label2index[x], y_aggregated_valid_labels)

    return y_aggregated_train_indexes, y_aggregated_valid_indexes, aggregated_label2index, aggregated_index2label

def read_embeddings_info(config_parser, config_features):
    config_embeddings = defaultdict(lambda : dict())

    for feat in ['w2v_embeddings', 'pos_embeddings', 'ner_embeddings', 'sent_nr_embeddings', 'tense_embeddings']:
        config_embeddings[feat]['source'] = config_parser.get(feat, 'source')
        config_embeddings[feat]['dim'] = config_parser.get(feat, 'dim')
        config_embeddings[feat]['embedding_item'] = config_parser.get(feat, 'embedding_item')

    return config_embeddings

def read_config_file(config_ini_filename):
    config_features = None

    if config_ini_filename:
        logger.info('Reading configuration file')
        config_path = get_config_path(config_ini_filename)
        config_parser = ConfigParser.ConfigParser()
        config_parser.read(config_path)
        multi_feats = [name for name in config_parser.sections() if name.startswith('feature')]

        config_features = defaultdict(lambda: dict())

        config_embeddings = read_embeddings_info(config_parser, config_features)

        for feat in multi_feats:
            feat_name = config_parser.get(feat, 'name')
            config_features[feat_name]['n_filters'] = np.int(config_parser.get(feat, 'n_filters'))
            config_features[feat_name]['region_sizes'] = [np.int(rs) for rs in
                                                     config_parser.get(feat, 'region_sizes').split()]
            embeddings = None
            if feat_name.startswith('w2v'):
                embeddings = 'w2v_embeddings'
            elif feat_name.startswith('pos'):
                embeddings = 'pos_embeddings'
            elif feat_name.startswith('ner'):
                embeddings = 'ner_embeddings'
            elif feat_name.startswith('sent_nr'):
                embeddings = 'sent_nr_embeddings'
            elif feat_name.startswith('tense'):
                embeddings = 'tense_embeddings'

            assert embeddings is not None
            config_features[feat_name]['embeddings'] = embeddings

    return config_features, config_embeddings

def get_train_features_dataset(nn_class, config_features, config_embeddings,
                               x_train, x_train_feats, x_valid, x_valid_feats, x_test, x_test_feats):
    x_train_pos = None
    x_train_ner = None
    x_valid_pos = None
    x_valid_ner = None
    x_test_pos = None
    x_test_ner = None

    for feat in config_features.keys():
        emb_item = config_embeddings[config_features[feat]['embeddings']]['embedding_item']

        if feat.startswith('pos'):
            feat_position = nn_class.get_features_crf_position([feat])[0]
            if emb_item == 'word':
                x_train_pos = x_train
                x_valid_pos = x_valid
                x_test_pos = x_test
            elif emb_item == 'tag':
                x_train_pos = x_train_feats[feat_position]
                x_valid_pos = x_valid_feats[feat_position]
                x_test_pos = x_test_feats[feat_position]
            else:
                raise Exception('Invalid value for feature %s property embedding_item' % feat)
        elif feat.startswith('ner'):
            feat_position = nn_class.get_features_crf_position([feat])[0]
            if emb_item == 'word':
                x_train_ner = x_train
                x_valid_ner = x_valid
                x_test_ner = x_test
            elif emb_item == 'tag':
                x_train_ner = x_train_feats[feat_position]
                x_valid_ner = x_valid_feats[feat_position]
                x_test_ner = x_test_feats[feat_position]
            else:
                raise Exception('Invalid value for feature %s property embedding_item' % feat)

    return x_train_pos, x_train_ner, x_valid_pos, x_valid_ner, x_test_pos, x_test_ner

def use_testing_dataset(nn_class,
                        hidden_f,
                        out_f,
                        n_window,
                        w2v_model,
                        w2v_vectors,
                        w2v_dims,
                        add_words,
                        add_tags,
                        add_feats,
                        tag_dim,
                        get_output_path,
                        multi_feats,
                        normalize_samples,
                        config_ini_filename,
                        max_epochs=None,
                        minibatch_size=None,
                        regularization=None,
                        tags=None,
                        meta_tags=False,
                        **kwargs
                        ):

    logger.info('Using CLEF testing data')

    results = dict()

    config_features, config_embeddings = read_config_file(config_ini_filename)

    logger.info('Loading CRF training data')

    feat_positions = nn_class.get_features_crf_position(config_features.keys())

    x_train, y_train, x_train_feats, \
    x_valid, y_valid, x_valid_feats, \
    x_test, y_test, x_test_feats,\
    word2index, index2word, \
    label2index, index2label, \
    features_indexes = \
        nn_class.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                          add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=n_window, feat_positions=feat_positions)

    if normalize_samples:
        logger.info('Normalizing number of samples')
        x_train, y_train = NeuralNetwork.perform_sample_normalization(x_train, y_train)

    x_train_sent_nr_feats = None
    x_valid_sent_nr_feats = None
    x_test_sent_nr_feats = None
    if any(map(lambda x: str(x).startswith('sent_nr'), multi_feats)):
        x_train_sent_nr_feats, x_valid_sent_nr_feats, x_test_sent_nr_feats = \
            nn_class.get_word_sentence_number_features(clef_training=True, clef_validation=True, clef_testing=True)

    x_train_tense_feats = None
    x_valid_tense_feats = None
    x_test_tense_feats = None
    tense_probs = None
    if any(map(lambda x: str(x).startswith('tense'), multi_feats)):
        x_train_tense_feats, x_valid_tense_feats, x_test_tense_feats, tense_probs = \
            nn_class.get_tenses_features(clef_training=True, clef_validation=True, clef_testing=True)

    unique_words = word2index.keys()

    ner_embeddings, pos_embeddings, pretrained_embeddings, sent_nr_embeddings, tense_embeddings = initialize_embeddings(
        nn_class, unique_words, w2v_dims, w2v_model, w2v_vectors, word2index, config_embeddings, features_indexes,
        config_features.keys(), feat_positions)

    x_train_pos, x_train_ner, x_valid_pos, x_valid_ner, x_test_pos, x_test_ner = get_train_features_dataset(nn_class,
                                                                                    config_features, config_embeddings,
                                                                                    x_train, x_train_feats,
                                                                                    x_valid, x_valid_feats,
                                                                                    x_test, x_test_feats)

    if tags:
        tags = get_param(tags)
        y_train, y_valid, label2index, index2label = \
            Others.filter_tags_to_predict(y_train, y_valid, index2label, tags)

    if meta_tags:
        logger.info('Using aggregated tags')
        tag_mapping = get_hierarchical_mapping()
        y_train, y_valid, label2index, index2label = \
            get_aggregated_tags(y_train, y_valid, tag_mapping, index2label)

    n_out = len(label2index.keys())

    logger.info('Instantiating Neural network')

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index, word2index)

    try:
        # if im filtering tags, there might not be an "NA" tag
        na_tag = label2index['NA']
    except KeyError:
        na_tag = None

    params = {
        'x_train': x_train,
        'y_train': y_train,
        'x_valid': x_valid,
        'y_valid': y_valid,
        'x_test': x_test,
        'y_test': y_test,
        'hidden_activation_f': hidden_f,
        'out_activation_f': out_f,
        'n_window': n_window,
        'pretrained_embeddings': pretrained_embeddings,
        'n_out': n_out,
        'regularization': regularization,
        'pad_tag': pad_tag,
        'unk_tag': unk_tag,
        'pad_word': pad_word,
        'tag_dim': tag_dim,
        'get_output_path': get_output_path,
        'train_feats': x_train_feats,
        'valid_feats': x_valid_feats,
        'test_feats': x_test_feats,
        'features_indexes': features_indexes,
        'train_pos_feats': x_train_pos,
        'valid_pos_feats': x_valid_pos,
        'test_pos_feats': x_test_pos,
        'train_ner_feats': x_train_ner,
        'valid_ner_feats': x_valid_ner,
        'test_ner_feats': x_test_ner,
        'train_sent_nr_feats': x_train_sent_nr_feats,    #refers to sentence nr features.
        'valid_sent_nr_feats': x_valid_sent_nr_feats,    #refers to sentence nr features.
        'test_sent_nr_feats': x_test_sent_nr_feats,    #refers to sentence nr features.
        'train_tense_feats': x_train_tense_feats,    #refers to tense features.
        'valid_tense_feats': x_valid_tense_feats,    #refers to tense features.
        'test_tense_feats': x_test_tense_feats,    #refers to tense features.
        'tense_probs': tense_probs,
        'n_filters': args['n_filters'],
        'region_sizes': args['region_sizes'],
        'features_to_use': args['multi_features'],
        'static': args['static'],
        'na_tag': na_tag,
        'n_hidden': args['n_hidden'],
        'pos_embeddings': pos_embeddings,
        'ner_embeddings': ner_embeddings,
        'sent_nr_embeddings': sent_nr_embeddings,
        'tense_embeddings': tense_embeddings,
        'log_reg': args['log_reg'],
        'cnn_features': config_features
    }

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network', 'with' if regularization else 'without', 'regularization']))
    nn_trainer.train(batch_size=minibatch_size, max_epochs=max_epochs, save_params=False, nce=False, **kwargs)

    logger.info('Predicting on Training set')
    nnet_results = nn_trainer.predict(on_training_set=True, **kwargs)
    train_flat_predictions = nnet_results['flat_predictions']

    logger.info('Predicting on Validation set')
    nnet_results = nn_trainer.predict(on_validation_set=True, **kwargs)
    valid_flat_true = nnet_results['flat_trues']
    valid_flat_predictions = nnet_results['flat_predictions']

    logger.info('Predicting on Testing set')
    nnet_results = nn_trainer.predict(on_testing_set=True, **kwargs)
    test_flat_predictions = nnet_results['flat_predictions']

    assert valid_flat_true.__len__() == valid_flat_predictions.__len__()

    valid_flat_true = map(lambda x: index2label[x], valid_flat_true)
    valid_flat_predictions = map(lambda x: index2label[x], valid_flat_predictions)
    train_flat_predictions = map(lambda x: index2label[x], train_flat_predictions)
    test_flat_predictions = map(lambda x: index2label[x], test_flat_predictions)

    results[0] = (valid_flat_true, valid_flat_predictions, train_flat_predictions, test_flat_predictions)

    cPickle.dump(pretrained_embeddings, open(get_output_path('original_vectors.p'), 'wb'))
    try:
        # the CNN might not use this feature, and thus, not have a W1 weight
        cPickle.dump(nn_trainer.params['w1'].get_value(), open(get_output_path('trained_vectors.p'), 'wb'))
    except (KeyError, AttributeError):
        pass
    cPickle.dump(word2index, open(get_output_path('word2index.p'), 'wb'))

    return results, index2label

def initialize_w2v_embeddings(w2v_dims, w2v_model, w2v_vectors, config_embeddings, unique_words):
    if config_embeddings:
        source = config_embeddings['w2v_embeddings']['source']
        dim = np.int(config_embeddings['w2v_embeddings']['dim'])
        if source == 'random':
            n_unique_words = len(unique_words)
            pretrained_embeddings = utils.NeuralNetwork.initialize_weights(n_unique_words, dim, function='tanh')
        else:
            logger.info('Initializing w2v embeddings: cache')
            w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache({'w2v_vectors_cache': source})
            pretrained_embeddings = nn_class.initialize_w(dim, unique_words, w2v_vectors=w2v_vectors,
                                                          w2v_model=w2v_model)
    else:
        pretrained_embeddings = nn_class.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors,
                                                      w2v_model=w2v_model)

    return pretrained_embeddings

def initialize_feature_embedding(feat_name, word2index, initializer_func, config_embeddings, get_path_func,
                                 tag2index):
    '''
    Features can be initialized: 1) one-hot; 2) random 3) probabilistic; 4) from vectors_cache.
    '''
    if config_embeddings:
        source = config_embeddings[feat_name]['source']
        dim = np.int(config_embeddings[feat_name]['dim'])
        item = config_embeddings[feat_name]['embedding_item']

        if item == 'tag':
            item2index = tag2index
        elif item == 'word':
            item2index = word2index
        else:
            raise Exception('Invalid value in cfg file for feat %s property Embedding_item' % feat_name)
        n_unique_words = item2index.__len__()
        if source == 'random':
            logger.info('Initializing %s: random' % feat_name)
            pretrained_embeddings = utils.NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=dim, function='tanh')
        elif source == 'one-hot':
            logger.info('Initializing %s: one-hot' % feat_name)
            pretrained_embeddings = np.eye(N=n_unique_words, M=dim, dtype=float)
        elif source == 'probabilistic':
            logger.info('Initializing %s: probabilistic' % feat_name)
            pretrained_embeddings = initializer_func(item2index)
        else:
            logger.info('Initializing %s from vectors_cache: %s' % (feat_name, source))
            vectors = cPickle.load(open(get_path_func(source)))
            pretrained_embeddings = nn_class.initialize_w(dim, item2index.keys(), w2v_vectors=vectors,
                                                          w2v_model=None)
    else:
        # logger.info('Initializing %s embeddings: random' % feat_name)
        # pretrained_embeddings = utils.NeuralNetwork.initialize_weights(n_in=word2index.__len__(), n_out=dim, function='tanh')
        raise Exception('Provide a configuration file for %s initialisation' % feat_name)

    return pretrained_embeddings

def initialize_embeddings(nn_class, unique_words, w2v_dims, w2v_model, w2v_vectors, word2index, config_embeddings,
                          features_indexes, features, features_positions):

    pretrained_embeddings = initialize_w2v_embeddings(w2v_dims, w2v_model, w2v_vectors, config_embeddings, unique_words)

    features_positions = zip(features, features_positions)

    pos2index, index2pos, prob_dist = features_indexes[[pos for feat,pos in features_positions if feat.startswith('pos')][0]]
    pos_embeddings = initialize_feature_embedding('pos_embeddings', word2index, nn_class.initialize_w_pos,
                                                  config_embeddings, get_POS_nnet_path, pos2index)

    ner2index, index2ner, prob_dist = features_indexes[[pos for feat,pos in features_positions if feat.startswith('ner')][0]]
    ner_embeddings = initialize_feature_embedding('ner_embeddings', word2index, nn_class.initialize_w_ner,
                                                  config_embeddings, None, ner2index)

    sent_nr2index, index2sent_nr, prob_dist = features_indexes[[pos for feat,pos in features_positions if feat.startswith('sent_nr')][0]]
    sent_nr_embeddings = initialize_feature_embedding('sent_nr_embeddings', word2index, nn_class.initialize_w_sent_nr,
                                                      config_embeddings, None, sent_nr2index)

    tense2index, index2tense, prob_dist = features_indexes[[pos for feat,pos in features_positions if feat.startswith('tense')][0]]
    tense_embeddings = initialize_feature_embedding('tense_embeddings', word2index, nn_class.initialize_w_tense,
                                                    config_embeddings, None, tense2index)

    return ner_embeddings, pos_embeddings, pretrained_embeddings, sent_nr_embeddings, tense_embeddings


def write_to_file(fout_name, document_sentence_words, predictions, file_prefix, file_suffix):
    fout = open(fout_name, 'wb')
    word_count = 0
    for doc_nr, sentences in document_sentence_words.iteritems():
        doc_words = list(chain(*sentences))
        doc_len = doc_words.__len__()
        for word, tag in zip(doc_words, predictions[word_count:word_count+doc_len]):
            line = '\t'.join([file_prefix+str(doc_nr)+file_suffix, word, tag])
            fout.write(line+'\n')
        word_count += doc_len

    fout.close()

    return True

def save_predictions_to_file(train_y_pred, valid_y_pred, test_y_pred, get_output_path):
    train_fout_name = get_output_path('train_A.txt')
    valid_fout_name = get_output_path('validation_A.txt')
    test_fout_name = get_output_path('test_A.txt')

    _, _, document_sentence_words, _ = Dataset.get_clef_training_dataset()
    write_to_file(train_fout_name, document_sentence_words, train_y_pred, file_prefix='output', file_suffix='.txt')

    _, _, document_sentence_words, _ = Dataset.get_clef_validation_dataset()
    write_to_file(valid_fout_name, document_sentence_words, valid_y_pred, file_prefix='test', file_suffix='.xml.data')

    _, _, document_sentence_words, _ = Dataset.get_clef_testing_dataset()
    write_to_file(test_fout_name, document_sentence_words, test_y_pred, file_prefix='', file_suffix='.xml.data')

    return True

if __name__ == '__main__':
    start = time.time()

    # crf_training_data_filename = 'handoverdata.zip'
    # test_data_filename = 'handover-set2.zip'

    args = parse_arguments()

    check_arguments_consistency(args)

    w2v_vectors = None
    w2v_model = None
    w2v_dims = None

    if args['w2v_vectors_cache'] or args['w2v_model_name']:
        w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)
    elif args['w2v_random_dim']:
        logger.info('Using randomly initialised word embeddings of %d dimensions' % args['w2v_random_dim'])
        w2v_vectors = None
        w2v_model = None
        w2v_dims = args['w2v_random_dim']

    nn_class, hidden_f, out_f, add_words, add_tags, add_feats, tag_dim, n_window, get_output_path,\
        multi_feats, normalize_samples = determine_nnclass_and_parameters(args)

    logger.info('Using Neural class: %s with window size: %d for epochs: %d' % (args['nn_name'],n_window,args['max_epochs']))

    if args['use_leave_one_out']:
        results = perform_leave_one_out()
    else:
        results, index2label = use_testing_dataset(nn_class,
                                                   hidden_f,
                                                   out_f,
                                                   n_window,
                                                   w2v_model,
                                                   w2v_vectors,
                                                   w2v_dims,
                                                   add_words,
                                                   add_tags,
                                                   add_feats,
                                                   tag_dim,
                                                   get_output_path,
                                                   multi_feats,
                                                   normalize_samples,
                                                   **args)

    cPickle.dump(results, open(get_output_path('prediction_results.p'),'wb'))
    cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))

    valid_y_true = list(chain(*[true for true, _, _, _ in results.values()]))
    valid_y_pred = list(chain(*[pred for _, pred, _, _ in results.values()]))
    train_y_pred = list(chain(*[pred for _, _, pred, _ in results.values()]))
    test_y_pred = list(chain(*[pred for _, _, _, pred in results.values()]))

    if args['meta_tags']:
        labels_list = get_aggregated_classification_report_labels()
    else:
        if args['tags']:
            labels_list = list(set(valid_y_true))
        else:
            labels_list = get_classification_report_labels()

    if labels_list.__len__() > 2:
        results_micro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='micro')
        results_macro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='macro')

        print 'MICRO results'
        print results_micro

        print 'MACRO results'
        print results_macro

    assert labels_list is not None

    results_noaverage = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, labels=labels_list, average=None)

    print '...Saving no-averaged results to CSV file'
    df = pd.DataFrame(results_noaverage, index=labels_list)
    df.to_csv(get_output_path('no_average_results.csv'))

    print 'No-average results'
    print results_noaverage

    if args['plot']:
        cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
        plot_confusion_matrix(cm, labels=labels_list, output_filename=get_output_path('confusion_matrix.png'))

    print '...Saving predictions to file'
    save_predictions_to_file(train_y_pred, valid_y_pred, test_y_pred, get_output_path)

    print 'Elapsed time: ', time.time()-start

    logger.info('End')
