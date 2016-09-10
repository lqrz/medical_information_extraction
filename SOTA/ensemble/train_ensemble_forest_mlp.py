__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import cPickle
import logging
import numpy as np
import argparse
from itertools import chain
import time
from collections import Counter

from forests import Forest
from SOTA.neural_network.A_neural_network import A_neural_network
# from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
# from SOTA.neural_network.single_Layer_Context_Window_Net import Single_Layer_Context_Window_Net
# from SOTA.neural_network.last_tag_neural_network import Last_tag_neural_network_trainer
# from SOTA.neural_network.vector_Tag_Contex_Window_Net import Vector_Tag_Contex_Window_Net
# from SOTA.neural_network.recurrent_net import Recurrent_net
# from SOTA.neural_network.recurrent_Context_Window_net import Recurrent_Context_Window_net
# from SOTA.neural_network.multi_feature_type_hidden_layer_context_window_net import Multi_Feature_Type_Hidden_Layer_Context_Window_Net
from SOTA.neural_network.tensor_flow.multi_feat_mlp import Multi_feat_Neural_Net
from trained_models import get_ensemble_forest_mlp_path
from data import get_param
from utils import utils
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from utils.metrics import Metrics
from utils.plot_confusion_matrix import plot_confusion_matrix
from data import get_classification_report_labels
from data.dataset import Dataset
from SOTA.neural_network.train_neural_network import read_config_file, get_train_features_dataset, \
    get_sent_nr_tense_features_dataset, initialize_embeddings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

np.random.seed(1234)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['tf_mlp'], help='NNet type')
    parser.add_argument('--netwindow', type=int, action='store', required=True,
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
    parser.add_argument('--w2vvectorscache', action='store', type=str, required=True)
    parser.add_argument('--w2vmodel', action='store', type=str, default=None)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--sharedparams', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--tags', action='store', type=str, default=None)
    parser.add_argument('--classifier', action='store', type=str, default=None)
    parser.add_argument('--forestwindow', action='store', type=int, default=None, required=True)
    # parser.add_argument('--multifeats', action='store', type=str, nargs='*', default=[],
    #                        choices=Multi_Feature_Type_Hidden_Layer_Context_Window_Net.FEATURE_MAPPING.keys())
    parser.add_argument('--normalizesamples', action='store_true', default=False)
    parser.add_argument('--negativesampling', action='store_true', default=False)
    parser.add_argument('--forestconfig', action='store', required=True)
    parser.add_argument('--earlystop', action='store', default=None, type=int)
    parser.add_argument('--hidden', action='store', type=int, default=False)
    parser.add_argument('--picklelists', action='store_true', default=False)
    parser.add_argument('--logger', action='store', default=None, type=str)
    parser.add_argument('--alphana', action='store', type=float, default=None)
    parser.add_argument('--static', action='store_true', default=False)
    parser.add_argument('--lrtrain', action='store', type=float, default=.1)
    parser.add_argument('--lrtune', action='store', type=float, default=.001)
    parser.add_argument('--lrdecay', action='store', default=False, type=int)

    #parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['max_cv_iters'] = arguments.cviters
    args['nn_name'] = arguments.net
    args['net_window'] = arguments.netwindow
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
    args['classifier'] = arguments.classifier
    args['forest_window'] = arguments.forestwindow
    # args['multi_features'] = arguments.multifeats
    args['norm_samples'] = arguments.normalizesamples
    args['negative_sampling'] = arguments.negativesampling
    args['forest_config'] = arguments.forestconfig
    args['early_stopping_threshold'] = arguments.earlystop
    args['n_hidden'] = arguments.hidden
    args['pickle_lists'] = arguments.picklelists
    args['logger_filename'] = arguments.logger
    args['static'] = arguments.static
    args['alpha_na'] = arguments.alphana
    args['learning_rate_train'] = arguments.lrtrain
    args['learning_rate_tune'] = arguments.lrtune
    args['lr_decay'] = arguments.lrdecay

    return args

def check_arguments_consistency(args):
    if not args['w2v_vectors_cache'] and not args['w2v_model']:
        logger.error('Provide either a w2vmodel or a w2v vectors cache')
        exit()

    if args['nn_name'] == 'vector_tag' and not args['tagdim']:
        logger.error('Provide the tag dimensionality for Vector tag nnet')
        exit()

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

def load_w2v_model_and_vectors_cache(args):
    w2v_vectors = None
    w2v_model = None
    w2v_dims = None

    training_vectors_filename = get_w2v_training_data_vectors(args['w2v_vectors_cache'])

    if os.path.exists(training_vectors_filename):
        logger.info('Loading W2V vectors from pickle file')
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
        w2v_dims = len(w2v_vectors.values()[0])
    else:
        logger.info('Loading W2V model')
        W2V_PRETRAINED_FILENAME = args['w2v_model_name']
        w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
        w2v_dims = w2v_model.syn0.shape[0]

    return w2v_vectors, w2v_model, w2v_dims

def filter_tags_to_predict(y_train_labels, y_valid_labels, y_test_labels, tags, default_tag=None):

    #recreate indexes so they are continuous and start from 0.
    new_index2label = dict()
    new_label2index = dict()
    for i,tag in enumerate(tags):
        new_label2index[tag] = i
        new_index2label[i] = tag

    if default_tag:
        #add the default tag
        new_label2index[default_tag] = new_index2label.__len__()
        new_index2label[new_index2label.__len__()] = default_tag

    def replace_tag(tag):
        new_index = None
        try:
            new_index = new_label2index[tag]
        except KeyError:
            new_index = new_label2index[default_tag]

        return new_index

    y_train = []
    y_valid = []
    y_test = []
    if isinstance(y_train_labels[0], list):
        for i, sent in enumerate(y_train_labels):
            y_train.append(map(replace_tag, sent))
        for i, sent in enumerate(y_valid_labels):
            y_valid.append(map(replace_tag, sent))
        for i, sent in enumerate(y_test_labels):
            y_test.append(map(replace_tag, sent))
    else:
        y_train = map(replace_tag, y_train_labels)
        y_valid = map(replace_tag, y_valid_labels)
        y_test = map(replace_tag, y_test_labels)

    return y_train, y_valid, y_test, new_label2index, new_index2label

def get_original_labels(y_train, y_valid, y_test, index2label):
    y_train_labels = []
    y_valid_labels = []
    y_test_labels = []

    # recurrent nets use lists of lists.
    if isinstance(y_train[0], list):
        for i, sent in enumerate(y_train):
            y_train_labels.append(map(lambda x: index2label[x], sent))
        for i, sent in enumerate(y_valid):
            y_valid_labels.append(map(lambda x: index2label[x], sent))
        for i, sent in enumerate(y_test):
            y_test_labels.append(map(lambda x: index2label[x], sent))
    else:
        y_train_labels = map(lambda x: index2label[x], y_train)
        y_valid_labels = map(lambda x: index2label[x], y_valid)
        y_test_labels = map(lambda x: index2label[x] if x else None, y_test)

    return y_train_labels, y_valid_labels, y_test_labels

def determine_nnclass_and_parameters(args):

    hidden_f = utils.NeuralNetwork.tanh_activation_function
    out_f = utils.NeuralNetwork.softmax_activation_function
    add_words = []
    add_tags = []
    add_feats = []
    tag_dim = None
    net_window = args['net_window']
    nn_class = None
    multi_feats = []
    normalize_samples = False

    if args['nn_name'] == 'tf_mlp':
        from SOTA.neural_network.tensor_flow.feed_forward_mlp_net import Neural_Net
        nn_class = Neural_Net
        add_words = ['<PAD>']
        add_feats = ['<PAD>']

    return nn_class, hidden_f, out_f, add_words, add_tags, add_feats, tag_dim, net_window, multi_feats, \
           normalize_samples

def perform_sample_normalization(x_train, y_train):
    counts = Counter(y_train)

    higher_count = counts.most_common(n=1)[0][1]

    for tag, cnt in counts.iteritems():
        n_to_add = higher_count - cnt
        tag_idxs = np.where(np.array(y_train)==tag)[0]
        samples_to_add = np.random.choice(tag_idxs, n_to_add, replace=True)
        x_train.extend(np.array(x_train)[samples_to_add].tolist())
        y_train.extend(np.array(y_train)[samples_to_add].tolist())

    return x_train, y_train

def use_testing_dataset(nn_class,
                        hidden_f,
                        out_f,
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
                        classifier,
                        forest_window,
                        net_window,
                        tags,
                        max_epochs,
                        minibatch_size,
                        regularization,
                        forest_config,
                        **kwargs
                        ):

    logger.info('Using CLEF testing data')

    results = dict()

    logger.info('Loading CRF training data')

    config_features, config_embeddings = read_config_file(forest_config)
    feat_names_and_positions = Multi_feat_Neural_Net.get_features_crf_position(config_features.keys())
    feat_positions = [v for _,v in feat_names_and_positions]

    x_train, y_train_all, x_train_feats, \
    x_valid, y_valid_all, x_valid_feats, \
    x_test, y_test_all, x_test_feats,\
    word2index_all, index2word_all, \
    label2index_all, index2label_all, \
    features_indexes = \
        A_neural_network.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                          add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=forest_window, feat_positions=feat_positions)

    x_train_pos, x_train_ner, x_valid_pos, x_valid_ner, x_test_pos, x_test_ner = get_train_features_dataset(Multi_feat_Neural_Net,
                                                                                                            config_features,
                                                                                                            config_embeddings,
                                                                                                            x_train,
                                                                                                            x_train_feats,
                                                                                                            x_valid,
                                                                                                            x_valid_feats,
                                                                                                            x_test,
                                                                                                            x_test_feats)

    x_train_sent_nr, x_valid_sent_nr, x_test_sent_nr, sent_nr2index, index2sent_nr, \
    x_train_tense, x_valid_tense, x_test_tense, tense2index, index2tense = \
        get_sent_nr_tense_features_dataset(config_features, config_embeddings, forest_window, add_feats)

    # feat_positions = nn_class.get_features_crf_position(multi_feats)
    #
    # x_train, y_train_all, x_train_feats, x_valid, y_valid_all, x_valid_feats, x_test, y_test_all, x_test_feats, word2index_all, \
    # index2word_all, label2index_all, index2label_all, features_indexes = \
    #     nn_class.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
    #                       add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=forest_window, feat_positions=feat_positions)

    if normalize_samples:
        logger.info('Normalizing number of samples')
        x_train, y_train = perform_sample_normalization(x_train, y_train_all)

    unique_words = word2index_all.keys()

    ner_embeddings, pos_embeddings, pretrained_embeddings, sent_nr_embeddings, tense_embeddings = initialize_embeddings(
        A_neural_network, unique_words, w2v_dims, w2v_model, w2v_vectors, word2index_all, config_embeddings, features_indexes,
        config_features.keys(), feat_names_and_positions,
        sent_nr2index, tense2index)

    unique_words = word2index_all.keys()

    # x_train_sent_nr_feats = None
    # x_valid_sent_nr_feats = None
    # x_test_sent_nr_feats = None
    # if any(map(lambda x: str(x).startswith('sent_nr'), multi_feats)):
    #     x_train_sent_nr_feats, x_valid_sent_nr_feats, x_test_sent_nr_feats = \
    #         nn_class.get_word_sentence_number_features(clef_training=True, clef_validation=True, clef_testing=True)
    #
    # x_train_tense_feats = None
    # x_valid_tense_feats = None
    # x_test_tense_feats = None
    # tense_probs = None
    # if any(map(lambda x: str(x).startswith('tense'), multi_feats)):
    #     x_train_tense_feats, x_valid_tense_feats, x_test_tense_feats, tense_probs = \
    #         nn_class.get_tenses_features(clef_training=True, clef_validation=True, clef_testing=True)
    #
    # pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    # determine which tags to predict in the first step, and which ones in the second.
    y_train_labels, y_valid_labels, y_test_labels = get_original_labels(y_train_all, y_valid_all, y_test_all, index2label_all)
    tags = get_param(tags)
    tags_2nd_step = set(label2index_all.keys()) - set(tags)
    default_tag = '<OTHER>'
    y_train_1st_step, y_valid_1st_step, y_test_1st_step, label2index_1st_step, index2label_1st_step = \
        filter_tags_to_predict(y_train_labels, y_valid_labels, y_test_labels, tags, default_tag=default_tag)

    forest = Forest(classifier,
                    pretrained_embeddings, pos_embeddings,
                    ner_embeddings, sent_nr_embeddings, tense_embeddings,
                    forest_window)

    forest.train(x_train,
                 x_train_pos,
                 x_train_ner,
                 x_train_sent_nr,
                 x_train_tense,
                 y_train_1st_step)

    train_predictions_1st_step = forest.predict(x_train,
                                                x_train_pos,
                                                x_train_ner,
                                                x_train_sent_nr,
                                                x_train_tense)
    valid_predictions_1st_step = forest.predict(x_valid,
                                                x_valid_pos,
                                                x_valid_ner,
                                                x_valid_sent_nr,
                                                x_valid_tense)
    test_predictions_1st_step = forest.predict(x_test,
                                               x_test_pos,
                                               x_test_ner,
                                               x_test_sent_nr,
                                               x_test_tense)

    assert y_train_1st_step.__len__() == train_predictions_1st_step.__len__()
    train_score_1st_step = Metrics.compute_all_metrics(y_train_1st_step, train_predictions_1st_step, average=None)
    print 'Training scores'
    print train_score_1st_step

    assert y_valid_1st_step.__len__() == valid_predictions_1st_step.__len__()
    valid_score_1st_step = Metrics.compute_all_metrics(y_valid_1st_step, valid_predictions_1st_step, average=None)
    print 'Validation scores'
    print valid_score_1st_step

    assert y_test_1st_step.__len__() == test_predictions_1st_step.__len__()
    test_score_1st_step = Metrics.compute_all_metrics(y_test_1st_step, test_predictions_1st_step, average=None)
    print 'Testing scores'
    print test_score_1st_step

    if forest_window != net_window:
        # the forest and the nnet might use different window sizes.
        x_train, _, x_train_feats, x_valid, _, x_valid_feats, x_test, _, x_test_feats, _, _, _, _, features_indexes = \
            A_neural_network.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                              add_tags=add_tags, add_feats=add_feats, x_idx=None, n_window=net_window, feat_positions=feat_positions)
        # x_train, _, x_test, _, _, _, _, _ = \
        #     A_neural_network.get_data(crf_training_data_filename, test_data_filename, add_words, add_tags,
        #                               x_idx=None, n_window=net_window)

    #--- here starts the 2nd step ---#

    # get the subsamples to train and test on this 2nd step (all that were marked as '<OTHER>').
    # for the training set, im using the true_values to know which positions to train on now.
    # for the validation and testing set, im not using the true_values, but the predictions of the 1st step.
    indexes_to_train = np.where(np.array(y_train_1st_step) == label2index_1st_step[default_tag])[0]
    indexes_to_valid = np.where(valid_predictions_1st_step == label2index_1st_step[default_tag])[0]
    indexes_to_test = np.where(test_predictions_1st_step == label2index_1st_step[default_tag])[0]
    indexes_to_train_to_replace = np.where(train_predictions_1st_step == label2index_1st_step[default_tag])[0]
    x_train_2nd_step = np.array(x_train)[indexes_to_train]
    y_train_2nd_step = np.array(y_train_labels)[indexes_to_train]
    x_valid_2nd_step = np.array(x_valid)[indexes_to_valid]
    y_valid_2nd_step = np.array(y_valid_labels)[indexes_to_valid]
    x_test_2nd_step = np.array(x_test)[indexes_to_test]
    y_test_2nd_step = np.array(y_test_labels)[indexes_to_test]

    # i have to reconstruct the indexes, so that the tags im gonna use in this 2nd step are continuous.
    # because the validation and test indexes are based on the predictions from the 1st step, im gonna propagate the
    # errors i made in the first step, and so, there are gonna be true_labels that are included
    # in the tag_set to be predicted in the first tep; Im gonna map all those tags to a dummy_tag, which i will later
    # remove from the index (this is kind of nasty).
    dummy_tag = '#delete#'  #TODO: this is nasty.
    y_train_2nd_step, y_valid_2nd_step, y_test_2nd_step, label2index_2nd_step, index2label_2nd_step = \
        filter_tags_to_predict(y_train_2nd_step, y_valid_2nd_step, y_test_2nd_step, tags_2nd_step, default_tag=dummy_tag)

    # TODO: this is nasty.
    dummy_value = label2index_2nd_step[dummy_tag]
    del label2index_2nd_step[dummy_tag]
    del index2label_2nd_step[dummy_value]

    n_out = len(label2index_2nd_step.keys())

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index_2nd_step, word2index_all)

    params = {
        'x_train': x_train_2nd_step,
        'y_train': y_train_2nd_step,
        'x_valid': x_valid_2nd_step,
        'y_valid': y_valid_2nd_step,
        'x_test': x_test_2nd_step,
        'y_test': y_test_2nd_step,
        'hidden_activation_f': hidden_f,
        'out_activation_f': out_f,
        'n_window': net_window,
        'pretrained_embeddings': pretrained_embeddings,
        'n_out': n_out,
        'regularization': regularization,
        'pad_tag': pad_tag,
        'unk_tag': unk_tag,
        'pad_word': pad_word,
        'tag_dim': tag_dim,
        'get_output_path': get_output_path,
        'train_ner_feats': x_train_feats[1] if x_train_feats else None,  # refers to NER tag features.
        'valid_ner_feats': x_valid_feats[1] if x_train_feats else None,  # refers to NER tag features.
        'test_ner_feats': x_test_feats[1] if x_train_feats else None,  # refers to NER tag features.
        'train_pos_feats': x_train_feats[2] if x_train_feats else None,  # refers to POS tag features.
        'valid_pos_feats': x_valid_feats[2] if x_train_feats else None,  # refers to POS tag features.
        'test_pos_feats': x_test_feats[2] if x_train_feats else None,  # refers to POS tag features.
        'train_sent_nr_feats': x_train_sent_nr,  # refers to sentence nr features.
        'valid_sent_nr_feats': x_valid_sent_nr,  # refers to sentence nr features.
        'test_sent_nr_feats': x_test_sent_nr,  # refers to sentence nr features.
        # 'features_to_use': args['multi_features'],
        'validation_cost': False,
        'na_tag': None,
        'n_hidden': args['n_hidden'],
        'early_stopping_threshold': args['early_stopping_threshold'],
        'pickle_lists': args['pickle_lists'],
        'logger': logger
    }

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network', 'with' if regularization else 'without', 'regularization']))
    nn_trainer.train(learning_rate=.01, batch_size=minibatch_size, max_epochs=max_epochs, save_params=False, **kwargs)

    logger.info('Predicting tags %s' % ' ,'.join(tags_2nd_step))

    logger.info('Predicting on Training set')
    nn_trainer.x_train = np.array(x_train)[indexes_to_train_to_replace]
    training_set_predictions = nn_trainer.predict(on_training_set=True, **kwargs)
    train_flat_predictions_2nd_step = training_set_predictions['flat_predictions']

    logger.info('Predicting on Validation set')
    validation_set_predictions = nn_trainer.predict(on_validation_set=True, **kwargs)
    valid_flat_true_2nd_step = validation_set_predictions['flat_trues']
    valid_flat_predictions_2nd_step = validation_set_predictions['flat_predictions']

    logger.info('Predicting on Testing set')
    testing_set_predictions = nn_trainer.predict(on_testing_set=True, **kwargs)
    test_flat_predictions_2nd_step = testing_set_predictions['flat_predictions']

    assert valid_flat_true_2nd_step.__len__() == valid_flat_predictions_2nd_step.__len__()
    assert indexes_to_train_to_replace.__len__() == train_flat_predictions_2nd_step.__len__()
    assert indexes_to_valid.__len__() == valid_flat_predictions_2nd_step.__len__()
    assert indexes_to_test.__len__() == test_flat_predictions_2nd_step.__len__()

    # merge 1st and 2nd steps predictions.
    train_final_prediction = map(lambda x: index2label_1st_step[x], train_predictions_1st_step)
    valid_final_prediction = map(lambda x: index2label_1st_step[x], valid_predictions_1st_step)
    test_final_prediction = map(lambda x: index2label_1st_step[x], test_predictions_1st_step)
    # final_prediction[list(set(range(final_prediction.__len__())) - set(test_index_to_test))] = predictions_1st_step
    # if i index the list as np.array and do the replacement, some tags appear trimmed! I have to forloop them
    for idx, res in zip(indexes_to_train_to_replace, map(lambda x: index2label_2nd_step[x], train_flat_predictions_2nd_step)):
        train_final_prediction[idx] = res
    for idx, res in zip(indexes_to_valid, map(lambda x: index2label_2nd_step[x], valid_flat_predictions_2nd_step)):
        valid_final_prediction[idx] = res
    for idx, res in zip(indexes_to_test, map(lambda x: index2label_2nd_step[x], test_flat_predictions_2nd_step)):
        test_final_prediction[idx] = res

    assert train_final_prediction.__len__() == y_train_labels.__len__()
    assert valid_final_prediction.__len__() == y_valid_labels.__len__()
    assert test_final_prediction.__len__() == y_test_labels.__len__()
    assert default_tag not in train_final_prediction
    assert default_tag not in valid_final_prediction
    assert default_tag not in test_final_prediction

    # flat_true = y_test_labels
    # flat_predictions = map(lambda x: index2label[x], final_prediction)

    results[0] = (y_train_labels, train_final_prediction, y_valid_labels, valid_final_prediction, y_test_labels, test_final_prediction)

    return results

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
    train_fout_name = get_output_path('train_B.txt')
    valid_fout_name = get_output_path('validation_B.txt')
    test_fout_name = get_output_path('test_B.txt')

    _, _, document_sentence_words, _ = Dataset.get_clef_training_dataset()
    write_to_file(train_fout_name, document_sentence_words, train_y_pred, file_prefix='output', file_suffix='.txt')

    _, _, document_sentence_words, _ = Dataset.get_clef_validation_dataset()
    write_to_file(valid_fout_name, document_sentence_words, valid_y_pred, file_prefix='test', file_suffix='.xml.data')

    _, _, document_sentence_words, _ = Dataset.get_clef_testing_dataset()
    write_to_file(test_fout_name, document_sentence_words, test_y_pred, file_prefix='', file_suffix='.xml.data')

    return True

def perform_leave_one_out():
    raise Exception('Unimplemented.')


if __name__ == '__main__':

    start = time.time()

    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    args = parse_arguments()

    check_arguments_consistency(args)

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    nn_class, hidden_f, out_f, add_words, add_tags, add_feats, tag_dim, net_window, multi_feats, normalize_samples = \
        determine_nnclass_and_parameters(args)

    get_output_path = get_ensemble_forest_mlp_path

    logger.info('Using Forest: %s with window size: %d and Neural class: %s with window size: %d for epochs: %d'
                % ('Random forest' if args['classifier']=='rf' else 'GBDT', args['forest_window'], args['nn_name'],
                   net_window, args['max_epochs']))

    if args['use_leave_one_out']:
        results = perform_leave_one_out()
    else:
        results = use_testing_dataset(nn_class,
                                        hidden_f,
                                        out_f,
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
    # cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))

    train_y_true = list(chain(*[train_true for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))
    train_y_pred = list(chain(*[train_pred for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))
    valid_y_true = list(chain(*[valid_true for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))
    valid_y_pred = list(chain(*[valid_pred for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))
    test_y_true = list(chain(*[test_true for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))
    test_y_pred = list(chain(*[test_pred for train_true, train_pred, valid_true, valid_pred, test_true, test_pred in results.values()]))

    Metrics.print_metric_results(train_y_true=train_y_true, train_y_pred=train_y_pred,
                                 valid_y_true=valid_y_true, valid_y_pred=valid_y_pred,
                                 test_y_true=test_y_true, test_y_pred=test_y_pred,
                                 metatags=args['meta_tags'],
                                 get_output_path=get_output_path,
                                 additional_labels=add_tags,
                                 logger=logger)


    # print '...Plotting confusion matrix'
    # output_filename = get_ensemble_forest_mlp_path('confusion_matrix.png')
    # # labels_list = list(set(valid_y_true).union(set(valid_y_pred)))
    # labels_list = get_classification_report_labels()
    # cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
    # plot_confusion_matrix(confusion_matrix=cm, labels=labels_list, output_filename=output_filename)
    #
    # report = Metrics.compute_classification_report(valid_y_true, valid_y_pred, labels=get_classification_report_labels())
    #
    # # save classification report to file
    # fout = open(get_output_path('classification_report.txt'), 'wb')
    # for i,line in enumerate(report.strip().split('\n')):
    #     if line == u'':
    #         continue
    #     elif i == 0:
    #         fout.write("{: >68} {: >20} {: >20} {: >20}".format(*[c for c in line.strip().split('  ') if c != u'']))
    #         fout.write('\n')
    #     else:
    #         fout.write("{: >47} {: >20} {: >20} {: >20} {: >20}".format(*[c for c in line.strip().split('  ') if c != u'']))
    #         fout.write('\n')
    #
    # fout.close()
    #
    # results_micro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='micro')
    # results_macro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='macro')
    #
    # print 'MICRO results'
    # print results_micro
    # print 'MACRO results'
    # print results_macro
    #
    # print '...Saving predictions to file'
    # save_predictions_to_file(train_y_pred, valid_y_pred, test_y_pred, get_output_path)

    print 'Elapsed time: ', time.time()-start

    logger.info('End')
