__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from utils import utils
import cPickle
import logging
import argparse
import numpy as np
import time
from itertools import chain
from collections import Counter

from utils.metrics import Metrics
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from trained_models import get_cwnn_path
from trained_models import get_multi_hidden_cw_path
from trained_models import get_factored_model_path
from data import get_param
from utils.plot_confusion_matrix import plot_confusion_matrix
from data.dataset import Dataset
from data import get_classification_report_labels
from data import get_hierarchical_mapping
from trained_models import get_language_model_path
from SOTA.neural_network.multi_feature_type_hidden_layer_context_window_net import Multi_Feature_Type_Hidden_Layer_Context_Window_Net
from SOTA.language_model.language_model import Language_model

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

np.random.seed(1234)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_cw','hidden_cw','vector_tag','last_tag','rnn', 'cw_rnn', 'multi_hidden_cw'],
                        help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')
    parser.add_argument('--regularization', action='store_true', default=False)
    parser.add_argument('--minibatch', action='store', type=int, default=False)
    parser.add_argument('--gradmeans', action='store_true', default=False)

    group_w2v = parser.add_mutually_exclusive_group(required=True)
    group_w2v.add_argument('--w2vvectorscache', action='store', type=str)
    group_w2v.add_argument('--w2vmodel', action='store', type=str, default=None)

    parser.add_argument('--plot', action='store_true', default=False)
    parser.add_argument('--tags', action='store', type=str, default=None)

    group_cnn = parser.add_argument_group(title='cnn', description='Convolutional neural net specifics.')
    group_cnn.add_argument('--cnnfilters', action='store', type=int, default=None)
    group_cnn.add_argument('--static', action='store_true', default=False)
    group_cnn.add_argument('--maxpool', action='store_true', default=False)
    group_cnn.add_argument('--regionsizes', action='store', type=int, nargs='*')
    group_cnn.add_argument('--multifeats', action='store', type=str, nargs='*', default=[],
                           choices=Multi_Feature_Type_Hidden_Layer_Context_Window_Net.FEATURE_MAPPING.keys())

    parser.add_argument('--autoencoded', action='store_true', default=False)
    parser.add_argument('--aggregated', action='store_true', default=False)
    parser.add_argument('--normalizesamples', action='store_true', default=False)
    parser.add_argument('--negativesampling', action='store_true', default=False)

    #parse arguments
    arguments = parser.parse_args()

    args = dict()

    args['nn_name'] = arguments.net
    args['window_size'] = arguments.window
    args['max_epochs'] = arguments.epochs
    args['regularization'] = arguments.regularization
    args['minibatch_size'] = arguments.minibatch
    args['use_grad_means'] = arguments.gradmeans
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['w2v_model_name'] = arguments.w2vmodel
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

    return args

def check_arguments_consistency(args):
    if not args['w2v_vectors_cache'] and not args['w2v_model']:
        logger.error('Provide either a w2vmodel or a w2v vectors cache')
        exit()

    if args['nn_name'] == 'multi_hidden_cw' and not args['multi_features']:
        logger.error('Provide features for the nnet')
        exit()

    if args['nn_name'] == 'multi_hidden_cw':
        if np.max(args['region_sizes']) > args['window_size']:
            logger.error('Region size higher than window size.')
            exit()

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

    #TODO: im using get_factored_model for both nets

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

    if args['nn_name'] == 'hidden_cw':
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Hidden_Layer_Context_Window_Net
        get_output_path = get_factored_model_path
        add_words = ['<PAD>']
        multi_feats = args['multi_features']
        normalize_samples = args['norm_samples']
    elif args['nn_name'] == 'multi_hidden_cw':
        nn_class = Multi_Feature_Type_Hidden_Layer_Context_Window_Net
        get_output_path = get_factored_model_path
        add_words = ['<PAD>']
        add_feats = ['<PAD>']
        multi_feats = args['multi_features']

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

def viterbi_prediction(distribution, document_sentence_tags):
    # apply dynamic programming per document.
    top_n = 5
    tag_mapping = get_hierarchical_mapping()
    LM = Language_model.Instance()

    accum = 0
    for doc_sentences in document_sentence_tags.values():
        true_doc_tags = list(chain(*doc_sentences))

        # shortlist the top_n for the document predictions
        topn_doc_tag = distribution[range(true_doc_tags.__len__()), :].argsort()[::-1][:,:top_n]

        accum += true_doc_tags.__len__()

    return True

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
                        max_epochs=None,
                        minibatch_size=None,
                        regularization=None,
                        tags=None,
                        meta_tags=False,
                        **kwargs
                        ):

    logger.info('Using CLEF testing data')

    results = dict()

    logger.info('Loading CRF training data')

    feat_positions = nn_class.get_features_crf_position(multi_feats)

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
        x_train, y_train = perform_sample_normalization(x_train, y_train)

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

    pretrained_embeddings = nn_class.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    n_out = len(label2index.keys())

    logger.info('Instantiating Neural network')

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index, word2index)

    na_tag = label2index['NA']

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
        'train_ner_feats': x_train_feats[1] if x_train_feats else None,    #refers to NER tag features.
        'valid_ner_feats': x_valid_feats[1] if x_train_feats else None,    #refers to NER tag features.
        'test_ner_feats': x_test_feats[1] if x_train_feats else None,      #refers to NER tag features.
        'train_pos_feats': x_train_feats[2] if x_train_feats else None,    #refers to POS tag features.
        'valid_pos_feats': x_valid_feats[2] if x_train_feats else None,    #refers to POS tag features.
        'test_pos_feats': x_test_feats[2] if x_train_feats else None,    #refers to POS tag features.
        'train_sent_nr_feats': x_train_sent_nr_feats,    #refers to sentence nr features.
        'valid_sent_nr_feats': x_valid_sent_nr_feats,    #refers to sentence nr features.
        'test_sent_nr_feats': x_test_sent_nr_feats,    #refers to sentence nr features.
        'n_filters': args['n_filters'],
        'region_sizes': args['region_sizes'],
        'features_to_use': args['multi_features'],
        'static': args['static']
    }

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network', 'with' if regularization else 'without', 'regularization']))
    nn_trainer.train(learning_rate=.01, batch_size=minibatch_size, max_epochs=max_epochs, save_params=False, **kwargs)

    logger.info('Predicting on Training set')
    nnet_results = nn_trainer.predict_distribution(on_training_set=True, **kwargs)
    train_distribution = nnet_results['distribution']


    logger.info('Predicting on Validation set')
    nnet_results = nn_trainer.predict_distribution(on_validation_set=True, **kwargs)
    valid_flat_true = nnet_results['flat_trues']
    valid_distribution = nnet_results['distribution']

    logger.info('Predicting on Testing set')
    nnet_results = nn_trainer.predict_distribution(on_validation_set=False, **kwargs)
    test_distribution = nnet_results['distribution']

    cPickle.dump(train_distribution, open(get_output_path('train_output_distribution.p'), 'wb'))
    cPickle.dump(valid_distribution, open(get_output_path('valid_output_distribution.p'), 'wb'))
    cPickle.dump(test_distribution, open(get_output_path('test_output_distribution.p'), 'wb'))
    cPickle.dump(index2label, open(get_output_path('index2label.p'), 'wb'))

    # _, _, _, document_sentence_tags = Dataset.get_clef_validation_dataset()
    # viterbi_prediction(valid_distribution, document_sentence_tags)
    #

    #
    # valid_flat_true = map(lambda x: index2label[x], valid_flat_true)
    # valid_flat_predictions = map(lambda x: index2label[x], valid_flat_predictions)
    # train_flat_predictions = map(lambda x: index2label[x], train_flat_predictions)
    # test_flat_predictions = map(lambda x: index2label[x], test_flat_predictions)
    #
    # results[0] = (valid_flat_true, valid_flat_predictions, train_flat_predictions, test_flat_predictions)
    #
    # cPickle.dump(pretrained_embeddings, open(get_output_path('original_vectors.p'), 'wb'))
    # cPickle.dump(nn_trainer.params['w1'].get_value(), open(get_output_path('trained_vectors.p'), 'wb'))
    # cPickle.dump(word2index, open(get_output_path('word2index.p'), 'wb'))

    return results, index2label

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

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    nn_class, hidden_f, out_f, add_words, add_tags, add_feats, tag_dim, n_window, get_output_path, \
        multi_feats, normalize_samples = determine_nnclass_and_parameters(args)

    logger.info('Using Neural class: %s with window size: %d for epochs: %d' % (args['nn_name'],n_window,args['max_epochs']))

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

    # cPickle.dump(results, open(get_output_path('prediction_results.p'),'wb'))
    # cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))
    #
    # valid_y_true = list(chain(*[true for true, _, _, _ in results.values()]))
    # valid_y_pred = list(chain(*[pred for _, pred, _, _ in results.values()]))
    # train_y_pred = list(chain(*[pred for _, _, pred, _ in results.values()]))
    # test_y_pred = list(chain(*[pred for _, _, _, pred in results.values()]))
    #
    # results_micro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='micro')
    # results_macro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='macro')
    #
    # if args['plot']:
    #     labels_list = get_classification_report_labels()
    #     cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
    #     plot_confusion_matrix(cm, labels=labels_list, output_filename=get_output_path('confusion_matrix.png'))
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
