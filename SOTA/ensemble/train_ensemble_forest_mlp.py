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

from forests import Forest
from SOTA.neural_network.A_neural_network import A_neural_network
from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from SOTA.neural_network.single_Layer_Context_Window_Net import Single_Layer_Context_Window_Net
from SOTA.neural_network.last_tag_neural_network import Last_tag_neural_network_trainer
from SOTA.neural_network.vector_Tag_Contex_Window_Net import Vector_Tag_Contex_Window_Net
from SOTA.neural_network.recurrent_net import Recurrent_net
from SOTA.neural_network.recurrent_Context_Window_net import Recurrent_Context_Window_net
from trained_models import get_ensemble_forest_mlp_path
from data import get_param
from utils import utils
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from utils.metrics import Metrics
from utils.plot_confusion_matrix import plot_confusion_matrix
from data import get_classification_report_labels

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

np.random.seed(1234)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_cw','hidden_cw','vector_tag','last_tag','rnn', 'cw_rnn'], help='NNet type')
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

def filter_tags_to_predict(y_train_labels, y_test_labels, tags, default_tag=None):

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
    y_test = []
    if isinstance(y_train_labels[0], list):
        for i, sent in enumerate(y_train_labels):
            y_train.append(map(replace_tag, sent))
        for i, sent in enumerate(y_test_labels):
            y_test.append(map(replace_tag, sent))
    else:
        y_train = map(replace_tag, y_train_labels)
        y_test = map(replace_tag, y_test_labels)

    return y_train, y_test, new_label2index, new_index2label

def get_original_labels(y_test, y_train, index2label):
    y_train_labels = []
    y_test_labels = []

    # recurrent nets use lists of lists.
    if isinstance(y_train[0], list):
        for i, sent in enumerate(y_train):
            y_train_labels.append(map(lambda x: index2label[x], sent))
        for i, sent in enumerate(y_test):
            y_test_labels.append(map(lambda x: index2label[x], sent))
    else:
        y_train_labels = map(lambda x: index2label[x], y_train)
        y_test_labels = map(lambda x: index2label[x], y_test)

    return y_train_labels, y_test_labels

def determine_nnclass_and_parameters(args):

    hidden_f = utils.NeuralNetwork.tanh_activation_function
    out_f = utils.NeuralNetwork.softmax_activation_function
    add_words = []
    add_tags = []
    tag_dim = None
    net_window = args['net_window']
    nn_class = None

    if args['nn_name'] == 'single_cw':
        nn_class = Single_Layer_Context_Window_Net
        hidden_f = None #no hidden layer in the single MLP.
        add_words = ['<PAD>']
    if args['nn_name'] == 'hidden_cw':
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Hidden_Layer_Context_Window_Net
        add_words = ['<PAD>']
    elif args['nn_name'] == 'vector_tag':
        nn_class = Vector_Tag_Contex_Window_Net
        tag_dim = args['tagdim']
        add_words = ['<PAD>']
        add_tags = ['<PAD>', '<UNK>']
    elif args['nn_name'] == 'last_tag':
        nn_class = Last_tag_neural_network_trainer
        #TODO: no add tags? or add words?
    elif args['nn_name'] == 'rnn':
        #the RNN init function overwrites the n_window param and sets it to 1.
        nn_class = Recurrent_net
        net_window = 1
    elif args['nn_name'] == 'cw_rnn':
        nn_class = Recurrent_Context_Window_net
        add_words = ['<PAD>']

    return nn_class, hidden_f, out_f, add_words, add_tags, tag_dim, net_window

def use_testing_dataset(crf_training_data_filename,
                        test_data_filename,
                        nn_class,
                        hidden_f,
                        out_f,
                        w2v_model,
                        w2v_vectors,
                        w2v_dims,
                        add_words,
                        add_tags,
                        tag_dim,
                        get_output_path,
                        classifier,
                        forest_window,
                        net_window,
                        tags,
                        max_epochs,
                        minibatch_size,
                        regularization,
                        **kwargs
                        ):

    logger.info('Using CLEF testing data')

    results = dict()

    logger.info('Loading CRF training data')

    x_train, y_train_all, x_test, y_test_all, word2index_all, index2word_all, label2index_all, index2label_all = \
        A_neural_network.get_data(crf_training_data_filename, test_data_filename, add_words, add_tags,
                                  x_idx=None, n_window=forest_window)

    unique_words = word2index_all.keys()

    pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    # determine which tags to predict in the first step, and which ones in the second.
    y_train_labels, y_test_labels = get_original_labels(y_test_all, y_train_all, index2label_all)
    tags = get_param(tags)
    tags_2nd_step = set(label2index_all.keys()) - set(tags)
    default_tag = '<OTHER>'
    y_train_1st_step, y_test_1st_step, label2index_1st_step, index2label_1st_step = \
        filter_tags_to_predict(y_train_labels, y_test_labels, tags, default_tag=default_tag)

    forest = Forest(classifier, pretrained_embeddings, forest_window)
    forest.train(x_train, y_train_1st_step)
    predictions_1st_step = forest.predict(x_test)

    score_1st_step = Metrics.compute_all_metrics(y_test_1st_step, predictions_1st_step, average=None)
    print score_1st_step

    if forest_window != net_window:
        # the forest and the nnet might use different window sizes.
        x_train, _, x_test, _, _, _, _, _ = \
            A_neural_network.get_data(crf_training_data_filename, test_data_filename, add_words, add_tags,
                                      x_idx=None, n_window=net_window)

    #--- here starts the 2nd step ---#

    # get the subsamples to train and test on this 2nd step (all that were marked as '<OTHER>').
    indexes_to_train = np.where(np.array(y_train_1st_step) == label2index_1st_step[default_tag])[0]
    indexes_to_test = np.where(predictions_1st_step == label2index_1st_step[default_tag])[0]
    x_train_2nd_step = np.array(x_train)[indexes_to_train]
    y_train_2nd_step = np.array(y_train_labels)[indexes_to_train]
    x_test_2nd_step = np.array(x_test)[indexes_to_test]
    y_test_2nd_step = np.array(y_test_labels)[indexes_to_test]

    dummy_tag = '#delete#'  #TODO: this is nasty.
    y_train_2nd_step, y_test_2nd_step, label2index_2nd_step, index2label_2nd_step = \
        filter_tags_to_predict(y_train_2nd_step, y_test_2nd_step, tags_2nd_step, default_tag=dummy_tag)

    # TODO: this is nasty.
    dummy_value = label2index_2nd_step[dummy_tag]
    del label2index_2nd_step[dummy_tag]
    del index2label_2nd_step[dummy_value]

    n_out = len(label2index_2nd_step.keys())

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index_2nd_step, word2index_all)

    params = {
        'x_train': x_train_2nd_step,
        'y_train': y_train_2nd_step,
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
        'get_output_path': get_output_path
    }

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network', 'with' if regularization else 'without', 'regularization']))
    nn_trainer.train(learning_rate=.01, batch_size=minibatch_size, max_epochs=max_epochs, save_params=False, **kwargs)

    logger.info('Predicting tags %s' % ' ,'.join(tags_2nd_step))
    flat_true_2nd_step, flat_predictions_2nd_step, trues_2nd_step, preds_2nd_step = nn_trainer.predict(**kwargs)

    assert flat_true_2nd_step.__len__() == flat_predictions_2nd_step.__len__()
    assert indexes_to_test.__len__() == flat_predictions_2nd_step.__len__()

    # merge 1st and 2nd steps predictions.
    final_prediction = map(lambda x: index2label_1st_step[x], predictions_1st_step)
    # final_prediction[list(set(range(final_prediction.__len__())) - set(test_index_to_test))] = predictions_1st_step
    # if i index the list as np.array and do the replacement, some tags appear trimmed! I have to forloop them
    for idx, res in zip(indexes_to_test, map(lambda x: index2label_2nd_step[x], flat_predictions_2nd_step)):
        final_prediction[idx] = res

    assert final_prediction.__len__() == y_test_labels.__len__()
    assert default_tag not in final_prediction

    # flat_true = y_test_labels
    # flat_predictions = map(lambda x: index2label[x], final_prediction)

    results[0] = (y_test_labels, final_prediction)

    return results

def perform_leave_one_out():
    raise Exception('Unimplemented.')


if __name__ == '__main__':

    start = time.time()

    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    args = parse_arguments()

    check_arguments_consistency(args)

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    nn_class, hidden_f, out_f, add_words, add_tags, tag_dim, net_window = determine_nnclass_and_parameters(args)

    get_output_path = get_ensemble_forest_mlp_path

    logger.info('Using Forest: %s with window size: %d and Neural class: %s with window size: %d for epochs: %d'
                % ('Random forest' if args['classifier']=='rf' else 'GBDT', args['net_window'], args['nn_name'],
                   net_window, args['max_epochs']))

    if args['use_leave_one_out']:
        results = perform_leave_one_out()
    else:
        results = use_testing_dataset(crf_training_data_filename,
                                                   test_data_filename,
                                                   nn_class,
                                                   hidden_f,
                                                   out_f,
                                                   w2v_model,
                                                   w2v_vectors,
                                                   w2v_dims,
                                                   add_words,
                                                   add_tags,
                                                   tag_dim,
                                                   get_output_path,
                                                   **args)

    cPickle.dump(results, open(get_output_path('prediction_results.p'),'wb'))
    # cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))

    y_true = list(chain(*[true for true, _ in results.values()]))
    y_pred = list(chain(*[pred for _, pred in results.values()]))

    print '...Plotting confusion matrix'
    output_filename = get_ensemble_forest_mlp_path('confusion_matrix.png')
    labels_list = list(set(y_true).union(set(y_pred)))
    cm = Metrics.compute_confusion_matrix(y_true, y_pred, labels=labels_list)
    plot_confusion_matrix(confusion_matrix=cm, labels=labels_list, output_filename=output_filename)

    report = Metrics.compute_classification_report(y_true, y_pred, labels=get_classification_report_labels())

    # save classification report to file
    fout = open(get_output_path('classification_report.txt'), 'wb')
    for i,line in enumerate(report.strip().split('\n')):
        if line == u'':
            continue
        elif i == 0:
            fout.write("{: >68} {: >20} {: >20} {: >20}".format(*[c for c in line.strip().split('  ') if c != u'']))
            fout.write('\n')
        else:
            fout.write("{: >47} {: >20} {: >20} {: >20} {: >20}".format(*[c for c in line.strip().split('  ') if c != u'']))
            fout.write('\n')

    fout.close()

    results_micro = Metrics.compute_all_metrics(y_true, y_pred, average='micro')
    results_macro = Metrics.compute_all_metrics(y_true, y_pred, average='macro')

    print 'MICRO results'
    print results_micro
    print 'MACRO results'
    print results_macro

    print 'Elapsed time: ', time.time()-start

    logger.info('End')
