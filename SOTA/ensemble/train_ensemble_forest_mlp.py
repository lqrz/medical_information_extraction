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
from data.dataset import Dataset

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

def use_testing_dataset(nn_class,
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

    x_train, y_train_all, _, x_valid, y_valid_all, _, x_test, y_test_all, _, word2index_all, \
    index2word_all, label2index_all, index2label_all, _ = \
        nn_class.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                          add_tags=add_tags, x_idx=None, n_window=forest_window)

    unique_words = word2index_all.keys()

    pretrained_embeddings = A_neural_network.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    # determine which tags to predict in the first step, and which ones in the second.
    y_train_labels, y_valid_labels, y_test_labels = get_original_labels(y_train_all, y_valid_all, y_test_all, index2label_all)
    tags = get_param(tags)
    tags_2nd_step = set(label2index_all.keys()) - set(tags)
    default_tag = '<OTHER>'
    y_train_1st_step, y_valid_1st_step, y_test_1st_step, label2index_1st_step, index2label_1st_step = \
        filter_tags_to_predict(y_train_labels, y_valid_labels, y_test_labels, tags, default_tag=default_tag)

    forest = Forest(classifier, pretrained_embeddings, forest_window)
    forest.train(x_train, y_train_1st_step)
    train_predictions_1st_step = forest.predict(x_train)
    valid_predictions_1st_step = forest.predict(x_valid)
    test_predictions_1st_step = forest.predict(x_test)

    assert y_train_1st_step.__len__() == train_predictions_1st_step.__len__()
    train_score_1st_step = Metrics.compute_all_metrics(y_train_1st_step, train_predictions_1st_step, average=None)
    print 'Training scores'
    print train_score_1st_step

    assert y_valid_1st_step.__len__() == valid_predictions_1st_step.__len__()
    valid_score_1st_step = Metrics.compute_all_metrics(y_valid_1st_step, valid_predictions_1st_step, average=None)
    print 'Validation scores'
    print valid_score_1st_step

    if forest_window != net_window:
        # the forest and the nnet might use different window sizes.

        x_train, _, _, x_valid, _, _, x_test, _, _, _, _, _, _, _ = \
            nn_class.get_data(clef_training=True, clef_validation=True, clef_testing=True, add_words=add_words,
                              add_tags=add_tags, x_idx=None, n_window=net_window)
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
        'validation_cost': False
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
    testing_set_predictions = nn_trainer.predict(on_validation_set=False, **kwargs)
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

    results[0] = (y_valid_labels, valid_final_prediction, train_final_prediction, test_final_prediction)

    return results

def write_to_file(fout_name, document_sentence_words, predictions, file_prefix, file_suffix):
    fout = open(fout_name, 'wb')
    word_count = 0
    for doc_nr, sentences in document_sentence_words.iteritems():
        doc_words = list(chain(*sentences))
        doc_len = doc_words.__len__()
        for word, tag in zip(doc_words, predictions[word_count:doc_len]):
            line = '\t'.join([file_prefix+str(doc_nr)+file_suffix, word, tag])
            fout.write(line+'\n')

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

    nn_class, hidden_f, out_f, add_words, add_tags, tag_dim, net_window = determine_nnclass_and_parameters(args)

    get_output_path = get_ensemble_forest_mlp_path

    logger.info('Using Forest: %s with window size: %d and Neural class: %s with window size: %d for epochs: %d'
                % ('Random forest' if args['classifier']=='rf' else 'GBDT', args['net_window'], args['nn_name'],
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
                                        tag_dim,
                                        get_output_path,
                                        **args)

    cPickle.dump(results, open(get_output_path('prediction_results.p'),'wb'))
    # cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))

    valid_y_true = list(chain(*[true for true, _, _, _ in results.values()]))
    valid_y_pred = list(chain(*[pred for _, pred, _, _ in results.values()]))
    train_y_pred = list(chain(*[pred for _, _, pred, _ in results.values()]))
    test_y_pred = list(chain(*[pred for _, _, _, pred in results.values()]))

    print '...Plotting confusion matrix'
    output_filename = get_ensemble_forest_mlp_path('confusion_matrix.png')
    # labels_list = list(set(valid_y_true).union(set(valid_y_pred)))
    labels_list = get_classification_report_labels()
    cm = Metrics.compute_confusion_matrix(valid_y_true, valid_y_pred, labels=labels_list)
    plot_confusion_matrix(confusion_matrix=cm, labels=labels_list, output_filename=output_filename)

    report = Metrics.compute_classification_report(valid_y_true, valid_y_pred, labels=get_classification_report_labels())

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

    results_micro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='micro')
    results_macro = Metrics.compute_all_metrics(valid_y_true, valid_y_pred, average='macro')

    print 'MICRO results'
    print results_micro
    print 'MACRO results'
    print results_macro

    print '...Saving predictions to file'
    save_predictions_to_file(train_y_pred, valid_y_pred, test_y_pred, get_output_path)

    print 'Elapsed time: ', time.time()-start

    logger.info('End')
