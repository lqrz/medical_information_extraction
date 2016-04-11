__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from SOTA.crf.crf_sklearn_crfsuite import CRF
from sklearn.cross_validation import LeaveOneOut
from data.dataset import Dataset
import logging
import argparse
from trained_models import get_single_mlp_path

from hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net #sgd and adagrad
from single_Layer_Net import Single_Layer_Net
from vector_Tag_Contex_Window_Net import Vector_Tag_Contex_Window_Net

from utils.utils import NeuralNetwork
from utils.metrics import Metrics
from itertools import chain
import numpy as np
import copy
from collections import OrderedDict
import time
import cPickle

np.random.seed(1234)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def construct_indexes(crf_training_data_filename, crf_testing_data_filename=None, add_tags=[]):

    _, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_crf_training_data_by_sentence(crf_training_data_filename)

    document_sentence_words = []
    document_sentence_tags = []
    document_sentence_words.extend(train_document_sentence_words.values())
    document_sentence_tags.extend(train_document_sentence_tags.values())

    if crf_testing_data_filename:
        _, _, test_document_sentence_words, test_document_sentence_tags = \
            Dataset.get_crf_training_data_by_sentence(file_name=test_data_filename,
                                                      path=Dataset.TESTING_FEATURES_PATH+'test',
                                                      extension=Dataset.TESTING_FEATURES_EXTENSION)
        document_sentence_words.extend(test_document_sentence_words.values())
        document_sentence_tags.extend(test_document_sentence_tags.values())

    unique_words = list(set([word for doc_sentences in document_sentence_words for sentence in doc_sentences for word in sentence]))
    unique_labels = list(set([tag for doc_sentences in document_sentence_tags for sentence in doc_sentences for tag in sentence]))

    index2word = dict()
    word2index = dict()
    for i,word in enumerate(unique_words):
        index2word[i] = word
        word2index[word] = i
    if add_tags:
        for i, tag in enumerate(add_tags):
            word2index[tag] = len(unique_words) + i

    index2label = dict()
    label2index = dict()
    for i,label in enumerate(unique_labels):
        index2label[i] = label
        label2index[label] = i
    if add_tags:
        for i, tag in enumerate(add_tags):
            label2index[tag] = len(unique_labels) + i

    return index2word, word2index, index2label, label2index

def items_to_index(item_list, index):
    return map(lambda x: index[x], item_list)

def construct_word_representation_index(word_feat_sentences):

    index2rep = OrderedDict()
    rep2index = OrderedDict()

    for i,word_feat_rep in enumerate(word_feat_sentences):
        rep2index[str(word_feat_rep)] = i   #TODO: must exist a nicer way.
        index2rep[i] = word_feat_rep

    #add <PAD>
    index2rep[index2rep.__len__()] = np.zeros_like(index2rep[0])

    return index2rep, rep2index

def representation_sentences_to_indexes(representation_sentences, rep2index):
    return [representation_sentence_to_index(sent,rep2index) for sent in representation_sentences]

def representation_sentence_to_index(representation_sentence, rep2index):
    return map(lambda x: rep2index[str(x)], representation_sentence)

def perform_leave_one_out(training_data,
                          crf_model,
                          hidden_f,
                          out_f,
                          tag_dim,
                          n_window,
                          max_cv_iters=None,
                          regularization=None,
                          nn_name=None,
                          max_epochs=None,
                          **kwargs):

    logger.info('Using Leave one out cross validation')

    results = dict()

    logger.info('Constructing word and label indexes')
    index2word, word2index, index2label, label2index = construct_indexes(crf_training_data_filename, crf_testing_data_filename=None, add_tags=add_tags)

    logger.info('Getting CRF training data')
    x = crf_model.get_features_from_crf_training_data(crf_model.training_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.training_data)

    loo = LeaveOneOut(training_data.__len__())
    for i, (x_idx, y_idx) in enumerate(loo):

        if (max_cv_iters > 0) and ((i+1) > max_cv_iters):
            break

        logger.info('Cross validation '+str(i)+' (train+predict)')
        # print x_idx, y_idx

        x_train, y_train = crf_model.filter_by_doc_nr(x, y, x_idx)

        crf_model.train(x_train, y_train, verbose=True)

        x_test, y_test = crf_model.filter_by_doc_nr(x, y, y_idx)
        predictions, _, _, _, _ = crf_model.predict(x_test, y_test)

        logger.info('Getting training set CRF features')
        x_train_crf_features = crf_model.sentences_to_features(x_train, y_train)

        logger.info('Getting test set CRF features')
        x_test_crf_features = crf_model.sentences_to_features(x_test, predictions)

        x_rep = copy.copy(x_train_crf_features)
        x_rep.extend(x_test_crf_features)
        x_rep_flat = list(chain(*x_rep))

        index2rep, rep2index = construct_word_representation_index(x_rep_flat)

        w_x = np.matrix(index2rep.values())
        n_features = w_x.shape[1]

        x_train_index = representation_sentences_to_indexes(x_train_crf_features, rep2index)
        x_test_index = representation_sentences_to_indexes(x_test_crf_features, rep2index)

        x_train_index_cw = np.array([sent_cws for sent in x_train_index for sent_cws in
                      NeuralNetwork.context_window(sent, n_window, pad_idx=index2rep.__len__()-1)])

        x_test_index_cw = np.array([sent_cws for sent in x_test_index for sent_cws in
                      NeuralNetwork.context_window(sent, n_window, pad_idx=index2rep.__len__()-1)])

        bos_index = np.where(x_train_index_cw[:,0]==index2rep.__len__()-1)

        logger.info('Instantiating Neural network')

        #flatten x_train and x_test to word level
        x_train = np.array(list(chain(*x_train_crf_features)))#TODO: for no-cw-mlp
        x_test = np.array(list(chain(*x_test_crf_features)))#TODO: for no-cw-mlp
        y_train = list(chain(*y_train))
        # y_test = list(chain(*predictions))
        y_test = list(chain(*y_test))

        y_train = np.array(items_to_index(y_train, label2index))
        y_test = np.array(items_to_index(y_test, label2index))

        #TODO: for no-cw-mlp
        w = NeuralNetwork.initialize_weights(n_in=n_features*n_window, n_out=label2index.__len__(), function='softmax')

        pad_tag, unk_tag, pad_word = determine_key_indexes(label2index, index2rep)

        params = {
            'hidden_activation_f': hidden_f,
            'out_activation_f': out_f,
            # 'x_train': x_train,#TODO: for no-cw-mlp
            'x_train': x_train_index_cw,
            'y_train': y_train,
            # 'x_test': x_test,#TODO: for no-cw-mlp
            'x_test': x_test_index_cw,
            'y_test': y_test,
            'embeddings': w_x,
            # 'embeddings': w,#TODO: for no-cw-mlp
            'n_window': n_window,
            'tag_dim': tag_dim,#TODO: for vector_tag
            'regularization': regularization,
            'n_out': label2index.__len__()-add_tags.__len__(),
            'pad_tag': pad_tag,
            'unk_tag': unk_tag,
            'pad_word': pad_word,
            'bos_index': bos_index
        }

        if nn_name == 'single_mlp':
            params['x_train'] = x_train
            params['x_test'] = y_test
            params['embeddings'] = w

        nn_trainer = nn_class(**params)

        logger.info(' '.join(['Training Neural network','with' if regularization else 'without', 'regularization']))
        nn_trainer.train(
            learning_rate=0.01,
            batch_size=1,
            max_epochs=max_epochs,
            alpha_L1_reg=0.001,
            alpha_L2_reg=0.01,
            save_params=False,
            use_scan=False
        )

        logger.info('Predicting')
        y_true, y_pred = nn_trainer.predict()

        results[i] = (y_true, y_pred)

        cPickle.dump(results,open('cw_ada_sgd'+str(i)+'.p', 'wb'))

    return results

def use_testing_dataset(crf_model,
                        feature_function,
                        hidden_f,
                        out_f,
                        tag_dim,
                        n_window,
                        regularization=None,
                        nn_name=None,
                        max_epochs=None,
                        minibatch_size=None,
                        **kwargs):

    logger.info('Using CLEF testing data')

    results = dict()

    # train on 101 training documents, and predict on 100 testing documents.
    testing_data, testing_texts, _, _ = \
        Dataset.get_crf_training_data_by_sentence(file_name=test_data_filename,
                                                  path=Dataset.TESTING_FEATURES_PATH+'test',
                                                  extension=Dataset.TESTING_FEATURES_EXTENSION)

    # set the testing_data attribute of the model
    crf_model.testing_data = testing_data

    logger.info('Constructing word and label indexes')
    index2word, word2index, index2label, label2index = construct_indexes(crf_training_data_filename, test_data_filename, add_tags=add_tags)

    # get training features
    x = crf_model.get_features_from_crf_training_data(crf_model.training_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.training_data)

    x_train = list(chain(*x.values()))
    y_train = list(chain(*y.values()))

    logger.info('Training the CRF model')
    crf_model.train(x_train, y_train, verbose=True)

    # get testing features
    x = crf_model.get_features_from_crf_training_data(crf_model.testing_data, feature_function)
    y = crf_model.get_labels_from_crf_training_data(crf_model.testing_data)

    x_test = list(chain(*x.values()))
    y_test = list(chain(*y.values()))

    logger.info('Predicting with the CRF model')
    predictions, _, _, _, _ = crf_model.predict(x_test, y_test)

    logger.info('Getting training set CRF features')
    x_train_crf_features = crf_model.sentences_to_features(x_train, y_train)

    logger.info('Getting test set CRF features')
    x_test_crf_features = crf_model.sentences_to_features(x_test, predictions)

    x_rep = copy.copy(x_train_crf_features)
    x_rep.extend(x_test_crf_features)
    x_rep_flat = list(chain(*x_rep))

    index2rep, rep2index = construct_word_representation_index(x_rep_flat)

    w_x = np.matrix(index2rep.values())
    n_features = w_x.shape[1]

    x_train_index = representation_sentences_to_indexes(x_train_crf_features, rep2index)
    x_test_index = representation_sentences_to_indexes(x_test_crf_features, rep2index)

    x_train_index_cw = np.array([sent_cws for sent in x_train_index for sent_cws in
                  NeuralNetwork.context_window(sent, n_window, pad_idx=index2rep.__len__()-1)])

    x_test_index_cw = np.array([sent_cws for sent in x_test_index for sent_cws in
                  NeuralNetwork.context_window(sent, n_window, pad_idx=index2rep.__len__()-1)])

    bos_index = np.where(x_train_index_cw[:,0]==index2rep.__len__()-1)

    logger.info('Instantiating Neural network')

    #flatten x_train and x_test to word level
    x_train = np.array(list(chain(*x_train_crf_features)))#TODO: for no-cw-mlp
    x_test = np.array(list(chain(*x_test_crf_features)))#TODO: for no-cw-mlp
    y_train = list(chain(*y_train))
    # y_test = list(chain(*predictions))
    y_test = list(chain(*y_test))

    y_train = np.array(items_to_index(y_train, label2index))
    y_test = np.array(items_to_index(y_test, label2index))

    #TODO: for no-cw-mlp
    w = NeuralNetwork.initialize_weights(n_in=n_features*n_window, n_out=label2index.__len__(), function='softmax')

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index, index2rep)

    params = {
        'hidden_activation_f': hidden_f,
        'out_activation_f': out_f,
        # 'x_train': x_train,#TODO: for no-cw-mlp
        'x_train': x_train_index_cw,
        'y_train': y_train,
        # 'x_test': x_test,#TODO: for no-cw-mlp
        'x_test': x_test_index_cw,
        'y_test': y_test,
        'embeddings': w_x,
        # 'embeddings': w,#TODO: for no-cw-mlp
        'n_window': n_window,
        'tag_dim': tag_dim,#TODO: for vector_tag
        'regularization': regularization,
        'n_out': label2index.__len__()-add_tags.__len__(),
        'pad_tag': pad_tag,
        'unk_tag': unk_tag,
        'pad_word': pad_word,
        'bos_index': bos_index
    }

    if nn_name == 'single_mlp':
        params['x_train'] = x_train
        params['x_test'] = x_test
        params['embeddings'] = w

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network','with' if regularization else 'without', 'regularization']))

    nn_trainer.train(
        learning_rate=0.01,
        batch_size=minibatch_size,
        max_epochs=max_epochs,
        alpha_L1_reg=0.001,
        alpha_L2_reg=0.01,
        save_params=False,
        use_scan=False,
        **kwargs
    )

    logger.info('Predicting')
    y_true, y_pred = nn_trainer.predict()

    results[0] = (y_true, y_pred)

    return results

def determine_key_indexes(label2index, index2rep):
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
        pad_word = index2rep.__len__()-1    #its the last one
    except KeyError:
        pass

    return pad_tag, unk_tag, pad_word

def parse_arguments():
    parser = argparse.ArgumentParser(description='CRF-MLP Ensemble')
    parser.add_argument('--originalfeatures', action='store_true', default=False)
    parser.add_argument('--customfeatures', action='store_true', default=False)
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
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_mlp','hidden_cw','vector_tag','last_tag','rnn'], help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')
    parser.add_argument('--regularization', action='store_true', default=False)
    parser.add_argument('--minibatch', action='store', type=int, default=False)
    parser.add_argument('--tagdim', action='store', type=int, default=None)
    parser.add_argument('--cviters', action='store', type=int, default=0)
    parser.add_argument('--crfiters', action='store', type=int, default=50)
    parser.add_argument('--leaveoneout', action='store_true', default=False)
    parser.add_argument('--gradmeans', action='store_true', default=False)

    arguments = parser.parse_args()

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
    args['nn_name'] = arguments.net
    args['window_size'] = arguments.window
    args['max_epochs'] = arguments.epochs
    args['regularization'] = arguments.regularization
    args['tagdim'] = arguments.tagdim
    args['use_leave_one_out'] = arguments.leaveoneout
    args['crf_iters'] = arguments.crfiters
    args['minibatch_size'] = arguments.minibatch
    args['use_grad_means'] = arguments.gradmeans

    return args

def determine_and_assign_feature_function(crf_model, args):
    #TODO: im using "custom_word_features_for_nnet"

    if args['use_original_paper_features']:
        feature_function = crf_model.get_original_paper_word_features
    elif args['use_custom_features']:
        feature_function = crf_model.get_custom_word_features

    feature_function = crf_model.get_custom_word_features_for_nnet

    return feature_function

def determine_nnclass_and_parameters(args):
    #TODO: add output path

    nn_class = None
    hidden_f = None #no hidden layer in the single MLP.
    out_f = NeuralNetwork.softmax_activation_function
    get_output_path = get_single_mlp_path
    n_window = args['window_size']
    add_tags = []
    tag_dim = None

    if args['nn_name'] == 'hidden_cw':
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Hidden_Layer_Context_Window_Net
        hidden_f = NeuralNetwork.tanh_activation_function
        add_tags = ['<PAD>']
    #     get_output_path = get_single_mlp_path
    # elif nn_name == 'mlp':
    #     nn_class = MLP_neural_network_trainer
    #     get_output_path = get_cwnn_path
    elif args['nn_name'] == 'vector_tag':
        # nn_class = Vector_tag_CW_MLP_neural_network_trainer
        nn_class = Vector_Tag_Contex_Window_Net
        hidden_f = NeuralNetwork.tanh_activation_function
        tag_dim = args['tagdim']
        add_tags = ['<PAD>','<UNK>']
    # elif args['nn_name'] == 'vector_tag_rand':
    #     nn_class = Vector_tag_CW_MLP_neural_network_trainer_random
    #     hidden_f = NeuralNetwork.tanh_activation_function
    #     tag_dim = args['tagdim']
    #     add_tags = ['<PAD>','<UNK>']
    # elif args['nn_name'] == 'single_cw_mlp_mini':
    #     nn_class = Single_CW_MLP_neural_network_trainer_minibatch
    #     hidden_f = NeuralNetwork.tanh_activation_function
    #     add_tags = ['<PAD>']
    elif args['nn_name'] == 'single_mlp':
        nn_class = Single_Layer_Net
        n_window = 1
        hidden_f = None

    return nn_class, hidden_f, out_f, add_tags, tag_dim, n_window

def check_arguments_consistency(args):
    if (args['nn_name'] == 'vector_tag_rand' or args['nn_name'] == 'vector_tag') and \
            not args['tagdim']:
        logger.error('Must provide tag dimensionality for Vector tag NNets')
        exit()

    return True

if __name__=='__main__':

    start = time.time()

    args = parse_arguments()

    check_arguments_consistency(args)

    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    # get the CRF training data
    training_data, training_texts, _, _ = Dataset.get_crf_training_data_by_sentence(crf_training_data_filename)

    # instantiate the CRF
    crf_model = CRF(training_data=training_data,
                    testing_data=None,
                    output_model_filename=None,
                    **args)

    feature_function = determine_and_assign_feature_function(crf_model, args)

    nn_class, hidden_f, out_f, add_tags, tag_dim, n_window = determine_nnclass_and_parameters(args)

    logger.info('Extracting features with: '+feature_function.__str__())

    logger.info('Using w2v_similar_words:%s kmeans:%s lda:%s zip:%s' %
                (args['w2v_similar_words'], args['kmeans_features'], args['lda_features'], args['zip_features']))

    logger.info('Using w2v_model: %s and vector_dictionary: %s' % (args['w2v_model_file'], args['w2v_vectors_cache']))

    if args['use_leave_one_out']:
        results = perform_leave_one_out(training_data,
                                        crf_model,
                                        hidden_f,
                                        out_f,
                                        tag_dim,
                                        n_window,
                                        **args)
    else:
        results = use_testing_dataset(crf_model,
                                      feature_function,
                                      hidden_f,
                                      out_f,
                                      tag_dim,
                                      n_window,
                                      **args)

    y_true = list(chain(*[true for true, _ in results.values()]))
    y_pred = list(chain(*[pred for _, pred in results.values()]))
    results_micro = Metrics.compute_all_metrics(y_true, y_pred, average='micro')
    results_macro = Metrics.compute_all_metrics(y_true, y_pred, average='macro')

    print 'MICRO results'
    print results_micro
    print 'MACRO results'
    print results_macro

    print 'Elapsed time: ', time.time()-start

    logger.info('End')
