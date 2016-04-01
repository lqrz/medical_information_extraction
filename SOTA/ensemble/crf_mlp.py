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
from ensemble_single_mlp import Single_MLP_neural_network_trainer
from ensemble_single_cw_mlp import Single_CW_MLP_neural_network_trainer
from ensemble_vector_tag_mlp import Vector_tag_CW_MLP_neural_network_trainer
from ensemble_vector_tag_mlp_randomcut import Vector_tag_CW_MLP_neural_network_trainer_random
from ensemble_vector_tag_test import Vector_tag_CW_MLP_neural_network_trainer_test
from utils.utils import NeuralNetwork
from utils.metrics import Metrics
from itertools import chain
import numpy as np
import copy
from collections import OrderedDict
import time

np.random.seed(1234)

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

def construct_indexes(crf_training_data_filename, add_tags=[]):

    _, _, document_sentence_words, document_sentence_tags = Dataset.get_crf_training_data_by_sentence(crf_training_data_filename)

    unique_words = list(set([word for doc_sentences in document_sentence_words.values() for sentence in doc_sentences for word in sentence]))
    unique_labels = list(set([tag for doc_sentences in document_sentence_tags.values() for sentence in doc_sentences for tag in sentence]))

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


if __name__=='__main__':

    start = time.time()

    parser = argparse.ArgumentParser(description='CRF-MLP Ensemble')
    # parser.add_argument('--outputfolder', default='./', type=str, help='Output folder for the model and logs')
    parser.add_argument('--originalfeatures', action='store_true', default=False)
    parser.add_argument('--customfeatures', action='store_true', default=False)
    parser.add_argument('--w2vsimwords', action='store_true', default=False)
    parser.add_argument('--w2vvectors', action='store_true', default=False)
    parser.add_argument('--w2vmodel', action='store', type=str, default=None)
    parser.add_argument('--w2vvectorscache', action='store', type=str, default=None)
    parser.add_argument('--kmeans', action='store_true', default=False)
    parser.add_argument('--lda', action='store_true', default=False)
    parser.add_argument('--unkscore', action='store_true', default=False)
    parser.add_argument('--metamap', action='store_true', default=False)
    parser.add_argument('--zipfeatures', action='store_true', default=False)
    parser.add_argument('--outputaddid', default=None, type=str, help='Output folder for the model and logs')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_mlp','single_cw_mlp','vector_tag','last_tag','rnn', 'vector_tag_rand'], help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')
    parser.add_argument('--regularization', action='store_true', default=False)
    parser.add_argument('--tagdim', action='store', type=int, default=None)
    parser.add_argument('--cviters', action='store', type=int, default=0)

    arguments = parser.parse_args()
    use_original_paper_features = arguments.originalfeatures
    use_custom_features = arguments.customfeatures
    w2v_similar_words = arguments.w2vsimwords
    w2v_vector_features = arguments.w2vvectors
    w2v_model_file = arguments.w2vmodel
    w2v_vectors_cache = arguments.w2vvectorscache
    kmeans = arguments.kmeans
    lda = arguments.lda
    incl_unk_score = arguments.unkscore
    incl_metamap = arguments.metamap
    zip_features = arguments.zipfeatures
    max_cv_iters = arguments.cviters
    outputaddid = arguments.outputaddid
    nn_name = arguments.net
    n_window = arguments.window
    max_epochs = arguments.epochs
    regularization = arguments.regularization
    tagdim = arguments.tagdim

    training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    training_data, training_texts, _, _ = Dataset.get_crf_training_data_by_sentence(training_data_filename)

    crf_model = CRF(training_data=training_data,
                    training_texts=training_texts,
                    test_data=None,
                    output_model_filename=None,
                    w2v_vector_features=w2v_vector_features,
                    w2v_similar_words=w2v_similar_words,
                    kmeans_features=kmeans,
                    lda_features=lda,
                    zip_features=zip_features,
                    original_inc_unk_score=incl_unk_score,
                    original_include_metamap=incl_metamap,
                    w2v_model=w2v_model_file,
                    w2v_vectors_dict=w2v_vectors_cache)

    if use_original_paper_features:
        feature_function = crf_model.get_original_paper_word_features
    elif use_custom_features:
        feature_function = crf_model.get_custom_word_features

    feature_function = crf_model.get_custom_word_features_for_nnet

    nn_class = Single_MLP_neural_network_trainer
    hidden_f = None #no hidden layer in the single MLP.
    out_f = NeuralNetwork.softmax_activation_function
    get_output_path = get_single_mlp_path

    add_tags = []

    if nn_name == 'single_cw_mlp':
        nn_class = Single_CW_MLP_neural_network_trainer
        hidden_f = NeuralNetwork.tanh_activation_function
        add_tags = ['<PAD>']
    #     get_output_path = get_single_mlp_path
    # elif nn_name == 'mlp':
    #     nn_class = MLP_neural_network_trainer
    #     get_output_path = get_cwnn_path
    elif nn_name == 'vector_tag':
        # nn_class = Vector_tag_CW_MLP_neural_network_trainer
        nn_class = Vector_tag_CW_MLP_neural_network_trainer_test
        hidden_f = NeuralNetwork.tanh_activation_function
        if not arguments.tagdim:
            logger.error('Must provide tag dimensionality for Vector tag NNets')
            exit()
        tag_dim = arguments.tagdim
        add_tags = ['<PAD>','<UNK>']
    elif nn_name == 'vector_tag_rand':
        nn_class = Vector_tag_CW_MLP_neural_network_trainer_random
        hidden_f = NeuralNetwork.tanh_activation_function
        if not arguments.tagdim:
            logger.error('Must provide tag dimensionality for Vector tag NNets')
            exit()
        tag_dim = arguments.tagdim
        add_tags = ['<PAD>','<UNK>']
    #     get_output_path = get_vector_tag_path
    # elif nn_name == 'last_tag':
    #     nn_class = Last_tag_neural_network_trainer
    #     get_output_path = get_last_tag_path
    # elif nn_name == 'rnn':
    #     #the RNN init function overwrites the n_window param and sets it to 1.
    #     nn_class = RNN_trainer
    #     get_output_path = get_rnn_path

    logger.info('Extracting features with: '+feature_function.__str__())

    logger.info('Using w2v_similar_words:%s kmeans:%s lda:%s zip:%s' % (w2v_similar_words, kmeans, lda, zip_features))
    logger.info('Using w2v_model: %s and vector_dictionary: %s' % (w2v_model_file, w2v_vectors_cache))

    logger.info('Constructing word and label indexes')
    crf_training_data_filename = 'handoverdata.zip'
    index2word, word2index, index2label, label2index = construct_indexes(crf_training_data_filename, add_tags=add_tags)

    logger.info('Getting CRF training data')
    x = crf_model.get_features_from_crf_training_data(feature_function)
    y = crf_model.get_labels_from_crf_training_data()

    results = dict()

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
        y_test = list(chain(*predictions))

        y_train = np.array(items_to_index(y_train, label2index))
        y_test = np.array(items_to_index(y_test, label2index))

        #TODO: for no-cw-mlp
        w = NeuralNetwork.initialize_weights(n_in=n_features*n_window, n_out=label2index.__len__(), function='softmax')

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

        nn_trainer = nn_class(**params)

        logger.info(' '.join(['Training Neural network','with' if regularization else 'without', 'regularization']))
        nn_trainer.train(
            learning_rate=0.01,
            batch_size=512,
            max_epochs=max_epochs,
            alpha_L1_reg=0.001,
            alpha_L2_reg=0.01,
            save_params=False,
            use_scan=False
        )

        logger.info('Predicting')
        y_true, y_pred = nn_trainer.predict()

        results[i] = (y_true, y_pred)

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
