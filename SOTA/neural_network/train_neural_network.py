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
from utils.metrics import Metrics

from data import get_w2v_training_data_vectors
from data import get_w2v_model
from last_tag_neural_network import Last_tag_neural_network_trainer
from recurrent_net import Recurrent_net
from single_Layer_Context_Window_Net import Single_Layer_Context_Window_Net
from recurrent_Context_Window_net import Recurrent_Context_Window_net
from hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from vector_Tag_Contex_Window_Net import Vector_Tag_Contex_Window_Net
from trained_models import get_vector_tag_path
from trained_models import get_last_tag_path
from trained_models import get_rnn_path
from trained_models import get_single_mlp_path
from trained_models import get_cw_rnn_path
from trained_models import get_cwnn_path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

np.random.seed(1234)

def parse_arguments():
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['single_cw','hidden_cw','vector_tag','last_tag','rnn', 'cw_rnn'], help='NNet type')
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
    parser.add_argument('--w2vvectorscache', action='store', type=str, required=True)
    parser.add_argument('--w2vmodel', action='store', type=str, default=None)
    parser.add_argument('--bidirectional', action='store_true', default=False)
    parser.add_argument('--sharedparams', action='store_true', default=False)
    parser.add_argument('--plot', action='store_true', default=False)

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

    return args

def check_arguments_consistency(args):
    if not args['w2v_vectors_cache'] and not args['w2v_model']:
        logger.error('Provide either a w2vmodel or a w2v vectors cache')
        exit()

    if args['nn_name'] == 'vector_tag' and not args['tagdim']:
        logger.error('Provide the tag dimensionality for Vector tag nnet')
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
        logger.info('Loading W2V vectors from pickle file')
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
    add_tags = []
    tag_dim = None
    n_window = args['window_size']
    nn_class = None

    if args['nn_name'] == 'single_cw':
        nn_class = Single_Layer_Context_Window_Net
        hidden_f = None #no hidden layer in the single MLP.
        get_output_path = get_single_mlp_path
        add_tags = ['<PAD>']
    if args['nn_name'] == 'hidden_cw':
        # one hidden layer with context window. Either minibatch or SGD.
        nn_class = Hidden_Layer_Context_Window_Net
        get_output_path = get_cwnn_path
        add_tags = ['<PAD>']
    elif args['nn_name'] == 'vector_tag':
        nn_class = Vector_Tag_Contex_Window_Net
        get_output_path = get_vector_tag_path
        tag_dim = args['tagdim']
        add_tags = ['<PAD>','<UNK>']
    elif args['nn_name'] == 'last_tag':
        nn_class = Last_tag_neural_network_trainer
        get_output_path = get_last_tag_path
    elif args['nn_name'] == 'rnn':
        #the RNN init function overwrites the n_window param and sets it to 1.
        nn_class = Recurrent_net
        get_output_path = get_rnn_path
        n_window = 1
    elif args['nn_name'] == 'cw_rnn':
        nn_class = Recurrent_Context_Window_net
        get_output_path = get_cw_rnn_path
        add_tags = ['<PAD>']

    return nn_class, hidden_f, out_f, add_tags, tag_dim, n_window, get_output_path

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

def use_testing_dataset(nn_class,
                        hidden_f,
                        out_f,
                        n_window,
                        w2v_model,
                        w2v_vectors,
                        w2v_dims,
                        add_tags,
                        tag_dim,
                        get_output_path,
                        max_epochs=None,
                        minibatch_size=None,
                        regularization=None,
                        **kwargs
                        ):

    logger.info('Using CLEF testing data')

    results = dict()

    logger.info('Loading CRF training data')

    x_train, y_train, x_test, y_test, word2index, index2word, label2index, index2label = nn_class.get_data(crf_training_data_filename, test_data_filename,
                                                                add_tags, x_idx=None, n_window=n_window)
    n_out = len(label2index.keys())
    unique_words = word2index.keys()

    pretrained_embeddings = nn_class.initialize_w(w2v_dims, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    logger.info('Instantiating Neural network')

    pad_tag, unk_tag, pad_word = determine_key_indexes(label2index, word2index)

    params = {
        'x_train': x_train,
        'y_train': y_train,
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
        'get_output_path': get_output_path
    }

    nn_trainer = nn_class(**params)

    logger.info(' '.join(['Training Neural network','with' if regularization else 'without', 'regularization']))
    nn_trainer.train(learning_rate=.01, batch_size=minibatch_size, max_epochs=max_epochs, save_params=False, **kwargs)

    logger.info('Predicting')
    flat_true, flat_predictions = nn_trainer.predict(**kwargs)

    results[0] = (flat_true, flat_predictions)

    return results, index2label

if __name__ == '__main__':
    start = time.time()

    crf_training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'

    args = parse_arguments()

    check_arguments_consistency(args)

    w2v_vectors, w2v_model, w2v_dims = load_w2v_model_and_vectors_cache(args)

    nn_class, hidden_f, out_f, add_tags, tag_dim, n_window, get_output_path = determine_nnclass_and_parameters(args)

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
                                                   add_tags,
                                                   tag_dim,
                                                   get_output_path,
                                                   **args)

    cPickle.dump(results, open(get_output_path('prediction_results.p'),'wb'))
    cPickle.dump(index2label, open(get_output_path('index2labels.p'),'wb'))

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
