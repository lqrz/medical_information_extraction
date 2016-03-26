__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from sklearn.cross_validation import LeaveOneOut
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from utils import utils
import cPickle
import logging
from mlp_neural_network import MLP_neural_network_trainer
from last_tag_neural_network import Last_tag_neural_network_trainer
from vector_tag_neural_network import Vector_tag_neural_network_trainer
from recurrent_neural_network import RNN_trainer
import argparse
import numpy as np
from trained_models import get_vector_tag_path
from trained_models import get_cwnn_path
from trained_models import get_last_tag_path
from trained_models import get_rnn_path

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Neural net trainer')
    parser.add_argument('--train_filename', default='handoverdata.zip', type=str)
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['mlp','vector_tag','last_tag','rnn'], help='NNet type')
    parser.add_argument('--window', type=int, action='store', required=True,
                        help='Context window size. 1 for RNN')
    parser.add_argument('--epochs', type=int, action='store', required=True,
                        help='Nr of training epochs.')
    parser.add_argument('--cviters', type=int, action='store', required=True,
                        help='Nr of cross-validation iterations.')

    #parse arguments
    arguments = parser.parse_args()
    crf_training_data_filename = arguments.train_filename
    n_window = arguments.window
    nn_name = arguments.net
    max_epochs = arguments.epochs
    max_cross_validation = arguments.cviters

    training_vectors_filename = get_w2v_training_data_vectors()

    w2v_vectors = None
    w2v_model = None
    w2v_dims = None

    if os.path.exists(training_vectors_filename):
        logger.info('Loading W2V vectors from pickle file')
        w2v_vectors = cPickle.load(open(training_vectors_filename,'rb'))
        w2v_dims = len(w2v_vectors.values()[0])
    else:
        logger.info('Loading W2V model')
        W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
        w2v_model = utils.Word2Vec.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
        w2v_dims = w2v_model.syn0.shape[0]

    if nn_name == 'mlp':
        nn_class = MLP_neural_network_trainer
        get_output_path = get_cwnn_path
    elif nn_name == 'vector_tag':
        nn_class = Vector_tag_neural_network_trainer
        get_output_path = get_vector_tag_path
    elif nn_name == 'last_tag':
        nn_class = Last_tag_neural_network_trainer
        get_output_path = get_last_tag_path
    elif nn_name == 'rnn':
        #the RNN init function overwrites the n_window param and sets it to 1.
        nn_class = RNN_trainer
        get_output_path = get_rnn_path

    logger.info('Loading CRF training data')
    words_per_document, tags_per_document, document_sentences_words, document_sentences_tags, n_docs, unique_words, unique_labels, index2word, word2index, index2label, label2index, n_unique_words, n_out = nn_class.get_data(crf_training_data_filename)

    logger.info('Using Neural class: %s with window size: %d for epochs: %d' % (nn_name,n_window,max_epochs))

    #store crossvalidation results
    accuracy_results = []
    f1_score_results = []

    prediction_results = dict()

    loo = LeaveOneOut(n_docs)
    for cross_idx, (x_idx, y_idx) in enumerate(loo):

        if (cross_idx+1) > max_cross_validation:
            break

        logger.info('Cross-validation %d' % (cross_idx))

        logger.info('Instantiating Neural network')

        params = {
            'hidden_activation_f': utils.NeuralNetwork.tanh_activation_function,
            'out_activation_f': utils.NeuralNetwork.softmax_activation_function,
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

        # training_sentence_words = [word['word'] for doc_nr,archive in crf_training_dataset.iteritems()
        #                            for word in archive.values() if doc_nr in x_idx]
        # training_sentence_tags = [word['tag'] for doc_nr,archive in crf_training_dataset.iteritems()
        #                           for word in archive.values() if doc_nr in x_idx]

        # validation_sentence_words = [word['word'] for doc_nr,archive in crf_training_dataset.iteritems()
        #                              for word in archive.values() if doc_nr in y_idx]
        # validation_sentence_tags = [word['tag'] for doc_nr,archive in crf_training_dataset.iteritems()
        #                             for word in archive.values() if doc_nr in x_idx]

        # nn_trainer = Neural_network_trainer(x_train, y_train,
        #                                     n_out=n_labels,
        # hidden_activation_f=utils.NeuralNetwork.tanh_activation_function,
        # out_activation_f=utils.NeuralNetwork.softmax_activation_function,
        # pretrained_embeddings=w)

        logger.info('Training Neural network')
        nn_trainer.train(learning_rate=.01, batch_size=512, max_epochs=max_epochs, save_params=False)

        logger.info('Predicting')
        flat_true, flat_predictions = nn_trainer.predict()

        accuracy = utils.metrics.accuracy_score(flat_true, flat_predictions)
        accuracy_results.append(accuracy)

        f1_score = utils.metrics.f1_score(flat_true, flat_predictions)
        f1_score_results.append(f1_score)

        prediction_results[cross_idx] = (flat_true, flat_predictions)

        print 'Accuracy: ', accuracy
        print 'F1-score: ', f1_score
        # print utils.metrics.classification_report(map(lambda x: index2label[x],y_valid), map(lambda x: index2label[x], predictions[0]), unique_labels)

    print 'Mean accuracy: ', np.mean(accuracy_results)
    print 'Mean F1-score: ', np.mean(f1_score_results)

    logger.info('Pickling results')
    cPickle.dump(prediction_results, open(get_output_path('prediction_results.p'), 'wb'))
    cPickle.dump(accuracy_results, open(get_output_path('accuracy_results.p'), 'wb'))
    cPickle.dump(f1_score_results, open(get_output_path('f1_score_results.p'), 'wb'))

    logger.info('End')
