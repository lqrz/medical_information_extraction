__author__ = 'root'
from sklearn.cross_validation import LeaveOneOut
from data.dataset import Dataset
from data import get_w2v_training_data_vectors
from data import get_w2v_model
from utils import utils
import cPickle
import logging
import os
from collections import defaultdict
from SOTA.neural_network.mlp_neural_network import MLP_neural_network_trainer
from SOTA.neural_network.last_tag_neural_network import Last_tag_neural_network_trainer
from SOTA.neural_network.vector_tag_neural_network import Vector_tag_neural_network_trainer
from SOTA.neural_network.recurrent_neural_network import RNN_trainer
import numpy as np
from sklearn import metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)


if __name__=='__main__':
    crf_training_data_filename = 'handoverdata.zip'

    training_vectors_filename = get_w2v_training_data_vectors()

    n_window = 7 #TODO: make param.

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

    nn_name = 'rnn' #TODO: make param
    nn_name = 'last_tag' #TODO: make param

    if nn_name=='mlp':
        nn_class = MLP_neural_network_trainer
    elif nn_name=='vector_tag':
        nn_class = Vector_tag_neural_network_trainer
    elif nn_name=='last_tag':
        nn_class = Last_tag_neural_network_trainer
    elif nn_name=='rnn':
        nn_class = RNN_trainer
        #the RNN init function overwrites the n_window param and sets it to 1.

    logger.info('Loading CRF training data')
    words_per_document, tags_per_document, document_sentences_words, document_sentences_tags, n_docs, unique_words, unique_labels, index2word, word2index, index2label, label2index, n_unique_words, n_out = nn_class.get_data(crf_training_data_filename)

    #TODO: do this.
    logger.info('Using Neural class: '+nn_name)

    loo = LeaveOneOut(n_docs)
    for cross_idx, (x_idx, y_idx) in enumerate(loo):

        if cross_idx > 0:
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
        nn_trainer.train(learning_rate=.01, batch_size=512, max_epochs=2, save_params=False)

        logger.info('Predicting')
        flat_true, flat_predictions = nn_trainer.predict()

        print 'Accuracy: ', utils.metrics.accuracy_score(flat_true, flat_predictions)
        print 'F1-score: ', utils.metrics.f1_score(flat_true, flat_predictions)
        # print utils.metrics.classification_report(map(lambda x: index2label[x],y_valid), map(lambda x: index2label[x], predictions[0]), unique_labels)

    logger.info('End')