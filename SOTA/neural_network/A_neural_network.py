__author__ = 'root'
from abc import ABCMeta, abstractmethod
from data.dataset import Dataset
from collections import defaultdict
import numpy as np
from utils import utils

class A_neural_network():

    __metaclass__ = ABCMeta

    def __init__(self, n_window, words_per_document, tags_per_document, document_sentences_words, document_sentences_tags,
                 unique_words, unique_labels, index2word, word2index, index2label, label2index,
                 n_unique_words, n_out):
        """
        common constructor for all neural nets.

        :param n_window:
        :param words_per_document:
        :param tags_per_document:
        :param document_sentences_words:
        :param document_sentences_tags:
        :param unique_words:
        :param unique_labels:
        :param index2word:
        :param word2index:
        :param index2label:
        :param label2index:
        :param n_unique_words:
        :param n_out:
        :return:
        """

        #window size (in case of RNN, its init() function hardcodes it to 1.
        self.n_window = n_window

        #data structure of words and tags per document, used by all except RNN
        self.words_per_document = words_per_document
        self.tags_per_document = tags_per_document

        #data structure of words and tags (in sentences), used specially by RNN
        self.document_sentences_words = document_sentences_words
        self.document_sentences_tags = document_sentences_tags

        #unique words and tags lists
        self.unique_words = unique_words
        self.unique_labels = unique_labels
        self.n_unique_words = n_unique_words

        #mapping dictionaries for words/tags to indexes
        self.index2word = index2word
        self.word2index = word2index
        self.index2label = index2label
        self.label2index = label2index

        #last layer size
        self.n_out = n_out

        #training datasets
        self.x_train = None
        self.y_train = None

        #validation datasets
        self.x_valid = None
        self.y_valid = None

        #pretrained w1 matrix (word embeddings)
        self.pretrained_embeddings = None

    @abstractmethod
    def to_string(self):
        pass

    @classmethod
    def get_data(cls, crf_training_data_filename):
        """
        this method retrieves the training data, and structures it per document.
        it is overridden by RNN, cause they use a special structure with a further division of documents into sentences.
        this method is independent of the instance.

        :param crf_training_data_filename:
        :return:
        """

        crf_training_dataset,_ = Dataset.get_crf_training_data(crf_training_data_filename)
        document_sentences_words = None
        document_sentences_tags = None
        words_per_document, tags_per_document, all_words, all_tags = A_neural_network._get_data_per_document(crf_training_dataset)

        unique_words = list(set(all_words))
        unique_labels = list(set(all_tags))

        # logger.info('Creating word-index dictionaries')
        index2word = defaultdict(None)
        word2index = defaultdict(None)
        word2index['<PAD>'] = len(unique_words)
        for i,word in enumerate(unique_words):
            index2word[i] = word
            word2index[word] = i

        # logger.info('Creating label-index dictionaries')
        index2label = defaultdict(None)
        label2index = defaultdict(None)
        label2index['<PAD>'] = len(unique_labels)
        for i,label in enumerate(unique_labels):
            index2label[i] = label
            label2index[label] = i

        n_unique_words = len(unique_words)+1    # +1 for the '<PAD>' token
        n_out = len(unique_labels)

        n_docs = len(words_per_document)

        return words_per_document, tags_per_document, document_sentences_words, document_sentences_tags, n_docs, unique_words, unique_labels, index2word, word2index, index2label, label2index, n_unique_words, n_out

    def get_partitioned_data(self, x_idx, y_idx):
        """
        this method partitions the training data into training and validation, according to the x_idxs and y_idxs
        lists of indices. x_idxs corresponds to doc_nrs used for training, and y_idxs to doc_nrs used for testing.

        :param x_idx:
        :param y_idx:
        :return:
        """
        training_sentence_words = []
        validation_sentence_words = []
        training_sentence_tags = []
        validation_sentence_tags = []
        for doc_nr,words in self.words_per_document.iteritems():
            if doc_nr in x_idx:
                training_sentence_words.extend(words)
                training_sentence_tags.extend(self.tags_per_document[doc_nr])
            elif doc_nr in y_idx:
                validation_sentence_words.extend(words)
                validation_sentence_tags.extend(self.tags_per_document[doc_nr])

        #TODO: should i pad every end of sentence? TRY IT!
        self.x_train = np.matrix([map(lambda x: self.word2index[x], sentence) for sentence in
                             utils.NeuralNetwork.context_window(training_sentence_words,self.n_window)])

        self.y_train = np.array([self.label2index[tag] for tag in training_sentence_tags])

        self.x_valid = np.matrix([map(lambda x: self.word2index[x], sentence) for sentence in
                             utils.NeuralNetwork.context_window(validation_sentence_words,self.n_window)])

        self.y_valid = np.array([self.label2index[tag] for tag in validation_sentence_tags])

        return True

    @staticmethod
    def _get_data_per_document(crf_training_data):
        """
        this method structures the training data per document number. This is used later when filtering doc_nrs in
        the cross-validation.
        it is not used by the RNN, cause they use a different organization of the data (they override the get_data()
        method).

        :param crf_training_data:
        :return:
        """

        words_per_document = dict()
        tags_per_document = dict()
        all_words = []
        all_tags = []
        for doc_nr,archive in crf_training_data.iteritems():
            words_per_document[doc_nr] = [token['word'] for token in archive.values()]
            tags_per_document[doc_nr] = [token['tag'] for token in archive.values()]
            all_words.extend(words_per_document[doc_nr])
            all_tags.extend(tags_per_document[doc_nr])

        return words_per_document, tags_per_document, all_words, all_tags

    def initialize_w(self, w2v_dims, w2v_vectors=None, w2v_model=None):
        """
        this method is used by all neural net structures. It initializes the W1 matrix with the pretrained embeddings
        from word2vec.

        :param w2v_dims:
        :param w2v_vectors:
        :param w2v_model:
        :return:
        """

        w = utils.NeuralNetwork.initialize_weights(self.n_unique_words, w2v_dims, function='tanh')

        self.pretrained_embeddings = utils.NeuralNetwork.replace_with_word_embeddings(w, self.unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

        return True