__author__ = 'root'
from abc import ABCMeta, abstractmethod
from collections import defaultdict
import numpy as np
from itertools import chain
from collections import OrderedDict
import copy

from data.dataset import Dataset
from utils import utils
from utils import get_word_tenses
from utils import features_distributions

class A_neural_network():

    __metaclass__ = ABCMeta

    CRF_POSITIONS = {'ner': 1, 'pos': 2}

    FEATURE_MAPPING = {
        'w2v_c_nm': {'name': 'w2v', 'convolve': True, 'max_pool': False, 'use_cw': True, 'nr_filters': None,
                     'filter_width': 1, 'nr_region_sizes': None, 'crf_position': None},
        'w2v_nc_nm': {'name': 'w2v', 'convolve': False, 'max_pool': False, 'use_cw': True, 'nr_filters': 0,
                      'filter_width': 0, 'nr_region_sizes': 0, 'crf_position': None},
        'w2v_c_m': {'name': 'w2v', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None,
                    'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'pos_c_m': {'name': 'pos', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None,
                    'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['pos']},
        'ner_c_m': {'name': 'ner', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None,
                    'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['ner']},
        'sent_nr_c_m': {'name': 'sent_nr', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None,
                        'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'tense_c_m': {'name': 'tense', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None,
                      'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'pos_nc_nm': {'name': 'pos', 'convolve': False, 'max_pool': False, 'nr_filters': None,
                      'filter_width': None, 'nr_region_sizes': None, 'use_cw': True, 'crf_position': CRF_POSITIONS['pos']},
        'ner_nc_nm': {'name': 'ner', 'convolve': False, 'max_pool': False, 'nr_filters': None,
                      'filter_width': None, 'nr_region_sizes': None, 'use_cw': True, 'crf_position': CRF_POSITIONS['ner']},
        'sent_nr_nc_nm': {'name': 'sent_nr', 'convolve': False, 'max_pool': False, 'nr_filters': None,
                          'filter_width': None, 'nr_region_sizes': None, 'use_cw': True, 'crf_position': None},
        'tense_nc_nm': {'name': 'tense', 'convolve': False, 'max_pool': False, 'nr_filters': None,
                        'filter_width': None, 'nr_region_sizes': None, 'use_cw': True, 'crf_position': None}
    }

    def __init__(self,
                 x_train,
                 y_train,
                 x_valid,
                 y_valid,
                 x_test,
                 y_test,
                 n_out,
                 n_window,
                 pretrained_embeddings,
                 get_output_path,
                 **kwargs):
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
        # self.words_per_document = words_per_document
        # self.tags_per_document = tags_per_document

        #data structure of words and tags (in sentences), used specially by RNN
        # self.train_document_sentences_words = train_document_sentences_words
        # self.train_document_sentences_tags = train_document_sentences_tags
        # self.test_document_sentences_words = test_document_sentences_words
        # self.test_document_sentences_tags = test_document_sentences_tags

        #unique words and tags lists
        # self.unique_words = unique_words
        # self.unique_labels = unique_labels
        # self.n_unique_words = n_unique_words

        #mapping dictionaries for words/tags to indexes
        # self.index2word = index2word
        # self.word2index = word2index
        # self.index2label = index2label
        # self.label2index = label2index

        #last layer size
        self.n_out = n_out

        #training datasets
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)
        self.n_samples = self.x_train.shape[0]

        #validation datasets
        self.x_valid = np.array(x_valid)
        self.y_valid = np.array(y_valid)

        #testing datasets
        self.x_test = np.array(x_test)
        self.y_test = np.array(y_test)

        #pretrained w1 matrix (word embeddings)
        if pretrained_embeddings is not None:
            self.pretrained_embeddings = pretrained_embeddings
            self.n_emb = self.pretrained_embeddings.shape[1]

        #output path get function
        self.get_output_path = get_output_path

    @abstractmethod
    def to_string(self):
        pass

    @classmethod
    def get_tenses_features(cls, clef_training=True, clef_validation=False, clef_testing=False):

        if not clef_training and not clef_validation and not clef_testing:
            raise Exception('At least one dataset must be loaded')

        train_features = None
        valid_features = None
        test_features = None

        if clef_training:
            train_features = list(chain(*get_word_tenses.get_training_set_tenses().values()))

        if clef_validation:
            valid_features = list(chain(*get_word_tenses.get_validation_set_tenses().values()))

        if clef_testing:
            test_features = list(chain(*get_word_tenses.get_testing_set_tenses().values()))

        tenses = set(train_features + valid_features + test_features)
        mapping = dict(zip(tenses, range(tenses.__len__())))
        train_features_mapped = map(lambda x: mapping[x], train_features)
        valid_features_mapped = map(lambda x: mapping[x], valid_features)
        test_features_mapped = map(lambda x: mapping[x], test_features)

        probabilities = dict()
        total = train_features.__len__()
        for t in tenses:
            probabilities[t] = [f for f in train_features if f == t].__len__() / float(total)

        return train_features_mapped, valid_features_mapped, test_features_mapped, probabilities

    @classmethod
    def get_word_sentence_number_features(cls, clef_training=True, clef_validation=False, clef_testing=False):

        if not clef_training and not clef_validation and not clef_testing:
            raise Exception('At least one dataset must be loaded')

        train_features = None
        valid_features = None
        test_features = None

        if clef_training:
            _, _, train_document_sentence_words, _ = Dataset.get_clef_training_dataset()
            train_features = cls.get_sentences_numbers(train_document_sentence_words)

        if clef_validation:
            _, _, valid_document_sentence_words, _ = Dataset.get_clef_validation_dataset()
            valid_features = cls.get_sentences_numbers(valid_document_sentence_words)

        if clef_testing:
            _, _, test_document_sentence_words, _ = Dataset.get_clef_testing_dataset()
            test_features = cls.get_sentences_numbers(test_document_sentence_words)

        # doc_sent_items = [[train_features+valid_features+test_features]]
        # feat2index, index2feat = cls._construct_index(add_items=add_feats,
        #                                               document_sentence_item=doc_sent_items)

        return train_features, valid_features, test_features

    @classmethod
    def get_sentences_numbers(cls, document_sentence_words):
        """
        returns the sentence number for each word.
        meant to be used as a feature in the multifeature cnnet.
        :param document_sentence_words:
        :return:
        """
        train_features = []

        for doc_sent in document_sentence_words.values():
            for i, sent in enumerate(doc_sent):
                train_features.extend([i] * sent.__len__())

        return train_features

    @classmethod
    def get_data(cls, clef_training=True, clef_validation=False, clef_testing=False,
                 add_words=[], add_tags=[], add_feats=[], x_idx=None, n_window=None, feat_positions=None):
        """
        this is at sentence level.
        gets the training data and organizes it into sentences per document.

        :param crf_training_data_filename:
        :return:
        """

        if not clef_training and not clef_validation and not clef_testing:
            raise Exception('At least one dataset must be loaded')

        # test_document_sentence_tags, test_document_sentence_words, train_document_sentence_tags, train_document_sentence_words = cls.get_datasets(
        #     crf_training_data_filename, testing_data_filename)

        document_sentence_words = []
        document_sentence_tags = []

        x_train = None
        y_train = None
        y_valid = None
        x_valid = None
        y_test = None
        x_test = None

        x_train_features = defaultdict(list)
        x_valid_features = defaultdict(list)
        x_test_features = defaultdict(list)
        x_train_feats = None
        x_valid_feats = None
        x_test_feats = None

        if clef_training:
            train_features, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_clef_training_dataset()

            if feat_positions:
                x_train_features = cls.get_features_by_position(feat_positions, train_features)

            document_sentence_words.extend(train_document_sentence_words.values())
            document_sentence_tags.extend(train_document_sentence_tags.values())

        if clef_validation:
            valid_features, _, valid_document_sentence_words, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

            if feat_positions:
                x_valid_features = cls.get_features_by_position(feat_positions, valid_features)

            document_sentence_words.extend(valid_document_sentence_words.values())
            document_sentence_tags.extend(valid_document_sentence_tags.values())

        if clef_testing:
            test_features, _, test_document_sentence_words, test_document_sentence_tags = Dataset.get_clef_testing_dataset()

            if feat_positions:
                x_test_features = cls.get_features_by_position(feat_positions, test_features)

            document_sentence_words.extend(test_document_sentence_words.values())
            # document_sentence_tags.extend(test_document_sentence_tags.values())   # its all Nones.

        # label2index_del, index2label_del, word2index_del, index2word_del = cls._construct_indexes(add_words, add_tags,
        #                                                                         document_sentence_words,
        #                                                                         document_sentence_tags)

        word2index, index2word = cls._construct_index(add_words, document_sentence_words)
        label2index, index2label = cls._construct_index(add_tags, document_sentence_tags)

        features_indexes = None
        if feat_positions:
            features_indexes = dict()
            for feat_pos in feat_positions:
                features = copy.copy(x_train_features[feat_pos])
                features.extend(x_valid_features[feat_pos])
                features.extend(x_test_features[feat_pos])
                feat2index, index2feat = cls._construct_index(add_items=add_feats, document_sentence_item=features)

                all_train_items = list(chain(*chain(*x_train_features[feat_pos])))
                unique_items = set(chain(*chain(*features))).union(add_feats)
                total = all_train_items.__len__()
                probs = dict()
                for item in unique_items:
                    probs[feat2index[item]] = [f for f in all_train_items if f==item].__len__() / float(total)

                features_indexes[feat_pos] = (feat2index, index2feat, probs)

        if clef_training:
            x_train, y_train = cls.get_partitioned_data(x_idx=x_idx,
                                                        document_sentences_words=train_document_sentence_words,
                                                        document_sentences_tags=train_document_sentence_tags,
                                                        word2index=word2index,
                                                        label2index=label2index,
                                                        use_context_window=True,
                                                        n_window=n_window)
            if x_train_features:
                x_train_feats = cls.transform_features_with_context_window(x_train_features,
                                                                           features_indexes,
                                                                           n_window)

        if clef_validation:
            x_valid, y_valid = cls.get_partitioned_data(x_idx=x_idx,
                                                        document_sentences_words=valid_document_sentence_words,
                                                        document_sentences_tags=valid_document_sentence_tags,
                                                        word2index=word2index,
                                                        label2index=label2index,
                                                        use_context_window=True,
                                                        n_window=n_window)
            if x_valid_features:
                x_valid_feats = cls.transform_features_with_context_window(x_valid_features,
                                                                           features_indexes,
                                                                           n_window)

        if clef_testing:
            x_test, y_test = cls.get_partitioned_data(x_idx=x_idx,
                                                      document_sentences_words=test_document_sentence_words,
                                                      document_sentences_tags=test_document_sentence_tags,
                                                      word2index=word2index,
                                                      label2index=label2index,
                                                      use_context_window=True,
                                                      n_window=n_window)
            if x_test_features:
                x_test_feats = cls.transform_features_with_context_window(x_test_features,
                                                                           features_indexes,
                                                                           n_window)

        # n_docs = len(train_document_sentence_words)

        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats,\
               x_test, y_test, x_test_feats,\
               word2index, index2word, \
               label2index, index2label, \
               features_indexes

    @classmethod
    def transform_features_with_context_window(cls, dataset_features, features_indexes, n_window):

        x_feats = dict()
        for feat_pos, features in dataset_features.iteritems():
            feat2index, index2feat, _ = features_indexes[feat_pos]
            feats = []
            for doc_features in features:
                feats.extend(cls._get_partitioned_data_with_context_window(doc_features, n_window, feat2index))
            x_feats[feat_pos] = feats

        return x_feats

    @classmethod
    def get_features_by_position(cls, feat_positions, dataset_features):
        x_dataset_features = OrderedDict()
        for feat_pos in feat_positions:
            doc_features = []
            for doc_sentences in dataset_features.values():
                sent_features = []
                for sent in doc_sentences:
                    sent_features.append(map(lambda x: x['features'][feat_pos], sent))
                doc_features.append(sent_features)

            x_dataset_features[feat_pos] = doc_features
        return x_dataset_features

    @classmethod
    def get_datasets(cls, crf_training_data_filename, testing_data_filename):

        # document_sentences_words, document_sentences_tags = Dataset.get_training_file_tokenized_sentences(crf_training_data_filename)
        _, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_crf_training_data_by_sentence(
            crf_training_data_filename)

        test_document_sentence_words = None
        test_document_sentence_tags = None

        if testing_data_filename:
            _, _, test_document_sentence_words, test_document_sentence_tags = \
                Dataset.get_crf_training_data_by_sentence(file_name=testing_data_filename,
                                                          path=Dataset.TESTING_FEATURES_PATH + 'test',
                                                          extension=Dataset.TESTING_FEATURES_EXTENSION)

        return test_document_sentence_tags, test_document_sentence_words, train_document_sentence_tags, train_document_sentence_words

    @classmethod
    def _construct_index(cls, add_items, document_sentence_item):

        index2item = OrderedDict()
        item2index = OrderedDict()

        unique_items = list(set([item for doc_sentences in document_sentence_item for sentence in doc_sentences
                        for item in sentence]))

        for i, item in enumerate(unique_items):
            item2index[item] = i
            index2item[i] = item

        if add_items:
            for i, item in enumerate(add_items):
                item2index[item] = len(unique_items) + i
                index2item[len(unique_items)+i] = item

        assert item2index.keys() == index2item.values()

        return item2index, index2item

    @classmethod
    def get_partitioned_data(cls, x_idx, document_sentences_words, document_sentences_tags,
                             word2index, label2index, use_context_window=False, n_window=None):
        """
        this is at sentence level.
        it partitions the training data according to the x_idx doc_nrs used for training and y_idx doc_nrs used for
        testing while cross-validating.

        :param x_idx:
        :param y_idx:
        :return:
        """
        x = []
        y = []

        for doc_nr, doc_sentences in document_sentences_words.iteritems():
            if (not x_idx) or (doc_nr in x_idx):
                # training set
                if use_context_window:
                    x_doc = cls._get_partitioned_data_with_context_window(doc_sentences, n_window, word2index)
                    y_doc = [tag for tag in chain(*[map(lambda x: label2index[x] if x else None,sent)
                                               for sent in document_sentences_tags[doc_nr]])]
                else:
                    x_doc = cls._get_partitioned_data_without_context_window(doc_sentences, word2index)
                    y_doc = cls._get_partitioned_data_without_context_window(document_sentences_tags[doc_nr], label2index)

                x.extend(x_doc)
                y.extend(y_doc)

        return x, y

    @classmethod
    def _get_partitioned_data_with_context_window(cls, doc_sentences_item, n_window, item2index):

        return [map(lambda x: item2index[x], sent_window) for sentence in doc_sentences_item
             for sent_window in utils.NeuralNetwork.context_window(sentence, n_window)]

    @classmethod
    def _get_partitioned_data_without_context_window(cls, doc_sentences_item, dictionary_mapping):

        return [map(lambda x: dictionary_mapping[x] if x else None, sentence) for sentence in doc_sentences_item]
        # y.extend([map(lambda x: label2index[x], sentence) for sentence in doc_sentences_tags])

    @classmethod
    def get_data_by_document(cls, crf_training_data_filename):
        """
        this is at document level (no sentences involved).
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

    def get_partitioned_data_by_document(self, x_idx, y_idx):
        """
        this is by document (no sentence level involved).
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

        # it PADs per document.
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

    @classmethod
    def initialize_w(cls, w2v_dims, unique_words, w2v_vectors=None, w2v_model=None):
        """
        this method is used by all neural net structures. It initializes the W1 matrix with the pretrained embeddings
        from word2vec.

        :param w2v_dims:
        :param w2v_vectors:
        :param w2v_model:
        :return:
        """
        n_unique_words = len(unique_words)

        w = utils.NeuralNetwork.initialize_weights(n_unique_words, w2v_dims, function='tanh')

        pretrained_embeddings = utils.NeuralNetwork.replace_with_word_embeddings(w, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

        return pretrained_embeddings

    @classmethod
    def initialize_w_pos(cls, word2index):
        """
        this method is used by all neural net structures. It initializes a matrix with the POS probabilistic
        representation.
        """
        unique_words = word2index.keys()
        n_unique_words = len(unique_words)
        pos_word_representations = features_distributions.training_word_pos_representations()
        pos_dims = pos_word_representations.values()[0].__len__()
        w = utils.NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=pos_dims, function='tanh')

        for word, rep in pos_word_representations.iteritems():
            w[word2index[word]] = rep

        return w

    @classmethod
    def initialize_w_ner(cls, word2index):
        """
        this method is used by all neural net structures. It initializes a matrix with the NER probabilistic
        representation.
        """
        unique_words = word2index.keys()
        n_unique_words = len(unique_words)
        ner_word_representations = features_distributions.training_word_ner_representations()
        ner_dims = ner_word_representations.values()[0].__len__()
        w = utils.NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=ner_dims, function='tanh')

        for word, rep in ner_word_representations.iteritems():
            w[word2index[word]] = rep

        return w

    @classmethod
    def initialize_w_sent_nr(cls, word2index):
        """
        this method is used by all neural net structures. It initializes a matrix with the NER probabilistic
        representation.
        """
        unique_words = word2index.keys()
        n_unique_words = len(unique_words)
        sent_nr_word_representations = features_distributions.training_word_sent_nr_representations()
        sent_nr_dims = sent_nr_word_representations.values()[0].__len__()
        w = utils.NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=sent_nr_dims, function='tanh')

        for word, rep in sent_nr_word_representations.iteritems():
            w[word2index[word]] = rep

        return w

    @classmethod
    def initialize_w_tense(cls, word2index):
        """
        this method is used by all neural net structures. It initializes a matrix with the NER probabilistic
        representation.
        """
        unique_words = word2index.keys()
        n_unique_words = len(unique_words)
        tense_word_representations = features_distributions.training_word_tense_representations()
        tense_dims = tense_word_representations.values()[0].__len__()
        w = utils.NeuralNetwork.initialize_weights(n_in=n_unique_words, n_out=tense_dims, function='tanh')

        for word, rep in tense_word_representations.iteritems():
            w[word2index[word]] = rep

        return w
    
    def plot_training_cost_and_error(self, train_costs_list, train_errors_list, test_costs_list, test_errors_list, 
                                     actual_time):

        assert train_costs_list.__len__() == train_errors_list.__len__()
        assert train_costs_list.__len__() == test_costs_list.__len__()
        assert train_costs_list.__len__() == test_errors_list.__len__()

        data = {
            'epoch': np.arange(train_costs_list.__len__(), dtype='int'),
            'Train_cost': train_costs_list,
            'Train_error': train_errors_list,
            'Valid_cost': test_costs_list,
            'Valid_error': test_errors_list
        }
        output_filename = self.get_output_path('training_cost_plot_' + actual_time)
        utils.NeuralNetwork.plot(data, x_axis='epoch', x_label='Epochs', y_label='Value',
                                 title='Training and Testing cost and error evolution',
                                 output_filename=output_filename)
        
        return True
    
    def plot_scores(self, precision_list, recall_list, f1_score_list, actual_time):

        assert precision_list.__len__() == recall_list.__len__()
        assert precision_list.__len__() == f1_score_list.__len__()
        
        data = {
            'epoch': np.arange(precision_list.__len__(), dtype='int'),
            'Precision': precision_list,
            'Recall': recall_list,
            'F1_score': f1_score_list
        }
        output_filename = self.get_output_path('training_scores_plot' + actual_time)
        utils.NeuralNetwork.plot(data, x_axis='epoch', x_label='Epochs', y_label='Score ',
                                 title='Training scores evolution',
                                 output_filename=output_filename)

        return True

    def plot_penalties(self, l2_w1_list, l2_w2_list=None, l2_ww_fw_list=None, l2_ww_bw_list=None,
                       l2_wt_list=None, l2_w0_list=None, actual_time=None):

        data = {
            'epoch': np.arange(l2_w1_list.__len__(), dtype='int'),
            'L2_W1_sum': l2_w1_list,
        }

        if l2_w0_list:
            assert l2_w1_list.__len__() == l2_w0_list.__len__()
            data['l2_W0_sum'] = l2_w0_list

        if l2_w2_list:
            assert l2_w1_list.__len__() == l2_w2_list.__len__()
            data['L2_W2_sum'] = l2_w2_list

        if l2_ww_fw_list:
            assert l2_w1_list.__len__() == l2_ww_fw_list.__len__()
            data['L2_WW_Fw_sum'] = l2_ww_fw_list

        if l2_ww_bw_list:
            assert l2_w1_list.__len__() == l2_ww_bw_list.__len__()
            data['L2_WW_Bw_sum'] = l2_ww_bw_list

        if l2_wt_list:
            assert l2_w1_list.__len__() == l2_wt_list.__len__()
            data['L2_Wt_sum'] = l2_wt_list

        output_filename = self.get_output_path('training_L2_penalty_plot' + actual_time)
        utils.NeuralNetwork.plot(data, x_axis='epoch', x_label='Epochs', y_label='Penalty',
                                 title='Training weight penalties evolution',
                                 output_filename=output_filename)

        return True

    def plot_penalties_general(self, data_dict, actual_time=None):

        assert data_dict is not None

        data_dict['epoch'] = np.arange(data_dict.values()[0].__len__(), dtype='int')

        for values in data_dict.values():
            assert values.__len__() == data_dict['epoch'].__len__()

        output_filename = self.get_output_path('training_L2_penalty_plot' + actual_time)
        utils.NeuralNetwork.plot(data_dict, x_axis='epoch', x_label='Epochs', y_label='Penalty',
                                 title='Training weight penalties evolution',
                                 output_filename=output_filename)

        return True

    def plot_cross_entropies(self, train_cross_entropy, valid_cross_entropy, actual_time,
                             title='Cross-entropy evolution', output_name='cross_entropy_plot'):

        assert train_cross_entropy.__len__() == valid_cross_entropy.__len__()

        data = {
            'epoch': np.arange(train_cross_entropy.__len__(), dtype='int'),
            'train': train_cross_entropy,
            'valid': valid_cross_entropy
        }

        output_filename = self.get_output_path(output_name + actual_time)

        utils.NeuralNetwork.plot(data, x_axis='epoch', x_label='Epochs', y_label='Cross-entropy',
                                 title=title,
                                 output_filename=output_filename)

        return True


    @classmethod
    def get_features_crf_position(cls, features):
        return []
