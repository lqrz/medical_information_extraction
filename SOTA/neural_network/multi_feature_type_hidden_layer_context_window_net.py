__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import theano.tensor.signal.conv as conv
from theano.tensor.signal.pool import pool_2d
import cPickle
from collections import OrderedDict
from utils import utils
import time
from collections import defaultdict

from A_neural_network import A_neural_network
from trained_models import get_cwnn_path
from utils.metrics import Metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
theano.config.warn_float64='raise'
# theano.config.floatX='float64'

INT = 'int64'

class Multi_Feature_Type_Hidden_Layer_Context_Window_Net(A_neural_network):
    """
    Context window with SGD and adagrad
    """

    # (feature_name, convolve, max_pool, use_context_window, nr_filters, filter_width, nr_region_sizes)
    # if "None", it gets determined by the instance attribute
    # FEATURE_MAPPING = {
    #     'w2v_c_nm': ('w2v', True, False, True, 1, 1, 1),
    #     'w2v_nc_nm': ('w2v', False, False, True, 0, 0, 0),
    #     'w2v_c_m': ('w2v', True, True, True, None, None, None),
    #     'pos_c_m': ('pos', True, True, True, None, None, None),
    #     'ner_c_m': ('ner', True, True, True, None, None, None),
    #     'sent_nr_nc_nm': ('sent_nr', False, False, False, 0, 0, 0)
    # }

    CRF_POSITIONS = {'ner': 1, 'pos': 2}

    FEATURE_MAPPING = {
        'w2v_c_nm': {'name': 'w2v', 'convolve': True, 'max_pool': False, 'use_cw': True, 'nr_filters': None, 'filter_width': 1, 'nr_region_sizes': None, 'crf_position': None},
        'w2v_nc_nm': {'name': 'w2v', 'convolve': False, 'max_pool': False, 'use_cw': True, 'nr_filters': 0, 'filter_width': 0, 'nr_region_sizes': 0, 'crf_position': None},
        'w2v_c_m': {'name': 'w2v', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'pos_c_m': {'name': 'pos', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['pos']},
        'ner_c_m': {'name': 'ner', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['ner']},
        'sent_nr_c_m': {'name': 'sent_nr', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'tense_c_m': {'name': 'tense', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': None}
    }

    @classmethod
    def get_features_crf_position(cls, features):
        positions = []
        for feat in features:
            pos = cls.FEATURE_MAPPING[feat]['crf_position']
            if pos:
                positions.append(pos)

        return positions

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
                 n_filters,
                 features_to_use,
                 regularization=False,
                 train_feats=None,
                 valid_feats=None,
                 test_feats=None,
                 features_indexes=None,
                 train_sent_nr_feats=None,
                 valid_sent_nr_feats=None,
                 test_sent_nr_feats=None,
                 train_tense_feats=None,
                 valid_tense_feats=None,
                 test_tense_feats=None,
                 tense_probs=None,
                 region_sizes=None,
                 pos_embeddings=None,
                 ner_embeddings=None,
                 sent_nr_embeddings=None,
                 tense_embeddings=None,
                 **kwargs):

        super(Multi_Feature_Type_Hidden_Layer_Context_Window_Net, self).__init__(**kwargs)

        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_valid = x_test
        # self.y_valid = y_test

        # self.n_samples = self.x_train.shape[0]

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        # self.pretrained_embeddings = embeddings
        self.regularization = regularization

        self.params = OrderedDict()
        self.params_to_get_l2 = []

        self.n_pos_emb = None
        self.n_ner_emb = None

        #POS features
        if train_feats:
            try:
                train_pos_feats = np.array(train_feats[self.CRF_POSITIONS['pos']])
                valid_pos_feats = np.array(valid_feats[self.CRF_POSITIONS['pos']])
                test_pos_feats = np.array(test_feats[self.CRF_POSITIONS['pos']])

                pos_probs = features_indexes[self.CRF_POSITIONS['pos']][2]

                if pos_embeddings is not None:
                    # use these embeddings as initialisers
                    self.pos_embeddings = pos_embeddings
                    self.n_pos_emb = pos_embeddings.shape[1]

                # TODO: choose one.
                # self.n_pos_emb = np.max(train_pos_feats) + 1  # encoding for the POS tags
                # self.n_pos_emb = 40  # encoding for the POS tags

            except KeyError:
                train_pos_feats = None
                valid_pos_feats = None
                test_pos_feats = None
                pos_probs = None

            self.train_pos_feats = train_pos_feats
            self.valid_pos_feats = valid_pos_feats
            self.test_pos_feats = test_pos_feats
            self.pos_probs = pos_probs

            #NER features
            try:
                train_ner_feats = np.array(train_feats[self.CRF_POSITIONS['ner']])
                valid_ner_feats = np.array(valid_feats[self.CRF_POSITIONS['ner']])
                test_ner_feats = np.array(test_feats[self.CRF_POSITIONS['ner']])

                ner_probs = features_indexes[self.CRF_POSITIONS['ner']][2]

                if ner_embeddings is not None:
                    # use these embeddings as initialisers
                    self.ner_embeddings = ner_embeddings
                    self.n_ner_emb = ner_embeddings.shape[1]

                # TODO: choose one.
                # self.n_ner_emb = np.max(train_ner_feats) + 1  # encoding for the POS tags
                # self.n_ner_emb = 40  # encoding for the NER tags

            except KeyError:
                train_ner_feats = None
                valid_ner_feats = None
                test_ner_feats = None
                ner_probs = None

            self.train_ner_feats = np.array(train_ner_feats)
            self.valid_ner_feats = np.array(valid_ner_feats)
            self.test_ner_feats = np.array(test_ner_feats)
            self.ner_probs = ner_probs

            self.pos_filter_width = self.n_pos_emb
            self.ner_filter_width = self.n_ner_emb

        #sent_nr features
        self.train_sent_nr_feats = np.array(train_sent_nr_feats)
        self.valid_sent_nr_feats = np.array(valid_sent_nr_feats)
        self.test_sent_nr_feats = np.array(test_sent_nr_feats)

        # tense features
        self.train_tense_feats = np.array(train_tense_feats)
        self.valid_tense_feats = np.array(valid_tense_feats)
        self.test_tense_feats = np.array(test_tense_feats)

        if sent_nr_embeddings is not None:
            self.sent_nr_embeddings = sent_nr_embeddings
            self.n_sent_nr_emb = sent_nr_embeddings.shape[1]

            self.sent_nr_filter_width = self.n_sent_nr_emb

        if tense_embeddings is not None:
            self.tense_embeddings = tense_embeddings
            self.n_tense_emb = tense_embeddings.shape[1]

            self.tense_filter_width = self.n_tense_emb

        # self.tense_probs = tense_probs

        self.alpha_L2_reg = None
        self.max_pool = None

        self.n_filters = n_filters  # number of filters per region size

        self.region_sizes = region_sizes

        self.w2v_filter_width = self.n_emb

        self.concatenate = utils.NeuralNetwork.theano_gpu_concatenate
        self.concatenate = T.concatenate

        self.features_to_use = self.parse_features_to_use(features_to_use)

        self.features_dimensions = self.get_features_dimensions()

    def get_features_dimensions(self):
        d = dict()
        d['w2v'] = self.n_emb
        d['pos'] = self.n_pos_emb
        d['ner'] = self.n_ner_emb
        d['sent_nr'] = self.n_sent_nr_emb
        d['tense'] = self.n_tense_emb

        return d

    def parse_features_to_use(self, features_to_use):
        return [self.__class__.FEATURE_MAPPING[feat] for feat in features_to_use]

    def using_feature(self, feature, convolve, max_pool):
        result = None

        if convolve is None and max_pool is None:
            result = any([feat['name'] == feature for feat in self.features_to_use])
        else:
            result = any([feat['name'] == feature and feat['convolve'] == convolve and feat['max_pool'] == max_pool
                          for feat in self.features_to_use])

        assert result is not None

        return result

    def train(self, **kwargs):

        nnet_description = ' '.join(['Training with SGD',
                                     'context_win:', str(kwargs['window_size']),
                                     'statically' if kwargs['static'] else 'dynamically',
                                     'filters_per_region:', str(kwargs['n_filters']),
                                     'region_sizes:', '[', ','.join(map(str, kwargs['region_sizes'])), ']',
                                     'with' if kwargs['max_pool'] else 'without', 'max pooling'])

        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)

        elif kwargs['n_hidden']:
            logger.info('Training with SGD with two layers')
            logger.info(nnet_description)
            self.train_with_sgd_two_layers(**kwargs)
        else:
            # train with SGD
            logger.info('Training with SGD with one layer')
            logger.info(nnet_description)
            self.train_with_sgd(**kwargs)

        return True

    def perform_forward_pass_dense_step(self, features_hidden_state, ndim):
        """
        performs the last step in the nnet forward pass (the dense layer and softmax).

        :param w2v_conv:
        :param pos_conv:
        :return:
        """

        a = self.concatenate(features_hidden_state)

        if ndim == 1:
            a = self.concatenate(features_hidden_state)
        elif ndim == 2:
            a = self.concatenate(features_hidden_state, axis=1)

        h = self.hidden_activation_f(a)

        return self.out_activation_f(T.dot(h, self.params['w3']) + self.params['b3'])

    def perform_forward_pass_dense_step_two_layers(self, features_hidden_state, ndim):
        """
        performs the last step in the nnet forward pass (the dense layer and softmax).

        :param w2v_conv:
        :param pos_conv:
        :return:
        """

        a = self.concatenate(features_hidden_state)

        if ndim == 1:
            a = self.concatenate(features_hidden_state)
        elif ndim == 2:
            a = self.concatenate(features_hidden_state, axis=1)

        h = self.hidden_activation_f(a)

        h1 = self.hidden_activation_f(T.dot(h, self.params['w3']) + self.params['b3'])

        return self.out_activation_f(T.dot(h1, self.params['w4']) + self.params['b4'])

    def sgd_forward_pass(self, tensor_dim):

        hidden_state = self.compute_hidden_state(tensor_dim)

        return self.perform_forward_pass_dense_step(hidden_state, ndim=tensor_dim/2)

    def sgd_forward_pass_two_layers(self, tensor_dim):

        hidden_state = self.compute_hidden_state(tensor_dim)

        return self.perform_forward_pass_dense_step_two_layers(hidden_state, ndim=tensor_dim/2)

    def compute_hidden_state(self, tensor_dim):
        hidden_state = []

        if self.using_w2v_no_convolution_no_maxpool_feature():
            if tensor_dim == 2:
                w2v_directly = self.w_x_directly.reshape(shape=(-1,))
            elif tensor_dim == 4:
                w2v_directly = self.w_x_directly.reshape(shape=(self.w_x_directly.shape[0], -1))

            hidden_state.append(w2v_directly)

        if self.using_w2v_convolution_no_maxpool_feature():
            if tensor_dim == 2:
                w2v_conv_nmp = self.convolve_word_embeddings(self.w_x_c_nmp,
                                                             filter_width=1,
                                                             # n_filters=1,
                                                             # region_sizes=[self.n_window],
                                                             n_filters=self.n_filters,
                                                             region_sizes=self.region_sizes,
                                                             max_pool=False,
                                                             filter_prefix='w2v_c_nmp_filter_')
            elif tensor_dim == 4:
                w_x_c_nmp_4D = self.w_x_c_nmp.reshape(
                    shape=(self.w_x_c_nmp.shape[0], 1, self.w_x_c_nmp.shape[1], self.w_x_c_nmp.shape[2]))
                w2v_conv_nmp = self.perform_nnet_word_embeddings_conv2d(w_x_c_nmp_4D,
                                                                        region_sizes=self.region_sizes,
                                                                        max_pool=False,
                                                                        filter_prefix='w2v_c_nmp_filter_')

            hidden_state.append(w2v_conv_nmp)

        if self.using_w2v_convolution_maxpool_feature():
            if tensor_dim == 2:
                w2v_conv_mp = self.convolve_word_embeddings(self.w_x,
                                                         filter_width=self.w2v_filter_width,
                                                         n_filters=self.n_filters,
                                                         region_sizes=self.region_sizes,
                                                         max_pool=self.max_pool)
            elif tensor_dim == 4:
                w_x_4D = self.w_x.reshape(shape=(self.w_x.shape[0], 1, self.w_x.shape[1], self.w_x.shape[2]))
                w2v_conv_mp = self.perform_nnet_word_embeddings_conv2d(w_x_4D,
                                                                    region_sizes=self.region_sizes,
                                                                    max_pool=self.max_pool)

            hidden_state.append(w2v_conv_mp)

        if self.using_pos_convolution_maxpool_feature():
            if tensor_dim == 2:
                pos_conv = self.convolve_pos_features(self.w_pos_x, filter_width=self.pos_filter_width)
            elif tensor_dim == 4:
                w_pos_x_4D = self.w_pos_x.reshape(
                    shape=(self.w_pos_x.shape[0], 1, self.w_pos_x.shape[1], self.w_pos_x.shape[2]))
                pos_conv = self.perform_nnet_pos_conv2d(w_pos_x_4D)

            hidden_state.append(pos_conv)

        if self.using_ner_convolution_maxpool_feature():
            if tensor_dim == 2:
                ner_conv = self.convolve_ner_features(self.w_ner_x, filter_width=self.ner_filter_width)
            elif tensor_dim == 4:
                w_ner_x_4D = self.w_ner_x.reshape(
                    shape=(self.w_ner_x.shape[0], 1, self.w_ner_x.shape[1], self.w_ner_x.shape[2]))
                ner_conv = self.perform_nnet_ner_conv2d(w_ner_x_4D)

            hidden_state.append(ner_conv)

        if self.using_sent_nr_convolution_maxpool_feature():
            if tensor_dim == 2:
                sent_nr_conv = self.convolve_sent_nr_features(self.w_sent_nr_x, filter_width=self.sent_nr_filter_width)
            elif tensor_dim == 4:
                w_sent_nr_x_4D = self.w_sent_nr_x.reshape(
                    shape=(self.w_sent_nr_x.shape[0], 1, self.w_sent_nr_x.shape[1], self.w_sent_nr_x.shape[2]))
                sent_nr_conv = self.perform_nnet_sent_nr_conv2d(w_sent_nr_x_4D)

            hidden_state.append(sent_nr_conv)

        if self.using_tense_convolution_maxpool_feature():
            if tensor_dim == 2:
                tense_conv = self.convolve_tense_features(self.w_tense_x, filter_width=self.tense_filter_width)
            elif tensor_dim == 4:
                w_tense_x_4D = self.w_tense_x.reshape(
                    shape=(self.w_tense_x.shape[0], 1, self.w_tense_x.shape[1], self.w_tense_x.shape[2]))
                tense_conv = self.perform_nnet_tense_conv2d(w_tense_x_4D)

            hidden_state.append(tense_conv)

        return hidden_state

    def create_filters(self, filter_width, region_sizes, n_filters):
        filters = []
        for rs in region_sizes:

            # w_filter_bound = filter_height * filter_width
            w_filter_bound = rs * filter_width
            w_filter_shape = (n_filters, rs, filter_width)
            w_filter = theano.shared(
                value=np.asarray(np.random.uniform(low=-1. / w_filter_bound, high=1. / w_filter_bound,
                                                   size=w_filter_shape), dtype=theano.config.floatX), name="filter_"+str(rs))

            filters.append(w_filter)

        return filters

    def convolve_word_embeddings(self, w_x, filter_width, region_sizes, n_filters, filter_prefix='w2v_filter_',
                                 max_pool=True):

        sentence_matrix_size = (self.n_window, self.n_emb)

        convolutions = []

        for rs in region_sizes:
            w_filter_shape = (n_filters, rs, filter_width)

            convolution = conv.conv2d(input=w_x,
                                 filters=self.params[filter_prefix+str(rs)],
                                 filter_shape=w_filter_shape,
                                 image_shape=sentence_matrix_size)

            if max_pool:
                max_conv = pool_2d(input=convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_conv)
            else:
                convolutions.append(convolution)

        if max_pool:
            conv_concat = self.concatenate(convolutions)[:, 0, 0]
        else:
            conv_concat = self.concatenate(convolutions, axis=1).reshape(shape=(-1,))

        return conv_concat

    def convolve_pos_features(self, w_pos_x, filter_width):

        sentence_pos_shape = (self.n_window, self.n_pos_emb)

        convolutions = []

        for rs in self.region_sizes:
            pos_filter_4_shape = (self.n_filters, rs, filter_width)

            convolution = conv.conv2d(input=w_pos_x,
                                 filters=self.params['pos_filter_'+str(rs)],
                                 filter_shape=pos_filter_4_shape,
                                 image_shape=sentence_pos_shape)

            if self.max_pool:
                max_conv = pool_2d(input=convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_conv)
            else:
                convolutions.append(convolution)

        if self.max_pool:
            conv_concat = self.concatenate(convolutions)[:, 0, 0]
        else:
            conv_concat = self.concatenate(convolutions, axis=1).reshape(shape=(-1,))

        return conv_concat

    def convolve_ner_features(self, w_ner_x, filter_width):

        sentence_ner_shape = (self.n_window, self.n_ner_emb)

        convolutions = []

        for rs in self.region_sizes:
            ner_filter_4_shape = (self.n_filters, rs, filter_width)

            convolution = conv.conv2d(input=w_ner_x,
                                 filters=self.params['ner_filter_'+str(rs)],
                                 filter_shape=ner_filter_4_shape,
                                 image_shape=sentence_ner_shape)

            if self.max_pool:
                max_conv = pool_2d(input=convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_conv)
            else:
                convolutions.append(convolution)

        if self.max_pool:
            conv_concat = self.concatenate(convolutions)[:, 0, 0]
        else:
            conv_concat = self.concatenate(convolutions, axis=1).reshape(shape=(-1,))

        return conv_concat

    def convolve_sent_nr_features(self, w_sent_nr_x, filter_width):

        sentence_sent_nr_shape = (self.n_window, self.n_sent_nr_emb)

        convolutions = []

        for rs in self.region_sizes:
            filter_4_shape = (self.n_filters, rs, filter_width)

            convolution = conv.conv2d(input=w_sent_nr_x,
                                 filters=self.params['sent_nr_filter_'+str(rs)],
                                 filter_shape=filter_4_shape,
                                 image_shape=sentence_sent_nr_shape)

            if self.max_pool:
                max_conv = pool_2d(input=convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_conv)
            else:
                convolutions.append(convolution)

        if self.max_pool:
            conv_concat = self.concatenate(convolutions)[:, 0, 0]
        else:
            conv_concat = self.concatenate(convolutions, axis=1).reshape(shape=(-1,))

        return conv_concat

    def convolve_tense_features(self, w_tense_x, filter_width):

        sentence_tense_shape = (self.n_window, self.n_tense_emb)

        convolutions = []

        for rs in self.region_sizes:
            filter_4_shape = (self.n_filters, rs, filter_width)

            convolution = conv.conv2d(input=w_tense_x,
                                 filters=self.params['tense_filter_'+str(rs)],
                                 filter_shape=filter_4_shape,
                                 image_shape=sentence_tense_shape)

            if self.max_pool:
                max_conv = pool_2d(input=convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_conv)
            else:
                convolutions.append(convolution)

        if self.max_pool:
            conv_concat = self.concatenate(convolutions)[:, 0, 0]
        else:
            conv_concat = self.concatenate(convolutions, axis=1).reshape(shape=(-1,))

        return conv_concat

    def determine_nr_filters(self, feature):
        nr_filters = self.__class__.FEATURE_MAPPING[feature]['nr_filters']
        if not nr_filters:
            nr_filters = self.n_filters

        assert nr_filters is not None, 'Could not determine nr_filters for feature: %s' % feature

        return nr_filters

    def determine_filter_width(self, feature):
        filter_width = self.__class__.FEATURE_MAPPING[feature]['filter_width']
        if not filter_width:
            if feature == 'w2v':
                filter_width = self.n_emb
            elif feature == 'pos':
                filter_width = self.n_pos_emb
            elif feature == 'ner':
                filter_width = self.n_ner_emb
            elif feature == 'sent_nr':
                filter_width = self.n_sent_nr_emb
            elif feature == 'tense':
                filter_width = self.n_tense_emb

        assert filter_width is not None, 'Could not determine filter_width for feature: %s' % feature

        return filter_width

    def determine_nr_region_sizes(self, feature):
        nr_regions = self.__class__.FEATURE_MAPPING[feature]['nr_region_sizes']
        if not nr_regions:
            nr_regions = self.region_sizes.__len__()

        assert nr_regions is not None, 'Could not determine nr_region_sizes for feature: %s' % feature

        return nr_regions

    def determine_resulting_width(self, feature, feat_dim):
        resulting_width = None
        filter_width = self.__class__.FEATURE_MAPPING[feature]['filter_width']

        if filter_width is not None:
            resulting_width = feat_dim / filter_width
        else:
            resulting_width = 1

        assert resulting_width is not None, 'Could not determine nr_region_sizes for feature: %s' % feature

        return resulting_width

    def determine_hidden_layer_size(self):
        size = 0

        for desc in self.features_to_use:

            feature = desc['name']
            convolve = desc['convolve']
            max_pool = desc['max_pool']
            c_window = desc['use_cw']
            feat_dim = self.get_features_dimensions()[feature]

            feat_extended = '_'.join([feature, 'c' if convolve else 'nc', 'm' if max_pool else 'nm'])

            if convolve:
                window = 1
                n_filters = self.determine_nr_filters(feat_extended)
                n_regions = self.determine_nr_region_sizes(feat_extended)
                filter_windows = np.sum([self.n_window - rs + 1 for rs in self.region_sizes])
                filter_window_width = self.determine_resulting_width(feat_extended, feat_dim)
            else:
                window = self.n_window
                n_regions = 1
                n_filters = 1
                filter_window_width = feat_dim
                filter_windows = 1

            if max_pool:
                feat_dimension = 1
                filter_windows = self.determine_nr_region_sizes(feat_extended)
            else:
                feat_dimension = self.features_dimensions[feature]

            if not c_window:
                window = 1

            # size += feat_dimension * window * n_regions * n_filters
            size += filter_windows * filter_window_width * window * n_filters

        return size

    def using_w2v_feature(self):
        return self.using_feature(feature='w2v', convolve=None, max_pool=None)

    def using_w2v_no_convolution_no_maxpool_feature(self):
        return self.using_feature(feature='w2v', convolve=False, max_pool=False)

    def using_w2v_convolution_no_maxpool_feature(self):
        return self.using_feature(feature='w2v', convolve=True, max_pool=False)

    def using_w2v_convolution_maxpool_feature(self):
        return self.using_feature(feature='w2v', convolve=True, max_pool=True)

    def using_pos_feature(self):
        return self.using_feature(feature='pos', convolve=None, max_pool=None)

    def using_ner_feature(self):
        return self.using_feature(feature='ner', convolve=None, max_pool=None)

    def using_pos_convolution_maxpool_feature(self):
        return self.using_feature(feature='pos', convolve=True, max_pool=True)

    def using_ner_convolution_maxpool_feature(self):
        return self.using_feature(feature='ner', convolve=True, max_pool=True)

    def using_sent_nr_feature(self):
        return self.using_feature(feature='sent_nr', convolve=None, max_pool=None)

    def using_sent_nr_convolution_maxpool_feature(self):
        return self.using_feature(feature='sent_nr', convolve=True, max_pool=True)

    def using_tense_feature(self):
        return self.using_feature(feature='tense', convolve=None, max_pool=None)

    def using_tense_convolution_maxpool_feature(self):
        return self.using_feature(feature='tense', convolve=True, max_pool=True)

    def train_with_sgd(self, learning_rate,
                       max_epochs,
                       alpha_L2_reg=0.01,
                       save_params=False, plot=False,
                       static=False, max_pool=True,
                       lr_decay=True,
                       **kwargs):

        self.alpha_L2_reg = alpha_L2_reg
        self.max_pool = max_pool
        self.static = static

        # separate learning rate train from learning rate fine tune
        learning_rate_train = learning_rate

        if lr_decay:
            logger.info('Applying learning rate step decay.')

        # indexes the w2v embeddings
        w2v_idxs = T.vector(name="w2v_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        # indexes the POS matrix
        pos_idxs = T.vector(name="pos_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        # indexes the NER matrix
        ner_idxs = T.vector(name="ner_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        sent_nr_idxs = T.vector(name="sent_nr_idxs", dtype=INT)

        tense_idxs = T.vector(name="tense_idxs", dtype=INT)

        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)
        #
        # if self.using_w2v_feature():
        #     train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        #     valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)
        #
        # if self.using_pos_feature():
        #     # shared variable with training POS features
        #     train_pos_x = theano.shared(value=np.array(self.train_pos_feats, dtype=INT), name='train_pos_x', borrow=True)
        #     valid_pos_x = theano.shared(value=np.array(self.valid_pos_feats, dtype=INT), name='valid_pos_x', borrow=True)
        #
        # if self.using_ner_feature():
        #     # shared variable with training POS features
        #     train_ner_x = theano.shared(value=np.array(self.train_ner_feats, dtype=INT), name='train_ner_x', borrow=True)
        #     valid_ner_x = theano.shared(value=np.array(self.valid_ner_feats, dtype=INT), name='valid_ner_x', borrow=True)
        #
        # if self.using_sent_nr_feature():
        #     train_sent_nr_x = theano.shared(value=np.array(self.train_sent_nr_feats, dtype=INT), name='train_sent_nr_x', borrow=True)
        #     valid_sent_nr_x = theano.shared(value=np.array(self.valid_sent_nr_feats, dtype=INT), name='valid_sent_nr_x', borrow=True)
        #
        # if self.using_tense_feature():
        #     train_tense_x = theano.shared(value=self.train_tense_feats.astype(dtype=INT), name='train_tense_x', borrow=True)
        #     valid_tense_x = theano.shared(value=self.valid_tense_feats.astype(dtype=INT), name='valid_tense_x', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)

        # create dense layer and bias
        hidden_layer_size = self.determine_hidden_layer_size()

        logger.info('Hidden layer size: %d' % hidden_layer_size)

        w3 = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            n_in=hidden_layer_size, n_out=self.n_out, function='softmax').astype(dtype=theano.config.floatX),
                           name="w3",
                           borrow=True)
        b3 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX),
                           name='b3',
                           borrow=True)

        params = [w3, b3]
        param_names = ['w3', 'b3']
        params_to_get_l2 = ['w3', 'b3']
        params_to_get_grad = [w3, b3]
        params_to_get_grad_names = ['w3', 'b3']

        self.params = OrderedDict(zip(param_names, params))

        if self.using_w2v_no_convolution_no_maxpool_feature():
            # word embeddings to be used directly.
            w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                               name='w1', borrow=True)

            # index embeddings
            w_x_directly = w1[w2v_idxs]

            # add structures to self.params
            self.params['w1'] = w1

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_directly)
                params_to_get_grad_names.append('w_x_directly')
                params_to_get_l2.append('w_x_directly')

            self.w_x_directly = w_x_directly

        if self.using_w2v_convolution_no_maxpool_feature():
            w1_c_nmp = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                               name='w1', borrow=True)

            # create w2v filters
            # w2v_c_nmp_filters = self.create_filters(filter_width=1, region_sizes=[self.n_window],
            #                                            n_filters=1)
            w2v_c_nmp_filters = self.create_filters(filter_width=1, region_sizes=self.region_sizes,
                                                       n_filters=self.n_filters)
            w2v_c_nmp_filters_names = map(lambda x: 'w2v_c_nmp_%s' % x.name, w2v_c_nmp_filters)

            # index embeddings
            w_x_c_nmp = w1_c_nmp[w2v_idxs]

            # add structures to self.params
            self.params['w1_c_nmp'] = w1_c_nmp
            self.params.update(dict(zip(w2v_c_nmp_filters_names, w2v_c_nmp_filters)))
            params_to_get_l2.extend(w2v_c_nmp_filters_names)
            params_to_get_grad.extend(w2v_c_nmp_filters)
            params_to_get_grad_names.extend(w2v_c_nmp_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_c_nmp)
                params_to_get_grad_names.append('w_x_c_nmp')
                params_to_get_l2.append('w_x_c_nmp')

            self.w_x_c_nmp = w_x_c_nmp

        if self.using_w2v_convolution_maxpool_feature():
            # word embeddings for constructing higher order features
            w1_w2v = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                                   name='w1_w2v', borrow=True)

            # create w2v filters
            w2v_filters = self.create_filters(filter_width=self.w2v_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            w2v_filters_names = map(lambda x: 'w2v_%s' % x.name, w2v_filters)

            # index embeddings
            w_x = w1_w2v[w2v_idxs]

            # add structures to self.params
            self.params['w1_w2v'] = w1_w2v
            self.params.update(dict(zip(w2v_filters_names, w2v_filters)))
            params_to_get_l2.append('w1_w2v')
            params_to_get_l2.extend(w2v_filters_names)
            params_to_get_grad.extend(w2v_filters)
            params_to_get_grad_names.extend(w2v_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x)
                params_to_get_grad_names.append('w_x')
                params_to_get_l2.append('w_x')

            self.w_x = w_x

        if self.using_pos_convolution_maxpool_feature():
            # create POS embeddings
            #TODO: one-hot, random, probabilistic ?
            if self.pos_embeddings is not None:
                w1_pos = theano.shared(value=np.matrix(self.pos_embeddings, dtype=theano.config.floatX), name='w1_pos', borrow=True)
                self.n_pos_emb = self.pos_embeddings.shape[1]
            else:
                w1_pos = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                    n_in=np.max(self.train_pos_feats)+1, n_out=self.n_pos_emb, function='tanh').astype(dtype=theano.config.floatX), name='w1_pos', borrow=True)

            # w1_pos = theano.shared(value=np.matrix(self.pos_probs.values(), dtype=theano.config.floatX).reshape((-1,1)), name='w1_pos', borrow=True)
            # self.n_pos_emb = 1
            # self.pos_filter_width = self.n_pos_emb

            # create POS filters
            pos_filters = self.create_filters(filter_width=self.pos_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            pos_filters_names = map(lambda x: 'pos_%s' % x.name, pos_filters)

            # index embeddings
            w_pos_x = w1_pos[pos_idxs]

            # add structures to self.params
            self.params['w1_pos'] = w1_pos
            self.params.update(dict(zip(pos_filters_names,pos_filters)))
            params_to_get_l2.extend(pos_filters_names)
            params_to_get_grad.extend(pos_filters)
            params_to_get_grad_names.extend(pos_filters_names)

            if not self.static:
                params_to_get_grad.append(w_pos_x)
                params_to_get_grad_names.append('w_pos_x')
                params_to_get_l2.append('w_pos_x')

            self.w_pos_x = w_pos_x

        if self.using_ner_convolution_maxpool_feature():
            # w1_ner = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            #     n_in=np.max(self.train_ner_feats)+1, n_out=self.n_ner_emb, function='tanh').astype(dtype=theano.config.floatX),
            #                        name='w1_ner', borrow=True)

            # TODO: one-hot, random, probabilistic ?
            if self.ner_embeddings is not None:
                w1_ner = theano.shared(value=np.matrix(self.ner_embeddings, dtype=theano.config.floatX), name='w1_ner',
                                       borrow=True)
                self.n_ner_emb = self.ner_embeddings.shape[1]
            else:
                w1_ner = theano.shared(
                    value=np.eye(np.max(self.train_ner_feats) + 1, self.n_ner_emb).astype(dtype=theano.config.floatX),
                    name='w1_ner', borrow=True)

            # create NER filters
            ner_filters = self.create_filters(filter_width=self.ner_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            ner_filters_names = map(lambda x: 'ner_%s' % x.name, ner_filters)

            # index embeddings
            w_ner_x = w1_ner[ner_idxs]

            # add structures to self.params
            self.params['w1_ner'] = w1_ner
            self.params.update(dict(zip(ner_filters_names, ner_filters)))
            params_to_get_l2.extend(ner_filters_names)
            params_to_get_grad.extend(ner_filters)
            params_to_get_grad_names.extend(ner_filters_names)
            if not self.static:
                params_to_get_grad.append(w_ner_x)
                params_to_get_grad_names.append('w_ner_x')
                params_to_get_l2.append('w_ner_x')

            self.w_ner_x = w_ner_x

        if self.using_sent_nr_convolution_maxpool_feature():

            if self.sent_nr_embeddings is not None:
                w1_sent_nr = theano.shared(value=np.matrix(self.sent_nr_embeddings, dtype=theano.config.floatX), name='w1_sent_nr', borrow=True)
            else:
                w1_sent_nr = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                    n_in=np.max(self.train_sent_nr_feats)+1, n_out=self.n_sent_nr_emb, function='tanh').astype(dtype=theano.config.floatX),
                                       name='w1_sent_nr', borrow=True)

            # create NER filters
            sent_nr_filters = self.create_filters(filter_width=self.sent_nr_filter_width,
                                                  region_sizes=self.region_sizes, n_filters=self.n_filters)

            sent_nr_filters_names = map(lambda x: 'sent_nr_%s' % x.name, sent_nr_filters)

            # index embeddings
            w_sent_nr_x = w1_sent_nr[sent_nr_idxs]

            # add structures to self.params
            self.params['w1_sent_nr'] = w1_sent_nr
            self.params.update(dict(zip(sent_nr_filters_names, sent_nr_filters)))
            params_to_get_l2.extend(sent_nr_filters_names)
            params_to_get_grad.extend(sent_nr_filters)
            params_to_get_grad_names.extend(sent_nr_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_sent_nr_x)
                params_to_get_grad_names.append('w_sent_nr_x')
                params_to_get_l2.append('w_sent_nr_x')

            self.w_sent_nr_x = w_sent_nr_x

        if self.using_tense_convolution_maxpool_feature():
            if self.tense_embeddings is not None:
                w1_tense = theano.shared(value=np.matrix(self.tense_embeddings, dtype=theano.config.floatX),
                                         name='w1_tense', borrow=True)
            else:
                w1_tense = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                    n_in=np.max(self.train_tense_feats)+1, n_out=self.n_tense_emb, function='tanh').astype(dtype=theano.config.floatX),
                                       name='w1_tense', borrow=True)

            # create NER filters
            tense_filters = self.create_filters(filter_width=self.tense_filter_width,
                                                region_sizes=self.region_sizes, n_filters=self.n_filters)

            tense_filters_names = map(lambda x: 'tense_%s' % x.name, tense_filters)

            # index embeddings
            w_tense_x = w1_tense[tense_idxs]

            # add structures to self.params
            self.params['w1_tense'] = w1_tense
            self.params.update(dict(zip(tense_filters_names, tense_filters)))
            params_to_get_l2.extend(tense_filters_names)
            params_to_get_grad.extend(tense_filters)
            params_to_get_grad_names.extend(tense_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_tense_x)
                params_to_get_grad_names.append('w_tense_x')
                params_to_get_l2.append('w_tense_x')

            self.w_tense_x = w_tense_x

        self.params_to_get_l2 = params_to_get_l2

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            L2_w2v_nc_nmp_weight, L2_w2v_c_mp_filters, L2_w_x, L2_w2v_c_nmp_filters, \
            L2_w2v_c_nmp_weight, L2_pos_c_mp_filters, L2_w_pos_x, L2_ner_c_mp_filters, L2_w_ner_x, \
            L2_w_sent_nr_x, L2_w_sent_nr_filters, L2_w_tense_x, L2_w_tense_filters, L2_w3 = self.compute_regularization_cost()

            L2 = L2_w2v_nc_nmp_weight + L2_w2v_c_mp_filters + L2_w_x + L2_w2v_c_nmp_filters + \
                 L2_w2v_c_nmp_weight + L2_pos_c_mp_filters + L2_w_pos_x + L2_ner_c_mp_filters + \
                 L2_w_ner_x + L2_w_sent_nr_x + L2_w_sent_nr_filters + L2_w_tense_x + L2_w_tense_filters + L2_w3

        W1_tense_sum, W1_sent_nr_sum, W1_ner_sum, W1_pos_sum, W1_w2v_sum, W1_c_nmp_sum, W1_sum = self.compute_weight_evolution()

        out = self.sgd_forward_pass(tensor_dim=2)

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))
        if self.regularization:
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = mean_cross_entropy + self.alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        # cost_prediction = mean_cross_entropy + alpha_L2_reg * L2
        # cost_prediction = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # cost_prediction = alpha_L2_reg*L2

        errors = T.sum(T.neq(y_predictions, y))

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        # test = theano.function([w2v_idxs, pos_idxs, y], [out, grads[0]])
        # test(self.x_train[0], self.train_pos_feats[0], self.y_train[0])

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1_w2v'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_pos_x':
                eps = np.zeros_like(self.params['w1_pos'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_ner_x':
                eps = np.zeros_like(self.params['w1_ner'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_x_directly':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_x_c_nmp':
                eps = np.zeros_like(self.params['w1_c_nmp'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_sent_nr_x':
                eps = np.zeros_like(self.params['w1_sent_nr'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_tense_x':
                eps = np.zeros_like(self.params['w1_tense'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads, accumulated_grad):
            if name == 'w_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_w2v'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_pos_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[pos_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate_train * grad/(T.sqrt(accum[pos_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_pos'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_ner_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[ner_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[ner_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_ner'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_x_directly':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_x_c_nmp':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_c_nmp'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_sent_nr_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[sent_nr_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[sent_nr_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_sent_nr'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_tense_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[tense_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[tense_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_tense'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        train_idx = T.scalar(name="train_idx", dtype=INT)
        valid_idx = T.scalar(name="valid_idx", dtype=INT)

        train_givens = dict()
        valid_givens = dict()
        if self.using_w2v_feature():
            train_givens.update({
                w2v_idxs: train_x[train_idx]
            })
            valid_givens.update({
                w2v_idxs: valid_x[valid_idx]
            })

        if self.using_pos_feature():
            train_givens.update({
                # pos_idxs: train_pos_x[train_idx]
                pos_idxs: train_x[train_idx]
            })
            valid_givens.update({
                # pos_idxs: valid_pos_x[valid_idx]
                pos_idxs: valid_x[valid_idx]
            })

        if self.using_ner_feature():
            train_givens.update({
                # ner_idxs: train_ner_x[train_idx]
                ner_idxs: train_x[train_idx]
            })
            valid_givens.update({
                # ner_idxs: valid_ner_x[valid_idx]
                ner_idxs: valid_x[valid_idx]
            })

        if self.using_sent_nr_feature():
            train_givens.update({
                # sent_nr_idxs: train_sent_nr_x[train_idx]
                sent_nr_idxs: train_x[train_idx]
            })
            valid_givens.update({
                # sent_nr_idxs: valid_sent_nr_x[valid_idx]
                sent_nr_idxs: valid_x[valid_idx]
            })

        if self.using_tense_feature():
            train_givens.update({
                # tense_idxs: train_tense_x[train_idx]
                tense_idxs: train_x[train_idx]
            })
            valid_givens.update({
                # tense_idxs: valid_tense_x[valid_idx]
                tense_idxs: valid_x[valid_idx]
            })

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost, errors],
                                updates=updates,
                                givens=train_givens)

        train_get_cross_entropy = theano.function(inputs=[train_idx, y],
                                            outputs=mean_cross_entropy,
                                            updates=[],
                                            givens=train_givens)

        valid_get_cross_entropy = theano.function(inputs=[valid_idx, y],
                                            outputs=mean_cross_entropy,
                                            updates=[],
                                            givens=valid_givens)

        train_predict = theano.function(inputs=[valid_idx, y],
                                        outputs=[cost, errors, y_predictions],
                                        updates=[],
                                        on_unused_input='ignore',
                                        givens=valid_givens)

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[W1_tense_sum, W1_sent_nr_sum, W1_ner_sum, W1_pos_sum, W1_w2v_sum,
                                                        W1_c_nmp_sum, W1_sum, L2_w2v_c_mp_filters, L2_w2v_c_nmp_filters,
                                                        L2_pos_c_mp_filters, L2_ner_c_mp_filters, L2_w_sent_nr_filters,
                                                        L2_w_tense_filters, L2_w3],
                                               givens=[])

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        epoch_l2_w2v_nc_nmp_weight_list = []
        epoch_l2_w2v_c_mp_filters_list = []
        epoch_l2_w2v_c_mp_weight_list = []
        epoch_l2_w2v_c_nmp_filters_list = []
        epoch_l2_w2v_c_nmp_weight_list = []
        epoch_l2_pos_filters_list = []
        epoch_l2_pos_weight_list = []
        epoch_l2_ner_filters_list = []
        epoch_l2_ner_weight_list = []
        epoch_l2_sent_nr_weight_list = []
        epoch_l2_sent_nr_filters_list = []
        epoch_l2_tense_weight_list = []
        epoch_l2_tense_filters_list = []
        epoch_l2_w3_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        last_valid_errors = np.inf

        for epoch_index in range(max_epochs):
            start = time.time()
            train_cost = 0
            train_errors = 0
            epoch_l2_w2v_nc_nmp_weight = 0
            epoch_l2_w2v_c_nmp_filters = 0
            epoch_l2_w2v_c_mp_filters = 0
            epoch_l2_w2v_c_mp_weight = 0
            epoch_l2_w2v_c_nmp_weight = 0
            epoch_l2_pos_filters = 0
            epoch_l2_pos_weight = 0
            epoch_l2_ner_filters = 0
            epoch_l2_ner_weight = 0
            epoch_l2_sent_nr_weight = 0
            epoch_l2_sent_nr_filters = 0
            epoch_l2_tense_weight = 0
            epoch_l2_tense_filters = 0
            epoch_l2_w3 = 0
            train_cross_entropy = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i, [train_y.get_value()[i]])
                train_cost += cost_output
                train_errors += errors_output
                train_cross_entropy += train_get_cross_entropy(i, [train_y.get_value()[i]])

            if self.regularization:
                w1_tense_sum, w1_sent_nr_sum, w1_ner_sum, w1_pos_sum, w1_w2v_sum, w1_c_nmp_sum, w1_sum, l2_w2v_c_mp_filters,\
                l2_w2v_c_nmp_filters, l2_pos_c_mp_filters, l2_ner_c_mp_filters, l2_sent_nr_c_m_filters,\
                l2_tense_c_m_filters, l2_w3 = train_l2_penalty()

                epoch_l2_w2v_nc_nmp_weight += w1_sum
                epoch_l2_w2v_c_mp_filters += l2_w2v_c_mp_filters
                epoch_l2_w2v_c_mp_weight += w1_w2v_sum
                epoch_l2_w2v_c_nmp_filters += l2_w2v_c_nmp_filters
                epoch_l2_w2v_c_nmp_weight += w1_c_nmp_sum
                epoch_l2_pos_filters += l2_pos_c_mp_filters
                epoch_l2_pos_weight += w1_pos_sum
                epoch_l2_ner_filters += l2_ner_c_mp_filters
                epoch_l2_ner_weight += w1_ner_sum
                epoch_l2_sent_nr_weight += w1_sent_nr_sum
                epoch_l2_sent_nr_filters += l2_sent_nr_c_m_filters
                epoch_l2_tense_weight += w1_tense_sum
                epoch_l2_tense_filters += l2_tense_c_m_filters
                epoch_l2_w3 += l2_w3

                # self.predict(on_validation_set=True, compute_cost_error=True)

            valid_errors = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            start1 = time.time()
            for i, y_sample in enumerate(self.y_valid):
                # cost_output = 0 #TODO: in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                cost_output, errors_output, pred = train_predict(i, [y_sample])
                valid_cost += cost_output
                valid_errors += errors_output
                valid_predictions.append(np.asscalar(pred))
                valid_cross_entropy += valid_get_cross_entropy(i, [y_sample])

            print time.time()-start1
            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_errors)

            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            epoch_l2_w2v_nc_nmp_weight_list.append(epoch_l2_w2v_nc_nmp_weight)
            epoch_l2_w2v_c_mp_filters_list.append(epoch_l2_w2v_c_mp_filters)
            epoch_l2_w2v_c_mp_weight_list.append(epoch_l2_w2v_c_mp_weight)
            epoch_l2_w2v_c_nmp_filters_list.append(epoch_l2_w2v_c_nmp_filters)
            epoch_l2_w2v_c_nmp_weight_list.append(epoch_l2_w2v_c_nmp_weight)
            epoch_l2_pos_filters_list.append(epoch_l2_pos_filters)
            epoch_l2_pos_weight_list.append(epoch_l2_pos_weight)
            epoch_l2_ner_filters_list.append(epoch_l2_ner_filters)
            epoch_l2_ner_weight_list.append(epoch_l2_ner_weight)
            epoch_l2_sent_nr_weight_list.append(epoch_l2_sent_nr_weight)
            epoch_l2_sent_nr_filters_list.append(epoch_l2_sent_nr_filters)
            epoch_l2_tense_weight_list.append(epoch_l2_tense_weight)
            epoch_l2_tense_filters_list.append(epoch_l2_tense_filters)
            epoch_l2_w3_list.append(epoch_l2_w3)

            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, train_cost, train_errors, valid_cost, valid_errors, f1_score, end-start))

            if lr_decay and (valid_errors > last_valid_errors):
                logger.info('Changing learning rate from %f to %f' % (learning_rate, .5*learning_rate))
                learning_rate *= .5

            last_valid_errors = valid_errors

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list, actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)

            plot_data_dict = {
                'w2v_emb': epoch_l2_w2v_nc_nmp_weight_list,
                'w2v_c_mp_filters': epoch_l2_w2v_c_mp_filters_list,
                'w2v_c_mp_emb': epoch_l2_w2v_c_mp_weight_list,
                'w2v_c_nmp_filters': epoch_l2_w2v_c_nmp_filters_list,
                'w2v_c_nmp_emb': epoch_l2_w2v_c_nmp_weight_list,
                'pos_c_mp_filters': epoch_l2_pos_filters_list,
                'pos_c_mp_emb': epoch_l2_pos_weight_list,
                'ner_c_mp_filters': epoch_l2_ner_filters_list,
                'ner_c_mp_emb': epoch_l2_ner_weight_list,
                'sent_nr_c_mp_emb': epoch_l2_sent_nr_weight_list,
                'sent_nr_c_mp_filters': epoch_l2_sent_nr_filters_list,
                'tense_c_mp_emb': epoch_l2_tense_weight_list,
                'tense_c_mp_filters': epoch_l2_tense_filters_list,
                'w3': epoch_l2_w3_list
            }
            self.plot_penalties_general(plot_data_dict, actual_time=actual_time)

            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        return True

    def train_with_sgd_two_layers(self,
                                  learning_rate,
                                  max_epochs,
                                  n_hidden,
                                  alpha_L2_reg=0.01,
                                  save_params=False, plot=False,
                                  static=False, max_pool=True,
                                  lr_decay=True,
                                  **kwargs):

        self.alpha_L2_reg = alpha_L2_reg
        self.max_pool = max_pool
        self.static = static

        if lr_decay:
            logger.info('Applying learning rate step decay.')

        # indexes the w2v embeddings
        w2v_idxs = T.vector(name="w2v_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        # indexes the POS matrix
        pos_idxs = T.vector(name="pos_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        # indexes the NER matrix
        ner_idxs = T.vector(name="ner_train_idxs",
                            dtype=INT)  # columns: context window size/lines: tokens in the sentence

        sent_nr_id = T.scalar(name="sent_nr_id", dtype=INT)

        tense_id = T.scalar(name="tense_id", dtype=INT)

        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        if self.using_w2v_feature():
            train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
            valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        if self.using_pos_feature():
            # shared variable with training POS features
            train_pos_x = theano.shared(value=np.array(self.train_pos_feats, dtype=INT), name='train_pos_x', borrow=True)
            valid_pos_x = theano.shared(value=np.array(self.valid_pos_feats, dtype=INT), name='valid_pos_x', borrow=True)

        if self.using_ner_feature():
            # shared variable with training POS features
            train_ner_x = theano.shared(value=np.array(self.train_ner_feats, dtype=INT), name='train_ner_x', borrow=True)
            valid_ner_x = theano.shared(value=np.array(self.valid_ner_feats, dtype=INT), name='valid_ner_x', borrow=True)

        if self.using_sent_nr_feature():
            train_sent_nr_x = theano.shared(value=np.array(self.train_sent_nr_feats, dtype=INT), name='train_sent_nr_x', borrow=True)
            valid_sent_nr_x = theano.shared(value=np.array(self.valid_sent_nr_feats, dtype=INT), name='valid_sent_nr_x', borrow=True)

        if self.using_tense_feature():
            train_tense_x = theano.shared(value=self.train_tense_feats.astype(dtype=INT), name='train_tense_x', borrow=True)
            valid_tense_x = theano.shared(value=self.valid_tense_feats.astype(dtype=INT), name='valid_tense_x', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)

        # create dense layer and bias
        hidden_layer_size = self.determine_hidden_layer_size()

        logger.info('Hidden layer size: %d' % hidden_layer_size)

        w3 = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            n_in=hidden_layer_size, n_out=n_hidden, function='softmax').astype(dtype=theano.config.floatX),
                           name='w3',
                           borrow=True)
        b3 = theano.shared(value=np.zeros(n_hidden).astype(dtype=theano.config.floatX),
                           name='b3',
                           borrow=True)
        w4 = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            n_in=n_hidden, n_out=self.n_out, function='softmax').astype(dtype=theano.config.floatX),
                           name='w4',
                           borrow=True)
        b4 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX),
                           name='b4',
                           borrow=True)

        params = [w3, b3, w4, b4]
        param_names = ['w3', 'b3', 'w4', 'b4']
        params_to_get_l2 = ['w3', 'b3', 'w4', 'b4']
        params_to_get_grad = [w3, b3, w4, b4]
        params_to_get_grad_names = ['w3', 'b3', 'w4', 'b4']

        self.params = OrderedDict(zip(param_names, params))

        if self.using_w2v_no_convolution_no_maxpool_feature():
            # word embeddings to be used directly.
            w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                               name='w1', borrow=True)

            # index embeddings
            w_x_directly = w1[w2v_idxs]

            # add structures to self.params
            self.params['w1'] = w1

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_directly)
                params_to_get_grad_names.append('w_x_directly')
                params_to_get_l2.append('w_x_directly')

            self.w_x_directly = w_x_directly

        if self.using_w2v_convolution_no_maxpool_feature():
            w1_c_nmp = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                               name='w1', borrow=True)

            # create w2v filters
            # w2v_c_nmp_filters = self.create_filters(filter_width=1, region_sizes=[self.n_window],
            #                                            n_filters=1)
            w2v_c_nmp_filters = self.create_filters(filter_width=1, region_sizes=self.region_sizes,
                                                       n_filters=self.n_filters)
            w2v_c_nmp_filters_names = map(lambda x: 'w2v_c_nmp_%s' % x.name, w2v_c_nmp_filters)

            # index embeddings
            w_x_c_nmp = w1_c_nmp[w2v_idxs]

            # add structures to self.params
            self.params['w1_c_nmp'] = w1_c_nmp
            self.params.update(dict(zip(w2v_c_nmp_filters_names, w2v_c_nmp_filters)))
            params_to_get_l2.extend(w2v_c_nmp_filters_names)
            params_to_get_grad.extend(w2v_c_nmp_filters)
            params_to_get_grad_names.extend(w2v_c_nmp_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_c_nmp)
                params_to_get_grad_names.append('w_x_c_nmp')
                params_to_get_l2.append('w_x_c_nmp')

            self.w_x_c_nmp = w_x_c_nmp

        if self.using_w2v_convolution_maxpool_feature():
            # word embeddings for constructing higher order features
            w1_w2v = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                                   name='w1_w2v', borrow=True)

            # create w2v filters
            w2v_filters = self.create_filters(filter_width=self.w2v_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            w2v_filters_names = map(lambda x: 'w2v_%s' % x.name, w2v_filters)

            # index embeddings
            w_x = w1_w2v[w2v_idxs]

            # add structures to self.params
            self.params['w1_w2v'] = w1_w2v
            self.params.update(dict(zip(w2v_filters_names, w2v_filters)))
            params_to_get_l2.append('w1_w2v')
            params_to_get_l2.extend(w2v_filters_names)
            params_to_get_grad.extend(w2v_filters)
            params_to_get_grad_names.extend(w2v_filters_names)

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_x)
                params_to_get_grad_names.append('w_x')
                params_to_get_l2.append('w_x')

            self.w_x = w_x

        if self.using_pos_convolution_maxpool_feature():
            # create POS embeddings
            #TODO: one-hot, random, probabilistic ?
            w1_pos = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                n_in=np.max(self.train_pos_feats)+1, n_out=self.n_pos_emb, function='tanh').astype(dtype=theano.config.floatX), name='w1_pos', borrow=True)

            # w1_pos = theano.shared(value=np.eye(np.max(self.train_pos_feats)+1, self.n_pos_emb).astype(dtype=theano.config.floatX), name='w1_pos', borrow=True)

            # w1_pos = theano.shared(value=np.matrix(self.pos_probs.values(), dtype=theano.config.floatX).reshape((-1,1)), name='w1_pos', borrow=True)
            # self.n_pos_emb = 1
            # self.pos_filter_width = self.n_pos_emb

            # create POS filters
            pos_filters = self.create_filters(filter_width=self.pos_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            pos_filters_names = map(lambda x: 'pos_%s' % x.name, pos_filters)

            # index embeddings
            w_pos_x = w1_pos[pos_idxs]

            # add structures to self.params
            self.params['w1_pos'] = w1_pos
            self.params.update(dict(zip(pos_filters_names,pos_filters)))
            params_to_get_l2.extend(pos_filters_names)
            params_to_get_grad.extend(pos_filters)
            params_to_get_grad_names.extend(pos_filters_names)

            if not self.static:
                params_to_get_grad.append(w_pos_x)
                params_to_get_grad_names.append('w_pos_x')
                params_to_get_l2.append('w_pos_x')

            self.w_pos_x = w_pos_x

        if self.using_ner_convolution_maxpool_feature():
            # w1_ner = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            #     n_in=np.max(self.train_ner_feats)+1, n_out=self.n_ner_emb, function='tanh').astype(dtype=theano.config.floatX),
            #                        name='w1_ner', borrow=True)
            w1_ner = theano.shared(
                value=np.eye(np.max(self.train_ner_feats) + 1, self.n_ner_emb).astype(dtype=theano.config.floatX),
                name='w1_ner', borrow=True)

            # create NER filters
            ner_filters = self.create_filters(filter_width=self.ner_filter_width, region_sizes=self.region_sizes,
                                              n_filters=self.n_filters)
            ner_filters_names = map(lambda x: 'ner_%s' % x.name, ner_filters)

            # index embeddings
            w_ner_x = w1_ner[ner_idxs]

            # add structures to self.params
            self.params['w1_ner'] = w1_ner
            self.params.update(dict(zip(ner_filters_names, ner_filters)))
            params_to_get_l2.extend(ner_filters_names)
            params_to_get_grad.extend(ner_filters)
            params_to_get_grad_names.extend(ner_filters_names)
            if not self.static:
                params_to_get_grad.append(w_ner_x)
                params_to_get_grad_names.append('w_ner_x')
                params_to_get_l2.append('w_ner_x')

            self.w_ner_x = w_ner_x

        if self.using_sent_nr_convolution_maxpool_feature():
            w1_sent_nr = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                n_in=np.max(self.train_sent_nr_feats)+1, n_out=self.n_sent_nr_emb, function='tanh').astype(dtype=theano.config.floatX),
                                   name='w1_sent_nr', borrow=True)

            # index embeddings
            w_sent_nr_x = w1_sent_nr[sent_nr_id]

            # add structures to self.params
            self.params['w1_sent_nr'] = w1_sent_nr

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_sent_nr_x)
                params_to_get_grad_names.append('w_sent_nr_x')
                params_to_get_l2.append('w_sent_nr_x')

            self.w_sent_nr_x = w_sent_nr_x

        if self.using_tense_convolution_maxpool_feature():
            w1_tense = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                n_in=np.max(self.train_tense_feats)+1, n_out=self.n_tense_emb, function='tanh').astype(dtype=theano.config.floatX),
                                   name='w1_tense', borrow=True)

            # index embeddings
            w_tense_x = w1_tense[tense_id]

            # add structures to self.params
            self.params['w1_tense'] = w1_tense

            if not self.static:
                # learn word_embeddings
                params_to_get_grad.append(w_tense_x)
                params_to_get_grad_names.append('w_tense_x')
                params_to_get_l2.append('w_tense_x')

            self.w_tense_x = w_tense_x

        self.params_to_get_l2 = params_to_get_l2

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            L2_w2v_nc_nmp_weight, L2_w2v_c_mp_filters, L2_w_x, L2_w2v_c_nmp_filters, \
            L2_w2v_c_nmp_weight, L2_pos_c_mp_filters, L2_w_pos_x, L2_ner_c_mp_filters, L2_w_ner_x, \
            L2_w_sent_nr_x, L2_w_sent_nr_filters, L2_w_tense_x, L2_w_tense_filters, L2_w3 = self.compute_regularization_cost()

            L2_w4 = T.sum(self.params['w4'] ** 2)

            L2 = L2_w2v_nc_nmp_weight + L2_w2v_c_mp_filters + L2_w_x + L2_w2v_c_nmp_filters + \
                 L2_w2v_c_nmp_weight + L2_pos_c_mp_filters + L2_w_pos_x + L2_ner_c_mp_filters + \
                 L2_w_ner_x + L2_w_sent_nr_x + L2_w_sent_nr_filters + L2_w_tense_x + L2_w_tense_filters + \
                 L2_w3 + L2_w3 + L2_w4

        W1_tense_sum, W1_sent_nr_sum, W1_ner_sum, W1_pos_sum, W1_w2v_sum, W1_c_nmp_sum, W1_sum = self.compute_weight_evolution()

        out = self.sgd_forward_pass_two_layers(tensor_dim=2)

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))
        if self.regularization:
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = mean_cross_entropy + self.alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        # cost_prediction = mean_cross_entropy + alpha_L2_reg * L2
        # cost_prediction = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # cost_prediction = alpha_L2_reg*L2

        errors = T.sum(T.neq(y_predictions, y))

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        # test = theano.function([w2v_idxs, pos_idxs, y], [out, grads[0]])
        # test(self.x_train[0], self.train_pos_feats[0], self.y_train[0])

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1_w2v'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_pos_x':
                eps = np.zeros_like(self.params['w1_pos'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_ner_x':
                eps = np.zeros_like(self.params['w1_ner'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_x_directly':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_x_c_nmp':
                eps = np.zeros_like(self.params['w1_c_nmp'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_sent_nr_x':
                eps = np.zeros_like(self.params['w1_sent_nr'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_tense_x':
                eps = np.zeros_like(self.params['w1_tense'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads, accumulated_grad):
            if name == 'w_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_w2v'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_pos_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[pos_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[pos_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_pos'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_ner_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[ner_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[ner_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_ner'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_x_directly':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_x_c_nmp':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[w2v_idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[w2v_idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_c_nmp'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_sent_nr_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[sent_nr_id],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[sent_nr_id])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_sent_nr'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            elif name == 'w_tense_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[tense_id],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[tense_id])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_tense'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        train_idx = T.scalar(name="train_idx", dtype=INT)
        valid_idx = T.scalar(name="valid_idx", dtype=INT)

        train_givens = dict()
        valid_givens = dict()
        if self.using_w2v_feature():
            train_givens.update({
                w2v_idxs: train_x[train_idx]
            })
            valid_givens.update({
                w2v_idxs: valid_x[valid_idx]
            })

        if self.using_pos_feature():
            train_givens.update({
                pos_idxs: train_pos_x[train_idx]
            })
            valid_givens.update({
                pos_idxs: valid_pos_x[valid_idx]
            })

        if self.using_ner_feature():
            train_givens.update({
                ner_idxs: train_ner_x[train_idx]
            })
            valid_givens.update({
                ner_idxs: valid_ner_x[valid_idx]
            })

        if self.using_sent_nr_feature():
            train_givens.update({
                sent_nr_id: train_sent_nr_x[train_idx]
            })
            valid_givens.update({
                sent_nr_id: valid_sent_nr_x[valid_idx]
            })

        if self.using_tense_feature():
            train_givens.update({
                tense_id: train_tense_x[train_idx]
            })
            valid_givens.update({
                tense_id: valid_tense_x[valid_idx]
            })

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost, errors],
                                updates=updates,
                                givens=train_givens)

        train_get_cross_entropy = theano.function(inputs=[train_idx, y],
                                            outputs=mean_cross_entropy,
                                            updates=[],
                                            givens=train_givens)

        valid_get_cross_entropy = theano.function(inputs=[valid_idx, y],
                                            outputs=mean_cross_entropy,
                                            updates=[],
                                            givens=valid_givens)

        train_predict = theano.function(inputs=[valid_idx, y],
                                        outputs=[cost, errors, y_predictions],
                                        updates=[],
                                        on_unused_input='ignore',
                                        givens=valid_givens)

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[W1_tense_sum, W1_sent_nr_sum, W1_ner_sum, W1_pos_sum, W1_w2v_sum,
                                                        W1_c_nmp_sum, W1_sum, L2_w2v_c_mp_filters, L2_w2v_c_nmp_filters,
                                                        L2_pos_c_mp_filters, L2_ner_c_mp_filters, L2_w3, L2_w4],
                                               givens=[])

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        epoch_l2_w2v_nc_nmp_weight_list = []
        epoch_l2_w2v_c_mp_filters_list = []
        epoch_l2_w2v_c_mp_weight_list = []
        epoch_l2_w2v_c_nmp_filters_list = []
        epoch_l2_w2v_c_nmp_weight_list = []
        epoch_l2_pos_filters_list = []
        epoch_l2_pos_weight_list = []
        epoch_l2_ner_filters_list = []
        epoch_l2_ner_weight_list = []
        epoch_l2_sent_nr_weight_list = []
        epoch_l2_tense_weight_list = []
        epoch_l2_w3_list = []
        epoch_l2_w4_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        last_valid_errors = np.inf

        for epoch_index in range(max_epochs):
            start = time.time()
            train_cost = 0
            train_errors = 0
            epoch_l2_w2v_nc_nmp_weight = 0
            epoch_l2_w2v_c_nmp_filters = 0
            epoch_l2_w2v_c_mp_filters = 0
            epoch_l2_w2v_c_mp_weight = 0
            epoch_l2_w2v_c_nmp_weight = 0
            epoch_l2_pos_filters = 0
            epoch_l2_pos_weight = 0
            epoch_l2_ner_filters = 0
            epoch_l2_ner_weight = 0
            epoch_l2_sent_nr_weight = 0
            epoch_l2_tense_weight = 0
            epoch_l2_w3 = 0
            epoch_l2_w4 = 0
            train_cross_entropy = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i, [train_y.get_value()[i]])
                train_cost += cost_output
                train_errors += errors_output
                train_cross_entropy += train_get_cross_entropy(i, [train_y.get_value()[i]])

            if self.regularization:
                w1_tense_sum, w1_sent_nr_sum, w1_ner_sum, w1_pos_sum, w1_w2v_sum, w1_c_nmp_sum, w1_sum, l2_w2v_c_mp_filters,\
                l2_w2v_c_nmp_filters, l2_pos_c_mp_filters, l2_ner_c_mp_filters, l2_w3, l2_w4 = train_l2_penalty()

                epoch_l2_w2v_nc_nmp_weight += w1_sum
                epoch_l2_w2v_c_mp_filters += l2_w2v_c_mp_filters
                epoch_l2_w2v_c_mp_weight += w1_w2v_sum
                epoch_l2_w2v_c_nmp_filters += l2_w2v_c_nmp_filters
                epoch_l2_w2v_c_nmp_weight += w1_c_nmp_sum
                epoch_l2_pos_filters += l2_pos_c_mp_filters
                epoch_l2_pos_weight += w1_pos_sum
                epoch_l2_ner_filters += l2_ner_c_mp_filters
                epoch_l2_ner_weight += w1_ner_sum
                epoch_l2_sent_nr_weight += w1_sent_nr_sum
                epoch_l2_tense_weight += w1_tense_sum
                epoch_l2_w3 += l2_w3
                epoch_l2_w4 += l2_w4

                # self.predict(on_validation_set=True, compute_cost_error=True)

            valid_errors = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            start1 = time.time()
            for i, y_sample in enumerate(self.y_valid):
                # cost_output = 0 #TODO: in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                cost_output, errors_output, pred = train_predict(i, [y_sample])
                valid_cost += cost_output
                valid_errors += errors_output
                valid_predictions.append(np.asscalar(pred))
                valid_cross_entropy += valid_get_cross_entropy(i, [y_sample])

            print time.time()-start1
            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_errors)

            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            epoch_l2_w2v_nc_nmp_weight_list.append(epoch_l2_w2v_nc_nmp_weight)
            epoch_l2_w2v_c_mp_filters_list.append(epoch_l2_w2v_c_mp_filters)
            epoch_l2_w2v_c_mp_weight_list.append(epoch_l2_w2v_c_mp_weight)
            epoch_l2_w2v_c_nmp_filters_list.append(epoch_l2_w2v_c_nmp_filters)
            epoch_l2_w2v_c_nmp_weight_list.append(epoch_l2_w2v_c_nmp_weight)
            epoch_l2_pos_filters_list.append(epoch_l2_pos_filters)
            epoch_l2_pos_weight_list.append(epoch_l2_pos_weight)
            epoch_l2_ner_filters_list.append(epoch_l2_ner_filters)
            epoch_l2_ner_weight_list.append(epoch_l2_ner_weight)
            epoch_l2_sent_nr_weight_list.append(epoch_l2_sent_nr_weight)
            epoch_l2_tense_weight_list.append(epoch_l2_tense_weight)
            epoch_l2_w3_list.append(epoch_l2_w3)
            epoch_l2_w4_list.append(epoch_l2_w4)

            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, train_cost, train_errors, valid_cost, valid_errors, f1_score, end-start))

            if lr_decay and (valid_errors > last_valid_errors):
                logger.info('Changing learning rate from %f to %f' % (learning_rate, .5*learning_rate))
                learning_rate *= .5

            last_valid_errors = valid_errors

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list, actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)

            plot_data_dict = {
                'w2v_emb': epoch_l2_w2v_nc_nmp_weight_list,
                'w2v_c_mp_filters': epoch_l2_w2v_c_mp_filters_list,
                'w2v_c_mp_emb': epoch_l2_w2v_c_mp_weight_list,
                'w2v_c_nmp_filters': epoch_l2_w2v_c_nmp_filters_list,
                'w2v_c_nmp_emb': epoch_l2_w2v_c_nmp_weight_list,
                'pos_c_mp_filters': epoch_l2_pos_filters_list,
                'pos_c_mp_emb': epoch_l2_pos_weight_list,
                'ner_c_mp_filters': epoch_l2_ner_filters_list,
                'ner_c_mp_emb': epoch_l2_ner_weight_list,
                'sent_nr_c_mp_emb': epoch_l2_sent_nr_weight_list,
                'tense_c_mp_emb': epoch_l2_tense_weight_list,
                'w3': epoch_l2_w3_list,
                'w4': epoch_l2_w4_list
            }
            self.plot_penalties_general(plot_data_dict, actual_time=actual_time)

            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        return True

    def compute_weight_evolution(self):
        w1_tense_sum = T.constant(0., dtype=theano.config.floatX)
        w1_sent_nr_sum = T.constant(0., dtype=theano.config.floatX)
        w1_ner_sum = T.constant(0., dtype=theano.config.floatX)
        w1_pos_sum = T.constant(0., dtype=theano.config.floatX)
        w1_w2v_sum = T.constant(0., dtype=theano.config.floatX)
        w1_c_nmp_sum = T.constant(0., dtype=theano.config.floatX)
        w1_sum = T.constant(0., dtype=theano.config.floatX)

        if self.using_tense_convolution_maxpool_feature():
            w1_tense_sum = T.sum(self.params['w1_tense'] ** 2)
        if self.using_sent_nr_convolution_maxpool_feature():
            w1_sent_nr_sum = T.sum(self.params['w1_sent_nr'] ** 2)
        if self.using_ner_convolution_maxpool_feature():
            w1_ner_sum = T.sum(self.params['w1_ner'] ** 2)
        if self.using_pos_convolution_maxpool_feature():
            w1_pos_sum = T.sum(self.params['w1_pos'] ** 2)
        if self.using_w2v_convolution_maxpool_feature():
            w1_w2v_sum = T.sum(self.params['w1_w2v'] ** 2)
        if self.using_w2v_convolution_no_maxpool_feature():
            w1_c_nmp_sum = T.sum(self.params['w1_c_nmp'] ** 2)
        if self.using_w2v_no_convolution_no_maxpool_feature():
            w1_sum = T.sum(self.params['w1'] ** 2)

        return w1_tense_sum, w1_sent_nr_sum, w1_ner_sum, w1_pos_sum, w1_w2v_sum, w1_c_nmp_sum, w1_sum

    def compute_regularization_cost(self):
        # symbolic Theano variable that represents the squared L2 term

        # L2 on w2v no-convolution no-maxpool
        L2_w2v_nc_nmp_weight = self.L2_w2v_noconvolution_maxpool_weight()

        # L2 on w2v convolution maxpool filters
        L2_w2v_c_mp_filters = self.L2_w2v_convolution_maxpool_filters()
        L2_w2v_c_mp_weight = self.L2_w2v_convolution_maxpool_weight()

        # L2 on w2v convolution no-maxpool
        L2_w2v_c_nmp_filters = self.L2_w2v_convolution_nomaxpool_filters()
        L2_w2v_c_nmp_weight = self.L2_w2v_convolution_nomaxpool_weight()

        # L2 on pos convolution maxpool
        L2_pos_c_mp_filters = self.L2_pos_convolution_maxpool_filters()
        L2_pos_c_mp_weight = self.L2_pos_convolution_maxpool_weight()

        # L2 on ner convolution maxpool
        L2_ner_c_mp_filters = self.L2_ner_convolution_maxpool_filters()
        L2_ner_c_mp_weight = self.L2_ner_convolution_maxpool_weight()

        # L2 on sent_nr convolution maxpool
        L2_sent_nr_c_m_filters = self.L2_sent_nr_convolution_maxpool_filters()
        L2_sent_nr_c_m_weight = self.L2_sent_nr_convolution_maxpool()

        L2_tense_c_m_filters = self.L2_tense_convolution_maxpool_filters()
        L2_tense_c_m_weight = self.L2_tense_convolution_maxpool()

        L2_w3 = T.sum(self.params['w3'] ** 2)

        return L2_w2v_nc_nmp_weight, L2_w2v_c_mp_filters, L2_w2v_c_mp_weight, L2_w2v_c_nmp_filters, \
               L2_w2v_c_nmp_weight, L2_pos_c_mp_filters, L2_pos_c_mp_weight, L2_ner_c_mp_filters, L2_ner_c_mp_weight, \
               L2_sent_nr_c_m_weight, L2_sent_nr_c_m_filters, L2_tense_c_m_weight, L2_tense_c_m_filters, L2_w3

    def L2_tense_convolution_maxpool(self):
        tense_emb = []

        if self.using_tense_convolution_maxpool_feature() and not self.static:
            tense_emb.append(T.sum(self.w_tense_x ** 2))

        return T.sum(tense_emb, dtype=theano.config.floatX)

    def L2_tense_convolution_maxpool_filters(self):
        filter_penalties = []

        if self.using_tense_convolution_maxpool_feature():

            filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('tense_filter')]
            for filter_name in filter_names:
                filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(filter_penalties, dtype=theano.config.floatX)

    def L2_sent_nr_convolution_maxpool_filters(self):
        filter_penalties = []

        if self.using_sent_nr_convolution_maxpool_feature():

            filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('sent_nr_filter')]
            for filter_name in filter_names:
                filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(filter_penalties, dtype=theano.config.floatX)

    def L2_sent_nr_convolution_maxpool(self):
        sent_nr_emb = []

        if self.using_sent_nr_convolution_maxpool_feature() and not self.static:
            sent_nr_emb.append(T.sum(self.w_sent_nr_x ** 2))

        return T.sum(sent_nr_emb, dtype=theano.config.floatX)

    def L2_pos_convolution_maxpool_filters(self):
        pos_filter_penalties = []

        if self.using_pos_convolution_maxpool_feature():

            pos_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('pos_filter')]
            for filter_name in pos_filter_names:
                pos_filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(pos_filter_penalties, dtype=theano.config.floatX)

    def L2_pos_convolution_maxpool_weight(self):
        pos_emb = []

        if self.using_pos_convolution_maxpool_feature() and not self.static:
            pos_emb.append(T.sum(self.w_pos_x ** 2))

        return T.sum(pos_emb, dtype=theano.config.floatX)

    def L2_ner_convolution_maxpool_filters(self):
        ner_filter_penalties = []
        if self.using_ner_convolution_maxpool_feature():

            ner_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('ner_filter')]
            for filter_name in ner_filter_names:
                ner_filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(ner_filter_penalties, dtype=theano.config.floatX)

    def L2_ner_convolution_maxpool_weight(self):
        ner_emb = []

        if self.using_ner_convolution_maxpool_feature() and not self.static:
            ner_emb.append(T.sum(self.w_ner_x ** 2))

        return T.sum(ner_emb, dtype=theano.config.floatX)

    def L2_w2v_convolution_nomaxpool_filters(self):
        w2v_filter_penalties = []

        if self.using_w2v_convolution_no_maxpool_feature():

            w2v_dir_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('w2v_directly_filter')]
            for filter_name in w2v_dir_filter_names:
                w2v_filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(w2v_filter_penalties, dtype=theano.config.floatX)

    def L2_w2v_convolution_nomaxpool_weight(self):
        w2v_emb = []

        if self.using_w2v_convolution_no_maxpool_feature() and not self.static:
            w2v_emb.append(T.sum(self.w_x_c_nmp ** 2))

        return T.sum(w2v_emb, dtype=theano.config.floatX)

    def L2_w2v_convolution_maxpool_filters(self):
        w2v_conv_filter_penalties = []

        if self.using_w2v_convolution_maxpool_feature():
            w2v_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('w2v_filter')]
            for filter_name in w2v_filter_names:
                w2v_conv_filter_penalties.append(T.sum(self.params[filter_name] ** 2))

        return T.sum(w2v_conv_filter_penalties, dtype=theano.config.floatX)

    def L2_w2v_convolution_maxpool_weight(self):
        w2v_emb = []

        if self.using_w2v_convolution_maxpool_feature() and not self.static:
            w2v_emb.append(T.sum(self.w_x ** 2))

        return T.sum(w2v_emb, dtype=theano.config.floatX)

    def L2_w2v_noconvolution_maxpool_weight(self):
        w2v_emb = []

        if self.using_w2v_no_convolution_no_maxpool_feature() and not self.static:
            w2v_emb.append(T.sum(self.w_x_directly ** 2))

        return T.sum(w2v_emb, dtype=theano.config.floatX)

    def minibatch_forward_pass(self, weight_x, bias_1, weight_2, bias_2):
        h = self.hidden_activation_f(weight_x+bias_1)
        return self.out_activation_f(T.dot(h,weight_2)+bias_2)

    def train_with_minibatch(self, learning_rate=0.01, batch_size=512, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_scan=False, **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)
        # x = T.matrix(name='x', dtype=theano.config.floatX)
        minibatch_idx = T.scalar('minibatch_idx', dtype=INT)  # minibatch index

        idxs = T.matrix(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace n_win with self.n_win
        # n_features = train_x.get_value().shape[1]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=n_window*self.n_emb, n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((n_window*self.n_emb)).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        params = [w1,b1,w2,b2]
        param_names = ['w1','b1','w2','b2']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, self.n_emb*n_window))

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            L1 = T.sum(abs(w1))

            # symbolic Theano variable that represents the squared L2 term
            L2 = T.sum(w1 ** 2) + T.sum(w2 ** 2)

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.minibatch_forward_pass,
                                    sequences=[w_x],
                                    outputs_info=None,
                                    non_sequences=[b1,w2,b2])

            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.minibatch_forward_pass(w_x,b1,w2,b2)

            if self.regularization:
                # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
                cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out, y))

            y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        grads = [T.grad(cost, param) for param in params]

        # adagrad
        accumulated_grad = []
        for param in params:
            eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for param, grad, accum_grad in zip(params, grads, accumulated_grad):
            # accum = T.cast(accum_grad + T.sqr(grad), dtype=theano.config.floatX)
            accum = accum_grad + T.sqr(grad)
            updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
            updates.append((accum_grad, accum))

        train = theano.function(inputs=[minibatch_idx],
                                outputs=[cost,errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
                                    y: train_y[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size]
                                })

        for epoch_index in range(max_epochs):
            epoch_cost = 0
            epoch_errors = 0
            for minibatch_index in range(self.x_train.shape[0]/batch_size):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(minibatch_index)
                epoch_cost += cost_output
                epoch_errors += errors_output
            logger.info('Epoch %d Cost: %f Errors: %d' % (epoch_index+1, epoch_cost, epoch_errors))

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def perform_nnet_word_embeddings_conv2d(self, w_x_4D, region_sizes, max_pool, filter_prefix='w2v_filter_'):
        """
        it performs the convolution on 4D tensors.
        Calls theano.tensor.nnet.conv2d()
        :return:
        """
        convolutions = []

        for rs in region_sizes:
            w2v_filter = self.params[filter_prefix + str(rs)]

            w2v_filter_4D = w2v_filter.reshape(
                shape=(w2v_filter.shape[0], 1, w2v_filter.shape[1], w2v_filter.shape[2]))

            w2v_convolution = T.nnet.conv2d(input=w_x_4D,
                                       filters=w2v_filter_4D)

            if max_pool:
                w2v_max_convolution = pool_2d(input=w2v_convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(w2v_max_convolution)
            else:
                convolutions.append(w2v_convolution)

        if max_pool:
            conv_conc = self.concatenate(convolutions, axis=1)[:, :, 0, 0]
        else:
            conv_conc = self.concatenate(convolutions, axis=2).reshape(shape=(w_x_4D.shape[0], -1))

        return conv_conc

    def perform_nnet_pos_conv2d(self, w_pos_x_4D):
        convolutions = []

        for rs in self.region_sizes:
            pos_filter = self.params['pos_filter_'+str(rs)]

            pos_filter_4D = pos_filter.reshape(
                shape=(pos_filter.shape[0], 1, pos_filter.shape[1], pos_filter.shape[2]))

            pos_convolution = T.nnet.conv2d(input=w_pos_x_4D,
                                            filters=pos_filter_4D)

            if self.max_pool:
                pos_max_convolution = pool_2d(input=pos_convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(pos_max_convolution)
            else:
                convolutions.append(pos_convolution)

        if self.max_pool:
            conv_conc = self.concatenate(convolutions, axis=1)[:, :, 0, 0]
        else:
            conv_conc = self.concatenate(convolutions, axis=2)[:,:,:,0].reshape(shape=(w_pos_x_4D.shape[0], -1))

        return conv_conc

    def perform_nnet_ner_conv2d(self, w_ner_x_4D):
        convolutions = []

        for rs in self.region_sizes:
            ner_filter = self.params['ner_filter_'+str(rs)]

            ner_filter_4D = ner_filter.reshape(
                shape=(ner_filter.shape[0], 1, ner_filter.shape[1], ner_filter.shape[2]))

            pos_convolution = T.nnet.conv2d(input=w_ner_x_4D,
                                            filters=ner_filter_4D)

            if self.max_pool:
                pos_max_convolution = pool_2d(input=pos_convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(pos_max_convolution)
            else:
                convolutions.append(pos_convolution)

        if self.max_pool:
            conv_conc = self.concatenate(convolutions, axis=1)[:, :, 0, 0]
        else:
            conv_conc = self.concatenate(convolutions, axis=2)[:,:,:,0].reshape(shape=(w_ner_x_4D.shape[0], -1))

        return conv_conc

    def perform_nnet_sent_nr_conv2d(self, w_sent_nr_x_4D):
        convolutions = []

        for rs in self.region_sizes:
            filter = self.params['sent_nr_filter_'+str(rs)]

            filter_4D = filter.reshape(
                shape=(filter.shape[0], 1, filter.shape[1], filter.shape[2]))

            sent_nr_convolution = T.nnet.conv2d(input=w_sent_nr_x_4D, filters=filter_4D)

            if self.max_pool:
                max_convolution = pool_2d(input=sent_nr_convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_convolution)
            else:
                convolutions.append(sent_nr_convolution)

        if self.max_pool:
            conv_conc = self.concatenate(convolutions, axis=1)[:, :, 0, 0]
        else:
            conv_conc = self.concatenate(convolutions, axis=2)[:,:,:,0].reshape(shape=(w_sent_nr_x_4D.shape[0], -1))

        return conv_conc

    def perform_nnet_tense_conv2d(self, w_tense_x_4D):
        convolutions = []

        for rs in self.region_sizes:
            filter = self.params['tense_filter_'+str(rs)]

            filter_4D = filter.reshape(
                shape=(filter.shape[0], 1, filter.shape[1], filter.shape[2]))

            tense_convolution = T.nnet.conv2d(input=w_tense_x_4D, filters=filter_4D)

            if self.max_pool:
                max_convolution = pool_2d(input=tense_convolution, ds=(self.n_window-rs+1, 1), ignore_border=True, mode='max')
                convolutions.append(max_convolution)
            else:
                convolutions.append(tense_convolution)

        if self.max_pool:
            conv_conc = self.concatenate(convolutions, axis=1)[:, :, 0, 0]
        else:
            conv_conc = self.concatenate(convolutions, axis=2)[:,:,:,0].reshape(shape=(w_tense_x_4D.shape[0], -1))

        return conv_conc

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, compute_cost_error=False, **kwargs):

        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)

            # if self.using_pos_feature():
            #     x_pos_test = self.train_pos_feats.astype(dtype=INT)
            #
            # if self.using_ner_feature():
            #     x_ner_test = self.train_ner_feats.astype(dtype=INT)
            #
            # if self.using_sent_nr_feature():
            #     x_sent_nr_test = self.train_sent_nr_feats.astype(dtype=INT)
            #
            # if self.using_tense_feature():
            #     x_tense_test = self.train_tense_feats.astype(dtype=INT)

            y_test = self.y_train.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)

            # if self.using_pos_feature():
            #     x_pos_test = self.valid_pos_feats.astype(dtype=INT)
            #
            # if self.using_ner_feature():
            #     x_ner_test = self.valid_ner_feats.astype(dtype=INT)
            #
            # if self.using_sent_nr_feature():
            #     x_sent_nr_test = self.valid_sent_nr_feats.astype(dtype=INT)
            #
            # if self.using_tense_feature():
            #     x_tense_test = self.valid_tense_feats.astype(dtype=INT)

            y_test = self.y_valid.astype(dtype=INT)
        elif on_testing_set:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)

            # if self.using_pos_feature():
            #     x_pos_test = self.test_pos_feats.astype(dtype=INT)
            #
            # if self.using_ner_feature():
            #     x_ner_test = self.test_ner_feats.astype(dtype=INT)
            #
            # if self.using_sent_nr_feature():
            #     x_sent_nr_test = self.test_sent_nr_feats.astype(dtype=INT)
            #
            # if self.using_tense_feature():
            #     x_tense_test = self.test_tense_feats.astype(dtype=INT)

            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        y = T.vector(name='test_y', dtype=INT)

        w2v_idxs = T.matrix(name="test_w2v_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        pos_idxs = T.matrix(name="test_pos_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        ner_idxs = T.matrix(name="test_ner_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        sent_nr_idxs = T.matrix(name="test_sent_nr_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        tense_idxs = T.matrix(name="test_tense_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        if self.using_w2v_convolution_maxpool_feature():
            self.w_x = self.params['w1_w2v'][w2v_idxs]

        if self.using_w2v_convolution_no_maxpool_feature():
            self.w_x_c_nmp = self.params['w1_c_nmp'][w2v_idxs]

        if self.using_w2v_no_convolution_no_maxpool_feature():
            self.w_x_directly = self.params['w1'][w2v_idxs]

        if self.using_pos_convolution_maxpool_feature():
            self.w_pos_x = self.params['w1_pos'][pos_idxs]

        if self.using_ner_convolution_maxpool_feature():
            self.w_ner_x = self.params['w1_ner'][ner_idxs]

        if self.using_sent_nr_convolution_maxpool_feature():
            self.w_sent_nr_x = self.params['w1_sent_nr'][sent_nr_idxs]

        if self.using_tense_convolution_maxpool_feature():
            self.w_tense_x = self.params['w1_tense'][tense_idxs]

        givens = dict()
        if self.using_w2v_feature():
            givens.update({
                w2v_idxs: x_test
            })

        if self.using_pos_feature():
            givens.update({
                # pos_idxs: x_pos_test
                pos_idxs: x_test
            })

        if self.using_ner_feature():
            givens.update({
                # ner_idxs: x_ner_test
                ner_idxs: x_test
            })

        if self.using_sent_nr_feature():
            givens.update({
                # sent_nr_idxs: x_sent_nr_test
                sent_nr_idxs: x_test
            })

        if self.using_tense_feature():
            givens.update({
                # tense_idxs: x_tense_test
                tense_idxs: x_test
            })


        #TODO: choose
        # out = self.perform_forward_pass_dense_step(w2v_conc, pos_conc)
        out = self.sgd_forward_pass(tensor_dim=4)

        y_predictions = T.argmax(out, axis=1)

        if compute_cost_error:
            errors = T.sum(T.neq(y, y_predictions))

            L2_w2v_nc_nmp_weight, L2_w2v_c_mp_filters, L2_w_x, L2_w2v_c_nmp_filters, \
            L2_w2v_c_nmp_weight, L2_pos_c_mp_filters, L2_w_pos_x, L2_ner_c_mp_filters, L2_w_ner_x, \
            L2_w_sent_nr_x, L2_w_sent_nr_filters, L2_w_tense_x, L2_w_tense_filters, L2_w3 = self.compute_regularization_cost()

            L2 = L2_w2v_nc_nmp_weight + L2_w2v_c_mp_filters + L2_w_x + L2_w2v_c_nmp_filters + \
                 L2_w2v_c_nmp_weight + L2_pos_c_mp_filters + L2_w_pos_x + L2_ner_c_mp_filters + \
                 L2_w_ner_x + L2_w_sent_nr_x + L2_w_sent_nr_filters + L2_w_tense_x + L2_w_tense_filters + L2_w3

            cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + self.alpha_L2_reg * L2

            perform_prediction = theano.function(inputs=[y],
                                                 outputs=[cost, errors, y_predictions],
                                                 on_unused_input='ignore',
                                                 givens=givens)

            out_cost, out_errors, out_predictions = perform_prediction(y_test)
            results['errors'] = out_errors
            results['cost'] = out_cost
        else:
            perform_prediction = theano.function(inputs=[],
                                                 outputs=y_predictions,
                                                 on_unused_input='ignore',
                                                 givens=givens)

            out_predictions = perform_prediction()

        results['flat_predictions'] = out_predictions
        results['flat_trues'] = y_test

        return results

    def predict_two_layers(self, on_training_set=False, on_validation_set=False, on_testing_set=False, compute_cost_error=False, **kwargs):

        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)

            if self.using_pos_feature():
                x_pos_test = self.train_pos_feats.astype(dtype=INT)

            if self.using_ner_feature():
                x_ner_test = self.train_ner_feats.astype(dtype=INT)

            if self.using_sent_nr_feature():
                x_sent_nr_test = self.train_sent_nr_feats.astype(dtype=INT)

            if self.using_tense_feature():
                x_tense_test = self.train_tense_feats.astype(dtype=INT)

            y_test = self.y_train.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)
            if self.using_pos_feature():
                x_pos_test = self.valid_pos_feats.astype(dtype=INT)

            if self.using_ner_feature():
                x_ner_test = self.valid_ner_feats.astype(dtype=INT)

            if self.using_sent_nr_feature():
                x_sent_nr_test = self.valid_sent_nr_feats.astype(dtype=INT)

            if self.using_tense_feature():
                x_tense_test = self.valid_tense_feats.astype(dtype=INT)

            y_test = self.y_valid.astype(dtype=INT)
        elif on_testing_set:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)

            if self.using_pos_feature():
                x_pos_test = self.test_pos_feats.astype(dtype=INT)

            if self.using_ner_feature():
                x_ner_test = self.test_ner_feats.astype(dtype=INT)

            if self.using_sent_nr_feature():
                x_sent_nr_test = self.test_sent_nr_feats.astype(dtype=INT)

            if self.using_tense_feature():
                x_tense_test = self.test_tense_feats.astype(dtype=INT)

            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        y = T.vector(name='test_y', dtype=INT)

        w2v_idxs = T.matrix(name="test_w2v_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        pos_idxs = T.matrix(name="test_pos_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        ner_idxs = T.matrix(name="test_ner_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        sent_nr_idxs = T.vector(name="test_sent_nr_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        tense_idxs = T.vector(name="test_tense_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        if self.using_w2v_convolution_maxpool_feature():
            self.w_x = self.params['w1_w2v'][w2v_idxs]

        if self.using_w2v_convolution_no_maxpool_feature():
            self.w_x_c_nmp = self.params['w1_c_nmp'][w2v_idxs]

        if self.using_w2v_no_convolution_no_maxpool_feature():
            self.w_x_directly = self.params['w1'][w2v_idxs]

        if self.using_pos_convolution_maxpool_feature():
            self.w_pos_x = self.params['w1_pos'][pos_idxs]

        if self.using_ner_convolution_maxpool_feature():
            self.w_ner_x = self.params['w1_ner'][ner_idxs]

        if self.using_sent_nr_convolution_maxpool_feature():
            self.w_sent_nr_x = self.params['w1_sent_nr'][sent_nr_idxs]

        if self.using_tense_convolution_maxpool_feature():
            self.w_tense_x = self.params['w1_tense'][tense_idxs]

        givens = dict()
        if self.using_w2v_feature():
            givens.update({
                w2v_idxs: x_test
            })

        if self.using_pos_feature():
            givens.update({
                pos_idxs: x_pos_test
            })

        if self.using_ner_feature():
            givens.update({
                ner_idxs: x_ner_test
            })

        if self.using_sent_nr_feature():
            givens.update({
                sent_nr_idxs: x_sent_nr_test
            })

        if self.using_tense_feature():
            givens.update({
                tense_idxs: x_tense_test
            })


        #TODO: choose
        # out = self.perform_forward_pass_dense_step(w2v_conc, pos_conc)
        out = self.sgd_forward_pass_two_layers(tensor_dim=4)

        y_predictions = T.argmax(out, axis=1)

        if compute_cost_error:
            errors = T.sum(T.neq(y, y_predictions))

            L2_w2v_nc_nmp_weight, L2_w2v_c_mp_filters, L2_w_x, L2_w2v_c_nmp_filters, \
            L2_w2v_c_nmp_weight, L2_pos_c_mp_filters, L2_w_pos_x, L2_ner_c_mp_filters, L2_w_ner_x, \
            L2_w_sent_nr_x, L2_w_sent_nr_filters, L2_w_tense_x, L2_w_tense_filters, L2_w3 = self.compute_regularization_cost()

            L2_w4 = T.sum(self.params['w4'] ** 2)

            L2 = L2_w2v_nc_nmp_weight + L2_w2v_c_mp_filters + L2_w_x + L2_w2v_c_nmp_filters + \
                 L2_w2v_c_nmp_weight + L2_pos_c_mp_filters + L2_w_pos_x + L2_ner_c_mp_filters + \
                 L2_w_ner_x + L2_w_sent_nr_x + L2_w_sent_nr_filters + L2_w_tense_x +L2_w_tense_filters + \
                 L2_w3 + L2_w3 + L2_w4

            cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + self.alpha_L2_reg * L2

            perform_prediction = theano.function(inputs=[y],
                                                 outputs=[cost, errors, y_predictions],
                                                 on_unused_input='ignore',
                                                 givens=givens)

            out_cost, out_errors, out_predictions = perform_prediction(y_test)
            results['errors'] = out_errors
            results['cost'] = out_cost
        else:
            perform_prediction = theano.function(inputs=[],
                                                 outputs=y_predictions,
                                                 on_unused_input='ignore',
                                                 givens=givens)

            out_predictions = perform_prediction()

        results['flat_predictions'] = out_predictions
        results['flat_trues'] = y_test

        return results

    def to_string(self):
        return 'Lexical-Semantical convolutional context window neural network with no tags.'