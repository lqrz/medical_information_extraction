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
        'w2v_c_nm': {'name': 'w2v', 'convolve': True, 'max_pool': False, 'use_cw': True, 'nr_filters': 1, 'filter_width': 1, 'nr_region_sizes': 1, 'crf_position': None},
        'w2v_nc_nm': {'name': 'w2v', 'convolve': False, 'max_pool': False, 'use_cw': True, 'nr_filters': 0, 'filter_width': 0, 'nr_region_sizes': 0, 'crf_position': None},
        'w2v_c_m': {'name': 'w2v', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': None},
        'pos_c_m': {'name': 'pos', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['pos']},
        'ner_c_m': {'name': 'ner', 'convolve': True, 'max_pool': True, 'use_cw': True, 'nr_filters': None, 'filter_width': None, 'nr_region_sizes': None, 'crf_position': CRF_POSITIONS['ner']},
        'sent_nr_nc_nm': {'name': 'sent_nr', 'convolve': False, 'max_pool': False, 'use_cw': False, 'nr_filters': 0, 'filter_width': 0, 'nr_region_sizes': 0, 'crf_position': None}
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
                 train_sent_nr_feats=None,
                 valid_sent_nr_feats=None,
                 test_sent_nr_feats=None,
                 region_sizes=None,
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

        #POS features
        try:
            train_pos_feats = np.array(train_feats[self.CRF_POSITIONS['pos']])
            valid_pos_feats = np.array(valid_feats[self.CRF_POSITIONS['pos']])
            test_pos_feats = np.array(test_feats[self.CRF_POSITIONS['pos']])
        except KeyError:
            train_pos_feats = None
            valid_pos_feats = None
            test_pos_feats = None

        self.train_pos_feats = train_pos_feats
        self.valid_pos_feats = valid_pos_feats
        self.test_pos_feats = test_pos_feats

        #NER features
        try:
            train_ner_feats = np.array(train_feats[self.CRF_POSITIONS['ner']])
            valid_ner_feats = np.array(valid_feats[self.CRF_POSITIONS['ner']])
            test_ner_feats = np.array(test_feats[self.CRF_POSITIONS['ner']])
        except KeyError:
            train_ner_feats = None
            valid_ner_feats = None
            test_ner_feats = None
        self.train_ner_feats = np.array(train_ner_feats)
        self.valid_ner_feats = np.array(valid_ner_feats)
        self.test_ner_feats = np.array(test_ner_feats)

        #sent_nr features
        self.train_sent_nr_feats = np.array(train_sent_nr_feats)
        self.valid_sent_nr_feats = np.array(valid_sent_nr_feats)
        self.test_sent_nr_feats = np.array(test_sent_nr_feats)

        self.alpha_L2_reg = None
        self.max_pool = None

        self.n_filters = n_filters  # number of filters per region size
        #TODO: choose one.
        self.n_pos_emb = np.max(self.train_pos_feats)+1 #encoding for the POS tags
        self.n_pos_emb = 40 #encoding for the POS tags

        self.n_ner_emb = 100 #encoding for the NER tags
        self.n_sent_nr_emb = 100 #encoding for the sent_nr tags

        self.region_sizes = region_sizes

        self.w2v_filter_width = self.n_emb
        self.pos_filter_width = self.n_pos_emb
        self.ner_filter_width = self.n_ner_emb

        self.concatenate = T.concatenate
        self.concatenate = utils.NeuralNetwork.theano_gpu_concatenate

        self.features_to_use = self.parse_features_to_use(features_to_use)

        self.features_dimensions = self.get_features_dimensions()

    def get_features_dimensions(self):
        d = dict()
        d['w2v'] = self.n_emb
        d['pos'] = self.n_pos_emb
        d['ner'] = self.n_ner_emb
        d['sent_nr'] = self.n_sent_nr_emb

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
        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)
        else:
            # train with SGD
            nnet_description = ' '.join(['Training with SGD',
                                         'context_win:', str(kwargs['window_size']),
                                         'statically' if kwargs['static'] else 'dynamically',
                                         'filters_per_region:', str(kwargs['n_filters']),
                                         'region_sizes:', '[', ','.join(map(str,kwargs['region_sizes'])), ']',
                                         'with' if kwargs['max_pool'] else 'without', 'max pooling'])
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

    def sgd_forward_pass(self, tensor_dim):

        hidden_state = self.compute_hidden_state(tensor_dim)

        return self.perform_forward_pass_dense_step(hidden_state, ndim=tensor_dim/2)

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
                w2v_directly = self.convolve_word_embeddings(self.w_x_directly,
                                                             filter_width=1,
                                                             n_filters=1,
                                                             region_sizes=[self.n_window],
                                                             max_pool=False,
                                                             filter_prefix='w2v_directly_filter_')
            elif tensor_dim == 4:
                w_x_directly_4D = self.w_x_directly.reshape(
                    shape=(self.w_x_directly.shape[0], 1, self.w_x_directly.shape[1], self.w_x_directly.shape[2]))
                w2v_directly = self.perform_nnet_word_embeddings_conv2d(w_x_directly_4D,
                                                                        region_sizes=[self.n_window],
                                                                        max_pool=False,
                                                                        filter_prefix='w2v_directly_filter_')

            hidden_state.append(w2v_directly)

        if self.using_w2v_convolution_maxpool_feature():
            if tensor_dim == 2:
                w2v_conv = self.convolve_word_embeddings(self.w_x,
                                                         filter_width=self.w2v_filter_width,
                                                         n_filters=self.n_filters,
                                                         region_sizes=self.region_sizes,
                                                         max_pool=self.max_pool)
            elif tensor_dim == 4:
                w_x_4D = self.w_x.reshape(shape=(self.w_x.shape[0], 1, self.w_x.shape[1], self.w_x.shape[2]))
                w2v_conv = self.perform_nnet_word_embeddings_conv2d(w_x_4D,
                                                                    region_sizes=self.region_sizes,
                                                                    max_pool=self.max_pool)

            hidden_state.append(w2v_conv)

        if self.using_pos_convolution_maxpool_feature():
            if tensor_dim == 2:
                pos_conv = self.convolve_pos_features(self.w_pos_x,
                                                      filter_width=self.pos_filter_width)
            elif tensor_dim == 4:
                w_pos_x_4D = self.w_pos_x.reshape(
                    shape=(self.w_pos_x.shape[0], 1, self.w_pos_x.shape[1], self.w_pos_x.shape[2]))
                pos_conv = self.perform_nnet_pos_conv2d(w_pos_x_4D)

            hidden_state.append(pos_conv)

        if self.using_ner_convolution_maxpool_feature():
            if tensor_dim == 2:
                ner_conv = self.convolve_ner_features(self.w_ner_x,
                                                      filter_width=self.ner_filter_width)
            elif tensor_dim == 4:
                w_ner_x_4D = self.w_ner_x.reshape(
                    shape=(self.w_ner_x.shape[0], 1, self.w_ner_x.shape[1], self.w_ner_x.shape[2]))
                ner_conv = self.perform_nnet_ner_conv2d(w_ner_x_4D)

            hidden_state.append(ner_conv)

        if self.using_sent_nr_no_convolution_no_maxpool_feature():
            if tensor_dim == 2:
                sent_nr_vector = self.w_sent_nr_x.reshape(shape=(-1,))
            elif tensor_dim == 4:
                sent_nr_vector = self.w_sent_nr_x.reshape(shape=(self.w_sent_nr_x.shape[0], -1))

            hidden_state.append(sent_nr_vector)

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

        assert filter_width is not None, 'Could not determine filter_width for feature: %s' % feature

        return filter_width

    def determine_nr_region_sizes(self, feature):
        nr_regions = self.__class__.FEATURE_MAPPING[feature]['nr_region_sizes']
        if not nr_regions:
            nr_regions = self.region_sizes.__len__()

        assert nr_regions is not None, 'Could not determine nr_region_sizes for feature: %s' % feature

        return nr_regions

    def determine_hidden_layer_size(self):
        size = 0

        for desc in self.features_to_use:

            feature = desc['name']
            convolve = desc['convolve']
            max_pool = desc['max_pool']
            c_window = desc['use_cw']

            if max_pool:
                feat_dimension = 1
            else:
                feat_dimension = self.features_dimensions[feature]

            if convolve:
                feat_extended = '_'.join([feature, 'c' if convolve else 'nc', 'm' if max_pool else 'nm'])
                window = 1
                n_filters = self.determine_nr_filters(feat_extended)
                n_regions = self.determine_nr_region_sizes(feat_extended)
            else:
                window = self.n_window
                n_regions = 1
                n_filters = 1

            if not c_window:
                window = 1

            size += feat_dimension * window * n_regions * n_filters

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

    def using_sent_nr_no_convolution_no_maxpool_feature(self):
        return self.using_feature(feature='sent_nr', convolve=False, max_pool=False)

    def train_with_sgd(self, learning_rate=0.01, max_epochs=100,
                       alpha_L1_reg=0.001, alpha_L2_reg=0.01,
                       save_params=False, plot=False,
                       static=False, max_pool=True,
                       **kwargs):

        self.alpha_L2_reg = alpha_L2_reg
        self.max_pool = max_pool

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

        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        if self.using_w2v_feature():
            train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)

        if self.using_pos_feature():
            # shared variable with training POS features
            train_pos_x = theano.shared(value=np.array(self.train_pos_feats, dtype=INT), name='train_pos_x', borrow=True)

        if self.using_ner_feature():
            # shared variable with training POS features
            train_ner_x = theano.shared(value=np.array(self.train_ner_feats, dtype=INT), name='train_ner_x', borrow=True)

        if self.using_sent_nr_feature():
            train_sent_nr_x = theano.shared(value=np.array(self.train_sent_nr_feats, dtype=INT), name='train_sent_nr_x', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)

        # create dense layer and bias
        hidden_layer_size = self.determine_hidden_layer_size()
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

            if not static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_directly)
                params_to_get_grad_names.append('w_x_directly')

            self.w_x_directly = w_x_directly

        if self.using_w2v_convolution_no_maxpool_feature():
            # word embeddings to be used directly.
            w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                               name='w1', borrow=True)

            # create w2v filters
            w2v_directly_filters = self.create_filters(filter_width=1, region_sizes=[self.n_window],
                                                       n_filters=1)
            w2v_directly_filters_names = map(lambda x: 'w2v_directly_%s' % x.name, w2v_directly_filters)

            # index embeddings
            w_x_directly = w1[w2v_idxs]

            # add structures to self.params
            self.params['w1'] = w1
            self.params.update(dict(zip(w2v_directly_filters_names, w2v_directly_filters)))
            params_to_get_l2.extend(w2v_directly_filters_names)
            params_to_get_grad.extend(w2v_directly_filters)
            params_to_get_grad_names.extend(w2v_directly_filters_names)

            if not static:
                # learn word_embeddings
                params_to_get_grad.append(w_x_directly)
                params_to_get_grad_names.append('w_x_directly')

            self.w_x_directly = w_x_directly

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

            if not static:
                # learn word_embeddings
                params_to_get_grad.append(w_x)
                params_to_get_grad_names.append('w_x')

            self.w_x = w_x

        if self.using_pos_convolution_maxpool_feature():
            # create POS embeddings
            #TODO: one-hot or random?
            w1_pos = theano.shared(value=np.eye(np.max(self.train_pos_feats)+1, self.n_pos_emb).astype(dtype=theano.config.floatX), name='w1_pos', borrow=True)
            # w1_pos = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            #     n_in=np.max(self.train_pos_feats)+1, n_out=self.n_pos_emb, function='tanh').astype(dtype=theano.config.floatX), name='w1_pos', borrow=True)

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
            params_to_get_grad.append(w_pos_x)
            params_to_get_grad_names.append('w_pos_x')
            params_to_get_grad.extend(pos_filters)
            params_to_get_grad_names.extend(pos_filters_names)

            self.w_pos_x = w_pos_x

        if self.using_ner_convolution_maxpool_feature():
            w1_ner = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                n_in=np.max(self.train_ner_feats)+1, n_out=self.n_ner_emb, function='tanh').astype(dtype=theano.config.floatX),
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
            params_to_get_grad.append(w_ner_x)
            params_to_get_grad_names.append('w_ner_x')
            params_to_get_grad.extend(ner_filters)
            params_to_get_grad_names.extend(ner_filters_names)

            self.w_ner_x = w_ner_x

        if self.using_sent_nr_no_convolution_no_maxpool_feature():
            w1_sent_nr = theano.shared(value=utils.NeuralNetwork.initialize_weights(
                n_in=np.max(self.train_sent_nr_feats)+1, n_out=self.n_sent_nr_emb, function='tanh').astype(dtype=theano.config.floatX),
                                   name='w1_sent_nr', borrow=True)

            # index embeddings
            w_sent_nr_x = w1_sent_nr[sent_nr_id]

            # add structures to self.params
            self.params['w1_sent_nr'] = w1_sent_nr

            if not static:
                # learn word_embeddings
                params_to_get_grad.append(w_sent_nr_x)
                params_to_get_grad_names.append('w_sent_nr_x')

            self.w_sent_nr_x = w_sent_nr_x


        self.params_to_get_l2 = params_to_get_l2

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            L2_w2v_emb, L2_pos_emb, L2_w1_emb, L2_pos_filters, L2_w2v_filters, \
            L2_w3, L2_w2v_dir_filters, L2_ner_emb, L2_ner_filters, L2_w1_sent_nr = \
                self.compute_regularization_cost(w2v_idxs, pos_idxs, ner_idxs, sent_nr_id)

            L2 = L2_w2v_emb + L2_pos_emb + L2_w1_emb + L2_pos_filters + L2_w2v_filters + \
                 L2_w3 + L2_w2v_dir_filters + L2_ner_emb + L2_ner_filters + L2_w1_sent_nr

        #TODO: choose
        # out = self.sgd_forward_pass(w_x, w_pos_x)
        # out = self.sgd_forward_pass_with_non_convolution(w_x, w_pos_x, w_x_directly_res)

        out = self.sgd_forward_pass(tensor_dim=2)

        if self.regularization:
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + self.alpha_L2_reg*L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out, y))

        y_predictions = T.argmax(out, axis=1)

        cost_prediction = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
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
            elif name == 'w_sent_nr_x':
                eps = np.zeros_like(self.params['w1_sent_nr'].get_value(), dtype=theano.config.floatX)
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
            elif name == 'w_sent_nr_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[sent_nr_id],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[sent_nr_id])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1_sent_nr'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        train_idx = T.scalar(name="train_idx", dtype=INT)

        givens = dict()
        if self.using_w2v_feature():
            givens.update({
                w2v_idxs: train_x[train_idx]
            })

        if self.using_pos_feature():
            givens.update({
                pos_idxs: train_pos_x[train_idx]
            })

        if self.using_ner_feature():
            givens.update({
                ner_idxs: train_ner_x[train_idx]
            })

        if self.using_sent_nr_feature():
            givens.update({
                sent_nr_id: train_sent_nr_x[train_idx]
            })

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost, errors],
                                updates=updates,
                                givens=givens)

        # theano.printing.debugprint(train)

        train_predict = theano.function(inputs=[train_idx, y],
                                        outputs=[cost, errors, y_predictions],
                                        updates=[],
                                        on_unused_input='ignore',
                                        givens=givens)

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[train_idx],
                                               outputs=[L2_w2v_emb, L2_pos_emb, L2_w1_emb, L2_pos_filters,
                                                        L2_w2v_filters, L2_w3, L2_w2v_dir_filters, L2_ner_emb,
                                                        L2_ner_filters, L2_w1_sent_nr],
                                               givens=givens)

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        l2_w2v_emb_list = []
        l2_w2v_filters_list = []
        l2_pos_filters_list = []
        l2_w3_list = []

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            epoch_l2_w2v_emb = 0
            epoch_l2_w2v_filters = 0
            epoch_l2_pos_filters = 0
            epoch_l2_w3 = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i, [train_y.get_value()[i]])
                epoch_cost += cost_output
                epoch_errors += errors_output
                if self.regularization:
                    l2_w2v_emb, l2_pos_emb, l2_w1_emb, l2_pos_filters, l2_w2v_filters, \
                    l2_w3, l2_w2v_dir_filters, l2_ner_emb, l2_ner_filters, l2_sent_nr_emb = train_l2_penalty(i)

                if i==0:
                    epoch_l2_w2v_emb = l2_w2v_emb
                epoch_l2_w2v_filters += l2_w2v_filters
                epoch_l2_pos_filters += l2_pos_filters
                epoch_l2_w3 += l2_w3

                # self.predict(on_validation_set=True, compute_cost_error=True)

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            # errors_output, pred = train_predict(self.x_valid, self.valid_pos_feats, self.y_valid)
            # start1 = time.time()
            # valid_results = self.predict(on_validation_set=True, compute_cost_error=True)
            # print time.time()-start1
            # valid_error_1 = valid_results['errors']
            # valid_cost_1 = valid_results['cost']
            # valid_predictions_1 = valid_results['predictions']
            start1 = time.time()
            for i, y_sample in enumerate(self.y_valid):
                # cost_output = 0 #TODO: in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                cost_output, errors_output, pred = train_predict(i, [y_sample])
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.append(np.asscalar(pred))
            print time.time()-start1
            train_costs_list.append(epoch_cost)
            train_errors_list.append(epoch_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w2v_emb_list.append(epoch_l2_w2v_emb)
            l2_w2v_filters_list.append(epoch_l2_w2v_filters)
            l2_pos_filters_list.append(epoch_l2_pos_filters)
            l2_w3_list.append(epoch_l2_w3)

            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, epoch_cost, epoch_errors, valid_cost, valid_error, f1_score, end-start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)

            #TODO: redo this plotting. special case.
            # self.plot_penalties(l2_w1_list, l2_w2_list, actual_time=actual_time)

        return True

    def compute_regularization_cost(self, w2v_idxs, pos_idxs, ner_idxs, sent_nr_id):
        # symbolic Theano variable that represents the squared L2 term

        w2v_conv_filter_penalties = []
        w2v_emb = []
        if self.using_w2v_convolution_maxpool_feature():
            w2v_emb.append(T.sum(self.params['w1_w2v'][w2v_idxs] ** 2))
            w2v_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('w2v_filter')]

            for filter_name in w2v_filter_names:
                w2v_conv_filter_penalties.append(T.sum(self.params[filter_name] ** 2))
        L2_w2v_filters = T.sum(w2v_conv_filter_penalties, dtype=theano.config.floatX)
        L2_w2v_emb = T.sum(w2v_emb, dtype=theano.config.floatX)

        pos_filter_penalties = []
        pos_emb = []
        if self.using_pos_convolution_maxpool_feature():
            pos_emb.append(T.sum(self.params['w1_pos'][pos_idxs] ** 2))
            pos_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('pos_filter')]

            for filter_name in pos_filter_names:
                pos_filter_penalties.append(T.sum(self.params[filter_name] ** 2))
        L2_pos_filters = T.sum(pos_filter_penalties, dtype=theano.config.floatX)
        L2_pos_emb = T.sum(pos_emb, dtype=theano.config.floatX)

        ner_filter_penalties = []
        ner_emb = []
        if self.using_ner_convolution_maxpool_feature():
            ner_emb.append(T.sum(self.params['w1_ner'][ner_idxs] ** 2))
            ner_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('ner_filter')]

            for filter_name in ner_filter_names:
                ner_filter_penalties.append(T.sum(self.params[filter_name] ** 2))
        L2_ner_filters = T.sum(ner_filter_penalties, dtype=theano.config.floatX)
        L2_ner_emb = T.sum(ner_emb, dtype=theano.config.floatX)

        w1_emb = []
        if self.using_w2v_no_convolution_no_maxpool_feature():
            w1_emb.append(T.sum(self.params['w1'][w2v_idxs] ** 2))

        w2v_filter_penalties = []
        if self.using_w2v_convolution_no_maxpool_feature():
            w1_emb.append(T.sum(self.params['w1'][w2v_idxs] ** 2))
            w2v_dir_filter_names = [pn for pn in self.params_to_get_l2 if pn.startswith('w2v_directly_filter')]

            for filter_name in w2v_dir_filter_names:
                w2v_filter_penalties.append(T.sum(self.params[filter_name] ** 2))
        L2_w2v_dir_filters = T.sum(w2v_filter_penalties, dtype=theano.config.floatX)
        L2_w1_emb = T.sum(w1_emb, dtype=theano.config.floatX)

        sent_nr_emb = []
        if self.using_sent_nr_no_convolution_no_maxpool_feature():
            sent_nr_emb.append(T.sum(self.params['w1_sent_nr'][sent_nr_id] ** 2))
        L2_w1_sent_nr = T.sum(sent_nr_emb, dtype=theano.config.floatX)

        L2_w3 = T.sum(self.params['w3'] ** 2)

        return L2_w2v_emb, L2_pos_emb, L2_w1_emb, L2_pos_filters, L2_w2v_filters, \
               L2_w3, L2_w2v_dir_filters, L2_ner_emb, L2_ner_filters, L2_w1_sent_nr

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

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, compute_cost_error=False, **kwargs):

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

            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        y = T.vector(name='test_y', dtype=INT)

        w2v_idxs = T.matrix(name="test_w2v_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        pos_idxs = T.matrix(name="test_pos_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        ner_idxs = T.matrix(name="test_ner_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        sent_nr_idxs = T.vector(name="test_sent_nr_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        if self.using_w2v_convolution_maxpool_feature():
            self.w_x = self.params['w1_w2v'][w2v_idxs]

        if self.using_w2v_convolution_no_maxpool_feature():
            self.w_x_directly = self.params['w1'][w2v_idxs]

        if self.using_w2v_no_convolution_no_maxpool_feature():
            self.w_x_directly = self.params['w1'][w2v_idxs]

        if self.using_pos_convolution_maxpool_feature():
            self.w_pos_x = self.params['w1_pos'][pos_idxs]

        if self.using_ner_convolution_maxpool_feature():
            self.w_ner_x = self.params['w1_ner'][ner_idxs]

        if self.using_sent_nr_no_convolution_no_maxpool_feature():
            self.w_sent_nr_x = self.params['w1_sent_nr'][sent_nr_idxs]

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

        #TODO: choose
        # out = self.perform_forward_pass_dense_step(w2v_conc, pos_conc)
        out = self.sgd_forward_pass(tensor_dim=4)

        y_predictions = T.argmax(out, axis=1)

        if compute_cost_error:
            errors = T.sum(T.neq(y, y_predictions))

            L2_w2v_emb, L2_pos_emb, L2_w1_emb, L2_pos_filters, L2_w2v_filters, L2_w3, \
            L2_w2v_dir_filters, L2_ner_emb, L2_ner_filters, L2_sent_nr_emb = \
                self.compute_regularization_cost(w2v_idxs, pos_idxs, ner_idxs, sent_nr_idxs)
            L2 = L2_w2v_emb + L2_pos_emb + L2_pos_filters + L2_w2v_filters + L2_w3 + \
                 L2_w2v_dir_filters + L2_ner_emb + L2_ner_filters + L2_sent_nr_emb

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