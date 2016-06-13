__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
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
# theano.config.warn_float64='raise'
# theano.config.floatX='float64'

INT = 'int64'

class Hidden_Layer_Context_Window_Net(A_neural_network):
    "Context window with SGD and adagrad"

    CRF_POSITIONS = {'ner': 1, 'pos': 2}

    FEATURE_MAPPING = {
        'pos_c_m': {'name': 'pos', 'crf_position': CRF_POSITIONS['pos']},
        'ner_c_m': {'name': 'ner', 'crf_position': CRF_POSITIONS['ner']},
        'sent_nr_nc_nm': {'name': 'sent_nr', 'crf_position': None},
        'tense_nc_nm': {'name': 'tense', 'crf_position': None}
    }

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
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
                 features_to_use=None,
                 na_tag=None,
                 **kwargs):

        super(Hidden_Layer_Context_Window_Net, self).__init__(**kwargs)

        # self.x_train = x_train
        # self.y_train = y_train
        # self.x_valid = x_test
        # self.y_valid = y_test

        # self.n_samples = self.x_train.shape[0]

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        # self.pretrained_embeddings = embeddings
        self.regularization = regularization

        # sent_nr features
        self.train_sent_nr_feats = np.array(train_sent_nr_feats)
        self.valid_sent_nr_feats = np.array(valid_sent_nr_feats)
        self.test_sent_nr_feats = np.array(test_sent_nr_feats)

        # tense features
        self.train_tense_feats = np.array(train_tense_feats)
        self.valid_tense_feats = np.array(valid_tense_feats)
        self.test_tense_feats = np.array(test_tense_feats)
        self.tense_probs = tense_probs

        # POS features
        if train_feats:
            try:
                train_pos_feats = np.array(train_feats[self.__class__.CRF_POSITIONS['pos']])
                valid_pos_feats = np.array(valid_feats[self.__class__.CRF_POSITIONS['pos']])
                test_pos_feats = np.array(test_feats[self.__class__.CRF_POSITIONS['pos']])

                pos_probs = features_indexes[self.__class__.CRF_POSITIONS['pos']][2]

                # # TODO: choose one.
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

            # NER features
            try:
                train_ner_feats = np.array(train_feats[self.__class__.CRF_POSITIONS['ner']])
                valid_ner_feats = np.array(valid_feats[self.__class__.CRF_POSITIONS['ner']])
                test_ner_feats = np.array(test_feats[self.__class__.CRF_POSITIONS['ner']])

                ner_probs = features_indexes[self.__class__.CRF_POSITIONS['ner']][2]

                # # TODO: choose one.
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

        # self.n_emb = None

        self.features_to_use = self.parse_features_to_use(features_to_use)

        self.na_tag_idx = na_tag

        self.params = OrderedDict()

    def train(self, **kwargs):
        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)
        else:
            # train with SGD
            if self.features_to_use:
                logger.info('Training with SGD and multi features')
                self.train_with_sgd_with_extra_features(**kwargs)
            else:
                logger.info('Training with SGD and no multi features')
                # self.train_with_sgd(**kwargs)
                self.train_with_sgd(**kwargs)

        return True

    @classmethod
    def get_features_crf_position(cls, features):
        positions = []
        for feat in features:
            pos = cls.FEATURE_MAPPING[feat]['crf_position']
            if pos:
                positions.append(pos)

        return positions

    def parse_features_to_use(self, features_to_use):
        return [self.__class__.FEATURE_MAPPING[feat] for feat in features_to_use]

    def using_feature(self, feature):
        return any([feat['name'] == feature for feat in self.features_to_use])

    def using_pos_feature(self):
        return self.using_feature(feature='pos')

    def using_ner_feature(self):
        return self.using_feature(feature='ner')

    def using_tense_feature(self):
        return self.using_feature(feature='tense')

    def using_sent_nr_feature(self):
        return self.using_feature(feature='sent_nr')

    def compute_hidden_layer_size(self):
        size = self.n_window * self.n_emb

        if self.using_pos_feature():
            size += self.n_window * self.n_pos_emb

        if self.using_ner_feature():
            size += self.n_window * self.n_ner_emb

        if self.using_sent_nr_feature():
            size += self.sent_nr_emb

        if self.using_tense_feature():
            size += self.n_tense_emb

        return size

    def sgd_forward_pass_extra_features(self, concat, n_tokens):
        h = self.hidden_activation_f(concat + self.params['b1'])
        return self.out_activation_f(T.dot(h, self.params['w2']) + self.params['b2'])

    def sgd_forward_pass_extra_matrix(self, weight_x, n_tokens):
        a = T.dot(weight_x.reshape((n_tokens, self.n_emb * self.n_window)), self.params['w1'])
        h = self.hidden_activation_f(a + self.params['b1'])
        return self.out_activation_f(T.dot(h, self.params['w2']) + self.params['b2'])

    def sgd_forward_pass(self, weight_x, n_tokens):
        h = self.compute_hidden_state(n_tokens, weight_x)
        return self.out_activation_f(T.dot(h, self.params['w2']) + self.params['b2'])

    def compute_hidden_state(self, n_tokens, weight_x):
        return self.hidden_activation_f(weight_x.reshape((n_tokens, self.n_emb * self.n_window)) + self.params['b1'])

    def train_with_sgd_with_extra_features(self, learning_rate=0.01, max_epochs=100,
                       alpha_L1_reg=0.001, alpha_L2_reg=0.01,
                       save_params=False, plot=False,
                       validation_cost=True,
                       static=False,
                       **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)
        valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)
        valid_y = theano.shared(value=np.array(self.y_valid, dtype=INT), name='valid_y', borrow=True)

        if self.using_pos_feature():
            # shared variable with training POS features
            train_pos_x = theano.shared(value=np.array(self.train_pos_feats, dtype=INT), name='train_pos_x',
                                        borrow=True)
            valid_pos_x = theano.shared(value=np.array(self.valid_pos_feats, dtype=INT), name='valid_pos_x',
                                        borrow=True)
            w1_pos = theano.shared(value=np.matrix(self.pos_probs.values(), dtype=theano.config.floatX).reshape((-1,1)),
                                   name='w1_pos', borrow=True)

            # indexes the POS matrix
            pos_idxs = T.vector(name="pos_train_idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence

            w_pos_x = w1_pos[pos_idxs]

            self.n_pos_emb = 1

        if self.using_ner_feature():
            # shared variable with training POS features
            train_ner_x = theano.shared(value=np.array(self.train_ner_feats, dtype=INT), name='train_ner_x',
                                        borrow=True)
            valid_ner_x = theano.shared(value=np.array(self.valid_ner_feats, dtype=INT), name='valid_ner_x',
                                        borrow=True)
            w1_ner = theano.shared(value=np.matrix(self.ner_probs.values(), dtype=theano.config.floatX).reshape((-1,1)),
                                   name='w1_ner', borrow=True)

            # indexes the NER matrix
            ner_idxs = T.vector(name="ner_train_idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
            w_ner_x = w1_ner[ner_idxs]

            self.n_ner_emb = 1

        if self.using_sent_nr_feature():
            train_sent_nr_x = theano.shared(value=np.array(self.train_sent_nr_feats, dtype=theano.config.floatX), name='train_sent_nr_x',
                                            borrow=True)
            valid_sent_nr_x = theano.shared(value=np.array(self.valid_sent_nr_feats, dtype=theano.config.floatX), name='valid_sent_nr_x',
                                            borrow=True)

            sent_nr = T.scalar(name="sent_nr_id", dtype=theano.config.floatX)

            self.sent_nr_emb = 1

        if self.using_tense_feature():
            train_tense_x = theano.shared(value=self.train_tense_feats.astype(dtype=INT), name='train_tense_x',
                                          borrow=True)
            valid_tense_x = theano.shared(value=self.valid_tense_feats.astype(dtype=INT), name='valid_tense_x',
                                          borrow=True)

            w1_tense = theano.shared(value=np.matrix(self.tense_probs.values(), dtype=theano.config.floatX).
                                     reshape((-1,1)), name='w1_tense', borrow=True)

            tense_idxs = T.scalar(name="tense_idxs", dtype=INT)
            w_tense_x = w1_tense[tense_idxs]

            self.n_tense_emb = 1

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence

        n_tokens = T.scalar(name='n_tokens', dtype=INT)  # tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)

        h_size = self.compute_hidden_layer_size()

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(
            n_in=h_size, n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),name='w2', borrow=True)


        b1 = theano.shared(value=np.zeros((h_size)).astype(dtype=theano.config.floatX), name='b1',
                           borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        w_x = w1[idxs]

        params = [w1, b1, w2, b2, w1_pos, w1_ner, w1_tense]
        param_names = ['w1', 'b1', 'w2', 'b2', 'w1_pos', 'w1_ner', 'w1_tense']
        params_to_get_grad = [b1, w2, b2]
        params_to_get_grad_names = ['b1', 'w2', 'b2']

        if not static:
            params_to_get_grad.append(w_x)
            params_to_get_grad_names.append('w_x')

        self.params = OrderedDict(zip(param_names, params))

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1 ** 2)
            L2_wx = T.sum(w_x ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2 = L2_wx + L2_w2

        w2v_r = w_x.reshape(shape=(self.n_emb * self.n_window,))
        pos_r = w_pos_x.reshape(shape=(self.n_pos_emb * self.n_window,))
        ner_r = w_ner_x.reshape(shape=(self.n_ner_emb * self.n_window,))
        sent_nr_r = sent_nr.reshape(shape=(1,))

        concat = T.concatenate([w2v_r, pos_r, ner_r, sent_nr_r, w_tense_x])

        out = self.sgd_forward_pass_extra_features(concat, n_tokens)

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))
        if self.regularization:
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = mean_cross_entropy + alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions, y))

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads, accumulated_grad):
            if name == 'w_x':
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs], T.sqr(grad))
                # this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad / (T.sqrt(accum[idxs]) + 10 ** -5))
                # update whole structure with whole structure
                updates.append((self.params['w1'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
                updates.append((accum_grad, accum))

        train_idx = T.scalar(name="train_idx", dtype=INT)

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost, errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[train_idx],
                                    pos_idxs: train_pos_x[train_idx],
                                    ner_idxs: train_ner_x[train_idx],
                                    sent_nr: train_sent_nr_x[train_idx],
                                    tense_idxs: train_tense_x[train_idx]
                                })
        # theano.printing.debugprint(train)

        train_predict_with_cost = theano.function(inputs=[train_idx, y],
                                                  outputs=[cost, errors, y_predictions],
                                                    givens={
                                                        idxs: valid_x[train_idx],
                                                        pos_idxs: valid_pos_x[train_idx],
                                                        ner_idxs: valid_ner_x[train_idx],
                                                        sent_nr: valid_sent_nr_x[train_idx],
                                                        tense_idxs: valid_tense_x[train_idx]
                                                    })

        train_predict_without_cost = theano.function(inputs=[train_idx, y],
                                                     outputs=[errors, y_predictions],
                                                    givens={
                                                        idxs: valid_x[train_idx],
                                                        pos_idxs: valid_pos_x[train_idx],
                                                        ner_idxs: valid_ner_x[train_idx],
                                                        sent_nr: valid_sent_nr_x[train_idx],
                                                        tense_idxs: valid_tense_x[train_idx]
                                                    })

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[L2_w1, L2_w2],
                                               givens=[])

        train_get_cross_entropy = theano.function(inputs=[train_idx, y],
                                            outputs=mean_cross_entropy,
                                            givens={
                                                idxs: train_x[train_idx],
                                                pos_idxs: train_pos_x[train_idx],
                                                ner_idxs: train_ner_x[train_idx],
                                                sent_nr: train_sent_nr_x[train_idx],
                                                tense_idxs: train_tense_x[train_idx]
                                            })
        valid_get_cross_entropy = theano.function(inputs=[train_idx, y],
                                                outputs=mean_cross_entropy,
                                                givens={
                                                    idxs: valid_x[train_idx],
                                                    pos_idxs: valid_pos_x[train_idx],
                                                    ner_idxs: valid_ner_x[train_idx],
                                                    sent_nr: valid_sent_nr_x[train_idx],
                                                    tense_idxs: valid_tense_x[train_idx]
                                                })

        # hidden_activation = self.compute_hidden_state(n_tokens, w_x)
        # get_hidden_state = theano.function(inputs=[idxs],
        #                                    outputs=hidden_activation)

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        l2_w1_list = []
        l2_w2_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        hidden_activations = []

        for epoch_index in range(max_epochs):
            start = time.time()
            train_cost = 0
            train_errors = 0
            train_l2_emb = 0
            train_l2_w2 = 0
            train_cross_entropy = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i, [train_y.get_value()[i]])
                train_cost += cost_output
                train_errors += errors_output

                train_cross_entropy += train_get_cross_entropy(i, [train_y.get_value()[i]])
                # if epoch_index == max_epochs - 1:
                #     hidden_activations.append(get_hidden_state(train_x.get_value()[i]).reshape(-1, ))

            if self.regularization:
                l2_w1, l2_w2 = train_l2_penalty()
                train_l2_w2 += l2_w2
                train_l2_emb += l2_w1

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for j in range(self.x_valid.shape[0]):
                if validation_cost:
                    cost_output, errors_output, pred = train_predict_with_cost(j,
                                                                               [valid_y.get_value()[j]])
                else:
                    # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    cost_output = 0
                    errors_output, pred = train_predict_without_cost(j,
                                                                     [valid_y.get_value()[j]])
                valid_cross_entropy += valid_get_cross_entropy(j, [valid_y.get_value()[j]])
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.append(np.asscalar(pred))

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)

            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index + 1, train_cost, train_errors, valid_cost, valid_error, f1_score, end - start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list, l2_w2_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        cPickle.dump(np.array(hidden_activations), open(get_cwnn_path('hidden_activations.p'), 'wb'))

        return True


    def train_with_sgd(self, learning_rate=0.01, max_epochs=100,
                       alpha_L1_reg=0.001, alpha_L2_reg=0.01,
                       save_params=False, use_scan=False, plot=False,
                       validation_cost=True,
                       static=False,
                       use_autoencoded_weight=False,
                       negative_sampling=False,
                       **kwargs):

        if negative_sampling:
            logger.info('Training with negative sampling')

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)
        valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)
        valid_y = theano.shared(value=np.array(self.y_valid, dtype=INT), name='valid_y', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)
        y_negative = T.vector(name='y_negative', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = T.scalar(name='n_tokens', dtype=INT)    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)

        if use_autoencoded_weight:
            print '...Getting autoencoded W2 weight matrix'
            w2 = theano.shared(value=cPickle.load(open(self.get_output_path('autoencoded_w2.p'), 'rb')).astype(dtype=theano.config.floatX),
                               name='w2', borrow=True)
        else:
            w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_out, function='tanh').
                               astype(dtype=theano.config.floatX),
                               name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_window*self.n_emb)).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        w_x = w1[idxs]

        params = [w1,b1,w2,b2]
        param_names = ['w1','b1','w2','b2']
        params_to_get_grad = [b1,w2,b2]
        params_to_get_grad_names = ['b1','w2','b2']

        if not static:
            params_to_get_grad.append(w_x)
            params_to_get_grad_names.append('w_x')

        self.params = OrderedDict(zip(param_names, params))

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1 ** 2)
            L2_wx = T.sum(w_x ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2 = L2_wx + L2_w2

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.sgd_forward_pass,
                                    sequences=[w_x],
                                    outputs_info=None,
                                    non_sequences=[])

            mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out[:, -1, :], y))
            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = mean_cross_entropy + alpha_L2_reg * L2
            else:
                cost = mean_cross_entropy

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.sgd_forward_pass(w_x, n_tokens)

            mean_true_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))

            cost = mean_true_cross_entropy

            if self.regularization:
                # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
                cost += alpha_L2_reg * L2

            if negative_sampling:
                mean_negative_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out+10**-8, y_negative))
                # if T.isnan(mean_negative_cross_entropy):
                #     mean_negative_cross_entropy = T.constant(0.)
                negative_cost = cost - mean_negative_cross_entropy

            y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names,params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads, accumulated_grad):
            if name == 'w_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w1'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        # test_wx = theano.function(inputs=[idxs,y],outputs=[updates])
        #
        # test_wx(self.x_train[0], self.y_train[0])
        train_idx = T.scalar(name="train_idx", dtype=INT)

        givens_train = {
            idxs: train_x[train_idx],
            n_tokens: 1
        }
        givens_valid = {
            idxs: valid_x[train_idx],
            n_tokens: 1
        }
        if negative_sampling:
            givens_train[y_negative] = np.array([self.na_tag_idx], dtype=INT)
            givens_valid[y_negative] = np.array([self.na_tag_idx], dtype=INT)

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost, errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[train_idx],
                                    n_tokens: 1
                                })

        if negative_sampling:
            train_negative = theano.function(inputs=[train_idx, y],
                                    outputs=[negative_cost, errors],
                                    updates=updates,
                                    givens=givens_train)

        # theano.printing.debugprint(train)

        train_predict_with_cost = theano.function(inputs=[train_idx, y],
                                                  outputs=[cost, errors, y_predictions],
                                                  updates=[],
                                                  givens=givens_valid,
                                                  on_unused_input='ignore')

        train_predict_without_cost = theano.function(inputs=[train_idx, y],
                                             outputs=[errors, y_predictions],
                                             updates=[],
                                             givens=givens_valid,
                                             on_unused_input='ignore')

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[L2_w1, L2_w2],
                                               givens=[])

        get_true_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_true_cross_entropy,
                                            givens={
                                                n_tokens: 1
                                            })

        if negative_sampling:
            get_negative_cross_entropy = theano.function(inputs=[idxs],
                                                outputs=mean_negative_cross_entropy,
                                                givens={
                                                    n_tokens: 1,
                                                    y_negative: np.array([self.na_tag_idx], dtype=INT)
                                                })

        hidden_activation = self.compute_hidden_state(n_tokens, w_x)
        get_hidden_state = theano.function(inputs=[idxs],
                                           outputs=hidden_activation,
                                           givens={
                                               n_tokens: 1
                                           })

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        l2_w1_list = []
        l2_w2_list = []
        train_true_cross_entropy_list = []
        train_negative_cross_entropy_list = []
        valid_true_cross_entropy_list = []
        valid_negative_cross_entropy_list = []

        hidden_activations = []

        for epoch_index in range(max_epochs):
            start = time.time()
            train_cost = 0
            train_errors = 0
            train_l2_emb = 0
            train_l2_w2 = 0
            train_true_cross_entropy = 0
            train_negative_cross_entropy = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                y_sample = train_y.get_value()[i]
                if negative_sampling and y_sample != self.na_tag_idx:
                    cost_output, errors_output = train_negative(i, [y_sample])
                    train_negative_cross_entropy += get_negative_cross_entropy(train_x.get_value()[i])
                else:
                    cost_output, errors_output = train(i, [y_sample])

                train_cost += cost_output
                train_errors += errors_output

                train_true_cross_entropy += get_true_cross_entropy(train_x.get_value()[i], [y_sample])
                if epoch_index == max_epochs-1:
                    hidden_activations.append(get_hidden_state(train_x.get_value()[i]).reshape(-1,))

            if self.regularization:
                l2_w1, l2_w2 = train_l2_penalty()
                train_l2_w2 += l2_w2
                train_l2_emb += l2_w1

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_true_cross_entropy = 0
            valid_negative_cross_entropy = 0
            for j in range(self.x_valid.shape[0]):
                x_sample = valid_x.get_value()[j]
                y_sample = valid_y.get_value()[j]
                if validation_cost:
                    cost_output, errors_output, pred = train_predict_with_cost(j, [y_sample])
                else:
                    # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    cost_output = 0
                    errors_output, pred = train_predict_without_cost(j, [y_sample])

                valid_true_cross_entropy += get_true_cross_entropy(x_sample, [y_sample])

                if negative_sampling and y_sample != self.na_tag_idx:
                    valid_negative_cross_entropy += get_negative_cross_entropy(x_sample)
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.append(np.asscalar(pred))

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)

            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            train_true_cross_entropy_list.append(train_true_cross_entropy)
            valid_true_cross_entropy_list.append(valid_true_cross_entropy)
            train_negative_cross_entropy_list.append(train_negative_cross_entropy)
            valid_negative_cross_entropy_list.append(valid_negative_cross_entropy)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, train_cost, train_errors, valid_cost, valid_error, f1_score, end-start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list, l2_w2_list, actual_time=actual_time)
            self.plot_cross_entropies(train_true_cross_entropy_list, valid_true_cross_entropy_list, actual_time)
            self.plot_cross_entropies(train_negative_cross_entropy_list, valid_negative_cross_entropy_list, actual_time,
                                      title='Negative cross-entropy evolution', output_name='negative_cross_entropy_plot')

        cPickle.dump(np.array(hidden_activations), open(get_cwnn_path('hidden_activations.p'), 'wb'))

        return True

    def train_with_sgd_with_extra_matrix(self, learning_rate=0.01, max_epochs=100,
                       alpha_L1_reg=0.001, alpha_L2_reg=0.01,
                       save_params=False, use_scan=False, plot=False,
                       validation_cost=True,
                       static=False,
                       use_autoencoded_weight=False,
                       **kwargs):
        '''
        Adds an extra weight matrix (W1). The word embeddings are now W0.
        The hidden layer is computed as: h(T.dot(W0[idxs], W1) + b1)
        '''

        #TODO: make param
        self.n_hidden = 50

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = T.scalar(name='n_tokens', dtype=INT)    #tokens in sentence

        w0 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w1 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_hidden, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)

        if use_autoencoded_weight:
            print '...Getting autoencoded W2 weight matrix'
            w2 = theano.shared(value=cPickle.load(open(self.get_output_path('autoencoded_w2.p'), 'rb')).astype(dtype=theano.config.floatX),
                               name='w2', borrow=True)
        else:
            w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_hidden, n_out=self.n_out, function='tanh').
                               astype(dtype=theano.config.floatX),
                               name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_hidden)).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        # word embeddings indexed
        w_x = w0[idxs]

        params = [w0,w1,b1,w2,b2]
        param_names = ['w0','w1','b1','w2','b2']
        params_to_get_grad = [w1,b1,w2,b2]
        params_to_get_grad_names = ['w1','b1','w2','b2']

        if not static:
            # also learn word embeddings (W0)
            params_to_get_grad.append(w_x)
            params_to_get_grad_names.append('w_x')

        self.params = OrderedDict(zip(param_names, params))

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1 ** 2)
            L2_w0 = T.sum(w0 ** 2)
            L2_wx = T.sum(w_x ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2 = L2_wx + L2_w1 + L2_w2

        out = self.sgd_forward_pass_extra_matrix(w_x, n_tokens)

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))

        if self.regularization:
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = mean_cross_entropy + alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names,params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w0'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads, accumulated_grad):
            if name == 'w_x':
                #this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs],T.sqr(grad))
                #this will return the whole w1 structure decremented according to the idxs vector.
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[idxs])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['w0'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        # test_wx = theano.function(inputs=[idxs,y],outputs=[updates])
        #
        # test_wx(self.x_train[0], self.y_train[0])
        train_idx = T.scalar(name="train_idx", dtype=INT)

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost,errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[train_idx],
                                    n_tokens: 1
                                })
        # theano.printing.debugprint(train)

        train_predict_with_cost = theano.function(inputs=[idxs, y],
                                             outputs=[cost, errors, y_predictions],
                                             updates=[],
                                             givens={
                                                 n_tokens: 1
                                             })

        train_predict_without_cost = theano.function(inputs=[idxs, y],
                                             outputs=[errors, y_predictions],
                                             updates=[],
                                             givens={
                                                 n_tokens: 1
                                             })

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[L2_w0, L2_w1, L2_w2],
                                               givens=[])

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy,
                                            givens={
                                                n_tokens: 1
                                            })

        hidden_activation = self.compute_hidden_state(n_tokens, w_x)
        get_hidden_state = theano.function(inputs=[idxs],
                                           outputs=hidden_activation,
                                           givens={
                                               n_tokens: 1
                                           })

        valid_flat_true = self.y_valid

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        valid_costs_list = []
        valid_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        l2_w0_list = []
        l2_w1_list = []
        l2_w2_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        hidden_activations = []

        for epoch_index in range(max_epochs):
            start = time.time()
            train_cost = 0
            train_errors = 0
            train_l2_emb = 0
            train_l2_w1 = 0
            train_l2_w2 = 0
            train_cross_entropy = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i, [train_y.get_value()[i]])
                train_cost += cost_output
                train_errors += errors_output

                train_cross_entropy += get_cross_entropy(train_x.get_value()[i], [train_y.get_value()[i]])
                # if epoch_index == max_epochs-1:
                #     hidden_activations.append(get_hidden_state(train_x.get_value()[i]).reshape(-1,))

            if self.regularization:
                l2_w0, l2_w1, l2_w2 = train_l2_penalty()
                train_l2_emb += l2_w0
                train_l2_w1 += l2_w1
                train_l2_w2 += l2_w2

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for x_sample, y_sample in zip(self.x_valid, self.y_valid):
                if validation_cost:
                    cost_output, errors_output, pred = train_predict_with_cost(x_sample, [y_sample])
                else:
                    # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    cost_output = 0
                    errors_output, pred = train_predict_without_cost(x_sample, [y_sample])
                valid_cross_entropy += get_cross_entropy(x_sample, [y_sample])
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.append(np.asscalar(pred))

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w0_list.append(train_l2_emb)
            l2_w1_list.append(train_l2_w1)
            l2_w2_list.append(train_l2_w2)

            assert valid_flat_true.__len__() == valid_predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, train_cost, train_errors, valid_cost, valid_error, f1_score, end-start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w0_list=l2_w0_list, l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        cPickle.dump(np.array(hidden_activations), open(get_cwnn_path('hidden_activations.p'), 'wb'))

        return True

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

    def predict_with_extra_matrix(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):
        '''
        With extra matrix. To be used with the training function "train_with_sgd_with_extra_matrix".
        '''

        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)
            y_test = self.y_train.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)
            y_test = self.y_valid.astype(dtype=INT)
        elif on_testing_set:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)
            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        # y = T.vector(name='test_y', dtype=INT)

        idxs = T.matrix(name="test_idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]  # tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w0'][idxs]

        # the sgd_forward_pass is valid for either minibatch or sgd prediction.
        out = self.sgd_forward_pass_extra_matrix(w_x, n_tokens)

        y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                             outputs=y_predictions,
                                             updates=[],
                                             givens=[])

        predictions = perform_prediction(x_test)
        # predictions = perform_prediction(valid_x.get_value())

        results['flat_predictions'] = predictions
        results['flat_trues'] = y_test

        return results

    def predict(self, multi_features=None, **kwargs):

        results = None

        if multi_features:
            results = self.predict_with_extra_features(**kwargs)
        else:
            results = self.predict_with_no_extra_features(**kwargs)

        return results

    def predict_with_extra_features(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):
        '''
        Predict function when the regular architecture is used (No extra matrix). To be used with the training
        function "train_with_sgd".
        '''
        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)
            y_test = self.y_train.astype(dtype=INT)
            x_pos_test = self.train_pos_feats.astype(dtype=INT)
            x_ner_test = self.train_ner_feats.astype(dtype=INT)
            x_sent_nr_test = self.train_sent_nr_feats.astype(dtype=theano.config.floatX)
            x_tense_test = self.train_tense_feats.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)
            y_test = self.y_valid.astype(dtype=INT)
            x_pos_test = self.valid_pos_feats.astype(dtype=INT)
            x_ner_test = self.valid_ner_feats.astype(dtype=INT)
            x_sent_nr_test = self.valid_sent_nr_feats.astype(dtype=theano.config.floatX)
            x_tense_test = self.valid_tense_feats.astype(dtype=INT)
        elif on_testing_set:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)
            y_test = self.y_test
            x_pos_test = self.test_pos_feats.astype(dtype=INT)
            x_ner_test = self.test_ner_feats.astype(dtype=INT)
            x_sent_nr_test = self.test_sent_nr_feats.astype(dtype=theano.config.floatX)
            x_tense_test = self.test_tense_feats.astype(dtype=INT)

        idxs = T.matrix(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        pos_idxs = T.matrix(name="test_pos_idxs", dtype=INT)
        ner_idxs = T.matrix(name="test_ner_idxs", dtype=INT)
        sent_nr = T.vector(name="test_sent_nr", dtype=theano.config.floatX)
        tense_idxs = T.vector(name="test_tense_idxs", dtype=INT)

        n_tokens = idxs.shape[0]    #tokens in sentence

        w_x = self.params['w1'][idxs]
        w_pos_x = self.params['w1_pos'][pos_idxs]
        w_ner_x = self.params['w1_ner'][ner_idxs]
        w_tense_x = self.params['w1_tense'][tense_idxs]

        w_x_r = w_x.reshape(shape=(n_tokens, -1))
        w_pos_r = w_pos_x.reshape(shape=(n_tokens, -1))
        w_ner_r = w_ner_x.reshape(shape=(n_tokens, -1))
        sent_nr_r = sent_nr.reshape(shape=(n_tokens, -1))
        w_tense_r = w_tense_x.reshape(shape=(n_tokens, -1))

        concat = T.concatenate([w_x_r, w_pos_r, w_ner_r, sent_nr_r, w_tense_r], axis=1)
        out = self.sgd_forward_pass_extra_features(concat, n_tokens)

        y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs, pos_idxs, ner_idxs, sent_nr, tense_idxs],
                                outputs=y_predictions,
                                updates=[],
                                givens=[])

        predictions = perform_prediction(x_test, x_pos_test, x_ner_test, x_sent_nr_test, x_tense_test)
        # predictions = perform_prediction(valid_x.get_value())

        results['flat_predictions'] = predictions
        results['flat_trues'] = y_test

        return results

    def predict_with_no_extra_features(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):
        '''
        Predict function when the regular architecture is used (No extra matrix). To be used with the training
        function "train_with_sgd".
        '''
        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)
            y_test = self.y_train.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)
            y_test = self.y_valid.astype(dtype=INT)
        elif on_testing_set:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)
            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        # y = T.vector(name='test_y', dtype=INT)

        idxs = T.matrix(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][idxs]

        # the sgd_forward_pass is valid for either minibatch or sgd prediction.
        out = self.sgd_forward_pass(w_x, n_tokens)

        y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=y_predictions,
                                updates=[],
                                givens=[])

        predictions = perform_prediction(x_test)
        # predictions = perform_prediction(valid_x.get_value())

        results['flat_predictions'] = predictions
        results['flat_trues'] = y_test

        return results

    def predict_distribution(self, on_training_set=False, on_validation_set=False, **kwargs):

        results = defaultdict(None)

        if on_training_set:
            # predict on training set
            x_test = self.x_train.astype(dtype=INT)
            y_test = self.y_train.astype(dtype=INT)

        elif on_validation_set:
            # predict on validation set
            x_test = self.x_valid.astype(dtype=INT)
            y_test = self.y_valid.astype(dtype=INT)
        else:
            # predict on test set
            x_test = self.x_test.astype(dtype=INT)
            y_test = self.y_test

        # test_x = theano.shared(value=self.x_valid.astype(dtype=INT), name='test_x', borrow=True)
        # test_y = theano.shared(value=self.y_valid.astype(dtype=INT), name='test_y', borrow=True)

        # y = T.vector(name='test_y', dtype=INT)

        idxs = T.matrix(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][idxs]

        # the sgd_forward_pass is valid for either minibatch or sgd prediction.
        out = self.sgd_forward_pass(w_x, n_tokens)

        # y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=out,
                                updates=[],
                                givens=[])

        distribution = perform_prediction(x_test)
        # predictions = perform_prediction(valid_x.get_value())

        results['distribution'] = distribution
        results['flat_trues'] = y_test

        return results

    def to_string(self):
        return 'One hidden layer context window neural network with no tags.'