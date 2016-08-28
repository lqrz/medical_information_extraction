__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from collections import OrderedDict
import time
from itertools import chain

from trained_models import get_cwnn_path
from A_neural_network import A_neural_network
from utils import utils
from utils.metrics import Metrics
from data.dataset import Dataset

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
theano.config.warn_float64='raise'
# theano.config.floatX='float64'

INT = 'int64'

class Last_tag_neural_network_trainer(A_neural_network):

    def __init__(self, hidden_activation_f, out_activation_f, pad_tag, tag_dim=50, **kwargs):

        super(Last_tag_neural_network_trainer, self).__init__(**kwargs)

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f

        self.params = OrderedDict()

        self.tag_dim = tag_dim

        self.pad_tag = pad_tag

        self.forward_pass_function = None

    def train(self, **kwargs):
        if kwargs['batch_size']:
            # train with minibatch

            raise NotImplementedError

            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])

            if kwargs['n_hidden']:
                self.forward_pass_function = self.forward_pass_noscan_two_layers
                self.train_with_minibatch_two_layers(**kwargs)
            else:
                self.forward_pass_function = self.forward_pass_noscan_one_layer
                self.train_with_minibatch_one_layer(**kwargs)
        else:
            # train with SGD
            if kwargs['update_per_sentence']:
                if kwargs['n_hidden']:
                    # two layers
                    logger.info('Training with SGD two layers at end of sentence')
                    self.forward_pass_function = self.forward_pass_scan_two_layers
                    self.train_with_sgd_updates_at_end_of_sentence_two_layers(**kwargs)
                else:
                    logger.info('Training with SGD one layer at end of sentence')
                    self.forward_pass_function = self.forward_pass_scan_one_layer
                    self.train_with_sgd_updates_at_end_of_sentence_one_layer(**kwargs)
            if kwargs['n_hidden']:
                # two layers
                logger.info('Training with SGD two layers at each token')
                self.forward_pass_function = self.forward_pass_noscan_two_layers
                self.train_with_sgd_updates_at_each_token_two_layers(**kwargs)
            else:
                # one layer
                logger.info('Training with SGD one layer at each token')
                self.forward_pass_function = self.forward_pass_noscan_one_layer
                self.train_with_sgd_updates_at_each_token_one_layer(**kwargs)

        return True

    def forward_pass_noscan_one_layer(self, idxs, n_tokens):
        w_t = self.params['wt'][self.prev_pred]
        w_x = self.params['w1'][idxs]
        prev_rep = w_t.reshape(shape=(n_tokens,self.tag_dim))
        h = self.hidden_activation_f(T.concatenate([w_x.reshape(shape=(n_tokens,self.n_emb*self.n_window)),prev_rep], axis=1)+self.params['b1'])
        result = self.out_activation_f(T.dot(h, self.params['w2'])+self.params['b2'])
        pred = T.argmax(result)

        return pred, result, w_x, w_t

    def forward_pass_noscan_two_layers(self, idxs, n_tokens):
        w_t = self.params['wt'][self.prev_pred]
        w_x = self.params['w1'][idxs]
        prev_rep = w_t.reshape(shape=(n_tokens,self.tag_dim))
        h1 = self.hidden_activation_f(T.concatenate([w_x.reshape(shape=(n_tokens,self.n_emb*self.n_window)),prev_rep], axis=1)+self.params['b1'])
        h2 = self.hidden_activation_f(T.dot(h1, self.params['w2'])+self.params['b2'])
        out_activations = self.out_activation_f(T.dot(h2, self.params['w3'])+self.params['b3'])
        pred = T.argmax(out_activations)

        return pred, out_activations, w_x, w_t

    def forward_pass_scan_one_layer(self, weight_x, prev_rep, weight_t):
        # prev_rep = weight_t[prev_pred, :]
        h = self.hidden_activation_f(T.concatenate([weight_x, prev_rep], axis=0) + self.params['b1'])
        result = self.out_activation_f(T.dot(h, self.params['w2']) + self.params['b2'])
        pred = T.cast(T.argmax(result), dtype=INT)

        return [pred, result]

    def forward_pass_scan_two_layers(self, weight_x, prev_rep, weight_t):
        # prev_rep = weight_t[prev_pred, :]
        h1 = self.hidden_activation_f(T.concatenate([weight_x, prev_rep], axis=0) + self.params['b1'])
        h2 = self.hidden_activation_f(T.dot(h1, self.params['w2']) + self.params['b2'])
        result = self.out_activation_f(T.dot(h2, self.params['w3']) + self.params['b3'])
        pred = T.cast(T.argmax(result), dtype=INT)

        return [pred, result]

    def forward_pass_predict(self, weight_x, prev_pred):
        '''
        Use an initial tag (fixed as: <PAD>), and use the previous prediction for indexing wt.
        '''
        prev_rep = self.params['wt'][prev_pred, :]
        h = self.hidden_activation_f(T.concatenate([weight_x, prev_rep], axis=0) + self.params['b1'])
        result = self.out_activation_f(T.dot(h, self.params['w2']) + self.params['b2'])
        pred = T.cast(T.argmax(result), dtype=INT)

        return [pred, result]

    def train_with_sgd_updates_at_each_token_one_layer(self, learning_rate=0.01, max_epochs=100,
                       L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, validation_cost=True,
                       plot=True,
                       use_grad_means=False,
                       **kwargs):

        # train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        # train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_tokens = T.constant(1)

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb + self.tag_dim,
                                                                        n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window * self.n_emb + self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(self.n_out, dtype=theano.config.floatX),
                           name='b2', borrow=True)

        tag_lim = np.sqrt(6. / (self.n_window + self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim, tag_lim, (self.n_out, self.tag_dim)).astype(
            dtype=theano.config.floatX),
            name='wt', borrow=True)

        self.prev_pred = theano.shared(value=self.pad_tag, name='previous_prediction')

        params = [w1, b1, w2, b2, wt]
        param_names = ['w1', 'b1', 'w2', 'b2', 'wt']

        self.params = OrderedDict(zip(param_names, params))

        # w_x = w1[idxs].reshape(shape=(n_tokens, self.n_emb * self.n_window))
        #
        # w_t = wt[y]

        pred, result, w_x, w_t = self.forward_pass_noscan_one_layer(idxs, n_tokens)

        params_to_get_grad = [w_x, b1, w2, b2, w_t]
        params_to_get_grad_names = ['w_x', 'b1', 'w2', 'b2', 'w_t']

        # initial_tag = T.scalar(name='initial_tag', dtype=INT)

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=w_x,
        #                                outputs_info=[initial_tag, None],
        #                                non_sequences=wt)
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=[w_x, w_t],
        #                                outputs_info=[None, None],
        #                                non_sequences=wt)

        # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(result, y))

        L2_w1 = T.sum(w1 ** 2)
        L2_w_x = T.sum(w_x ** 2)
        L2_w2 = T.sum(w2 ** 2)
        L2_wt = T.sum(wt ** 2)
        L2_w_t = T.sum(w_t ** 2)
        L2 = L2_w_x + L2_w2 + L2_w_t

        cost = mean_cross_entropy + alpha_L2_reg * L2

        # This is the same as the output of the scan "y_pred"
        y_predictions = T.argmax(result, axis=1)

        errors = T.sum(T.neq(y_predictions, y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        # TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds, uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads, axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx, :], grad_mean)

            return res_grad

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads,
                                                 accumulated_grad):
            if name == 'w_x':
                # idxs_r = idxs.reshape(shape=(-1,))
                # param = self.params['w1'][idxs]
                # reshape the gradient so as to get the same dimensions as the embeddings (otherwise,
                # i wouldnt be able to do the updates - dimensions mismatch)
                # grad_r = grad.reshape((n_tokens * self.n_window, self.n_emb))
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs], T.sqr(grad))
                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = - learning_rate * grad / (T.sqrt(accum[idxs]) + 10 ** -5)
                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['w1'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            elif name == 'w_t':
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[y], T.sqr(grad))

                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = (- learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5)).reshape(shape=(-1,))

                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['wt'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
                updates.append((accum_grad, accum))

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost, errors, pred],
                                updates=updates
                                )

        predictions = T.argmax(result, axis=1)

        errors_predict = T.sum(T.neq(y_predictions, y))

        train_predict_with_cost = theano.function(inputs=[idxs, y],
                                                  outputs=[cost, errors_predict, predictions],
                                                  updates=[])

        train_predict_without_cost = theano.function(inputs=[idxs, y],
                                                     outputs=[errors_predict, predictions],
                                                     updates=[])

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy
                                            )

        train_l2_penalty = theano.function(inputs=[],
                                           outputs=[L2_w1, L2_w2, L2_wt],
                                           updates=[],
                                           givens=[])

        valid_flat_true = list(chain(*self.y_valid))

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
        l2_wt_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        early_stopping_cnt_since_last_update = 0
        early_stopping_min_validation_cost = np.inf
        early_stopping_min_iteration = None
        model_update = None

        for epoch_index in range(max_epochs):
            start = time.time()

            if self.early_stopping_threshold is not None:
                if early_stopping_cnt_since_last_update >= self.early_stopping_threshold:
                    assert early_stopping_min_iteration is not None
                    self.logger.info('Training early stopped at iteration %d' % early_stopping_min_iteration)
                    break

            train_cost = 0
            train_errors = 0
            train_cross_entropy = 0
            for x_train_sentence, y_train_sentence in zip(self.x_train, self.y_train):
                self.prev_pred.set_value(self.pad_tag)
                for word_cw, word_tag in zip(x_train_sentence, y_train_sentence):
                    cost_output, errors_output, pred_output = train(word_cw, [word_tag])
                    self.prev_pred.set_value(word_tag)    #do not propagate the prediction, but use the true_tag instead.
                    train_cost += cost_output
                    train_errors += errors_output
                    train_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            l2_w1, l2_w2, l2_wt = train_l2_penalty()
            train_l2_emb = l2_w1
            train_l2_w2 = l2_w2
            train_l2_wt = l2_wt

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for x_sentence, y_sentence in zip(self.x_valid, self.y_valid):
                self.prev_pred.set_value(self.pad_tag)
                for word_cw, word_tag in zip(x_sentence, y_sentence):
                    if validation_cost:
                        cost_output, errors_output, pred = train_predict_with_cost(word_cw, [word_tag])
                    else:
                        # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                        cost_output = 0
                        errors_output, pred = train_predict_without_cost(word_cw, [word_tag])
                    self.prev_pred.set_value(np.asscalar(pred))
                    valid_cross_entropy += get_cross_entropy(word_cw, [word_tag])
                    valid_cost += cost_output
                    valid_error += errors_output
                    valid_predictions.extend(pred)

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)
            l2_wt_list.append(train_l2_wt)

            assert valid_flat_true.__len__() == valid_predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            assert train_cross_entropy_list.__len__() == valid_cross_entropy_list.__len__()
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()

            if valid_cost < early_stopping_min_validation_cost:
                self.save_params()
                early_stopping_min_iteration = epoch_index
                early_stopping_min_validation_cost = valid_cost
                early_stopping_cnt_since_last_update = 0
                model_update = True
            else:
                early_stopping_cnt_since_last_update += 1
                model_update = False

            assert model_update is not None

            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f upd: %s Took: %f'
                        % (epoch_index + 1, train_cost, train_errors, valid_cost, valid_error, f1_score, model_update, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            self.save_params()

        return True

    def train_with_sgd_updates_at_each_token_two_layers(self,
                                                        n_hidden,
                                                        learning_rate=0.01, max_epochs=100,
                                                        L1_reg=0.001, alpha_L2_reg=0.01,
                                                        save_params=False,
                                                        validation_cost=True,
                                                        plot=True,
                                                        use_grad_means=False,
                                                        **kwargs):

        # train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        # train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_tokens = T.constant(1)

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb + self.tag_dim,
                                                                        n_out=n_hidden, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        w3 = theano.shared(
            value=utils.NeuralNetwork.initialize_weights(n_in=n_hidden, n_out=self.n_out, function='tanh').
            astype(dtype=theano.config.floatX),
            name='w3', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window * self.n_emb + self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX),
                           name='b2', borrow=True)
        b3 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b3', borrow=True)

        tag_lim = np.sqrt(6. / (self.n_window + self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim, tag_lim, (self.n_out, self.tag_dim)).astype(
            dtype=theano.config.floatX),
            name='wt', borrow=True)

        self.prev_pred = theano.shared(value=self.pad_tag, name='previous_prediction')

        params = [w1, b1, w2, b2, wt, w3, b3]
        param_names = ['w1', 'b1', 'w2', 'b2', 'wt', 'w3', 'b3']

        self.params = OrderedDict(zip(param_names, params))

        # w_x = w1[idxs].reshape(shape=(n_tokens, self.n_emb * self.n_window))
        #
        # w_t = wt[y]

        pred, result, w_x, w_t = self.forward_pass_function(idxs, n_tokens)

        params_to_get_grad = [w_x, b1, w2, b2, w_t, w3, b3]
        params_to_get_grad_names = ['w_x', 'b1', 'w2', 'b2', 'w_t', 'w3', 'b3']

        # initial_tag = T.scalar(name='initial_tag', dtype=INT)

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=w_x,
        #                                outputs_info=[initial_tag, None],
        #                                non_sequences=wt)
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=[w_x, w_t],
        #                                outputs_info=[None, None],
        #                                non_sequences=wt)

        # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(result, y))

        # if self.regularization:
        L2_w1 = T.sum(w1 ** 2)
        L2_w_x = T.sum(w_x ** 2)
        L2_w2 = T.sum(w2 ** 2)
        L2_w3 = T.sum(w3 ** 2)
        L2_wt = T.sum(wt ** 2)
        L2_w_t = T.sum(w_t ** 2)
        L2 = L2_w_x + L2_w2 + L2_w_t + L2_w3

        cost = mean_cross_entropy + alpha_L2_reg * L2
        # else:
        #     cost = mean_cross_entropy

        # This is the same as the output of the scan "y_pred"
        y_predictions = T.argmax(result, axis=1)

        errors = T.sum(T.neq(y_predictions, y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        # TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds, uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads, axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx, :], grad_mean)

            return res_grad

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads,
                                                 accumulated_grad):
            if name == 'w_x':
                # idxs_r = idxs.reshape(shape=(-1,))
                # param = self.params['w1'][idxs]
                # reshape the gradient so as to get the same dimensions as the embeddings (otherwise,
                # i wouldnt be able to do the updates - dimensions mismatch)
                # grad_r = grad.reshape((n_tokens * self.n_window, self.n_emb))
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs], T.sqr(grad))
                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = - learning_rate * grad / (T.sqrt(accum[idxs]) + 10 ** -5)
                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['w1'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            elif name == 'w_t':
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[y], T.sqr(grad))

                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = (- learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5)).reshape(shape=(-1,))

                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['wt'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
                updates.append((accum_grad, accum))

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost, errors, pred],
                                updates=updates
                                )

        predictions = T.argmax(result, axis=1)

        errors_predict = T.sum(T.neq(y_predictions, y))

        train_predict_with_cost = theano.function(inputs=[idxs, y],
                                                  outputs=[cost, errors_predict, predictions],
                                                  updates=[])

        train_predict_without_cost = theano.function(inputs=[idxs, y],
                                                     outputs=[errors_predict, predictions],
                                                     updates=[])

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy
                                            )

        # if self.regularization:
        train_l2_penalty = theano.function(inputs=[],
                                           outputs=[L2_w1, L2_w2, L2_wt, L2_w3],
                                           updates=[],
                                           givens=[])

        valid_flat_true = list(chain(*self.y_valid))

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
        l2_w3_list = []
        l2_wt_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        early_stopping_cnt_since_last_update = 0
        early_stopping_min_validation_cost = np.inf
        early_stopping_min_iteration = None
        model_update = None

        for epoch_index in range(max_epochs):
            start = time.time()

            if self.early_stopping_threshold is not None:
                if early_stopping_cnt_since_last_update >= self.early_stopping_threshold:
                    assert early_stopping_min_iteration is not None
                    self.logger.info('Training early stopped at iteration %d' % early_stopping_min_iteration)
                    break

            train_cost = 0
            train_errors = 0
            train_cross_entropy = 0
            for x_train_sentence, y_train_sentence in zip(self.x_train, self.y_train):
                self.prev_pred.set_value(self.pad_tag)
                for word_cw, word_tag in zip(x_train_sentence, y_train_sentence):
                    cost_output, errors_output, pred_output = train(word_cw, [word_tag])
                    self.prev_pred.set_value(word_tag)    #do not propagate the prediction, but use the true_tag instead.
                    train_cost += cost_output
                    train_errors += errors_output
                    train_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            # if self.regularization:
            l2_w1, l2_w2, l2_wt, l2_w3 = train_l2_penalty()
            train_l2_emb = l2_w1
            train_l2_w2 = l2_w2
            train_l2_w3 = l2_w3
            train_l2_wt = l2_wt

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for x_sentence, y_sentence in zip(self.x_valid, self.y_valid):
                self.prev_pred.set_value(self.pad_tag)
                for word_cw, word_tag in zip(x_sentence, y_sentence):
                    if validation_cost:
                        cost_output, errors_output, pred = train_predict_with_cost(word_cw, [word_tag])
                    else:
                        # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                        cost_output = 0
                        errors_output, pred = train_predict_without_cost(word_cw, [word_tag])
                    self.prev_pred.set_value(np.asscalar(pred))
                    valid_cross_entropy += get_cross_entropy(word_cw, [word_tag])
                    valid_cost += cost_output
                    valid_error += errors_output
                    valid_predictions.extend(pred)

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)
            l2_w3_list.append(train_l2_w3)
            l2_wt_list.append(train_l2_wt)

            assert valid_flat_true.__len__() == valid_predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            assert train_cross_entropy_list.__len__() == valid_cross_entropy_list.__len__()
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()

            if valid_cost < early_stopping_min_validation_cost:
                self.save_params()
                early_stopping_min_iteration = epoch_index
                early_stopping_min_validation_cost = valid_cost
                early_stopping_cnt_since_last_update = 0
                model_update = True
            else:
                early_stopping_cnt_since_last_update += 1
                model_update = False

            assert model_update is not None

            logger.info(
                'Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f upd: %s Took: %f'
                % (epoch_index + 1, train_cost, train_errors, valid_cost, valid_error, f1_score, model_update,
                   end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list,
                                l2_w3_list=l2_w3_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            self.save_params()

        return True

    # deprecated
    def train_with_sgd_updates_at_end_of_sentence_one_layer(self, learning_rate=0.01, max_epochs=100,
                       L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, validation_cost=True,
                       plot=True,
                       use_grad_means=False,
                       **kwargs):

        # train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        # train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.matrix(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb + self.tag_dim,
                                                                        n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window * self.n_emb + self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(self.n_out, dtype=theano.config.floatX),
                           name='b2', borrow=True)

        tag_lim = np.sqrt(6. / (self.n_window + self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim, tag_lim, (self.n_out, self.tag_dim)).astype(
            dtype=theano.config.floatX),
            name='wt', borrow=True)

        params = [w1, b1, w2, b2, wt]
        param_names = ['w1', 'b1', 'w2', 'b2', 'wt']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape(shape=(n_tokens, self.n_emb * self.n_window))

        w_t = wt[y]

        params_to_get_grad = [w_x, b1, w2, b2, w_t]
        params_to_get_grad_names = ['w_x', 'b1', 'w2', 'b2', 'w_t']

        initial_tag = T.scalar(name='initial_tag', dtype=INT)

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=w_x,
        #                                outputs_info=[initial_tag, None],
        #                                non_sequences=wt)
        [y_pred, out], _ = theano.scan(fn=self.forward_pass_function,
                                       sequences=[w_x, w_t],
                                       outputs_info=[None, None],
                                       non_sequences=wt)

        # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out[:, -1, :], y))

        L2_w1 = T.sum(w1 ** 2)
        L2_w_x = T.sum(w_x ** 2)
        L2_w2 = T.sum(w2 ** 2)
        L2_wt = T.sum(wt ** 2)
        L2_w_t = T.sum(w_t ** 2)
        L2 = L2_w_x + L2_w2 + L2_w_t

        cost = mean_cross_entropy + alpha_L2_reg * L2

        # This is the same as the output of the scan "y_pred"
        y_predictions = T.argmax(out[:, -1, :], axis=1)

        errors = T.sum(T.neq(y_predictions, y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        # TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds, uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads, axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx, :], grad_mean)

            return res_grad

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads,
                                                 accumulated_grad):
            if name == 'w_x':
                idxs_r = idxs.reshape(shape=(-1,))
                param = self.params['w1'][idxs_r]
                # reshape the gradient so as to get the same dimensions as the embeddings (otherwise,
                # i wouldnt be able to do the updates - dimensions mismatch)
                grad_r = grad.reshape((n_tokens * self.n_window, self.n_emb))
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs_r], T.sqr(grad_r))
                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = - learning_rate * grad_r / (T.sqrt(accum[idxs_r]) + 10 ** -5)
                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['w1'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            elif name == 'w_t':

                if use_grad_means:

                    uniq_values, uniq_values_start_idxs, _ = T.extra_ops.Unique(True, True, False)(y)
                    out_grad = T.zeros(shape=(grad.shape[0], self.tag_dim), dtype=theano.config.floatX)

                    mean_grad, _ = theano.scan(fn=take_mean,
                                               sequences=[uniq_values, uniq_values_start_idxs],
                                               outputs_info=out_grad,
                                               non_sequences=[y, grad])

                    grad_n = T.set_subtensor(grad[:], mean_grad[-1])
                    # accum_grad_r = accumulated_grad[1]

                    accum = T.inc_subtensor(accum_grad[y], T.sqr(grad_n))

                    upd = - learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5)
                    update = T.inc_subtensor(param, upd)
                else:
                    # this will return the whole accum_grad structure incremented in the specified idxs
                    accum = T.inc_subtensor(accum_grad[y], T.sqr(grad))
                    # this will return the whole w1 structure decremented according to the idxs vector.
                    update = T.inc_subtensor(param, - learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5))
                # update whole structure with whole structure
                updates.append((self.params['wt'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
                updates.append((accum_grad, accum))

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost, errors],
                                updates=updates
                                )
                                # givens={
                                #     initial_tag: np.int32(36) if INT == 'int32' else 36
                                # })

        [_, out_predict], _ = theano.scan(fn=self.forward_pass_predict,
                                       sequences=w_x,
                                       outputs_info=[initial_tag, None],
                                       non_sequences=[])

        mean_cross_entropy_predict = T.mean(T.nnet.categorical_crossentropy(out_predict[:, -1, :], y))

        cost_predict = mean_cross_entropy_predict + alpha_L2_reg * L2

        predictions = T.argmax(out_predict[:, -1, :], axis=1)

        errors_predict = T.sum(T.neq(y_predictions, y))

        train_predict_with_cost = theano.function(inputs=[idxs, y],
                                                  outputs=[cost_predict, errors_predict, predictions],
                                                  updates=[],
                                                  givens={
                                                      initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                                  })

        train_predict_without_cost = theano.function(inputs=[idxs, y],
                                                     outputs=[errors_predict, predictions],
                                                     updates=[],
                                                     givens={
                                                         initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                                     })

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy
                                            )

        get_cross_entropy_predict = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy_predict,
                                            givens={
                                                initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                            })

        train_l2_penalty = theano.function(inputs=[],
                                           outputs=[L2_w1, L2_w2, L2_wt],
                                           updates=[],
                                           givens=[])

        valid_flat_true = list(chain(*self.y_valid))

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
        l2_wt_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        early_stopping_cnt_since_last_update = 0
        early_stopping_min_validation_cost = np.inf
        early_stopping_min_iteration = None
        model_update = None

        for epoch_index in range(max_epochs):
            start = time.time()

            if self.early_stopping_threshold is not None:
                if early_stopping_cnt_since_last_update >= self.early_stopping_threshold:
                    assert early_stopping_min_iteration is not None
                    self.logger.info('Training early stopped at iteration %d' % early_stopping_min_iteration)
                    break

            train_cost = 0
            train_errors = 0
            train_cross_entropy = 0
            for idx in range(self.x_train.shape[0]):
                # error = train(self.x_train, self.y_train)
                pad_y = [self.pad_tag] + self.y_train[idx][:-1]
                cost_output, errors_output = train(self.x_train[idx], pad_y)
                train_cost += cost_output
                train_errors += errors_output

                train_cross_entropy += get_cross_entropy(self.x_train[idx], pad_y)

            l2_w1, l2_w2, l2_wt = train_l2_penalty()
            train_l2_emb = l2_w1
            train_l2_w2 = l2_w2
            train_l2_wt = l2_wt

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for x_sample, y_sample in zip(self.x_valid, self.y_valid):
                if validation_cost:
                    pad_y_sample = [self.pad_tag] + y_sample[:-1]
                    cost_output, errors_output, pred = train_predict_with_cost(x_sample, pad_y_sample)
                else:
                    # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    cost_output = 0
                    errors_output, pred = train_predict_without_cost(x_sample, pad_y_sample)
                valid_cross_entropy += get_cross_entropy_predict(x_sample, pad_y_sample)
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.extend(pred)

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)
            l2_wt_list.append(train_l2_wt)

            assert valid_flat_true.__len__() == valid_predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            assert train_cross_entropy_list.__len__() == valid_cross_entropy_list.__len__()
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()

            if valid_cost < early_stopping_min_validation_cost:
                self.save_params()
                early_stopping_min_iteration = epoch_index
                early_stopping_min_validation_cost = valid_cost
                early_stopping_cnt_since_last_update = 0
                model_update = True
            else:
                early_stopping_cnt_since_last_update += 1
                model_update = False

            assert model_update is not None

            logger.info(
                'Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f upd: %s Took: %f'
                % (epoch_index + 1, train_cost, train_errors, valid_cost, valid_error, f1_score, model_update, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            self.save_params()

        return True

    # deprecated
    def train_with_sgd_updates_at_end_of_sentence_two_layers(self, n_hidden, learning_rate=0.01, max_epochs=100,
                       L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, validation_cost=True,
                       plot=True,
                       use_grad_means=False,
                       **kwargs):

        # train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        # train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.matrix(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb + self.tag_dim,
                                                                        n_out=n_hidden, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window * self.n_emb + self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(n_hidden, dtype=theano.config.floatX),
                           name='b2', borrow=True)

        w3 = theano.shared(
            value=utils.NeuralNetwork.initialize_weights(n_in=n_hidden, n_out=self.n_out, function='tanh').
            astype(dtype=theano.config.floatX),
            name='w3', borrow=True)

        b3 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b3', borrow=True)

        tag_lim = np.sqrt(6. / (self.n_window + self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim, tag_lim, (self.n_out, self.tag_dim)).astype(
            dtype=theano.config.floatX),
            name='wt', borrow=True)

        params = [w1, b1, w2, b2, wt, w3, b3]
        param_names = ['w1', 'b1', 'w2', 'b2', 'wt', 'w3', 'b3']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape(shape=(n_tokens, self.n_emb * self.n_window))

        w_t = wt[y]

        params_to_get_grad = [w_x, b1, w2, b2, w_t, w3, b3]
        params_to_get_grad_names = ['w_x', 'b1', 'w2', 'b2', 'w_t', 'w3', 'b3']

        initial_tag = T.scalar(name='initial_tag', dtype=INT)

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        # [y_pred, out], _ = theano.scan(fn=self.forward_pass,
        #                                sequences=w_x,
        #                                outputs_info=[initial_tag, None],
        #                                non_sequences=wt)
        [y_pred, out], _ = theano.scan(fn=self.forward_pass_function,
                                       sequences=[w_x, w_t],
                                       outputs_info=[None, None],
                                       non_sequences=wt)

        # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out[:, -1, :], y))

        # if self.regularization:
        L2_w1 = T.sum(w1 ** 2)
        L2_w_x = T.sum(w_x ** 2)
        L2_w2 = T.sum(w2 ** 2)
        L2_w3 = T.sum(w3 ** 2)
        L2_wt = T.sum(wt ** 2)
        L2_w_t = T.sum(w_t ** 2)
        L2 = L2_w_x + L2_w2 + L2_w_t + L2_w3

        cost = mean_cross_entropy + alpha_L2_reg * L2

        # This is the same as the output of the scan "y_pred"
        y_predictions = T.argmax(out[:, -1, :], axis=1)

        errors = T.sum(T.neq(y_predictions, y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        # TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in params_to_get_grad]

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds, uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads, axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx, :], grad_mean)

            return res_grad

        # adagrad
        accumulated_grad = []
        for name, param in zip(params_to_get_grad_names, params_to_get_grad):
            if name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for name, param, grad, accum_grad in zip(params_to_get_grad_names, params_to_get_grad, grads,
                                                 accumulated_grad):
            if name == 'w_x':
                idxs_r = idxs.reshape(shape=(-1,))
                param = self.params['w1'][idxs_r]
                # reshape the gradient so as to get the same dimensions as the embeddings (otherwise,
                # i wouldnt be able to do the updates - dimensions mismatch)
                grad_r = grad.reshape((n_tokens * self.n_window, self.n_emb))
                # this will return the whole accum_grad structure incremented in the specified idxs
                accum = T.inc_subtensor(accum_grad[idxs_r], T.sqr(grad_r))
                # this will return the whole w1 structure decremented according to the idxs vector.
                upd = - learning_rate * grad_r / (T.sqrt(accum[idxs_r]) + 10 ** -5)
                update = T.inc_subtensor(param, upd)
                # update whole structure with whole structure
                updates.append((self.params['w1'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            elif name == 'w_t':

                if use_grad_means:

                    uniq_values, uniq_values_start_idxs, _ = T.extra_ops.Unique(True, True, False)(y)
                    out_grad = T.zeros(shape=(grad.shape[0], self.tag_dim), dtype=theano.config.floatX)

                    mean_grad, _ = theano.scan(fn=take_mean,
                                               sequences=[uniq_values, uniq_values_start_idxs],
                                               outputs_info=out_grad,
                                               non_sequences=[y, grad])

                    grad_n = T.set_subtensor(grad[:], mean_grad[-1])
                    # accum_grad_r = accumulated_grad[1]

                    accum = T.inc_subtensor(accum_grad[y], T.sqr(grad_n))

                    upd = - learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5)
                    update = T.inc_subtensor(param, upd)
                else:
                    # this will return the whole accum_grad structure incremented in the specified idxs
                    accum = T.inc_subtensor(accum_grad[y], T.sqr(grad))
                    # this will return the whole w1 structure decremented according to the idxs vector.
                    update = T.inc_subtensor(param, - learning_rate * grad / (T.sqrt(accum[y]) + 10 ** -5))
                # update whole structure with whole structure
                updates.append((self.params['wt'], update))
                # update whole structure with whole structure
                updates.append((accum_grad, accum))

            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad / (T.sqrt(accum) + 10 ** -5)))
                updates.append((accum_grad, accum))

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost, errors],
                                updates=updates
                                )
                                # givens={
                                #     initial_tag: np.int32(36) if INT == 'int32' else 36
                                # })

        [_, out_predict], _ = theano.scan(fn=self.forward_pass_predict,
                                       sequences=w_x,
                                       outputs_info=[initial_tag, None],
                                       non_sequences=[])

        mean_cross_entropy_predict = T.mean(T.nnet.categorical_crossentropy(out_predict[:, -1, :], y))

        cost_predict = mean_cross_entropy_predict + alpha_L2_reg * L2

        predictions = T.argmax(out_predict[:, -1, :], axis=1)

        errors_predict = T.sum(T.neq(y_predictions, y))

        train_predict_with_cost = theano.function(inputs=[idxs, y],
                                                  outputs=[cost_predict, errors_predict, predictions],
                                                  updates=[],
                                                  givens={
                                                      initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                                  })

        train_predict_without_cost = theano.function(inputs=[idxs, y],
                                                     outputs=[errors_predict, predictions],
                                                     updates=[],
                                                     givens={
                                                         initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                                     })

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy
                                            )

        get_cross_entropy_predict = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy_predict,
                                            givens={
                                                initial_tag: np.int32(self.pad_tag) if INT == 'int32' else self.pad_tag
                                            })

        # if self.regularization:
        train_l2_penalty = theano.function(inputs=[],
                                           outputs=[L2_w1, L2_w2, L2_wt, L2_w3],
                                           updates=[],
                                           givens=[])

        valid_flat_true = list(chain(*self.y_valid))

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
        l2_w3_list = []
        l2_wt_list = []
        train_cross_entropy_list = []
        valid_cross_entropy_list = []

        early_stopping_cnt_since_last_update = 0
        early_stopping_min_validation_cost = np.inf
        early_stopping_min_iteration = None
        model_update = None

        for epoch_index in range(max_epochs):
            start = time.time()

            if self.early_stopping_threshold is not None:
                if early_stopping_cnt_since_last_update >= self.early_stopping_threshold:
                    assert early_stopping_min_iteration is not None
                    self.logger.info('Training early stopped at iteration %d' % early_stopping_min_iteration)
                    break

            train_cost = 0
            train_errors = 0
            train_cross_entropy = 0
            for idx in range(self.x_train.shape[0]):
                # error = train(self.x_train, self.y_train)
                pad_y = [self.pad_tag] + self.y_train[idx][:-1]
                cost_output, errors_output = train(self.x_train[idx], pad_y)
                train_cost += cost_output
                train_errors += errors_output

                train_cross_entropy += get_cross_entropy(self.x_train[idx], pad_y)

            # if self.regularization:
            l2_w1, l2_w2, l2_wt, l2_w3 = train_l2_penalty()
            train_l2_emb = l2_w1
            train_l2_w2 = l2_w2
            train_l2_w3 = l2_w3
            train_l2_wt = l2_wt

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            valid_cross_entropy = 0
            for x_sample, y_sample in zip(self.x_valid, self.y_valid):
                if validation_cost:
                    pad_y_sample = [self.pad_tag] + y_sample[:-1]
                    cost_output, errors_output, pred = train_predict_with_cost(x_sample, pad_y_sample)
                else:
                    # in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    cost_output = 0
                    errors_output, pred = train_predict_without_cost(x_sample, pad_y_sample)
                valid_cross_entropy += get_cross_entropy_predict(x_sample, pad_y_sample)
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.extend(pred)

            train_costs_list.append(train_cost)
            train_errors_list.append(train_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(train_l2_emb)
            l2_w2_list.append(train_l2_w2)
            l2_w3_list.append(train_l2_w3)
            l2_wt_list.append(train_l2_wt)

            assert valid_flat_true.__len__() == valid_predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=valid_flat_true, y_pred=valid_predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            assert train_cross_entropy_list.__len__() == valid_cross_entropy_list.__len__()
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            end = time.time()

            if valid_cost < early_stopping_min_validation_cost:
                self.save_params()
                early_stopping_min_iteration = epoch_index
                early_stopping_min_validation_cost = valid_cost
                early_stopping_cnt_since_last_update = 0
                model_update = True
            else:
                early_stopping_cnt_since_last_update += 1
                model_update = False

            assert model_update is not None

            logger.info(
                'Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f upd: %s Took: %f'
                % (epoch_index + 1, train_cost, train_errors, valid_cost, valid_error, f1_score, model_update, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list,
                                l2_w3_list=l2_w3_list, actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            self.save_params()

        return True

    def train_with_minibatch_one_layer(self, learning_rate=0.01, batch_size=512, max_epochs=100,
              L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.matrix(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb+self.tag_dim,
                                                                        n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX),name='w2', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window*self.n_emb+self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(self.n_out, dtype=theano.config.floatX),
                           name='b2', borrow=True)

        tag_lim = np.sqrt(6./(self.n_window+self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim,tag_lim,(self.n_out,self.tag_dim)).astype(
                            dtype=theano.config.floatX),
                           name='wt', borrow=True)

        params = [w1,b1,w2,b2,wt]
        param_names = ['w1','b1','w2','b2','wt']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, self.n_emb*self.n_window))

        minibatch_idx = T.scalar(name='minibatch_idxs', dtype=INT)  # minibatch index

        initial_tag = T.scalar(name='initial_tag', dtype=INT)

        # if use_scan:
        #     TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [y_pred,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[initial_tag,None],
                                non_sequences=[b1,w2,b2,wt])

        # if self.regularization:
        L2_w1 = T.sum(w1[idxs]**2)
        L2_w2 = T.sum(w2**2)
        L2_wt = T.sum(wt**2)
        L2 = L2_w1 + L2_w2 + L2_wt

        cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L2_reg * L2
        # else:
        #     # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        #     cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

        y_predictions = T.argmax(out[:,-1,:], axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        #TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

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
                                    y: train_y[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
                                    initial_tag: np.int32(36) if INT=='int32' else 36
                                })

        for epoch_index in range(max_epochs):
            epoch_cost = 0
            epoch_errors = 0
            for minibatch_index in range(np.int(np.ceil(self.x_train.shape[0]/float(batch_size)))):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(minibatch_index)
                epoch_cost += cost_output
                epoch_errors += errors_output
            print 'Epoch %d Cost: %f Errors: %d' % (epoch_index+1, epoch_cost, epoch_errors)

        if save_params:
            self.save_params()

        return True

    def save_params(self):
        for param_name, param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(self.get_output_path(param_name+'.p'),'wb'))

        return True

    def load_params(self):
        for param_name, param_obj in self.params.iteritems():
            self.params[param_name] = cPickle.load(open(self.get_output_path(param_name+'.p'), 'rb'))

        return True

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):

        results = dict()

        assert on_training_set or on_validation_set or on_testing_set

        self.load_params()

        if on_training_set:
            x_test = self.x_train
            y_test = self.y_train
        elif on_validation_set:
            x_test = self.x_valid
            y_test = self.y_valid
        elif on_testing_set:
            x_test = self.x_test
            y_test = self.y_test

        y = T.vector(name='valid_y', dtype=INT)

        idxs = T.matrix(name="valid_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        n_tokens = idxs.shape[0]    #tokens in sentence

        w_x = self.params['w1'][idxs].reshape((n_tokens, self.n_emb*self.n_window))

        initial_tag = T.scalar(name='initial_tag', dtype=INT)

        [y_pred,out], _ = theano.scan(fn=self.forward_pass_predict,
                                sequences=w_x,
                                outputs_info=[initial_tag, None],
                                non_sequences=[])

        # out = self.forward_pass(w_x, 36)
        y_predictions = T.argmax(out[:, -1, :], axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=y_predictions,
                                updates=[],
                                givens={
                                    initial_tag: np.int32(self.pad_tag) if INT=='int32' else self.pad_tag
                                })

        predictions = []
        for i in range(x_test.shape[0]):
            predictions.extend(perform_prediction(x_test[i]))

        flat_true = list(chain(*y_test))

        assert flat_true.__len__() == predictions.__len__()

        results['flat_trues'] = flat_true
        results['flat_predictions'] = predictions

        return results

    def to_string(self):
        return 'MLP NN with last predicted tag.'

    @classmethod
    def _get_partitioned_data_with_context_window(cls, doc_sentences, n_window, item2index):
        x = []
        for sentence in doc_sentences:
            x.append([map(lambda x: item2index[x], sent_window)
                      for sent_window in utils.NeuralNetwork.context_window(sentence, n_window)])

        return x

    @classmethod
    def get_partitioned_data(cls, x_idx, document_sentences_words, document_sentences_tags,
                             word2index, label2index, use_context_window=False, n_window=None, **kwargs):
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
                else:
                    x_doc = cls._get_partitioned_data_without_context_window(doc_sentences, word2index)

                y_doc = [map(lambda x: label2index[x] if x else None, sent) for sent in document_sentences_tags[doc_nr]]

                x.extend(x_doc)
                y.extend(y_doc)

        return x, y

    @classmethod
    def get_data(cls, clef_training=True, clef_validation=False, clef_testing=False, add_words=[], add_tags=[],
                 add_feats=[], x_idx=None, n_window=None, **kwargs):
        """
        overrides the inherited method.
        gets the training data and organizes it into sentences per document.
        RNN overrides this method, cause other neural nets dont partition into sentences.

        :param crf_training_data_filename:
        :return:
        """

        x_train_feats = None
        x_valid_feats = None
        x_test_feats = None
        features_indexes = None

        if not clef_training and not clef_validation and not clef_testing:
            raise Exception('At least one dataset must be loaded')

        document_sentence_words = []
        document_sentence_tags = []

        if clef_training:
            _, _, train_document_sentence_words, train_document_sentence_tags = Dataset.get_clef_training_dataset()

            document_sentence_words.extend(train_document_sentence_words.values())
            document_sentence_tags.extend(train_document_sentence_tags.values())

        if clef_validation:
            _, _, valid_document_sentence_words, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

            document_sentence_words.extend(valid_document_sentence_words.values())
            document_sentence_tags.extend(valid_document_sentence_tags.values())

        if clef_testing:
            _, _, test_document_sentence_words, test_document_sentence_tags = Dataset.get_clef_testing_dataset()

            document_sentence_words.extend(test_document_sentence_words.values())
            document_sentence_tags.extend(test_document_sentence_tags.values())

        word2index, index2word = cls._construct_index(add_words, document_sentence_words)
        label2index, index2label = cls._construct_index(add_tags, document_sentence_tags)

        if clef_training:
            x_train, y_train = cls.get_partitioned_data(x_idx=x_idx,
                                                        document_sentences_words=train_document_sentence_words,
                                                        document_sentences_tags=train_document_sentence_tags,
                                                        word2index=word2index,
                                                        label2index=label2index,
                                                        use_context_window=True,
                                                        n_window=n_window)

        if clef_validation:
            x_valid, y_valid = cls.get_partitioned_data(x_idx=x_idx,
                                                        document_sentences_words=valid_document_sentence_words,
                                                        document_sentences_tags=valid_document_sentence_tags,
                                                        word2index=word2index,
                                                        label2index=label2index,
                                                        use_context_window=True,
                                                        n_window=n_window)

        if clef_testing:
            x_test, y_test = cls.get_partitioned_data(x_idx=x_idx,
                                                      document_sentences_words=test_document_sentence_words,
                                                      document_sentences_tags=test_document_sentence_tags,
                                                      word2index=word2index,
                                                      label2index=label2index,
                                                      use_context_window=True,
                                                      n_window=n_window)
        # x_train, y_train = cls.get_partitioned_data(x_idx=x_idx,
        #                                             document_sentences_words=train_document_sentence_words,
        #                                             document_sentences_tags=train_document_sentence_tags,
        #                                             word2index=word2index,
        #                                             label2index=label2index,
        #                                             use_context_window=True,
        #                                             n_window=n_window)
        #
        # x_test, y_test = cls.get_partitioned_data(x_idx=x_idx,
        #                                           document_sentences_words=test_document_sentence_words,
        #                                           document_sentences_tags=test_document_sentence_tags,
        #                                           word2index=word2index,
        #                                           label2index=label2index,
        #                                           use_context_window=True,
        #                                           n_window=n_window)

        # return x_train, y_train, x_test, y_test, word2index, index2word, label2index, index2label


        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats, \
               x_test, y_test, x_test_feats, \
               word2index, index2word, \
               label2index, index2label, \
               features_indexes

    def get_hidden_activations(self, on_training_set, on_validation_set, on_testing_set):
        # not implemented
        return []

    def get_output_logits(self, on_training_set, on_validation_set, on_testing_set):
        # not implemented
        return []
