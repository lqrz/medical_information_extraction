__author__ = 'root'

from data import get_w2v_model
from data import get_w2v_training_data_vectors
from data.dataset import Dataset
import logging
import numpy as np
from collections import defaultdict
import theano
import theano.tensor as T
import os.path
import cPickle
from trained_models import get_cwnn_path
from collections import OrderedDict
from utils import utils
from A_neural_network import A_neural_network
from itertools import chain
import time
from utils.utils import NeuralNetwork

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.exception_verbosity='high'
# theano.config.optimizer='fast_run'
# theano.config.exception_verbosity='low'
theano.config.warn_float64='raise'
# theano.config.floatX='float64'

print theano.config.floatX

INT = 'int64'

class Recurrent_net(A_neural_network):

    def __init__(self, hidden_activation_f, out_activation_f, regularization, **kwargs):
        super(Recurrent_net, self).__init__(**kwargs)

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        self.n_window = 1

        self.regularization = regularization

        self.params = OrderedDict()

    def train(self, bidirectional=None, shared_params=None, **kwargs):
        if bidirectional and shared_params:
            logger.info('Training bidirectional RNN with shared hidden layer parameters.')
            self.train_bidirectional_with_shared_params(**kwargs)
        elif bidirectional and not shared_params:
            logger.info('Training bidirectional RNN without shared hidden layer parameters.')
            self.train_bidirectional_without_shared_params(**kwargs)
        else:
            logger.info('Training unidirectional RNN.')
            self.train_unidirectional(**kwargs)

        return True

    def predict(self, bidirectional=None, shared_params=None, **kwargs):
        predictions = None

        if bidirectional and shared_params:
            predictions = self.predict_bidirectional_with_shared_params(**kwargs)
        elif bidirectional and not shared_params:
            predictions = self.predict_bidirectional_without_shared_params(**kwargs)
        else:
            predictions = self.predict_unidirectional(**kwargs)

        return predictions

    def forward_pass(self, weight_x, h_previous,bias_1, weight_2, bias_2, weight_w):
        h_tmp = self.hidden_activation_f(weight_x + T.dot(weight_w, h_previous) + bias_1)
        forward_result = self.out_activation_f(T.dot(h_tmp, weight_2) + bias_2)

        return [h_tmp,forward_result]

    def train_unidirectional(self, learning_rate=0.01, batch_size=512, max_epochs=100, alpha_l1_reg=0.001, alpha_l2_reg=0.01,
              save_params=False, **kwargs):

        # train_x = theano.shared(value=self.x_train, name='train_x_shared')
        # train_y = theano.shared(value=self.y_train, name='train_y_shared')

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX), name='w1', borrow=True)
        w2 = theano.shared(value=NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)
        ww = theano.shared(value=NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_window*self.n_emb,
                                                                  function='tanh').astype(dtype=theano.config.floatX), name='ww', borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_window*self.n_emb).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        params = [w1,b1,w2,b2,ww]
        param_names = ['w1','b1','w2','b2','ww']

        self.params = OrderedDict(zip(param_names, params))

        grad_params_names = ['w_x','b1','w2','b2','ww']
        # grad_params_names = ['w_x','w2','ww']

        w_x = w1[idxs].reshape((n_tokens, self.n_emb*self.n_window))

        self.params['w_x'] = w_x

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[dict(initial=T.zeros(self.n_window*self.n_emb)), None],
                                non_sequences=[b1,w2,b2,ww])

        if self.regularization:
            L2 = T.sum(w1[idxs] ** 2) + T.sum(w2 ** 2) + T.sum(ww ** 2)
            # L2 = T.sum(w1 ** 2) + T.sum(w2 ** 2) + T.sum(ww ** 2)
            #TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_l2_reg * L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

        y_predictions = T.argmax(out[:,-1,:], axis=1)
        errors = T.sum(T.neq(y_predictions,y))

        grads = [T.grad(cost, self.params[param_name]) for param_name in grad_params_names]

        # adagrad
        accumulated_grad = []
        for param_name in grad_params_names:
            if param_name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(self.params[param_name].get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for param_name, grad, accum_grad in zip(grad_params_names, grads, accumulated_grad):
            param = self.params[param_name]
            if param_name == 'w_x':
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

        train = theano.function(inputs=[theano.In(idxs,borrow=True), theano.In(y,borrow=True)],
                                outputs=[cost,errors],
                                updates=updates)

        predict = theano.function(inputs=[theano.In(idxs,borrow=True), theano.In(y,borrow=True)],
                                outputs=[cost,errors],
                                updates=[])

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            for j,(sentence_idxs, tags_idxs) in enumerate(zip(self.x_train, self.y_train)):
                # for j,(sentence_idxs, tags_idxs) in enumerate(zip(train_x.get_value(borrow=True), train_y.get_value(borrow=True))):
                # error = train(self.x_train, self.y_train)
                # print 'Epoch %d Sentence %d' % (epoch_index, j)
                cost_output, errors_output = train(sentence_idxs, tags_idxs)
                epoch_cost += cost_output
                epoch_errors += errors_output

            test_error = 0
            test_cost = 0
            for sentence_idxs,tags_idxs in zip(self.x_test, self.y_test):
                cost_output, errors_output = predict(sentence_idxs, tags_idxs)
                test_cost += cost_output
                test_error += errors_output
            print 'Epoch %d  Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d Took: %f' % \
                  (epoch_index+1, epoch_cost, epoch_errors, test_cost, test_error, time.time()-start)

        if save_params:
            self.save_params()

        return True

    def train_bidirectional_with_shared_params(self, learning_rate=0.01, batch_size=512, max_epochs=100, alpha_l1_reg=0.001, alpha_l2_reg=0.01,
              save_params=False, **kwargs):

        # train_x = theano.shared(value=self.x_train, name='train_x_shared')
        # train_y = theano.shared(value=self.y_train, name='train_y_shared')

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX), name='w1', borrow=True)
        w2 = theano.shared(value=NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb * 2, n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        ww = theano.shared(value=NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_window*self.n_emb,
                                                                  function='tanh').astype(dtype=theano.config.floatX), name='ww', borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_window*self.n_emb).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        params = [w1,b1,w2,b2,ww]
        param_names = ['w1','b1','w2','b2','ww']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, self.n_emb*self.n_window))
        w_x_flipped = w_x[::-1]

        self.params['w_x'] = w_x
        self.params['w_x_flipped'] = w_x_flipped

        grad_params_names = ['w_x', 'w_x_flipped', 'b1', 'w2', 'b2', 'ww']

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[dict(initial=T.zeros(self.n_window*self.n_emb)), None],
                                non_sequences=[b1,w2,b2,ww])

        [h_flipped, out_flipped], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x_flipped,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * self.n_emb)), None],
                                  non_sequences=[b1, w2, b2, ww])

        h_flipped_flipped = h_flipped[::-1]

        h_bidirectional = T.concatenate([h, h_flipped_flipped], axis=1)

        out_bidirectional = self.out_activation_f(T.dot(h_bidirectional,self.params['w2']) + self.params['b2'])

        # test = theano.function(inputs=[idxs], outputs=[out_bidirectional], allow_input_downcast=True)
        # test(self.x_train[0])

        if self.regularization:
            L2 = T.sum(w1[idxs] ** 2) + T.sum(w2 ** 2) + T.sum(ww ** 2)
            # L2 = T.sum(w1 ** 2) + T.sum(w2 ** 2) + T.sum(ww ** 2)
            #TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = T.mean(T.nnet.categorical_crossentropy(out_bidirectional, y)) + alpha_l2_reg * L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out_bidirectional, y))

        y_predictions = T.argmax(out_bidirectional, axis=1)
        errors = T.sum(T.neq(y_predictions,y))

        # test_probs = theano.function(inputs=[idxs, y], outputs=[cost, errors], allow_input_downcast=True)
        # test_probs(self.x_train[0], self.y_train[0])

        grads = [T.grad(cost, self.params[param_name]) for param_name in grad_params_names]

        tmp_grads = OrderedDict(zip(grad_params_names, grads))
        # compute w_x aggregated gradient.
        w_x_agg_grad = T.sum(T.concatenate([tmp_grads['w_x'].reshape(shape=(tmp_grads['w_x'].shape[0],tmp_grads['w_x'].shape[1],1),ndim=3),
                              tmp_grads['w_x_flipped'][::-1].reshape(shape=(tmp_grads['w_x_flipped'].shape[0],tmp_grads['w_x_flipped'].shape[1],1),ndim=3)],axis=2),axis=2)

        grad_params_names_agg = ['w_x', 'b1', 'w2', 'b2', 'ww']
        grads_agg = [w_x_agg_grad,tmp_grads['b1'],tmp_grads['w2'],tmp_grads['b2'],tmp_grads['ww']]

        # adagrad
        accumulated_grad = []
        for param_name in grad_params_names_agg:
            if param_name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(self.params[param_name].get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, name=param_name, borrow=True))

        # grad = grads_agg[0]
        # param = self.params['w_x']
        # accum = T.inc_subtensor(accumulated_grad[0][idxs], T.sqr(grad))
        # upd = - learning_rate * grad / (T.sqrt(accum[idxs]) + 10 ** -5)
        # update = T.inc_subtensor(param, upd)
        #
        # test = theano.function([idxs,y], [grad,param,accum,upd,update], allow_input_downcast=True)
        # test(self.x_train[0], self.y_train[0])

        updates = []
        for param_name, grad, accum_grad in zip(grad_params_names_agg, grads_agg, accumulated_grad):
            param = self.params[param_name]
            if param_name == 'w_x':
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

        train = theano.function(inputs=[theano.In(idxs,borrow=True), theano.In(y,borrow=True)],
                                outputs=[cost,errors],
                                updates=updates)

        predict = theano.function(inputs=[theano.In(idxs,borrow=True), theano.In(y,borrow=True)],
                                outputs=[cost,errors],
                                updates=[])

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            for j,(sentence_idxs, tags_idxs) in enumerate(zip(self.x_train, self.y_train)):
                # for j,(sentence_idxs, tags_idxs) in enumerate(zip(train_x.get_value(borrow=True), train_y.get_value(borrow=True))):
                # error = train(self.x_train, self.y_train)
                # print 'Epoch %d Sentence %d' % (epoch_index, j)
                cost_output, errors_output = train(sentence_idxs, tags_idxs)
                epoch_cost += cost_output
                epoch_errors += errors_output

            test_error = 0
            test_cost = 0
            for sentence_idxs,tags_idxs in zip(self.x_test, self.y_test):
                cost_output, errors_output = predict(sentence_idxs, tags_idxs)
                test_cost += cost_output
                test_error += errors_output
            print 'Epoch %d  Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d Took: %f' % \
                  (epoch_index+1, epoch_cost, epoch_errors, test_cost, test_error, time.time()-start)

        if save_params:
            self.save_params()

        return True

    def train_bidirectional_without_shared_params(self, learning_rate=0.01, batch_size=512, max_epochs=100,
                                               alpha_l1_reg=0.001, alpha_l2_reg=0.01,
                                               save_params=False, **kwargs):

        # train_x = theano.shared(value=self.x_train, name='train_x_shared')
        # train_y = theano.shared(value=self.y_train, name='train_y_shared')

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_tokens = idxs.shape[0]  # tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX), name='w1',
                           borrow=True)
        w2 = theano.shared(value=NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb * 2, n_out=self.n_out,
                                                                  function='softmax').
                           astype(dtype=theano.config.floatX), name='w2', borrow=True)

        ww_forward = theano.shared(
            value=NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb, n_out=self.n_window * self.n_emb,
                                                   function='tanh').astype(dtype=theano.config.floatX), name='ww',
            borrow=True)
        ww_backwards = theano.shared(
            value=NeuralNetwork.initialize_weights(n_in=self.n_window * self.n_emb, n_out=self.n_window * self.n_emb,
                                                   function='tanh').astype(dtype=theano.config.floatX), name='ww',
            borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_window * self.n_emb).astype(dtype=theano.config.floatX), name='b1',
                           borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        params = [w1, b1, w2, b2, ww_forward, ww_backwards]
        param_names = ['w1', 'b1', 'w2', 'b2', 'ww_forward', 'ww_backwards']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, self.n_emb * self.n_window))
        w_x_flipped = w_x[::-1]

        self.params['w_x'] = w_x
        self.params['w_x_flipped'] = w_x_flipped

        grad_params_names = ['w_x', 'w_x_flipped', 'b1', 'w2', 'b2', 'ww_forward', 'ww_backwards']

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h, out], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * self.n_emb)), None],
                                  non_sequences=[b1, w2, b2, ww_forward])

        [h_flipped, out_flipped], _ = theano.scan(fn=self.forward_pass,
                                                  sequences=w_x_flipped,
                                                  outputs_info=[dict(initial=T.zeros(self.n_window * self.n_emb)), None],
                                                  non_sequences=[b1, w2, b2, ww_backwards])

        h_flipped_flipped = h_flipped[::-1]

        h_bidirectional = T.concatenate([h, h_flipped_flipped], axis=1)

        out_bidirectional = self.out_activation_f(T.dot(h_bidirectional, self.params['w2']) + self.params['b2'])

        # test = theano.function(inputs=[idxs], outputs=[out_bidirectional], allow_input_downcast=True)
        # test(self.x_train[0])

        if self.regularization:
            L2 = T.sum(w1[idxs] ** 2) + T.sum(w2 ** 2) + T.sum(ww_forward ** 2) + T.sum(ww_backwards ** 2)
            cost = T.mean(T.nnet.categorical_crossentropy(out_bidirectional, y)) + alpha_l2_reg * L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out_bidirectional, y))

        y_predictions = T.argmax(out_bidirectional, axis=1)
        errors = T.sum(T.neq(y_predictions, y))

        # test_probs = theano.function(inputs=[idxs, y], outputs=[cost, errors], allow_input_downcast=True)
        # test_probs(self.x_train[0], self.y_train[0])

        grads = [T.grad(cost, self.params[param_name]) for param_name in grad_params_names]

        tmp_grads = OrderedDict(zip(grad_params_names, grads))
        # compute w_x aggregated gradient.
        w_x_agg_grad = T.sum(T.concatenate(
            [tmp_grads['w_x'].reshape(shape=(tmp_grads['w_x'].shape[0], tmp_grads['w_x'].shape[1], 1), ndim=3),
             tmp_grads['w_x_flipped'][::-1].reshape(
                 shape=(tmp_grads['w_x_flipped'].shape[0], tmp_grads['w_x_flipped'].shape[1], 1), ndim=3)], axis=2), axis=2)

        grad_params_names_agg = ['w_x', 'b1', 'w2', 'b2', 'ww_forward', 'ww_backwards']
        grads_agg = [w_x_agg_grad, tmp_grads['b1'], tmp_grads['w2'], tmp_grads['b2'], tmp_grads['ww_forward'], tmp_grads['ww_backwards']]

        # adagrad
        accumulated_grad = []
        for param_name in grad_params_names_agg:
            if param_name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(self.params[param_name].get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, name=param_name, borrow=True))

        # grad = grads_agg[0]
        # param = self.params['w_x']
        # accum = T.inc_subtensor(accumulated_grad[0][idxs], T.sqr(grad))
        # upd = - learning_rate * grad / (T.sqrt(accum[idxs]) + 10 ** -5)
        # update = T.inc_subtensor(param, upd)
        #
        # test = theano.function([idxs,y], [grad,param,accum,upd,update], allow_input_downcast=True)
        # test(self.x_train[0], self.y_train[0])

        updates = []
        for param_name, grad, accum_grad in zip(grad_params_names_agg, grads_agg, accumulated_grad):
            param = self.params[param_name]
            if param_name == 'w_x':
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

        train = theano.function(inputs=[theano.In(idxs, borrow=True), theano.In(y, borrow=True)],
                                outputs=[cost, errors],
                                updates=updates)

        predict = theano.function(inputs=[theano.In(idxs, borrow=True), theano.In(y, borrow=True)],
                                  outputs=[cost, errors],
                                  updates=[])

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            for j, (sentence_idxs, tags_idxs) in enumerate(zip(self.x_train, self.y_train)):
                # for j,(sentence_idxs, tags_idxs) in enumerate(zip(train_x.get_value(borrow=True), train_y.get_value(borrow=True))):
                # error = train(self.x_train, self.y_train)
                # print 'Epoch %d Sentence %d' % (epoch_index, j)
                cost_output, errors_output = train(sentence_idxs, tags_idxs)
                epoch_cost += cost_output
                epoch_errors += errors_output

            test_error = 0
            test_cost = 0
            for sentence_idxs, tags_idxs in zip(self.x_test, self.y_test):
                cost_output, errors_output = predict(sentence_idxs, tags_idxs)
                test_cost += cost_output
                test_error += errors_output
            print 'Epoch %d  Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d Took: %f' % \
                  (epoch_index + 1, epoch_cost, epoch_errors, test_cost, test_error, time.time() - start)

        if save_params:
            self.save_params()

        return True

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict_unidirectional(self, **kwargs):

        # self.x_valid = x_valid.astype(dtype=int)
        # self.y_valid = y_valid.astype(dtype=int)

        y = T.vector(name='test_y', dtype=INT)

        idxs = T.vector(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb*self.n_window))

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[dict(initial=T.zeros(self.n_window*n_emb)), None],
                                non_sequences=[self.params['b1'],self.params['w2'],self.params['b2'],self.params['ww']])

        # out = self.forward_pass(w_x, 36)
        y_predictions = T.argmax(out[:,-1,:], axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[theano.In(idxs)],
                                outputs=[y_predictions],
                                updates=[],
                                givens=[])

        predictions = []
        for sentence_idxs in self.x_test:
            predictions.append(perform_prediction(sentence_idxs)[-1])

        flat_predictions = list(chain(*predictions))
        flat_true = list(chain(*self.y_test))

        return flat_true, flat_predictions

    def predict_bidirectional_with_shared_params(self, **kwargs):

        idxs = T.vector(name="test_idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1]  # embeddings dimension
        n_tokens = idxs.shape[0]  # tokens in sentence

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb * self.n_window))
        w_x_flipped = w_x[::-1]

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h_forward, out_forward], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * n_emb)), None],
                                  non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'],
                                                 self.params['ww']])

        [h_backwards, out_backwards], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x_flipped,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * n_emb)), None],
                                  non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'],
                                                 self.params['ww']])

        h_backwards_flipped = h_backwards[::-1]

        h_bidirectional = T.concatenate([h_forward, h_backwards_flipped], axis=1)

        out_bidirectional = self.out_activation_f(T.dot(h_bidirectional, self.params['w2']) + self.params['b2'])

        y_predictions = T.argmax(out_bidirectional, axis=1)

        perform_prediction = theano.function(inputs=[theano.In(idxs)],
                                             outputs=y_predictions,
                                             updates=[],
                                             givens=[])

        predictions = []
        for sentence_idxs in self.x_test:
            predictions.append(perform_prediction(sentence_idxs))

        flat_predictions = list(chain(*predictions))
        flat_true = list(chain(*self.y_test))

        return flat_true, flat_predictions

    def predict_bidirectional_without_shared_params(self, **kwargs):

        idxs = T.vector(name="test_idxs", dtype=INT)  # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1]  # embeddings dimension
        n_tokens = idxs.shape[0]  # tokens in sentence

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb * self.n_window))
        w_x_flipped = w_x[::-1]

        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [h_forward, out_forward], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * n_emb)), None],
                                  non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'],
                                                 self.params['ww_forward']])

        [h_backwards, out_backwards], _ = theano.scan(fn=self.forward_pass,
                                  sequences=w_x_flipped,
                                  outputs_info=[dict(initial=T.zeros(self.n_window * n_emb)), None],
                                  non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'],
                                                 self.params['ww_backwards']])

        h_backwards_flipped = h_backwards[::-1]

        h_bidirectional = T.concatenate([h_forward, h_backwards_flipped], axis=1)

        out_bidirectional = self.out_activation_f(T.dot(h_bidirectional, self.params['w2']) + self.params['b2'])

        y_predictions = T.argmax(out_bidirectional, axis=1)

        perform_prediction = theano.function(inputs=[theano.In(idxs)],
                                             outputs=y_predictions,
                                             updates=[],
                                             givens=[])

        predictions = []
        for sentence_idxs in self.x_test:
            predictions.append(perform_prediction(sentence_idxs))

        flat_predictions = list(chain(*predictions))
        flat_true = list(chain(*self.y_test))

        return flat_true, flat_predictions

    def to_string(self):
        return 'RNN.'

    @classmethod
    def get_data(cls, crf_training_data_filename, testing_data_filename=None, add_tags=[], x_idx=None, n_window=None):
        """
        overrides the inherited method.
        gets the training data and organizes it into sentences per document.
        RNN overrides this method, cause other neural nets dont partition into sentences.

        :param crf_training_data_filename:
        :return:
        """

        test_document_sentence_tags, test_document_sentence_words, train_document_sentence_tags, train_document_sentence_words = cls.get_datasets(
            crf_training_data_filename, testing_data_filename)

        document_sentence_words = []
        document_sentence_tags = []
        document_sentence_words.extend(train_document_sentence_words.values())
        document_sentence_words.extend(test_document_sentence_words.values())
        document_sentence_tags.extend(train_document_sentence_tags.values())
        document_sentence_tags.extend(test_document_sentence_tags.values())

        label2index, index2label, word2index, index2word = cls._construct_indexes(add_tags,
                                                                                document_sentence_words,
                                                                                document_sentence_tags)

        x_train, y_train = cls.get_partitioned_data(x_idx=x_idx,
                                 document_sentences_words=train_document_sentence_words,
                                 document_sentences_tags=train_document_sentence_tags,
                                 word2index=word2index,
                                 label2index=label2index,
                                 use_context_window=False,
                                 n_window=n_window)

        x_test, y_test = cls.get_partitioned_data(x_idx=x_idx,
                                 document_sentences_words=test_document_sentence_words,
                                 document_sentences_tags=test_document_sentence_tags,
                                 word2index=word2index,
                                 label2index=label2index,
                                 use_context_window=False,
                                 n_window=n_window)

        return x_train, y_train, x_test, y_test, word2index, index2word, label2index, index2label

if __name__ == '__main__':
    crf_training_data_filename = 'handoverdata.zip'

    training_vectors_filename = get_w2v_training_data_vectors()

    #TODO: make param.
    use_context_window = False
    n_window = 1

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

    logger.info('Loading CRF training data')

    #TODO: if i tokenize myself, i get a diff split
    # training_sentences = Dataset.get_words_in_training_dataset(crf_training_data_filename)

    # crf_training_dataset,_ = Dataset.get_crf_training_data(crf_training_data_filename)

    # annotated_data = transform_crf_training_data(crf_training_dataset)
    # training_sentence_words/ = [word['word'] for archive in crf_training_dataset.values() for word in archive.values()]
    # training_sentence_tags = [word['tag'] for archive in crf_training_dataset.values() for word in archive.values()]

    # sentences_words, sentences_tags = Dataset.get_training_file_tokenized_sentences(crf_training_data_filename)
    _, _, document_sentence_words, document_sentence_tags = Dataset.get_crf_training_data_by_sentence(crf_training_data_filename)

    unique_words = list(set([word for doc_sentences in document_sentence_words.values() for sentence in doc_sentences for word in sentence]))
    unique_labels = list(set([tag for doc_sentences in document_sentence_tags.values() for sentence in doc_sentences for tag in sentence]))
    # unique_words = list(set([word_dict['word'] for doc_sentences in document_sentence.values() for sentence in doc_sentences for word_dict in sentence]))
    # unique_labels = list(set([word_dict['tag'] for doc_sentences in document_sentence.values() for sentence in doc_sentences for word_dict in sentence]))

    logger.info('Creating word-index dictionaries')
    index2word = defaultdict(None)
    word2index = defaultdict(None)
    word2index['<PAD>'] = len(unique_words)
    for i,word in enumerate(unique_words):
        index2word[i] = word
        word2index[word] = i

    logger.info('Creating label-index dictionaries')
    index2label = defaultdict(None)
    label2index = defaultdict(None)
    label2index['<PAD>'] = len(unique_labels)
    for i,label in enumerate(unique_labels):
        index2label[i] = label
        label2index[label] = i

    n_out = len(label2index)

    n_unique_words = len(unique_words)+1    # +1 for the <PAD>
    n_labels = len(unique_labels)+1

    lim = np.sqrt(6./(n_unique_words+w2v_dims))

    w = np.random.uniform(-lim,lim,(n_unique_words,w2v_dims))

    w = utils.NeuralNetwork.replace_with_word_embeddings(w, unique_words, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    words_per_document, tags_per_document, document_sentences_words, document_sentences_tags, n_docs, unique_words, unique_labels, index2word, word2index, index2label, label2index, n_unique_words, n_out = RNN_trainer.get_data(crf_training_data_filename)

    logger.info('Instantiating RNN')
    params = {
    'hidden_activation_f': utils.NeuralNetwork.tanh_activation_function,
    'out_activation_f': utils.NeuralNetwork.softmax_activation_function,
    'n_window': n_window,
    'words_per_document': None,
    'tags_per_document': None,
    'document_sentences_words': document_sentence_words,
    'document_sentences_tags': document_sentence_tags,
    'unique_words': unique_words,
    'unique_labels': unique_labels,
    'index2word': index2word,
    'word2index': word2index,
    'index2label': index2label,
    'label2index': label2index,
    'n_unique_words': n_unique_words,
    'n_out': n_out
    }
    nn_trainer = RNN_trainer(**params)

    #TODO: change to make sentences.
    if use_context_window:
        nn_trainer.x_train = np.array([map(lambda x: word2index[x],sentence_window) for sentences in document_sentence_words.values()
                   for sentence in sentences
                   for sentence_window in utils.NeuralNetwork.context_window(sentence,n_window)])
    else:
        # x_train = np.array([map(lambda x: word2index[x], sentence) for sentence in sentences_words.values()])
        # x_train = np.array([map(lambda x: word2index[x], sentence) for sentence in delete_words])
        # x_train = np.array([map(lambda x: word2index[x], sentence) for doc_sentences in sentences_words.values() for sentence in doc_sentences])
        nn_trainer.x_train = np.array([map(lambda x:word2index[x],sentence) for doc_sentences in document_sentence_words.values() for sentence in doc_sentences])

    # y_train = np.array([map(lambda x: label2index[x], sentence) for sentence in delete_tags])
    # y_train = np.array([map(lambda x: label2index[x], sentence) for doc_sentences in sentences_tags.values() for sentence in doc_sentences])
    nn_trainer.y_train = np.array([map(lambda x:label2index[x],sentence) for doc_sentences in document_sentence_tags.values() for sentence in doc_sentences])

    nn_trainer.initialize_w(w2v_dims, w2v_vectors=w2v_vectors, w2v_model=w2v_model)

    logger.info('Training RNN')
    nn_trainer.train(learning_rate=.01, batch_size=512, max_epochs=50, save_params=False)

    logger.info('End')