__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from collections import OrderedDict
import time

from trained_models import get_cwnn_path
from A_neural_network import A_neural_network
from utils.metrics import Metrics

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.warn_float64='raise'
# theano.config.floatX='float64'

INT = 'int64'

class Single_Layer_Context_Window_Net(A_neural_network):
    """
    w1: embeddings
    b1: bias
    """

    def __init__(self, hidden_activation_f, out_activation_f, regularization=False, **kwargs):

        super(Single_Layer_Context_Window_Net, self).__init__(**kwargs)
        #
        # self.x_train = x_train.astype(dtype=int)
        # self.y_train = y_train.astype(dtype=int)
        # self.x_valid = None
        # self.y_valid = None
        #
        # self.n_out = n_out
        self.out_activation_f = out_activation_f
        self.regularization = regularization
        # self.pretrained_embeddings = np.array(self.pretrained_embeddings, dtype=theano.config.floatX)

        self.params = OrderedDict()

    def train(self, **kwargs):
        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)
        else:
            # train with SGD
            logger.info('Training with SGD')
            self.train_with_sgd(**kwargs)

        return True

    def sgd_forward_pass(self, weight_x, bias_1, n_tokens):
        return self.out_activation_f(weight_x.reshape((n_tokens, self.n_emb*self.n_window))+bias_1)

    def train_with_sgd(self, learning_rate=0.01, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_scan=False, **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = T.scalar(name='n_tokens', dtype=INT)    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_window*self.n_emb)).astype(dtype=theano.config.floatX), name='b1', borrow=True)

        w_x = w1[idxs]

        params = [w1,b1]
        param_names = ['w1','b1']
        params_to_get_grad = [w_x,b1]
        params_to_get_grad_names = ['w_x','b1']

        self.params = OrderedDict(zip(param_names, params))

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            L1 = T.sum(abs(w1))

            # symbolic Theano variable that represents the squared L2 term
            L2 = T.sum(w1[idxs] ** 2)

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.sgd_forward_pass,
                                    sequences=[w_x],
                                    outputs_info=None,
                                    non_sequences=[b1])

            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.sgd_forward_pass(w_x, self.params['b1'], n_tokens)

            if self.regularization:
                # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
                cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out, y))

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

        train = theano.function(inputs=[train_idx, y],
                                outputs=[cost,errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[train_idx],
                                    n_tokens: 1
                                })
        # theano.printing.debugprint(train)

        train_predict = theano.function(inputs=[idxs, y],
                                        outputs=[cost, errors, y_predictions],
                                        updates=[],
                                        givens={
                                            n_tokens: 1
                                        })

        flat_true = self.y_test

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i,[train_y.get_value()[i]])
                epoch_cost += cost_output
                epoch_errors += errors_output

            test_error = 0
            test_cost = 0
            predictions = []
            for x_sample, y_sample in zip(self.x_test, self.y_test):
                cost_output, errors_output, pred = train_predict(x_sample, [y_sample])
                test_cost += cost_output
                test_error += errors_output
                predictions.append(np.asscalar(pred))

            results = Metrics.compute_all_metrics(y_true=flat_true, y_pred=predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d F1-score: %f Took: %f'
                        % (epoch_index+1, epoch_cost, epoch_errors, test_cost, test_error, f1_score, end-start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def minibatch_forward_pass(self, weight_x, bias_1):
        return self.out_activation_f(weight_x+bias_1)

    def train_with_minibatch(self, learning_rate=0.01, batch_size=512, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_scan=False, **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype='int32'), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype='int32'), name='train_y', borrow=True)

        y = T.vector(name='y', dtype='int32')
        # x = T.matrix(name='x', dtype=theano.config.floatX)
        minibatch_idx = T.scalar('minibatch_idx', dtype='int32')  # minibatch index

        idxs = T.matrix(name="idxs", dtype='int32') # columns: context window size/lines: tokens in the sentence
        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace n_win with self.n_win
        # n_features = train_x.get_value().shape[1]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        b1 = theano.shared(value=np.zeros((n_window*self.n_emb)).astype(dtype=theano.config.floatX), name='b1', borrow=True)

        params = [w1,b1]
        param_names = ['w1','b1']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, self.n_emb*n_window))

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            L1 = T.sum(abs(w1))

            # symbolic Theano variable that represents the squared L2 term
            L2 = T.sum(w1 ** 2)

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.minibatch_forward_pass,
                                    sequences=[w_x],
                                    outputs_info=None,
                                    non_sequences=[b1])

            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.minibatch_forward_pass(w_x,b1)

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

    def predict(self, **kwargs):

        self.x_test = self.x_test.astype(dtype=INT)
        self.y_test = self.y_test.astype(dtype=INT)

        y = T.vector(name='test_y', dtype=INT)

        idxs = T.matrix(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        n_tokens = idxs.shape[0]    #tokens in sentence

        w_x = self.params['w1'][idxs]

        # the sgd_forward_pass is valid for either minibatch or sgd prediction.
        out = self.sgd_forward_pass(w_x, self.params['b1'], n_tokens)

        y_predictions = T.argmax(out, axis=1)
        cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=[y_predictions],
                                updates=[],
                                givens=[])

        predictions = perform_prediction(self.x_test)
        # predictions = perform_prediction(valid_x.get_value())

        return self.y_test, predictions[-1], self.y_test, predictions[-1]

    def to_string(self):
        return 'Single layer context window neural network with no tags.'