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

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
                 regularization=False,
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

        # self.n_emb = None

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

    def sgd_forward_pass(self, weight_x, bias_1, weight_2, bias_2, n_tokens):
        h = self.hidden_activation_f(weight_x.reshape((n_tokens, self.n_emb*self.n_window))+bias_1)
        return self.out_activation_f(T.dot(h, weight_2)+bias_2)

    def train_with_sgd(self, learning_rate=0.01, max_epochs=100,
                       alpha_L1_reg=0.001, alpha_L2_reg=0.01,
                       save_params=False, use_scan=False, plot=False,
                       validation_cost=True, **kwargs):

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        # valid_x = theano.shared(value=np.array(self.x_valid, dtype=INT), name='valid_x', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = T.scalar(name='n_tokens', dtype=INT)    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*self.n_emb, n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_window*self.n_emb)).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        w_x = w1[idxs]

        params = [w1,b1,w2,b2]
        param_names = ['w1','b1','w2','b2']
        params_to_get_grad = [w_x,b1,w2,b2]
        params_to_get_grad_names = ['w_x','b1','w2','b2']

        self.params = OrderedDict(zip(param_names, params))

        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1[idxs] ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2 = L2_w1 + L2_w2

        if use_scan:
            #TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
            # Unchanging variables are passed to scan as non_sequences.
            # Initialization occurs in outputs_info
            out, _ = theano.scan(fn=self.sgd_forward_pass,
                                    sequences=[w_x],
                                    outputs_info=None,
                                    non_sequences=[b1,w2,b2])

            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.sgd_forward_pass(w_x, self.params['b1'], self.params['w2'], self.params['b2'], n_tokens)

            if self.regularization:
                # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
                cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out, y))

            y_predictions = T.argmax(out, axis=1)

        cost_prediction = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
        # cost_prediction = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # cost_prediction = alpha_L2_reg*L2

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
            train_l2_penalty = theano.function(inputs=[train_idx],
                                               outputs=[L2_w1, L2_w2],
                                               givens={
                                                   idxs: train_x[train_idx]
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

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            epoch_l2_w1 = 0
            epoch_l2_w2 = 0
            for i in np.random.permutation(self.n_samples):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(i,[train_y.get_value()[i]])
                epoch_cost += cost_output
                epoch_errors += errors_output
                if self.regularization:
                    l2_w1, l2_w2 = train_l2_penalty(i)

                if i==0:
                    epoch_l2_w1 = l2_w1
                epoch_l2_w2 += l2_w2

            valid_error = 0
            valid_cost = 0
            valid_predictions = []
            for x_sample, y_sample in zip(self.x_valid, self.y_valid):
                if validation_cost:
                    cost_output, errors_output, pred = train_predict_with_cost(x_sample, [y_sample])
                else:
                    cost_output = 0 #TODO: in the forest prediction, computing the cost yield and error (out of bounds for 1st misclassification).
                    errors_output, pred = train_predict_without_cost(x_sample, [y_sample])
                valid_cost += cost_output
                valid_error += errors_output
                valid_predictions.append(np.asscalar(pred))

            train_costs_list.append(epoch_cost)
            train_errors_list.append(epoch_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(epoch_l2_w1)
            l2_w2_list.append(epoch_l2_w2)

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
            self.plot_penalties(l2_w1_list, l2_w2_list, actual_time=actual_time)

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

    def predict(self, on_training_set=False, on_validation_set=False, **kwargs):

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
        out = self.sgd_forward_pass(w_x, self.params['b1'], self.params['w2'], self.params['b2'], n_tokens)

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

    def to_string(self):
        return 'One hidden layer context window neural network with no tags.'