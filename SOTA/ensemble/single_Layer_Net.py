__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from trained_models import get_cwnn_path
from collections import OrderedDict

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.warn_float64='raise'
# theano.config.floatX='float64'

class Single_Layer_Net(object):

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 embeddings,
                 n_out,
                 regularization=False,
                 **kwargs):

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.n_out = n_out
        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        self.pretrained_embeddings = embeddings
        self.regularization = regularization

        self.params = OrderedDict()

    def train(self, **kwargs):
        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)
        else:
            # train with SGD
            logger.info('Training with SGD')
            raise Exception('SGD Not implemented')
            #TODO: implement
            # self.train_with_sgd(**kwargs)

        return True

    def forward_pass(self, x, weight_1, bias_1):
        return self.out_activation_f(T.dot(x,weight_1)+bias_1)

    def train_with_minibatch(self, learning_rate=0.01, batch_size=128, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_scan=False, **kwargs):

        # i need these theano vars for the minibatch (the givens in the train function). Otherwise i wouldnt.
        train_x = theano.shared(value=np.array(self.x_train, dtype=theano.config.floatX), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype='int32'), name='train_y', borrow=True)

        y = T.vector(name='y', dtype='int32')
        x = T.matrix(name='x', dtype=theano.config.floatX)
        minibatch_idx = T.scalar('minibatch_idx', dtype='int32')  # minibatch index

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        b1 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b1', borrow=True)

        params = [w1,b1]
        param_names = ['w1','b1']

        self.params = OrderedDict(zip(param_names, params))

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
            out, _ = theano.scan(fn=self.forward_pass,
                                    sequences=x,
                                    outputs_info=None,
                                    non_sequences=[w1,b1])

            if self.regularization:
                # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

            y_predictions = T.argmax(out[:,-1,:], axis=1)

        else:
            out = self.forward_pass(x,w1,b1)

            if self.regularization:
                cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
            else:
                cost = T.mean(T.nnet.categorical_crossentropy(out, y))

            y_predictions = T.argmax(out, axis=1)

        # test
        # test = theano.function(inputs=[x],
        #                 outputs=[out,y_predictions],
        #                 updates=[],
        #                 givens={})
        # print test(np.matrix(train_x.get_value()[0]))

        # test_scan = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])

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
                                    x: train_x[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
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

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict(self):

        self.x_test = self.x_test.astype(dtype=theano.config.floatX)
        self.y_test = self.y_test.astype(dtype='int32')

        # valid_x = theano.shared(value=self.x_valid.astype(dtype='int32'), name='valid_x', borrow=True)
        # valid_y = theano.shared(value=self.y_valid.astype(dtype='int32'), name='valid_y', borrow=True)

        y = T.vector(name='test_y', dtype='int32')
        x = T.matrix(name='test_x', dtype=theano.config.floatX)

        # idxs = T.matrix(name="valid_idxs", dtype='int32') # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        # n_tokens = idxs.shape[0]    #tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        # w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb* n_window))

        out = self.forward_pass(x, self.params['w1'], self.params['b1'])
        y_predictions = T.argmax(out, axis=1)
        cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[x],
                                outputs=[y_predictions],
                                updates=[],
                                givens=[])

        predictions = perform_prediction(self.x_test)
        # predictions = perform_prediction(valid_x.get_value())

        return self.y_test, predictions[-1]

    def to_string(self):
        return 'Single layer MLP NN with no context window and no tags.'