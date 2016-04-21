__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from collections import OrderedDict

from trained_models import get_cwnn_path
from A_neural_network import A_neural_network
from utils import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
theano.config.warn_float64='raise'
# theano.config.floatX='float64'


class Last_tag_neural_network_trainer(A_neural_network):

    def __init__(self, hidden_activation_f, out_activation_f, tag_dim=50, **kwargs):

        super(Last_tag_neural_network_trainer, self).__init__(**kwargs)

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f

        self.params = OrderedDict()

        self.tag_dim = tag_dim

    def forward_pass(self, weight_x, prev_pred, bias_1, weight_2, bias_2, weight_t):
        prev_rep = weight_t[prev_pred,:]
        h = self.hidden_activation_f(T.concatenate([weight_x,prev_rep])+bias_1)
        result = self.out_activation_f(T.dot(h, weight_2)+bias_2)
        pred = T.cast(T.argmax(result), dtype='int32')

        return [pred,result]

    def train(self, learning_rate=0.01, batch_size=512, max_epochs=100,
              L1_reg=0.001, alpha_L2_reg=0.01, save_params=False):

        train_x = theano.shared(value=np.array(self.x_train, dtype='int32'), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype='int32'), name='train_y', borrow=True)

        y = T.vector(name='y', dtype='int32')

        idxs = T.matrix(name="idxs", dtype='int32') # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        n_tokens = idxs.shape[0]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings, dtype=theano.config.floatX),
                           name='w1', borrow=True)

        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*n_emb+self.tag_dim,
                                                                        n_out=self.n_out, function='softmax').
                           astype(dtype=theano.config.floatX),name='w2', borrow=True)

        b1 = theano.shared(value=np.zeros(self.n_window*n_emb+self.tag_dim, dtype=theano.config.floatX),
                           name='b1', borrow=True)

        b2 = theano.shared(value=np.zeros(self.n_out, dtype=theano.config.floatX),
                           name='b2', borrow=True)

        n_tags = self.n_out+1 # adding the '<PAD>' tag. Should be 37 in total

        tag_lim = np.sqrt(6./(self.n_window+self.tag_dim))
        wt = theano.shared(value=np.random.uniform(-tag_lim,tag_lim,(n_tags,self.tag_dim)).astype(
                            dtype=theano.config.floatX),
                           name='wt', borrow=True)

        params = [w1,b1,w2,b2,wt]
        param_names = ['w1','b1','w2','b2','wt']

        self.params = OrderedDict(zip(param_names, params))

        w_x = w1[idxs].reshape((n_tokens, n_emb*self.n_window))

        minibatch_idx = T.scalar(name='minibatch_idxs', dtype='int32')  # minibatch index

        initial_tag = T.scalar(name='initial_tag',dtype='int32')

        # if use_scan:
        #     TODO: DO I NEED THE SCAN AT ALL: NO! Im leaving it for reference only.
        # Unchanging variables are passed to scan as non_sequences.
        # Initialization occurs in outputs_info
        [y_pred,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[initial_tag,None],
                                non_sequences=[b1,w2,b2,wt])
        # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
        cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        y_predictions = T.argmax(out[:,-1,:], axis=1)
        # else:
        #     out = self.forward_pass(w_x,b1,w2,b2)
        #     y_predictions = T.argmax(out, axis=1)
        #     cost = T.mean(T.nnet.categorical_crossentropy(out, y))

        # test_scan = theano.function(inputs=[idxs,y,initial_tag], outputs=[out,cost,y_pred], updates=[])
        # test_scan(self.x_train,self.y_train,37)

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
                                    initial_tag: np.int32(36)
                                })

        for epoch_index in range(max_epochs):
            epoch_cost = 0
            epoch_errors = 0
            for minibatch_index in range(self.x_train.shape[0]/batch_size):
                # error = train(self.x_train, self.y_train)
                cost_output, errors_output = train(minibatch_index)
                epoch_cost += cost_output
                epoch_errors += errors_output
            print 'Epoch %d Cost: %f Errors: %d' % (epoch_index+1, epoch_cost, epoch_errors)

        if save_params:
            self.save_params()

        return True

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict(self):

        self.x_valid = self.x_valid.astype(dtype='int32')
        self.y_valid = self.y_valid.astype(dtype='int32')

        y = T.vector(name='valid_y', dtype='int32')

        idxs = T.matrix(name="valid_idxs", dtype='int32') # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb*self.n_window))

        initial_tag = T.scalar(name='initial_tag',dtype='int32')

        [y_pred,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[initial_tag,None],
                                non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'], self.params['wt']])

        # out = self.forward_pass(w_x, 36)
        y_predictions = T.argmax(out[:,-1,:], axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out, y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=[y_predictions],
                                updates=[],
                                givens={
                                    initial_tag: np.int32(36)
                                })

        predictions = perform_prediction(self.x_valid)

        return self.y_valid, predictions[-1]

    def to_string(self):
        return 'MLP NN with last predicted tag.'