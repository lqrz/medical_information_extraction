__author__ = 'root'

from data import get_w2v_model
from data import get_w2v_training_data_vectors
import logging
from data.dataset import Dataset
import numpy as np
from collections import defaultdict
import theano
import theano.tensor as T
import os.path
import cPickle
from trained_models import get_cwnn_path
from collections import OrderedDict
from utils import utils

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
theano.config.exception_verbosity='high'
theano.config.warn_float64='raise'
# theano.config.floatX='float64'

def transform_crf_training_data(dataset):
    return [(token['word'],token['tag']) for archive in dataset.values() for token in archive.values()]

class Vector_tag_CW_MLP_neural_network_trainer_test(object):

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
                 x_train,
                 y_train,
                 x_test,
                 y_test,
                 embeddings,
                 n_out,
                 n_window,
                 tag_dim,
                 regularization,
                 pad_tag,
                 unk_tag,
                 bos_index,
                 pad_word,
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
        self.n_window = n_window

        self.tag_dim = tag_dim
        self.pad_tag = pad_tag
        self.unk_tag = unk_tag

        self.pad_word = pad_word

        self.bos_index = bos_index  #index to begining of sentence

        self.params = OrderedDict()

    # def forward_pass(self, weight_x, prev_preds, bias_1, weight_2, bias_2, weight_t):
    def forward_pass(self, idxs, prev_preds, weight_1, bias_1, weight_2, bias_2, weight_t):
        ret = T.eq(idxs, self.pad_word).nonzero()[0]
        weight_x = weight_1[idxs].reshape(shape=(self.n_emb*self.n_window,), ndim=1)
        prev_rep = weight_t[prev_preds].reshape(shape=(self.tag_dim*self.n_window,), ndim=1)
        # return [prev_preds,weight_x]
        h = self.hidden_activation_f(T.concatenate([weight_x,prev_rep])+bias_1)
        result = self.out_activation_f(T.dot(h, weight_2)+bias_2)
        pred = T.argmax(result)

        #TODO: change with T.set_subtensor()
        prev_preds = T.concatenate([prev_preds[1:(self.n_window/2)],T.stack(pred),[self.unk_tag]*((self.n_window/2)+1)])

        return [idxs, result,ret]

    def train(self, learning_rate=0.01, batch_size=128, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, **kwargs):

        # self.x_train = self.x_train[:2,:] #TODO: remove this. debug only.
        # self.y_train = self.y_train[:2] #TODO: remove this. debug only.

        # i need these theano vars for the minibatch (the givens in the train function). Otherwise i wouldnt.
        train_x = theano.shared(value=np.array(self.x_train, dtype='int64'), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype='int64'), name='train_y', borrow=True)

        y = T.vector(name='y', dtype='int64')
        # x = T.matrix(name='x', dtype=theano.config.floatX)
        minibatch_idx = T.scalar('minibatch_idx', dtype='int64')  # minibatch index

        idxs = T.matrix(name="idxs", dtype='int64') # columns: context window size/lines: tokens in the sentence
        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        self.n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace n_win with self.n_win
        # n_features = train_x.get_value().shape[1]    #tokens in sentence

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=n_window*(self.n_emb+self.tag_dim), n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((n_window*(self.n_emb+self.tag_dim))).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        n_tags = self.n_out + 2
        #include tag structure
        wt = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=n_tags,n_out=self.tag_dim,function='tanh').astype(
            dtype=theano.config.floatX), name='wt', borrow=True)

        initial_tags = T.vector(name='initial_tags', dtype='int64')

        params = [w1,b1,w2,b2,wt]
        param_names = ['w1','b1','w2','b2','wt']

        self.params = OrderedDict(zip(param_names, params))

        # w_x = w1[idxs].reshape((n_tokens, n_emb*n_window))

        # def forward_pass(weight_x, bias_1, weight_2, bias_2):
        #     h = self.hidden_activation_f(weight_x+bias_1)
        #     return self.out_activation_f(T.dot(h, weight_2)+bias_2)

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            L1 = T.sum(abs(w1))

            # symbolic Theano variable that represents the squared L2 term
            L2 = T.sum(w1 ** 2)

        # [y_preds,out], _ = theano.scan(fn=self.forward_pass,
        #                         sequences=[w_x],
        #                         outputs_info=[initial_tags,None],
        #                         non_sequences=[b1,w2,b2,wt])
        [_,out,ret], _ = theano.scan(fn=self.forward_pass,
                                sequences=[idxs],
                                outputs_info=[initial_tags,None,None],
                                non_sequences=[w1,b1,w2,b2,wt])

        test_scan = theano.function(inputs=[idxs, initial_tags], outputs=[out,ret], updates=[])
        test_scan(np.matrix(self.x_train[0]),np.array([self.pad_tag]*n_window, dtype='int64'))
        # test_wx = theano.function(inputs=[idxs], outputs=w_x, updates=[])
        # test_scan(self.x_train, np.array([self.pad_tag]*n_window, dtype='int64'))

        if self.regularization:
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))

        y_predictions = T.argmax(out[:,-1,:], axis=1)

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

        # train = theano.function(inputs=[minibatch_idx],
        #                         outputs=[cost,errors],
        #                         updates=updates,
        #                         givens={
        #                             idxs: train_x[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
        #                             y: train_y[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
        #                             initial_tags: np.array([self.pad_tag]*n_window, dtype='int64')
        #                         })
        start = T.scalar(name='start', dtype='int64')
        end = T.scalar(name='end', dtype='int64')
        train = theano.function(inputs=[start,end],
                                outputs=[cost,errors],
                                updates=updates,
                                givens={
                                    idxs: train_x[start:end],
                                    y: train_y[start:end],
                                    initial_tags: np.array([self.pad_tag]*n_window, dtype='int64')
                                })
        #TODO: im treating <PAD> and <UNK> tags the same (idx 36) does it make a diff to separate them?

        for epoch_index in range(max_epochs):
            epoch_cost = 0
            epoch_errors = 0
            # for minibatch_index in range(self.x_train.shape[0]/batch_size):
                #TODO: finish thinking this.
            for i,sent_idx in enumerate(list(self.bos_index[0])):
                # start = np.min(np.where(self.bos_index[0]>acc_batch_idx))

                end_ix = (i+1)*2
                if end_ix >= list(self.bos_index[0]).__len__():
                    end = self.x_train.shape[0]-1
                else:
                    end = list(self.bos_index[0])[end_ix]
                # cost_output, errors_output = train(minibatch_index)
                cost_output, errors_output = train(sent_idx, end)
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

        self.x_test = self.x_test.astype(dtype='int32')
        self.y_test = self.y_test.astype(dtype='int32')

        # valid_x = theano.shared(value=self.x_valid.astype(dtype='int32'), name='valid_x', borrow=True)
        # valid_y = theano.shared(value=self.y_valid.astype(dtype='int32'), name='valid_y', borrow=True)

        y = T.vector(name='test_y', dtype='int32')

        idxs = T.matrix(name="valid_idxs", dtype='int32') # columns: context window size/lines: tokens in the sentence
        n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        n_tokens = idxs.shape[0]    #tokens in sentence
        n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][idxs].reshape((n_tokens, n_emb*n_window))

        initial_tags = T.vector(name='initial_tags', dtype='int64')

        [y_pred,out], _ = theano.scan(fn=self.forward_pass,
                                sequences=w_x,
                                outputs_info=[initial_tags,None],
                                non_sequences=[self.params['b1'], self.params['w2'], self.params['b2'], self.params['wt']])

        y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[idxs],
                                outputs=[y_predictions],
                                updates=[],
                                givens={
                                    initial_tags: np.array([36]*self.n_window, dtype='int64')
                                })

        predictions = perform_prediction(self.x_test)
        # predictions = perform_prediction(valid_x.get_value())

        return self.y_test, predictions[-1]

    def to_string(self):
        return 'Ensemble single layer MLP NN with no tags.'