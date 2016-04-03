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
import time

INT = 'int32'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.warn_float64='raise'
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

        self.n_out = np.int32(n_out)
        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f
        self.pretrained_embeddings = embeddings
        self.regularization = regularization
        self.n_window = n_window

        self.tag_dim = np.int32(tag_dim)
        self.pad_tag = np.int32(pad_tag)
        self.unk_tag = np.int32(unk_tag)

        self.pad_word = np.int32(pad_word)

        self.bos_index = bos_index  #index to begining of sentence

        self.n_samples = np.int32(self.x_train.shape[0])

        self.params = OrderedDict()

    # def forward_pass(self, weight_x, prev_preds, bias_1, weight_2, bias_2, weight_t):
    def forward_pass(self, weight_x, weight_t, bias_1, weight_2, bias_2):
        # ret = T.eq(idxs, self.pad_word).nonzero()[0]
        # prev_preds = T.set_subtensor(prev_preds[ret], 5, inplace=False)
        prev_rep = weight_t.reshape(shape=(self.tag_dim*self.n_window,), ndim=1)
        # return [prev_preds,weight_x]
        h = self.hidden_activation_f(T.concatenate([weight_x.reshape((self.n_emb*self.n_window,)),prev_rep])+bias_1)
        result = self.out_activation_f(T.dot(h, weight_2)+bias_2)
        pred = T.argmax(result)

        return [pred,result]

        # i changed it to T.set_subtensor()
        # prev_preds = T.set_subtensor(prev_preds[self.n_window/2],pred)
        # prev_preds = T.set_subtensor(prev_preds[:-1],prev_preds[1:])
        # prev_preds = T.set_subtensor(prev_preds[-((self.n_window/2)+1):],self.unk_tag)
        # prev_preds = T.concatenate([prev_preds[1:(self.n_window/2)],T.stack(pred),[self.unk_tag]*((self.n_window/2)+1)])

        # return [prev_preds, result]

    def forward_pass_matrix(self, weight_x, weight_t, bias_1, weight_2, bias_2):
        # ret = T.eq(idxs, self.pad_word).nonzero()[0]
        # prev_preds = T.set_subtensor(prev_preds[ret], 5, inplace=False)
        prev_rep = weight_t.reshape(shape=(self.tag_dim*self.n_window,))
        # return [prev_preds,weight_x]
        w_x_res = weight_x.reshape((self.n_emb*self.n_window,))
        h = self.hidden_activation_f(T.concatenate([w_x_res,prev_rep])+bias_1)
        result = self.out_activation_f(T.dot(h, weight_2)+bias_2)
        pred = T.argmax(result)

        return [pred,result]

    def train(self, learning_rate=0.01, batch_size=128, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, **kwargs):

        # self.x_train = self.x_train[:2,:] #TODO: remove this. debug only.
        # self.y_train = self.y_train[:2] #TODO: remove this. debug only.

        # i need these theano vars for the minibatch (the givens in the train function). Otherwise i wouldnt.
        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)
        # x = T.matrix(name='x', dtype=theano.config.floatX)
        # minibatch_idx = T.scalar('minibatch_idx', dtype='int64')  # minibatch index

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
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

        initial_tags = T.vector(name='initial_tags', dtype=INT)

        # w_x = w1[idxs].reshape((self.n_tokens, self.n_emb*self.n_window))
        w_x = w1[idxs]

        w_t = wt[initial_tags]

        params = [w_x,w1,b1,w2,b2,wt]
        param_names = ['w_x','w1','b1','w2','b2','wt']
        grad_params = [w_x,b1,w2,b2,w_t]
        grad_params_names = ['w_x','b1','w2','b2','w_t']

        self.params = OrderedDict(zip(param_names, params))

        # def forward_pass(weight_x, bias_1, weight_2, bias_2):
        #     h = self.hidden_activation_f(weight_x+bias_1)
        #     return self.out_activation_f(T.dot(h, weight_2)+bias_2)

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2 = T.sum(w1 ** 2) + T.sum(w2 ** 2)

        # [y_preds,out], _ = theano.scan(fn=self.forward_pass,
        #                         sequences=[w_x],
        #                         outputs_info=[initial_tags,None],
        #                         non_sequences=[b1,w2,b2,wt])

        # [pred,out], _ = theano.scan(fn=self.forward_pass,
        #                         sequences=[w_x, idxs],
        #                         outputs_info=[initial_tags,None],
        #                         non_sequences=[b1,w2,b2,wt])
        pred,out= self.forward_pass(w_x,w_t,b1,w2,b2)

        # def test(weight_x, idxs):
        #     return [weight_x,idxs]
        # [wx_vec, idxs_vec], _ = theano.scan(fn=test,
        #                         sequences=[w_x, idxs],
        #                         outputs_info=[],
        #                         non_sequences=[])
        # test_multivar_scan = theano.function(inputs=[idxs], outputs=[wx_vec, idxs_vec])

        # test_scan = theano.function(inputs=[idxs, initial_tags], outputs=[prev_pred,out,ret], updates=[])
        # test_scan(np.matrix(self.x_train[0]),np.array([self.pad_tag]*n_window, dtype='int64'))
        # test_wx = theano.function(inputs=[idxs], outputs=w_x, updates=[])
        # test_scan(self.x_train, np.array([self.pad_tag]*n_window, dtype='int64'))

        if self.regularization:
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
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

        # test = theano.function(inputs=[idxs, y, initial_tags], outputs=[cost,errors])
        #
        # test(train_x.get_value()[0],[train_y.get_value()[0]], [36]*self.n_window)

        # test_train = theano.function(inputs=[idxs,y], outputs=[out,cost], updates=[])
        # test_train_error = theano.function(inputs=[idxs,y], outputs=[cost], updates=[])

        #TODO: here im testing cost, probabilities, and error calculation. All ok.
        # test_predictions = theano.function(inputs=[idxs,y], outputs=[cost,out,errors], updates=[])
        # cost_out, probs_out, errors_out = test_predictions(self.x_train,self.y_train)

        # y_probabilities, error = test_train(self.x_train, self.y_train)
        # computed_error = test_train_error(self.x_train, self.y_train)

        # y_probabilities = test_scan(self.x_train)
        # y_predictions = np.argmax(y_probabilities[-1][:,0],axis=1)

        grads = [T.grad(cost, param) for param in grad_params]

        # def test_gradient():
        #
        #     grad_wx = T.grad(cost,w_x)
        #     acum = np.zeros((batch_size,self.n_emb), dtype=theano.config.floatX) + T.sqr(grad_wx)
        #     # upd_wx = T.inc_subtensor(w_x, - learning_rate * grad_wx/(T.sqrt(acum)+10**-5))
        #     upd_wx = - learning_rate * grad_wx/(T.sqrt(acum)+10**-5)
        #     grad_w1 = T.grad(cost,w1)
        #     acum_w1 = np.zeros_like(w1, dtype=theano.config.floatX) + T.sqr(grad_w1)
        #     upd_w1 = - learning_rate * grad_w1/(T.sqrt(acum_w1)+10**-5)
        #
        #     updates = []
        #     for param_name, param, grad, accum_grad in zip(grad_params_names, grad_params, grads, accumulated_grad):
        #         # accum = T.cast(accum_grad + T.sqr(grad), dtype=theano.config.floatX)
        #         if param_name == 'w_x':
        #             accum = accum_grad + T.sqr(grad)
        #             update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum)+10**-5))
        #             updates.append((self.params['w1'],update))
        #         elif param_name == 'w_t':
        #             accum = accum_grad + T.sqr(grad)
        #             update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum)+10**-5))
        #             updates.append((self.params['wt'],update))
        #         else:
        #             accum = accum_grad + T.sqr(grad)
        #             updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
        #             updates.append((accum_grad, accum))
        #
        #     ret = w_x

        # test_gradient()

        # adagrad
        accumulated_grad = []
        for param_name, param in zip(grad_params_names,grad_params):
            if param_name == 'w_x':
                eps = np.zeros((self.n_window,self.n_emb), dtype=theano.config.floatX)
            elif param_name == 'w_t':
                    eps = np.zeros((self.n_window,self.tag_dim), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(param.get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        updates = []
        for param_name, param, grad, accum_grad in zip(grad_params_names, grad_params, grads, accumulated_grad):
            # accum = T.cast(accum_grad + T.sqr(grad), dtype=theano.config.floatX)
            if param_name == 'w_x':
                accum = accum_grad + T.sqr(grad)
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum)+10**-5))
                updates.append((self.params['w1'],update))
            elif param_name == 'w_t':
                accum = accum_grad + T.sqr(grad)
                update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum)+10**-5))
                updates.append((self.params['wt'],update))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        # train_index = T.scalar(name='train_index', dtype='int32')

        train = theano.function(inputs=[idxs, y, initial_tags],
                                outputs=[cost,errors,pred],
                                updates=updates,
                                givens=[])

        # test_grad = theano.function(inputs=[minibatch_idx],
        #                             outputs=grads,
        #                             updates=[],
        #                         givens={
        #                             idxs: train_x[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
        #                             y: train_y[minibatch_idx*batch_size:(minibatch_idx+1)*batch_size],
        #                             initial_tags: np.array([self.pad_tag]*n_window, dtype='int64')
        #                         })
        # start = T.scalar(name='start', dtype='int64')
        # end = T.scalar(name='end', dtype='int64')
        # train = theano.function(inputs=[start,end],
        #                         outputs=[cost,errors],
        #                         updates=[(w1,updates)],
        #                         givens={
        #                             idxs: train_x[start:end],
        #                             y: train_y[start:end],
        #                             initial_tags: np.array([self.pad_tag]*n_window, dtype='int64')
        #                         })
        #TODO: im treating <PAD> and <UNK> tags the same (idx 36) does it make a diff to separate them?

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            predicted_tags = np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT)
            for i in range(self.n_samples):
                train_x_sample = train_x.get_value()[i]
                idxs_to_replace = np.where(train_x_sample==self.pad_word)
                predicted_tags[idxs_to_replace] = self.pad_tag
                cost_output, errors_output, pred = train(train_x_sample, [train_y.get_value()[i]], predicted_tags)
                predicted_tags[(n_window/2)] = pred
                predicted_tags = np.concatenate((predicted_tags[1:],[self.unk_tag]))
                epoch_cost += cost_output
                epoch_errors += errors_output
            logger.info('Epoch %d Cost: %f Errors: %d Took: %f' % (epoch_index+1, epoch_cost, epoch_errors,time.time()-start))

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict(self):

        self.x_test = self.x_test.astype(dtype=INT)
        self.y_test = self.y_test.astype(dtype=INT)

        test_x = theano.shared(value=self.x_test.astype(dtype=INT), name='test_x', borrow=True)
        test_y = theano.shared(value=self.y_test.astype(dtype=INT), name='test_y', borrow=True)

        # y = T.vector(name='test_y', dtype='int64')

        valid_idxs = T.vector(name="valid_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        # n_tokens = valid_idxs.shape[0]    #tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][valid_idxs]
        # w_x2_ix = w_x2[valid_idxs]
        # w_res = w_x2_ix.shape
        # w_x_res = w_x.reshape((self.n_tokens, self.n_emb*self.n_window))

        # test = theano.function(inputs=[valid_idxs], outputs=[w_res])
        # test(test_x.get_value()[0])

        initial_tags = T.vector(name='initial_tags', dtype=INT)

        w_t = self.params['wt'][initial_tags]

        y_pred,out = self.forward_pass_matrix(w_x,w_t,self.params['b1'],self.params['w2'],self.params['b2'])

        y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # errors = T.sum(T.neq(y_predictions,y))

        perform_prediction = theano.function(inputs=[valid_idxs, initial_tags],
                                outputs=y_predictions[-1],
                                updates=[],
                                givens=[])

        predictions = []
        predicted_tags = np.array(np.concatenate(([self.pad_tag]*(self.n_window/2),[self.unk_tag]*((self.n_window/2)+1))), dtype=INT)
        for i in range(test_x.get_value().shape[0]):
            test_x_sample = test_x.get_value()[i]
            idxs_to_replace = np.where(test_x_sample==self.pad_word)
            predicted_tags[idxs_to_replace] = self.pad_tag
            pred = perform_prediction(test_x_sample, predicted_tags)
            predicted_tags[(self.n_window/2)] = pred
            predicted_tags = np.concatenate((predicted_tags[1:],[self.unk_tag]))
            predictions.append(pred)

        # predictions = perform_prediction(self.x_test)
        # predictions = perform_prediction(valid_x.get_value())

        return self.y_test, predictions

    def to_string(self):
        return 'Ensemble single layer MLP NN with no tags.'