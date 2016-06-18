__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from collections import OrderedDict
import time
from itertools import chain

from A_neural_network import A_neural_network
from trained_models import get_cwnn_path
from utils import utils
from utils.metrics import Metrics
from data.dataset import Dataset

INT = 'int64'

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# theano.config.optimizer='fast_compile'
# theano.config.exception_verbosity='high'
# theano.config.warn_float64='raise'
# theano.config.floatX='float64'

class Vector_Tag_Contex_Window_Net(A_neural_network):

    def __init__(self,
                 hidden_activation_f,
                 out_activation_f,
                 tag_dim,
                 regularization,
                 pad_tag,
                 unk_tag,
                 pad_word,
                 **kwargs):

        super(Vector_Tag_Contex_Window_Net, self).__init__(**kwargs)

        self.hidden_activation_f = hidden_activation_f
        self.out_activation_f = out_activation_f

        self.regularization = regularization

        self.tag_dim = tag_dim
        self.pad_tag = pad_tag
        self.unk_tag = unk_tag

        self.pad_word = pad_word

        self.prev_preds = None

        self.params = OrderedDict()

    def train(self, **kwargs):
        if kwargs['batch_size']:
            # train with minibatch
            logger.info('Training with minibatch size: %d' % kwargs['batch_size'])
            self.train_with_minibatch(**kwargs)
        else:
            # train with SGD
            if kwargs['n_hidden']:
                logger.info('Training with SGD two hidden layer')
                self.train_with_sgd_two_layers(**kwargs)
            else:
                logger.info('Training with SGD one hidden layer')
                self.train_with_sgd_one_layer(**kwargs)

        return True

    def sgd_forward_pass(self, idxs, n_tokens):
            idxs_to_replace = T.eq(idxs, self.pad_word).nonzero()[0]
            prev_preds = T.set_subtensor(self.prev_preds[idxs_to_replace], self.pad_tag)
            w_t = self.params['wt'][prev_preds]
            w_x = self.params['w1'][idxs]
            prev_rep = w_t.reshape(shape=(n_tokens,self.tag_dim*self.n_window))
            # return [prev_preds,weight_x]
            h = self.hidden_activation_f(T.concatenate([w_x.reshape(shape=(n_tokens,self.n_emb*self.n_window)),prev_rep], axis=1)+self.params['b1'])
            result = self.out_activation_f(T.dot(h, self.params['w2'])+self.params['b2'])
            pred = T.argmax(result)

            next_preds = T.set_subtensor(prev_preds[self.n_window/2],pred)
            next_preds = T.set_subtensor(next_preds[:-1],next_preds[1:])
            next_preds = T.set_subtensor(next_preds[-((self.n_window/2)+1):],self.unk_tag)

            return [pred,result,prev_preds,next_preds,w_x,w_t]

    def sgd_forward_pass_two_layers(self, idxs, n_tokens):
            idxs_to_replace = T.eq(idxs, self.pad_word).nonzero()[0]
            prev_preds = T.set_subtensor(self.prev_preds[idxs_to_replace], self.pad_tag)
            w_t = self.params['wt'][prev_preds]
            w_x = self.params['w1'][idxs]
            prev_rep = w_t.reshape(shape=(n_tokens,self.tag_dim*self.n_window))
            # return [prev_preds,weight_x]
            h1 = self.hidden_activation_f(T.concatenate([w_x.reshape(shape=(n_tokens,self.n_emb*self.n_window)),prev_rep], axis=1)+self.params['b1'])
            h2 = self.hidden_activation_f(T.dot(h1,self.params['w2']) + self.params['b2'])
            result = self.out_activation_f(T.dot(h2, self.params['w3']) + self.params['b3'])
            pred = T.argmax(result)

            next_preds = T.set_subtensor(prev_preds[self.n_window/2],pred)
            next_preds = T.set_subtensor(next_preds[:-1],next_preds[1:])
            next_preds = T.set_subtensor(next_preds[-((self.n_window/2)+1):],self.unk_tag)

            return [pred,result,prev_preds,next_preds,w_x,w_t]

    def train_with_sgd_one_layer(self, learning_rate=0.01, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_grad_means=False, plot=False, **kwargs):

        logger.info('Mean gradients: '+str(use_grad_means))

        # train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        # train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        # self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # self.n_tokens = idxs.shape[0]    #tokens in sentence
        # n_features = train_x.get_value().shape[1]    #tokens in sentence
        n_tokens = T.scalar(name='n_tokens', dtype=INT)

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*(self.n_emb+self.tag_dim), n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_window*(self.n_emb+self.tag_dim))).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b2', borrow=True)

        # #include tag structure
        wt = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_out,n_out=self.tag_dim,function='tanh').astype(
            dtype=theano.config.floatX), name='wt', borrow=True)
        # wt = theano.shared(value=np.zeros((n_tags,self.tag_dim),dtype=theano.config.floatX), name='wt', borrow=True)  #this was for test only

        # prev_preds = theano.shared(value=np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT),
        self.prev_preds = theano.shared(value=np.array([self.unk_tag]*self.n_window, dtype=INT),
                                   name='previous_predictions', borrow=True)

        # w_x = w1[idxs].reshape((self.n_tokens, self.n_emb*self.n_window))
        # w_x = theano.shared(value=np.zeros(shape=(self.n_window,self.n_emb),dtype=INT),
        #                    name='w_x', borrow=True)
        # w_t = theano.shared(value=np.zeros(shape=(self.n_window,self.tag_dim),dtype=theano.config.floatX),
        #                    name='w_t', borrow=True)
        # w_x = w1[idxs]

        # test = theano.function(inputs=[idxs], outputs=w_x)
        # test(self.x_train[0])

        # w_t = T.vector(name='w_t', dtype=theano.config.floatX)

        params = [w1,b1,w2,b2,wt]
        param_names = ['w1','b1','w2','b2','wt']

        self.params = OrderedDict(zip(param_names, params))

        # w_x = self.params['w1'][idxs]
        # w_t = self.params['wt'][prev_preds]
        # self.params['w_x'] = w_x
        # self.params['w_t'] = w_t

        # grad_params = [self.params['b1'], self.params['w_x'],self.params['w2'],self.params['b2'],self.params['w_t']]
        grad_params_names = ['w_x','w_t','b1','w2','b2']

        pred,out,prev_preds,next_preds,w_x,w_t = self.sgd_forward_pass(idxs, n_tokens)

        self.params['w_x'] = w_x
        self.params['w_t'] = w_t

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1 ** 2)
            L2_w_x = T.sum(w_x ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2_wt = T.sum(wt ** 2)
            L2_w_t = T.sum(w_t ** 2)

            L2 = L2_w_x + L2_w2 + L2_w_t

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))

        if self.regularization:
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = mean_cross_entropy + alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        # for reference: grad_params_names = ['w_x','w_t','b1','w2','b2']
        grads = [T.grad(cost, self.params[param_name]) for param_name in grad_params_names]

        # adagrad
        accumulated_grad = []
        for param_name in grad_params_names:
            if param_name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif param_name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(self.params[param_name].get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        # for reference: grad_params_names = ['w_x','w_t','b1','w2','b2']
        # grad = grads[1]
        # accum_grad = accumulated_grad[1]
        # acum_idx = accum_grad[prev_preds]
        # accum = T.inc_subtensor(acum_idx, T.sqr(grad))
        # upd = - learning_rate * grad / (T.sqrt(accum[prev_preds]) + 10 ** -5)
        # update = T.inc_subtensor(self.params['w_t'], upd)
        # uniq_values, uniq_values_start_idxs,_ = T.extra_ops.Unique(True, True, False)(prev_preds)
        # idxs_to_replace = T.eq(idxs, self.pad_word).nonzero()[0]
        # prev_preds = T.set_subtensor(self.prev_preds[idxs_to_replace], self.pad_tag)
        # w_test = self.params['w1'][idxs].reshape(shape=(n_tokens,self.n_emb*self.n_window))
        # wt_test = self.params['wt'][prev_preds].reshape(shape=(n_tokens,self.tag_dim*self.n_window))
        # conc = T.concatenate([w_test,wt_test], axis=1)
        # test = theano.function(inputs=[idxs,y, n_tokens], outputs=[pred,out,prev_preds], allow_input_downcast=True, on_unused_input='ignore')
        # self.prev_preds.set_value(prev_preds)
        # test = theano.function(inputs=[idxs,y, n_tokens], outputs=[prev_preds,grad,accum_grad,accum,upd, update], allow_input_downcast=True)
        # test1 = theano.function(inputs=[idxs,y, n_tokens], outputs=[cost,errors,next_preds], allow_input_downcast=True)

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds,uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads,axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx,:], grad_mean)

            return res_grad
        #
        # out_grad = T.zeros(shape=(grad.shape[0], self.tag_dim),dtype=theano.config.floatX)
        #
        # # test = theano.function(inputs=[idxs,y,n_tokens],outputs=out_grad, allow_input_downcast=True,on_unused_input='ignore')
        # # test(self.x_train[0], [self.y_train[0]], 1)
        #
        # mean_grad, _ = theano.scan(fn=take_mean,
        #             sequences=[uniq_values,uniq_values_start_idxs],
        #             outputs_info=out_grad,
        #             non_sequences=[prev_preds,grad])

        # f_take_mean = theano.function(inputs=[idxs, y, n_tokens], outputs=[grad,res_grad],
        #                               allow_input_downcast=True)
        # grad,mean_grad = f_take_mean(self.x_train[0], [self.y_train[0]], 1)  #TODO: check
        #
        # grad = T.set_subtensor(grad[:],mean_grad[-1])
        # accum_grad = accumulated_grad[1]
        # acum_idx = accum_grad[prev_preds]
        # accum = T.inc_subtensor(acum_idx, T.sqr(grad))
        # upd = - learning_rate * grad / (T.sqrt(accum[prev_preds]) + 10 ** -5)
        # update = T.inc_subtensor(self.params['w_t'], upd)

        # test = theano.function(inputs=[idxs,y, n_tokens], outputs=[prev_preds,grad,accum_grad,accum,upd, update], allow_input_downcast=True)
        # test(self.x_train[0], [self.y_train[0]], 1)

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
            elif param_name == 'w_t':

                if use_grad_means:

                    uniq_values, uniq_values_start_idxs,_ = T.extra_ops.Unique(True, True, False)(prev_preds)
                    out_grad = T.zeros(shape=(grad.shape[0], self.tag_dim),dtype=theano.config.floatX)

                    mean_grad, _ = theano.scan(fn=take_mean,
                                sequences=[uniq_values,uniq_values_start_idxs],
                                outputs_info=out_grad,
                                non_sequences=[prev_preds,grad])

                    grad = T.set_subtensor(grad[:],mean_grad[-1])
                    accum_grad = accumulated_grad[1]
                    acum_idx = accum_grad[prev_preds]
                    accum = T.inc_subtensor(acum_idx, T.sqr(grad))
                    upd = - learning_rate * grad / (T.sqrt(accum[prev_preds]) + 10 ** -5)
                    update = T.inc_subtensor(self.params['w_t'], upd)
                else:
                    #this will return the whole accum_grad structure incremented in the specified idxs
                    accum = T.inc_subtensor(accum_grad[prev_preds],T.sqr(grad))
                    #this will return the whole w1 structure decremented according to the idxs vector.
                    update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[prev_preds])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['wt'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        # train_index = T.scalar(name='train_index', dtype=INT)

        # test = theano.function(inputs=[train_index], outputs=[cost,errors,updates], givens={
        #                             idxs: train_x[train_index],
        #                             y: train_y[train_index:train_index+1],
        #                             n_tokens: np.int32(1)
        # })
        # test(0)

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost,errors,prev_preds,pred,next_preds],
                                updates=updates,
                                givens={
                                    n_tokens: 1
                                })

        train_predict = theano.function(inputs=[idxs, y],
                                        outputs=[cost, errors, y_predictions, next_preds],
                                        updates=[],
                                        givens={
                                            n_tokens: 1
                                        })

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy,
                                            givens={
                                                n_tokens: 1
                                            })

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[L2_w1, L2_w2, L2_wt],
                                               givens=[])

        flat_true = list(chain(*self.y_valid))

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

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            epoch_l2_w1 = 0
            epoch_l2_w2 = 0
            epoch_l2_wt = 0
            train_cross_entropy = 0
            # predicted_tags = np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT)
            for x_train_sentence, y_train_sentence in zip(self.x_train, self.y_train):
                self.prev_preds.set_value(np.array([self.unk_tag] * self.n_window, dtype=INT))
                # train_x_sample = train_x.get_value()[i]
                # idxs_to_replace = np.where(train_x_sample==self.pad_word)
                # predicted_tags[idxs_to_replace] = self.pad_tag
                for word_cw, word_tag in zip(x_train_sentence, y_train_sentence):
                    cost_output, errors_output, prev_preds_output, pred_output, next_preds_output = train(word_cw, [word_tag])
                    next_preds_output[(self.n_window/2)-1] = word_tag   #do not propagate the prediction, but use the true_tag instead.
                    self.prev_preds.set_value(next_preds_output)
                    epoch_cost += cost_output
                    epoch_errors += errors_output
                    train_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            if self.regularization:
                l2_w1, l2_w2, l2_wt = train_l2_penalty()
                epoch_l2_w1 += l2_w1
                epoch_l2_w2 += l2_w2
                epoch_l2_wt += l2_wt

            valid_error = 0
            valid_cost = 0
            valid_cross_entropy = 0
            predictions = []
            for x_valid_sentence, y_valid_sentence in zip(self.x_valid, self.y_valid):
                self.prev_preds.set_value(np.array([self.unk_tag] * self.n_window, dtype=INT))
                for word_cw, word_tag in zip(x_valid_sentence, y_valid_sentence):
                    cost_output, errors_output, pred, next_preds_output = train_predict(word_cw, [word_tag])
                    valid_cost += cost_output
                    valid_error += errors_output
                    predictions.append(np.asscalar(pred))
                    self.prev_preds.set_value(next_preds_output)
                    valid_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            train_costs_list.append(epoch_cost)
            train_errors_list.append(epoch_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(epoch_l2_w1)
            l2_w2_list.append(epoch_l2_w2)
            l2_wt_list.append(epoch_l2_wt)
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            assert flat_true.__len__() == predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=flat_true, y_pred=predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index + 1, epoch_cost, epoch_errors, valid_cost, valid_error, f1_score, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list=precision_list, recall_list=recall_list, f1_score_list=f1_score_list,
                             actual_time=actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list,
                                actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def train_with_sgd_two_layers(self,
                                  n_hidden,
                                  learning_rate=0.01,
                                  max_epochs=100,
                                  alpha_L2_reg=0.01,
                                  save_params=False,
                                  use_grad_means=False,
                                  plot=False,
                                  **kwargs):

        self.n_hidden = n_hidden

        logger.info('Mean gradients: '+str(use_grad_means))

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        n_tokens = T.scalar(name='n_tokens', dtype=INT)

        w1 = theano.shared(value=np.array(self.pretrained_embeddings).astype(dtype=theano.config.floatX),
                           name='w1', borrow=True)
        w2 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_window*(self.n_emb+self.tag_dim), n_out=n_hidden, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w2', borrow=True)
        w3 = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=n_hidden, n_out=self.n_out, function='tanh').
                           astype(dtype=theano.config.floatX),
                           name='w3', borrow=True)
        b1 = theano.shared(value=np.zeros((self.n_window*(self.n_emb+self.tag_dim))).astype(dtype=theano.config.floatX), name='b1', borrow=True)
        b2 = theano.shared(value=np.zeros(n_hidden).astype(dtype=theano.config.floatX), name='b2', borrow=True)
        b3 = theano.shared(value=np.zeros(self.n_out).astype(dtype=theano.config.floatX), name='b3', borrow=True)

        # #include tag structure
        wt = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=self.n_out,n_out=self.tag_dim,function='tanh').astype(
            dtype=theano.config.floatX), name='wt', borrow=True)
        # wt = theano.shared(value=np.zeros((n_tags,self.tag_dim),dtype=theano.config.floatX), name='wt', borrow=True)  #this was for test only

        # prev_preds = theano.shared(value=np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT),
        self.prev_preds = theano.shared(value=np.array([self.unk_tag]*self.n_window, dtype=INT),
                                   name='previous_predictions', borrow=True)

        params = [w1,b1,w2,b2,wt,w3,b3]
        param_names = ['w1','b1','w2','b2','wt','w3','b3']

        self.params = OrderedDict(zip(param_names, params))

        # grad_params = [self.params['b1'], self.params['w_x'],self.params['w2'],self.params['b2'],self.params['w_t']]
        grad_params_names = ['w_x','w_t','b1','w2','b2','w3','b3']

        pred,out,prev_preds,next_preds,w_x,w_t = self.sgd_forward_pass_two_layers(idxs, n_tokens)

        self.params['w_x'] = w_x
        self.params['w_t'] = w_t

        #TODO: with regularization??
        if self.regularization:
            # symbolic Theano variable that represents the L1 regularization term
            # L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1 ** 2)
            L2_w_x = T.sum(w_x ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2_wt = T.sum(wt ** 2)
            L2_w_t = T.sum(w_t ** 2)

            L2 = L2_w_x + L2_w2 + L2_w_t

        mean_cross_entropy = T.mean(T.nnet.categorical_crossentropy(out, y))

        if self.regularization:
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            cost = mean_cross_entropy + alpha_L2_reg * L2
        else:
            cost = mean_cross_entropy

        y_predictions = T.argmax(out, axis=1)

        errors = T.sum(T.neq(y_predictions,y))

        # for reference: grad_params_names = ['w_x','w_t','b1','w2','b2']
        grads = [T.grad(cost, self.params[param_name]) for param_name in grad_params_names]

        # adagrad
        accumulated_grad = []
        for param_name in grad_params_names:
            if param_name == 'w_x':
                eps = np.zeros_like(self.params['w1'].get_value(), dtype=theano.config.floatX)
            elif param_name == 'w_t':
                eps = np.zeros_like(self.params['wt'].get_value(), dtype=theano.config.floatX)
            else:
                eps = np.zeros_like(self.params[param_name].get_value(), dtype=theano.config.floatX)
            accumulated_grad.append(theano.shared(value=eps, borrow=True))

        def take_mean(uniq_val, uniq_val_start_idx, res_grad, prev_preds, grad):
            same_idxs = T.eq(prev_preds,uniq_val).nonzero()[0]
            same_grads = grad[same_idxs]
            grad_mean = T.mean(same_grads,axis=0)
            res_grad = T.set_subtensor(res_grad[uniq_val_start_idx,:], grad_mean)

            return res_grad

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
            elif param_name == 'w_t':

                if use_grad_means:

                    uniq_values, uniq_values_start_idxs,_ = T.extra_ops.Unique(True, True, False)(prev_preds)
                    out_grad = T.zeros(shape=(grad.shape[0], self.tag_dim),dtype=theano.config.floatX)

                    mean_grad, _ = theano.scan(fn=take_mean,
                                sequences=[uniq_values,uniq_values_start_idxs],
                                outputs_info=out_grad,
                                non_sequences=[prev_preds,grad])

                    grad = T.set_subtensor(grad[:],mean_grad[-1])
                    accum_grad = accumulated_grad[1]
                    acum_idx = accum_grad[prev_preds]
                    accum = T.inc_subtensor(acum_idx, T.sqr(grad))
                    upd = - learning_rate * grad / (T.sqrt(accum[prev_preds]) + 10 ** -5)
                    update = T.inc_subtensor(self.params['w_t'], upd)
                else:
                    #this will return the whole accum_grad structure incremented in the specified idxs
                    accum = T.inc_subtensor(accum_grad[prev_preds],T.sqr(grad))
                    #this will return the whole w1 structure decremented according to the idxs vector.
                    update = T.inc_subtensor(param, - learning_rate * grad/(T.sqrt(accum[prev_preds])+10**-5))
                #update whole structure with whole structure
                updates.append((self.params['wt'],update))
                #update whole structure with whole structure
                updates.append((accum_grad,accum))
            else:
                accum = accum_grad + T.sqr(grad)
                updates.append((param, param - learning_rate * grad/(T.sqrt(accum)+10**-5)))
                updates.append((accum_grad, accum))

        train = theano.function(inputs=[idxs, y],
                                outputs=[cost,errors,prev_preds,pred,next_preds],
                                updates=updates,
                                givens={
                                    n_tokens: 1
                                })

        train_predict = theano.function(inputs=[idxs, y],
                                        outputs=[cost, errors, y_predictions, next_preds],
                                        updates=[],
                                        givens={
                                            n_tokens: 1
                                        })

        get_cross_entropy = theano.function(inputs=[idxs, y],
                                            outputs=mean_cross_entropy,
                                            givens={
                                                n_tokens: 1
                                            })

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[],
                                               outputs=[L2_w1, L2_w2, L2_wt],
                                               givens=[])

        flat_true = list(chain(*self.y_valid))

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

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            epoch_l2_w1 = 0
            epoch_l2_w2 = 0
            epoch_l2_wt = 0
            train_cross_entropy = 0
            # predicted_tags = np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT)
            for x_train_sentence, y_train_sentence in zip(self.x_train, self.y_train):
                self.prev_preds.set_value(np.array([self.unk_tag] * self.n_window, dtype=INT))
                # train_x_sample = train_x.get_value()[i]
                # idxs_to_replace = np.where(train_x_sample==self.pad_word)
                # predicted_tags[idxs_to_replace] = self.pad_tag
                for word_cw, word_tag in zip(x_train_sentence, y_train_sentence):
                    cost_output, errors_output, prev_preds_output, pred_output, next_preds_output = train(word_cw, [word_tag])
                    next_preds_output[(self.n_window/2)-1] = word_tag   #do not propagate the prediction, but use the true_tag instead.
                    self.prev_preds.set_value(next_preds_output)
                    epoch_cost += cost_output
                    epoch_errors += errors_output
                    train_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            if self.regularization:
                l2_w1, l2_w2, l2_wt = train_l2_penalty()
                epoch_l2_w1 += l2_w1
                epoch_l2_w2 += l2_w2
                epoch_l2_wt += l2_wt

            valid_error = 0
            valid_cost = 0
            valid_cross_entropy = 0
            predictions = []
            for x_valid_sentence, y_valid_sentence in zip(self.x_valid, self.y_valid):
                self.prev_preds.set_value(np.array([self.unk_tag] * self.n_window, dtype=INT))
                for word_cw, word_tag in zip(x_valid_sentence, y_valid_sentence):
                    cost_output, errors_output, pred, next_preds_output = train_predict(word_cw, [word_tag])
                    valid_cost += cost_output
                    valid_error += errors_output
                    predictions.append(np.asscalar(pred))
                    self.prev_preds.set_value(next_preds_output)
                    valid_cross_entropy += get_cross_entropy(word_cw, [word_tag])

            train_costs_list.append(epoch_cost)
            train_errors_list.append(epoch_errors)
            valid_costs_list.append(valid_cost)
            valid_errors_list.append(valid_error)
            l2_w1_list.append(epoch_l2_w1)
            l2_w2_list.append(epoch_l2_w2)
            l2_wt_list.append(epoch_l2_wt)
            train_cross_entropy_list.append(train_cross_entropy)
            valid_cross_entropy_list.append(valid_cross_entropy)

            assert flat_true.__len__() == predictions.__len__()
            results = Metrics.compute_all_metrics(y_true=flat_true, y_pred=predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Valid_cost: %f Valid_errors: %d F1-score: %f Took: %f'
                        % (epoch_index + 1, epoch_cost, epoch_errors, valid_cost, valid_error, f1_score, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, valid_costs_list,
                                              valid_errors_list,
                                              actual_time)
            self.plot_scores(precision_list=precision_list, recall_list=recall_list, f1_score_list=f1_score_list,
                             actual_time=actual_time)
            self.plot_penalties(l2_w1_list=l2_w1_list, l2_w2_list=l2_w2_list, l2_wt_list=l2_wt_list,
                                actual_time=actual_time)
            self.plot_cross_entropies(train_cross_entropy_list, valid_cross_entropy_list, actual_time)

        if save_params:
            logger.info('Saving parameters to File system')
            self.save_params()

        return True

    def train_with_minibatch(self):
        raise Exception('Vector tag with minibatch not implemented!')

    def save_params(self):
        for param_name,param_obj in self.params.iteritems():
            cPickle.dump(param_obj, open(get_cwnn_path(param_name+'.p'),'wb'))

        return True

    def predict(self, on_training_set=False, on_validation_set=False, on_testing_set=False, **kwargs):

        results = dict()

        if on_training_set:
            x_test = self.x_train
            y_test = self.y_train
        elif on_validation_set:
            x_test = self.x_valid
            y_test = self.y_valid
        elif on_testing_set:
            x_test = self.x_test
            y_test = self.y_test

        # y = T.vector(name='test_y', dtype='int64')

        test_idxs = T.vector(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence

        n_tokens = T.scalar(name='n_tokens', dtype=INT)

        self.prev_preds = theano.shared(value=np.array([self.unk_tag]*self.n_window, dtype=INT),
                                        name='previous_predictions', borrow=True)

        pred, _, _, next_preds, _, _ = self.sgd_forward_pass(test_idxs,n_tokens)

        perform_prediction = theano.function(inputs=[test_idxs],
                                             outputs=[pred, next_preds],
                                             givens={
                                                 n_tokens: 1
                                             })

        predictions = []
        for test_sent in x_test:
            # test_x_sample = test_x.get_value()[i]
            # idxs_to_replace = np.where(test_x_sample==self.pad_word)
            # predicted_tags[idxs_to_replace] = self.pad_tag
            self.prev_preds.set_value(np.array([self.unk_tag]*self.n_window, dtype=INT))
            for word_cw in test_sent:
                pred, next_preds = perform_prediction(word_cw)
                # pred, next_preds_output = perform_prediction(i)
                self.prev_preds.set_value(next_preds)
                # predicted_tags[(self.n_window/2)] = pred
                # predicted_tags = np.concatenate((predicted_tags[1:],[self.unk_tag]))
                predictions.append(np.asscalar(pred))

        flat_true = list(chain(*y_test))

        assert flat_true.__len__() == predictions.__len__()

        results['flat_trues'] = flat_true
        results['flat_predictions'] = predictions

        return results

    def to_string(self):
        return 'Ensemble single layer MLP NN with no tags.'


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


        return x_train, y_train, x_train_feats, \
               x_valid, y_valid, x_valid_feats, \
               x_test, y_test, x_test_feats, \
               word2index, index2word, \
               label2index, index2label, \
               features_indexes