__author__ = 'root'

import logging
import numpy as np
import theano
import theano.tensor as T
import cPickle
from collections import OrderedDict
import time

from A_neural_network import A_neural_network
from trained_models import get_cwnn_path
from utils import utils
from utils.metrics import Metrics

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
            logger.info('Training with SGD')
            self.train_with_sgd(**kwargs)

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

    # def forward_pass_matrix(self, weight_x, weight_t, bias_1, weight_2, bias_2):
    #     # ret = T.eq(idxs, self.pad_word).nonzero()[0]
    #     # prev_preds = T.set_subtensor(prev_preds[ret], 5, inplace=False)
    #     prev_rep = weight_t.reshape(shape=(self.tag_dim*self.n_window,))
    #     # return [prev_preds,weight_x]
    #     w_x_res = weight_x.reshape((self.n_emb*self.n_window,))
    #     h = self.hidden_activation_f(T.concatenate([w_x_res,prev_rep])+bias_1)
    #     result = self.out_activation_f(T.dot(h, weight_2)+bias_2)
    #     pred = T.argmax(result)
    #
    #     return [pred,result]

    def train_with_sgd(self, learning_rate=0.01, max_epochs=100,
              alpha_L1_reg=0.001, alpha_L2_reg=0.01, save_params=False, use_grad_means=False, plot=False, **kwargs):

        logger.info('Mean gradients: '+str(use_grad_means))

        train_x = theano.shared(value=np.array(self.x_train, dtype=INT), name='train_x', borrow=True)
        train_y = theano.shared(value=np.array(self.y_train, dtype=INT), name='train_y', borrow=True)

        y = T.vector(name='y', dtype=INT)

        idxs = T.vector(name="idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        self.n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
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

        n_tags = self.n_out
        # #include tag structure
        wt = theano.shared(value=utils.NeuralNetwork.initialize_weights(n_in=n_tags,n_out=self.tag_dim,function='tanh').astype(
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
            L1 = T.sum(abs(w1)) + T.sum(abs(w2))

            # symbolic Theano variable that represents the squared L2 term
            L2_w1 = T.sum(w1[idxs] ** 2)
            L2_w2 = T.sum(w2 ** 2)
            L2_wt = T.sum(wt[prev_preds] ** 2)

            L2 = L2_w1 + L2_w2 + L2_wt

        if self.regularization:
            # TODO: not passing a 1-hot vector for y. I think its ok! Theano realizes it internally.
            # cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L1_reg*L1 + alpha_L2_reg*L2
            cost = T.mean(T.nnet.categorical_crossentropy(out, y)) + alpha_L2_reg*L2
        else:
            cost = T.mean(T.nnet.categorical_crossentropy(out, y))

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

        train_index = T.scalar(name='train_index', dtype=INT)

        # test = theano.function(inputs=[train_index], outputs=[cost,errors,updates], givens={
        #                             idxs: train_x[train_index],
        #                             y: train_y[train_index:train_index+1],
        #                             n_tokens: np.int32(1)
        # })
        # test(0)

        train = theano.function(inputs=[train_index],
                                outputs=[cost,errors,prev_preds,pred,next_preds],
                                updates=updates,
                                givens={
                                    idxs: train_x[train_index],
                                    y: train_y[train_index:train_index+1],
                                    n_tokens: 1
                                })

        train_predict = theano.function(inputs=[idxs, y],
                                        outputs=[cost, errors, y_predictions],
                                        updates=[],
                                        givens={
                                            n_tokens: 1
                                        })

        if self.regularization:
            train_l2_penalty = theano.function(inputs=[train_index],
                                               outputs=[L2_w1, L2_w2, L2_wt],
                                               givens={
                                                   idxs: train_x[train_index]
                                               })

        flat_true = self.y_test

        # plotting purposes
        train_costs_list = []
        train_errors_list = []
        test_costs_list = []
        test_errors_list = []
        precision_list = []
        recall_list = []
        f1_score_list = []
        l2_w1_list = []
        l2_w2_list = []
        l2_wt_list = []

        for epoch_index in range(max_epochs):
            start = time.time()
            epoch_cost = 0
            epoch_errors = 0
            epoch_l2_w1 = 0
            epoch_l2_w2 = 0
            epoch_l2_wt = 0
            # predicted_tags = np.array(np.concatenate(([self.pad_tag]*(n_window/2),[self.unk_tag]*((n_window/2)+1))), dtype=INT)
            for i in range(self.n_samples):
                # train_x_sample = train_x.get_value()[i]
                # idxs_to_replace = np.where(train_x_sample==self.pad_word)
                # predicted_tags[idxs_to_replace] = self.pad_tag
                cost_output, errors_output, prev_preds_output, pred_output, next_preds_output = train(i)
                self.prev_preds.set_value(next_preds_output)
                # predicted_tags[(n_window/2)] = pred
                # predicted_tags = np.concatenate((predicted_tags[1:],[self.unk_tag]))
                epoch_cost += cost_output
                epoch_errors += errors_output

                if self.regularization:
                    l2_w1, l2_w2, l2_wt = train_l2_penalty(i)

                if i==0:
                    epoch_l2_w1 = l2_w1
                epoch_l2_w2 += l2_w2
                epoch_l2_wt += l2_wt

            test_error = 0
            test_cost = 0
            predictions = []
            for x_sample, y_sample in zip(self.x_test, self.y_test):
                cost_output, errors_output, pred = train_predict(x_sample, [y_sample])
                test_cost += cost_output
                test_error += errors_output
                predictions.append(np.asscalar(pred))

            train_costs_list.append(epoch_cost)
            train_errors_list.append(epoch_errors)
            test_costs_list.append(test_cost)
            test_errors_list.append(test_error)
            l2_w1_list.append(epoch_l2_w1)
            l2_w2_list.append(epoch_l2_w2)
            l2_wt_list.append(epoch_l2_wt)

            results = Metrics.compute_all_metrics(y_true=flat_true, y_pred=predictions, average='macro')
            f1_score = results['f1_score']
            precision = results['precision']
            recall = results['recall']
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1_score)

            end = time.time()
            logger.info('Epoch %d Train_cost: %f Train_errors: %d Test_cost: %f Test_errors: %d F1-score: %f Took: %f'
                        % (epoch_index + 1, epoch_cost, epoch_errors, test_cost, test_error, f1_score, end - start))

        if plot:
            actual_time = str(time.time())
            self.plot_training_cost_and_error(train_costs_list, train_errors_list, test_costs_list,
                                              test_errors_list,
                                              actual_time)
            self.plot_scores(precision_list, recall_list, f1_score_list, actual_time)
            self.plot_penalties(l2_w1_list, l2_w2_list, actual_time=actual_time)

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

    def predict(self, **kwargs):

        self.x_test = self.x_test.astype(dtype=INT)
        self.y_test = self.y_test.astype(dtype=INT)

        test_x = theano.shared(value=self.x_test.astype(dtype=INT), name='test_x', borrow=True)
        test_y = theano.shared(value=self.y_test.astype(dtype=INT), name='test_y', borrow=True)

        # y = T.vector(name='test_y', dtype='int64')

        test_idxs = T.vector(name="test_idxs", dtype=INT) # columns: context window size/lines: tokens in the sentence
        # n_emb = self.pretrained_embeddings.shape[1] #embeddings dimension
        # n_tokens = self.x_train.shape[0]    #tokens in sentence
        # n_tokens = valid_idxs.shape[0]    #tokens in sentence
        # n_window = self.x_train.shape[1]    #context window size    #TODO: replace with self.n_win

        w_x = self.params['w1'][test_idxs]
        # w_x2_ix = w_x2[valid_idxs]
        # w_res = w_x2_ix.shape
        # w_x_res = w_x.reshape((self.n_tokens, self.n_emb*self.n_window))

        # test = theano.function(inputs=[valid_idxs], outputs=[w_res])
        # test(test_x.get_value()[0])

        initial_tags = T.vector(name='initial_tags', dtype=INT)
        n_tokens = T.scalar(name='n_tokens', dtype=INT)
        test_index = T.scalar(name='test_index', dtype=INT)
        # test_idxs = T.vector(name='test_idxs', dtype=INT)

        self.prev_preds = theano.shared(value=np.array([self.unk_tag]*self.n_window, dtype=INT),
                                   name='previous_predictions', borrow=True)

        pred, _, _, next_preds, _, _ = self.sgd_forward_pass(test_idxs,n_tokens)

        # y_predictions = T.argmax(out, axis=1)
        # cost = T.mean(T.nnet.categorical_crossentropy(out[:,-1,:], y))
        # errors = T.sum(T.neq(y_predictions,y))

        # perform_prediction = theano.function(inputs=[valid_idxs, initial_tags],
        #                         outputs=y_predictions[-1],
        #                         updates=[],
        #                         givens=[])

        perform_prediction = theano.function(inputs=[test_index],
                                outputs=[pred,next_preds],
                                updates=[],
                                givens={
                                    test_idxs: test_x[test_index],
                                    n_tokens: 1
                                })

        predictions = []
        for i in range(test_x.get_value().shape[0]):
            # test_x_sample = test_x.get_value()[i]
            # idxs_to_replace = np.where(test_x_sample==self.pad_word)
            # predicted_tags[idxs_to_replace] = self.pad_tag
            pred, next_preds = perform_prediction(i)
            # pred, next_preds_output = perform_prediction(i)
            self.prev_preds.set_value(next_preds)
            # predicted_tags[(self.n_window/2)] = pred
            # predicted_tags = np.concatenate((predicted_tags[1:],[self.unk_tag]))
            predictions.append(np.asscalar(pred))

        # predictions = perform_prediction(self.x_test)
        # predictions = perform_prediction(valid_x.get_value())

        return self.y_test, predictions, self.y_test, predictions

    def to_string(self):
        return 'Ensemble single layer MLP NN with no tags.'