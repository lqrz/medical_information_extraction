__author__ = 'root'
import gensim
import theano.tensor as T
import numpy as np
import time
import pandas
from ggplot import *

class Word2Vec:

    def __init__(self):
        pass

    @staticmethod
    def load_w2v(model_filename):
        return gensim.models.Word2Vec.load_word2vec_format(model_filename, binary=True)


class NeuralNetwork:
    def __init__(self):
        pass

    @staticmethod
    def replace_with_word_embeddings(w, unique_words, w2v_vectors=None, w2v_model=None):
        for i,word in enumerate(unique_words):
            try:
                if w2v_vectors:
                    w[i,:] = w2v_vectors[word.lower()]
                else:
                    w[i,:] = w2v_model[word.lower()] #TODO: lower?
            except KeyError:
                continue

        return w

    @staticmethod
    def context_window(sentence, n_window, pad_idx=None):
        # make sure its uneven
        assert (n_window % 2) == 1, 'Window size must be uneven.'

        # add '<PAD>' tokens at begining and end of sentence
        if pad_idx:
            l_padded = n_window //2 * [pad_idx] + sentence + n_window // 2 * [pad_idx]
        else:
            l_padded = n_window //2 * ['<PAD>'] + sentence + n_window // 2 * ['<PAD>']

        # slide the window
        return [l_padded[i:(i+n_window)] for i in range(len(sentence))]

    @staticmethod
    def theano_gpu_concatenate(tensor_list, axis=0):
        """
        Alternative implementation of `theano.tensor.concatenate`.
        This function does exactly the same thing, but contrary to Theano's own
        implementation, the gradient is implemented on the GPU.
        Backpropagating through `theano.tensor.concatenate` yields slowdowns
        because the inverse operation (splitting) needs to be done on the CPU.
        This implementation does not have that problem.
        :usage:
            x, y = theano.tensor.matrices('x', 'y')
            c = concatenate([x, y], axis=1)
        :parameters:
            - tensor_list : list
                list of Theano tensor expressions that should be concatenated.
            - axis : int
                the tensors will be joined along this axis.
        :returns:
            - out : tensor
                the concatenated tensor expression.
        """

        import theano
        import theano.tensor as tensor

        concat_size = sum(tt.shape[axis] for tt in tensor_list)

        output_shape = ()
        for k in range(axis):
            output_shape += (tensor_list[0].shape[k],)
        output_shape += (concat_size,)
        for k in range(axis + 1, tensor_list[0].ndim):
            output_shape += (tensor_list[0].shape[k],)

        out = tensor.zeros(output_shape, dtype=tensor_list[0].dtype)
        offset = 0
        for tt in tensor_list:
            indices = ()
            for k in range(axis):
                indices += (slice(None),)
            indices += (slice(offset, offset + tt.shape[axis]),)
            for k in range(axis + 1, tensor_list[0].ndim):
                indices += (slice(None),)

            out = tensor.set_subtensor(out[indices], tt)
            offset += tt.shape[axis]

        return out

    @staticmethod
    def initialize_weights(n_in, n_out, function):
        lim = np.sqrt(6./(n_in+n_out))

        if function=='sigmoid':
            lim *= 4

        return np.random.uniform(-lim,lim,(n_in,n_out))

    @staticmethod
    def linear_activation_function(x):
        return x

    @staticmethod
    def tanh_activation_function(x):
        return T.tanh(x)

    @staticmethod
    def softmax_activation_function(x):
        return T.nnet.softmax(x)

    @staticmethod
    def plot(data, x_axis, x_label, y_label, title, output_filename=str(time.time())):
        df = pandas.DataFrame(data)

        p = ggplot(pandas.melt(df, id_vars=[x_axis]), aes(x=x_axis, y='value', color='variable')) + \
            geom_line() + \
            labs(x=x_label, y=y_label) + \
            ggtitle(title)

        ggsave(output_filename+'.png', p, dpi=100)

        return True

    @staticmethod
    def relu(x):
        return T.nnet.relu(x)