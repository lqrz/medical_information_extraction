__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from itertools import chain
import numpy as np
from collections import Counter

from data.dataset import Dataset
from SOTA.neural_network.A_neural_network import A_neural_network
from SOTA.neural_network.two_hidden_Layer_Context_Window_Net import Two_Hidden_Layer_Context_Window_Net
from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
from utils import utils
from trained_models import get_POS_nnet_path

def choose_negative_sample(word_idx, pad_idx, n_unique_words):
    choice = np.random.randint(0,n_unique_words)

    if choice == word_idx or choice == pad_idx:
        return choose_negative_sample(word_idx, pad_idx, n_unique_words)

    return choice

if __name__ == '__main__':
    n_window = 5

    hidden_f = utils.NeuralNetwork.tanh_activation_function
    out_f = utils.NeuralNetwork.softmax_activation_function
    get_output_path = get_POS_nnet_path
    args = dict()
    args['n_hidden'] = 100
    args['static'] = False
    minibatch_size = None
    max_epochs = 10
    n_emb = 50
    k = 4

    tokens, tags = Dataset.get_wsj_dataset()

    unique_words = set(chain(*tokens))
    unique_words_and_pad = unique_words.union(['<PAD>'])
    word2index = dict(zip(unique_words_and_pad, range(unique_words_and_pad.__len__())))
    index2word = dict(zip(range(unique_words_and_pad.__len__()), unique_words_and_pad))
    label2index = dict(zip(set(chain(*tags)), range(set(chain(*tags)).__len__())))
    # index2label = dict(zip(range(set(chain(*tags)).__len__()), set(chain(*tags))))

    n_unique_words = word2index.keys().__len__()
    pad_idx = word2index['<PAD>']

    # construct probability dict for NCE expectation calc
    total_tokens = list(chain(*tokens)).__len__()
    word_counts = Counter(list(chain(*tokens)))
    word_counts_probs = dict()
    for word,cnt in word_counts.iteritems():
        word_counts_probs[word] = cnt / float(total_tokens)

    x_train_index = np.floor(tokens.__len__() * .9).astype(int)

    x_train = tokens[:x_train_index]
    y_train = tags[:x_train_index]
    x_valid = tokens[x_train_index:]
    y_valid = tags[x_train_index:]

    x_train_positive = A_neural_network._get_partitioned_data_with_context_window(x_train, n_window, word2index)
    y_train = list(chain(*A_neural_network._get_partitioned_data_without_context_window(y_train, label2index)))
    x_valid = A_neural_network._get_partitioned_data_with_context_window(x_valid, n_window, word2index)
    y_valid = list(chain(*A_neural_network._get_partitioned_data_without_context_window(y_valid, label2index)))

    threshold = 10e-5   #from the paper
    x_train = []
    x_train_probs = []
    # construct the NCE samples
    for win in x_train_positive:
        nce = [win]
        prob = [1.]
        before = win[:n_window / 2]
        after = win[(n_window+1) / 2:]
        word_idx = win[n_window / 2]
        # discard_prob = 1 - np.sqrt(threshold/word_counts[index2word[word_idx]])
        # if np.random.random() > discard_prob:
        #     continue
        for i in range(k):
            negative_sample = choose_negative_sample(word_idx, pad_idx, n_unique_words)
            nce.append(before+[negative_sample]+after)
            prob.append(word_counts_probs[index2word[negative_sample]])
        x_train.append(nce)
        x_train_probs.append(prob)

    n_out = label2index.keys().__len__()

    initial_embeddings = utils.NeuralNetwork.initialize_weights(n_in=word2index.keys().__len__(), n_out=n_emb, function='tanh')

    params = {
        'x_train': np.array(x_train).astype(int),
        'y_train': np.array(y_train).astype(int),
        'x_valid': np.array(x_valid).astype(int),
        'y_valid': np.array(y_valid).astype(int),
        'x_test': None,
        'y_test': None,
        'hidden_activation_f': hidden_f,
        'out_activation_f': out_f,
        'n_window': n_window,
        'pretrained_embeddings': initial_embeddings,
        'n_out': n_out,
        'regularization': True,
        'pad_tag': None,
        'unk_tag': None,
        'pad_word': word2index['<PAD>'],
        'tag_dim': None,
        'get_output_path': get_output_path,
        'train_feats': None,
        'valid_feats': None,
        'test_feats': None,
        'features_indexes': None,
        'train_sent_nr_feats': None,    #refers to sentence nr features.
        'valid_sent_nr_feats': None,    #refers to sentence nr features.
        'test_sent_nr_feats': None,    #refers to sentence nr features.
        'train_tense_feats': None,    #refers to tense features.
        'valid_tense_feats': None,    #refers to tense features.
        'test_tense_feats': None,    #refers to tense features.
        'tense_probs': None,
        'n_filters': None,
        'region_sizes': None,
        'features_to_use': [],
        'static': args['static'],
        'na_tag': None,
        'n_hidden': args['n_hidden'],
        'nce': True,
        'x_train_probs': x_train_probs,
        'k': k
    }

    nnet = Hidden_Layer_Context_Window_Net(**params)

    nnet.train(batch_size=minibatch_size, max_epochs=max_epochs, save_params=True, **params)

    print 'End'