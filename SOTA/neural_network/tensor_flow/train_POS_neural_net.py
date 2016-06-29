__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))))

from itertools import chain
import numpy as np
from collections import Counter
from collections import deque
from numpy import random
import cPickle
from collections import defaultdict

from data.dataset import Dataset
from trained_models import get_POS_nnet_path
# from SOTA.neural_network.A_neural_network import A_neural_network
# from SOTA.neural_network.two_hidden_Layer_Context_Window_Net import Two_Hidden_Layer_Context_Window_Net
# from SOTA.neural_network.hidden_Layer_Context_Window_Net import Hidden_Layer_Context_Window_Net
# from utils import utils

import tensorflow as tf
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def choose_negative_sample(word_idx, pad_idx, n_unique_words):
    choice = np.random.randint(0,n_unique_words)

    if choice == word_idx or choice == pad_idx:
        return choose_negative_sample(word_idx, pad_idx, n_unique_words)

    return choice


data_index = 0

def generate_batch_skipgram_postag(data, batch_size, num_skips, skip_window, **kwargs):
    global data_index

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]

        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

def generate_batch_skipgram_words(data, batch_size, num_skips, skip_window, tags_indexes):
    global data_index

    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 # [ skip_window target skip_window ]
    buffer = deque(maxlen=span)
    buffer_tags = deque(maxlen=span)

    for _ in range(span):
        buffer.append(data[data_index])
        buffer_tags.append(tags_indexes[data_index])
        data_index = (data_index + 1) % len(data)

    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer_tags[target]

        buffer.append(data[data_index])
        buffer_tags.append(tags_indexes[data_index])
        data_index = (data_index + 1) % len(data)

    return batch, labels

# batch, labels = generate_batch_skipgram(batch_size=8, num_skips=2, skip_window=1)

def train_pos_embeddings(batch_size, embedding_size, get_output_path, epochs, batch_func, skip_window=1, n_samples=64):
    tokens, tags = Dataset.get_wsj_dataset()
    tags_flat = list(chain(*tags))

    unique_tags = set(tags_flat)
    n_unique_tags = unique_tags.__len__()
    tag2index = dict(zip(unique_tags, range(n_unique_tags)))
    index2tag = dict(zip(range(n_unique_tags), unique_tags))

    tags_flat_indexes = map(lambda x: tag2index[x], tags_flat)

    # batch_inputs, batch_labels = generate_batch_skipgram(tags_flat_indexes, batch_size=8, num_skips=skip_window*2, skip_window=skip_window)

    final_embeddings = train_graph(tags_flat_indexes,
                                   vocabulary_size=n_unique_tags,
                                   embedding_size=embedding_size,
                                   batch_size=batch_size,
                                   num_sampled=n_samples,
                                   num_skips=skip_window * 2,
                                   skip_window=skip_window,
                                   epochs=epochs,
                                   batch_func=batch_func)

    representations = dict(zip(map(lambda x: index2tag[x], range(final_embeddings.shape[0])), final_embeddings))
    cPickle.dump(representations, open(get_output_path('pos_final_embeddings.p'), 'wb'))

    plot_with_labels(final_embeddings, index2tag, filename=get_output_path('pos_embeddings_tsne.png'),
                     plot_only=final_embeddings.shape[0])

    return

def train_word_embeddings(batch_size, embedding_size, get_output_path, epochs, batch_func, skip_window=1,
                          n_samples=64, min_count=5):
    tokens, tags = Dataset.get_wsj_dataset()
    tokens_flat = list(chain(*tokens))

    tokens_flat = [w.lower() for w in tokens_flat]

    counts = Counter(tokens_flat)

    word_tags = defaultdict(set)
    for w,t in zip(tokens_flat, list(chain(*tags))):
        word_tags[w] = word_tags[w].union([t])

    word_min_count = set([w for w, c in counts.iteritems() if c >= min_count])
    unique_tokens = word_min_count.union(['<UNK>'])
    n_unique_tokens = unique_tokens.__len__()
    word2index = dict(zip(unique_tokens, range(n_unique_tokens)))
    index2word = dict(zip(range(n_unique_tokens), unique_tokens))
    tags_flat = list(chain(*tags))

    unique_tags = set(tags_flat).union(['<UNK>'])
    n_unique_tags = unique_tags.__len__()
    tag2index = dict(zip(unique_tags, range(n_unique_tags)))

    tokens_flat = map(lambda x: x if x in word_min_count else '<UNK>', tokens_flat)

    tokens_flat_indexes = map(lambda x: word2index[x], tokens_flat)

    tags_flat_indexes = np.array(map(lambda x: tag2index[x], tags_flat))
    tags_flat_indexes[np.where(np.array(tokens_flat_indexes)==word2index['<UNK>'])[0]] = tag2index['<UNK>']
    # batch_inputs, batch_labels = generate_batch_skipgram(tags_flat_indexes, batch_size=8, num_skips=skip_window*2, skip_window=skip_window)

    final_embeddings = train_graph(tokens_flat_indexes,
                                   vocabulary_size=n_unique_tokens,
                                   embedding_size=embedding_size,
                                   batch_size=batch_size,
                                   num_sampled=n_samples,
                                   num_skips=skip_window * 2,
                                   skip_window=skip_window,
                                   epochs=epochs,
                                   tags_indexes=tags_flat_indexes,
                                   batch_func=batch_func)

    representations = dict(zip(map(lambda x: index2word[x], range(final_embeddings.shape[0])), final_embeddings))
    cPickle.dump(representations, open(get_output_path('word_final_embeddings.p'), 'wb'))

    plot_index2label = dict()
    for ix,w in index2word.iteritems():
        plot_index2label[ix] = w + '(' + ','.join(word_tags[w]) + ')'

    plot_with_labels(final_embeddings, plot_index2label, filename=get_output_path('word_embeddings_tsne.png'),
                     plot_only=250)

    return

def plot_with_labels(final_embeddings, index2tag, filename, plot_only):

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    final_embeddings = np.asfarray(final_embeddings, dtype='float')
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [index2tag[i] for i in xrange(plot_only)]

    assert low_dim_embs.shape[0] >= len(labels), "More labels than embeddings"

    plt.figure(figsize=(18, 18))  #in inches

    for i, label in enumerate(labels):
        x, y = low_dim_embs[i,:]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)

def train_graph(tags_flat_indexes, vocabulary_size, embedding_size, batch_size, num_sampled,
                num_skips, skip_window, epochs, batch_func, tags_indexes=None):

    graph = tf.Graph()

    with graph.as_default():
        # Input data.
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        # valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Ops and variables pinned to the CPU because of missing GPU implementation
        with tf.device('/cpu:0'):
            # Look up embeddings for inputs.
            embeddings = tf.Variable(
                tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

            # Construct the variables for the NCE loss
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                    stddev=1.0 / np.sqrt(embedding_size)))
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # Compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each
        # time we evaluate the loss.
        loss = tf.reduce_mean(
          tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels,
                         num_sampled, vocabulary_size))

        # Construct the SGD optimizer using a learning rate of 1.0.
        optimizer = tf.train.AdamOptimizer(1.0).minimize(loss)

        # Compute the cosine similarity between minibatch examples and all embeddings.
        square_emb = tf.reduce_sum(tf.square(embeddings), keep_dims=False)

        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm

        # Add variable initializer.
        init = tf.initialize_all_variables()

    with tf.Session(graph=graph) as session:
        # We must initialize all variables before we use them.
        init.run()
        print("Initialized")

        samples_per_batch = skip_window * 2 + 1 + batch_size / num_skips

        n_batches = int(np.ceil(tags_flat_indexes.__len__() / float(samples_per_batch)))

        for epoch_ix in range(epochs):

            average_loss = 0

            for batch_ix in xrange(n_batches):
                batch_inputs, batch_labels = batch_func(tags_flat_indexes,
                                                        batch_size, num_skips, skip_window,
                                                        tags_indexes=tags_indexes)
                feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

                # We perform one update step by evaluating the optimizer op (including it
                # in the list of returned values for session.run()
                _, loss_val, emb_sum = session.run([optimizer, loss, square_emb], feed_dict=feed_dict)
                average_loss += loss_val

                if batch_ix % 2000 == 0:
                    if batch_ix > 0:
                        average_loss /= 2000

                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print("Average loss at step %d: %f emb_sum: %f data_idx: %d (epoch: %d)" %
                          (batch_ix, average_loss, emb_sum, (data_index % tags_flat_indexes.__len__()),
                           epoch_ix))

                    average_loss = 0

                    # Note that this is expensive (~20% slowdown if computed every 500 steps)
                    # if step % 10000 == 0:
                    #   sim = similarity.eval()
                    #   for i in xrange(valid_size):
                    #     valid_word = reverse_dictionary[valid_examples[i]]
                    #     top_k = 8 # number of nearest neighbors
                    #     nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    #     log_str = "Nearest to %s:" % valid_word
                    #     for k in xrange(top_k):
                    #       close_word = reverse_dictionary[nearest[k]]
                    #       log_str = "%s %s," % (log_str, close_word)
                    #     print(log_str)
        final_embeddings = normalized_embeddings.eval()

    return final_embeddings

if __name__ == '__main__':
    train_pos = True
    train_words = False
    context = 1 # how many words to the left and to the right
    epochs = 5
    batch_size = 128
    embedding_size = 100
    n_samples = 20  # negative samples
    get_output_path = get_POS_nnet_path

    if train_pos:
        print 'Training POS embeddings'
        train_pos_embeddings(batch_size, embedding_size, get_output_path, epochs, skip_window=context,
                             n_samples=n_samples, batch_func=generate_batch_skipgram_postag)

    if train_words:
        print 'Training word embeddings'
        train_word_embeddings(batch_size=32, embedding_size=100, get_output_path=get_output_path,
                              epochs=4, skip_window=1, n_samples=20, min_count=2, batch_func=generate_batch_skipgram_words)

    exit(0)
    # n_window = 5
    #
    # hidden_f = utils.NeuralNetwork.tanh_activation_function
    # out_f = utils.NeuralNetwork.softmax_activation_function
    # get_output_path = get_POS_nnet_path
    # args = dict()
    # args['n_hidden'] = 100
    # args['static'] = False
    # minibatch_size = None
    # max_epochs = 10
    # n_emb = 50
    # k = 4
    # learning_rate = .1
    # min_count = 4
    #
    # tokens, tags = Dataset.get_wsj_dataset()
    #
    # # tokens = tokens[:1000]
    # # tags = tags[:1000]
    #
    # # to lower
    # tokens = [map(lambda x: x.lower(), sent) for sent in tokens]
    #
    # # apply min_count. Replace by UNK
    # word_counts = Counter(list(chain(*tokens)))
    # tokens_higher_min_count = set([w for w,c in word_counts.iteritems() if c>=min_count])
    # unique_words = set(tokens_higher_min_count)
    # unique_words_pad_unk = unique_words.union(['<PAD>', '<UNK>'])
    # transformed_tokens = []
    # for sent in tokens:
    #     transformed_tokens.append(map(lambda x: x if x in tokens_higher_min_count else '<UNK>', sent))
    #
    # word2index = dict(zip(unique_words_pad_unk, range(unique_words_pad_unk.__len__())))
    # index2word = dict(zip(range(unique_words_pad_unk.__len__()), unique_words_pad_unk))
    #
    # # label2index = dict(zip(set(chain(*tags)), range(set(chain(*tags)).__len__())))
    # # index2label = dict(zip(range(set(chain(*tags)).__len__()), set(chain(*tags))))
    #
    # n_unique_words = word2index.keys().__len__()
    # pad_idx = word2index['<PAD>']
    #
    # # construct probability dict for NCE expectation calc
    # # total_tokens = list(chain(*transformed_tokens)).__len__()
    # # word_counts_probs = dict()
    # # for word,cnt in word_counts.iteritems():
    # #     word_counts_probs[word] = cnt**.75 / float(total_tokens)
    #
    # x_train_index = np.floor(transformed_tokens.__len__() * .9).astype(int)
    #
    # x_train = transformed_tokens[:x_train_index]
    # y_train = tags[:x_train_index]
    # x_valid = transformed_tokens[x_train_index:]
    # y_valid = tags[x_train_index:]
    #
    # x_train_positive = A_neural_network._get_partitioned_data_with_context_window(x_train, n_window, word2index)
    # # y_train = list(chain(*A_neural_network._get_partitioned_data_without_context_window(y_train, label2index)))
    # x_valid_positive = A_neural_network._get_partitioned_data_with_context_window(x_valid, n_window, word2index)
    # # y_valid = list(chain(*A_neural_network._get_partitioned_data_without_context_window(y_valid, label2index)))
    #
    # threshold = 10e-5   #from the paper
    # x_train = []
    # x_train_probs = []
    # y_train = []
    # # construct the NCE samples
    # for win in x_train_positive:
    #     nce = [win]
    #     prob = [1.]
    #     before = win[:n_window / 2]
    #     after = win[(n_window+1) / 2:]
    #     word_idx = win[n_window / 2]
    #     # discard_prob = 1 - np.sqrt(threshold/word_counts[index2word[word_idx]])
    #     # if np.random.random() > discard_prob:
    #     #     continue
    #     for i in range(k):
    #         negative_sample = choose_negative_sample(word_idx, pad_idx, n_unique_words)
    #         nce.append(before+[negative_sample]+after)
    #         prob.append(word_counts_probs[index2word[negative_sample]])
    #     x_train.append(nce)
    #     x_train_probs.append(prob)
    #
    #     y_train.extend([1] + [0] * k)
    #
    # x_valid = []
    # x_valid_probs = []
    # y_valid = []
    # # construct the NCE samples
    # for win in x_valid_positive:
    #     nce = [win]
    #     prob = [1.]
    #     before = win[:n_window / 2]
    #     after = win[(n_window+1) / 2:]
    #     word_idx = win[n_window / 2]
    #     # discard_prob = 1 - np.sqrt(threshold/word_counts[index2word[word_idx]])
    #     # if np.random.random() > discard_prob:
    #     #     continue
    #     for i in range(k):
    #         negative_sample = choose_negative_sample(word_idx, pad_idx, n_unique_words)
    #         nce.append(before+[negative_sample]+after)
    #         prob.append(word_counts_probs[index2word[negative_sample]])
    #     x_valid.append(nce)
    #     x_valid_probs.append(prob)
    #
    #     y_valid.extend([1]+[0]*k)
    #
    # n_out = label2index.keys().__len__()
    #
    # initial_embeddings = utils.NeuralNetwork.initialize_weights(n_in=word2index.keys().__len__(), n_out=n_emb, function='tanh')
    #
    # params = {
    #     'x_train': np.array(x_train).astype(int),
    #     'y_train': np.array(y_train).astype(int),
    #     'x_valid': np.array(x_valid).astype(int),
    #     'y_valid': np.array(y_valid).astype(int),
    #     'x_test': None,
    #     'y_test': None,
    #     'hidden_activation_f': hidden_f,
    #     'out_activation_f': out_f,
    #     'n_window': n_window,
    #     'pretrained_embeddings': initial_embeddings,
    #     'n_out': n_out,
    #     'regularization': True,
    #     'pad_tag': None,
    #     'unk_tag': None,
    #     'pad_word': word2index['<PAD>'],
    #     'tag_dim': None,
    #     'get_output_path': get_output_path,
    #     'train_feats': None,
    #     'valid_feats': None,
    #     'test_feats': None,
    #     'features_indexes': None,
    #     'train_sent_nr_feats': None,    #refers to sentence nr features.
    #     'valid_sent_nr_feats': None,    #refers to sentence nr features.
    #     'test_sent_nr_feats': None,    #refers to sentence nr features.
    #     'train_tense_feats': None,    #refers to tense features.
    #     'valid_tense_feats': None,    #refers to tense features.
    #     'test_tense_feats': None,    #refers to tense features.
    #     'tense_probs': None,
    #     'n_filters': None,
    #     'region_sizes': None,
    #     'features_to_use': [],
    #     'static': args['static'],
    #     'na_tag': None,
    #     'n_hidden': args['n_hidden'],
    #     'nce': True,
    #     'x_train_probs': x_train_probs,
    #     'k': k,
    #     'learning_rate': learning_rate
    # }
    #
    # nnet = Hidden_Layer_Context_Window_Net(**params)
    #
    # nnet.train(batch_size=minibatch_size, max_epochs=max_epochs, save_params=True, **params)
    #
    # print 'End'