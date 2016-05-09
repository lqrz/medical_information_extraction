__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict
import argparse
import numpy as np
import cPickle as pickle
from sklearn.decomposition import PCA as sklearnPCA
from itertools import chain
from ggplot import *
import pandas as pd

from data.dataset import Dataset
from data import get_w2v_training_data_vectors


def get_test_data(testing_filename):
    training_data, _, _, _ = Dataset.get_crf_training_data_by_sentence(testing_filename,
                                                                       path=Dataset.TESTING_FEATURES_PATH + 'test',
                                                                       extension=Dataset.TESTING_FEATURES_EXTENSION)

    return training_data


def get_words_per_tag(training_data):

    words_per_tag = defaultdict(list)
    for doc_nr, doc_sentences in training_data.iteritems():
        for doc_sent in doc_sentences:
            for word_dict in doc_sent:
                words_per_tag[word_dict['tag']].append(word_dict['word'])

    return words_per_tag

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test dataset analyser.')
    parser.add_argument('--w2vvectorscache', action='store', type=str)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--resize', action='store_true', default=False)

    arguments = parser.parse_args()

    args = dict()
    args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['plot_pca'] = arguments.pca
    args['resize'] = arguments.resize

    return args

def check_arguments(args):
    if args['plot_pca'] and not args['w2v_vectors_cache']:
        print 'Provide w2v vectors for PCA plotting.'
        exit()

def load_w2v_cache(filename):
    return pickle.load(open(get_w2v_training_data_vectors(filename), 'rb'))

def get_vector_matrix(words_per_tag, w2v_vectors_cache):

    representations = defaultdict(lambda : defaultdict(int))

    processed_words = []
    repeated = []
    for tag, words in words_per_tag.iteritems():
        for word in words:
            try:
                word = word.lower()
                rep = w2v_vectors_cache[word]
                if word in processed_words:
                    representations[word]['count'] += 1
                    if (word,tag) not in repeated:
                        repeated.append((word,tag))
                else:
                    representations[word]['rep'] = rep
                    representations[word]['tag'] = tag
                    representations[word]['count'] += 1
                    # reps.append(rep)
                    # tags.append(tag)
                    processed_words.append(word)
            except KeyError:
                pass

    reps = [d['rep'] for d in representations.values()]
    tags = [d['tag'] for d in representations.values()]
    counts = [d['count'] for d in representations.values()]

    return np.array(reps), np.array(tags), np.array(counts), repeated

def plot_pca(reps, tags, counts, dimensions=2, resize=False, **kwargs):

    plt = None

    if dimensions == 2:
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(reps)

        if resize:
            data = np.concatenate([sklearn_transf, np.matrix(tags).T, np.matrix(counts).T], axis=1)
            df = pd.DataFrame(data, columns=['x', 'y', 'tag', 'count'])

            p = ggplot(df, aes(x='x', y='y', color='tag', size='count')) + \
                  geom_point() + \
                  labs(x='1st-component', y='2nd-component', title='PCA')
        else:
            data = np.concatenate([sklearn_transf, np.matrix(tags).T], axis=1)
            df = pd.DataFrame(data, columns=['x', 'y', 'tag'])

            p = ggplot(df, aes(x='x', y='y', color='tag')) + \
                  geom_point(size=20) + \
                  labs(x='1st-component', y='2nd-component', title='PCA')

    output_filename = ''.join(['test_dataset_pca-%s', '-resize' if resize else '-no_resize', '.png'])
    ggsave(output_filename % dimensions, p, dpi=100, bbox_inches='tight')

    return True


if __name__ == '__main__':
    testing_filename = 'handover-set2.zip'

    args = parse_arguments()

    check_arguments(args)

    training_data = get_test_data(testing_filename)

    words_per_tag = get_words_per_tag(training_data)

    w2v_vectors_cache = load_w2v_cache(args['w2v_vectors_cache'])

    if args['plot_pca']:
        reps, tags, counts, repeated = get_vector_matrix(words_per_tag, w2v_vectors_cache)
        plot_pca(reps, tags, counts, dimensions=2, **args)

    for tag, words in words_per_tag.iteritems():
        print tag+'\t'+', '.join(words)

    print 'End'