__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict
import argparse
import numpy as np
from sklearn.decomposition import PCA as sklearnPCA
from itertools import chain
from ggplot import *
import pandas as pd
import cPickle

from data.dataset import Dataset
from trained_models import get_tf_cwnn_path
from ggplot_lqrz import ggplot_lqrz


def get_data():
    training_data, _, document_sentence_words, document_sentence_tags = \
        Dataset.get_clef_training_dataset(lowercase=True)

    return training_data, document_sentence_words, document_sentence_tags

def get_words_per_tag(training_data):

    words_per_tag = defaultdict(list)
    for doc_nr, doc_sentences in training_data.iteritems():
        for doc_sent in doc_sentences:
            for word_dict in doc_sent:
                words_per_tag[word_dict['tag']].append(word_dict['word'])

    return words_per_tag

def parse_arguments():
    parser = argparse.ArgumentParser(description='Test dataset analyser.')
    # parser.add_argument('--w2vvectorscache', action='store', type=str)
    parser.add_argument('--pca', action='store_true', default=False)
    parser.add_argument('--resize', action='store_true', default=False)
    parser.add_argument('--model', action='store', choices=['mlp'], required=True)

    arguments = parser.parse_args()

    args = dict()
    # args['w2v_vectors_cache'] = arguments.w2vvectorscache
    args['plot_pca'] = arguments.pca
    args['resize'] = arguments.resize
    args['model'] = arguments.model

    return args

# def check_arguments(args):
#     if args['plot_pca'] and not args['w2v_vectors_cache']:
#         print 'Provide w2v vectors for PCA plotting.'
#         exit()

def get_vector_matrix(words_per_tag, word2index, original_embeddings):

    representations = defaultdict(lambda : defaultdict(int))

    processed_words = []
    # repeated = []
    for tag, words in words_per_tag.iteritems():
        for word in words:
            # word = word.lower()
            rep = original_embeddings[word2index[word]]
            if word in processed_words:
                representations[word]['count'] += 1
                representations[word]['tag'] = "overlapped"
                # if (word,tag) not in repeated:
                    # repeated.append((word,tag))
            else:
                representations[word]['rep'] = rep
                representations[word]['tag'] = tag
                representations[word]['count'] += 1
                # reps.append(rep)
                # tags.append(tag)
                processed_words.append(word)

    reps = [d['rep'] for d in representations.values()]
    tags = [d['tag'] for d in representations.values()]
    counts = [d['count'] for d in representations.values()]

    return np.array(reps), np.array(tags), np.array(counts)

# def plot_pca(reps, tags, counts, dimensions, get_output_path, resize=False, title=None, **kwargs):
#
#     plt = None
#
#     if dimensions == 2:
#         sklearn_pca = sklearnPCA(n_components=2)
#         sklearn_transf = sklearn_pca.fit_transform(reps)
#
#         if resize:
#             data = np.concatenate([sklearn_transf, np.matrix(tags).T, np.matrix(counts).T], axis=1)
#             df = pd.DataFrame(data, columns=['x', 'y', 'tag', 'count'])
#
#             p = ggplot_lqrz(df, aes(x='x', y='y', color='tag', size='count')) + \
#                 geom_point() + \
#                 labs(x='1st-component', y='2nd-component', title='PCA') + \
#                 xlim(-5, 5) + \
#                 ylim(-5, 5)
#         else:
#             data = np.concatenate([sklearn_transf, np.matrix(tags).T], axis=1)
#             df = pd.DataFrame(data, columns=['x', 'y', 'tag'])
#
#             p = ggplot_lqrz(df, aes(x='x', y='y', color='tag')) + \
#                 geom_point(size=20) + \
#                 labs(x='1st-component', y='2nd-component', title='PCA') + \
#                 xlim(-5, 5) + \
#                 ylim(-5, 5)
#
#
#     output_filename = ''.join(['test_dataset_pca-%s', '-resize' if resize else '-no_resize',
#                                '-', title if title else '', '.png'])
#     ggsave(get_output_path(output_filename % dimensions), p, dpi=100, bbox_inches='tight')
#
#     return True

def plot_pca(reps, tags, counts, dimensions, get_output_path, resize=False, title=None, **kwargs):
    import rpy2.robjects as robj
    import rpy2.robjects.pandas2ri  # for dataframe conversion
    from rpy2.robjects.packages import importr

    plt = None

    if dimensions == 2:
        sklearn_pca = sklearnPCA(n_components=2)
        sklearn_transf = sklearn_pca.fit_transform(reps)

        if resize:
            data = np.concatenate([sklearn_transf, np.matrix(tags).T, np.matrix(counts).T], axis=1)
            df = pd.DataFrame(data, columns=['x', 'y', 'tag', 'count'])

            p = ggplot_lqrz(df, aes(x='x', y='y', color='tag', size='count')) + \
                geom_point() + \
                labs(x='1st-component', y='2nd-component', title='PCA') + \
                xlim(-5, 5) + \
                ylim(-5, 5)
        else:
            data = np.concatenate([sklearn_transf, np.matrix(tags).T], axis=1)
            df = pd.DataFrame(data, columns=['x', 'y', 'tag'])

            # p = ggplot_lqrz(df, aes(x='x', y='y', color='tag')) + \
            #     geom_point(size=20) + \
            #     labs(x='1st-component', y='2nd-component', title='PCA') + \
            #     xlim(-5, 5) + \
            #     ylim(-5, 5)

            plotFunc = robj.r("""
                library(ggplot2)

                function(df, title){
                    str(df)
                    df$x <- as.numeric(as.character(df$x))
                    df$y <- as.numeric(as.character(df$y))
                    str(df)
                    p <- ggplot(df, aes(x=x, y=y, colour=tag)) +
                    geom_point() +
                    scale_colour_manual(values =
                        c("overlapped"="black")
                        ) +
                    labs(x='1st-component', y='2nd-component', title='PCA') +
                    xlim(-1,1) +
                    ylim(-1,1)

                    print(p)

                    ggsave(title, plot=p, width=14, height=7)

                    }
                """)

            gr = importr('grDevices')
            robj.pandas2ri.activate()
            conv_df = robj.conversion.py2ri(df)

            output_filename = ''.join(['test_dataset_pca-%s', '-resize' if resize else '-no_resize',
                                       '-', title if title else '', '.png'])

            plotFunc(conv_df, get_output_path(output_filename % dimensions))

            gr.dev_off()

    # output_filename = ''.join(['test_dataset_pca-%s', '-resize' if resize else '-no_resize',
    #                            '-', title if title else '', '.png'])
    # ggsave(get_output_path(output_filename % dimensions), p, dpi=100, bbox_inches='tight')

    return True

def determine_output_path(args):
    get_output_path = None

    if args['model'] == 'mlp':
        get_output_path = get_tf_cwnn_path

    return get_output_path

if __name__ == '__main__':

    args = parse_arguments()

    get_output_path = determine_output_path(args)

    # check_arguments(args)

    _, document_sentence_words, document_sentence_tags = get_data()
    all_words = list(chain(*chain(*document_sentence_words.values())))
    all_tags = list(chain(*chain(*document_sentence_tags.values())))

    word_tags = defaultdict(set)
    tags_words = defaultdict(list)

    for word, tag in zip(all_words, all_tags):
        word_tags[word].add(tag)
        tags_words[tag].append(word)

    tag_overlapping_word_tag = defaultdict(dict)
    tag_overlapping_tag = defaultdict(dict)
    tag_overlap = defaultdict(lambda: defaultdict(int))

    words_per_tag = defaultdict(list)
    for tag, words in tags_words.iteritems():
        # word_types = set(words)
        for word in words:
            if list(word_tags[word]).__len__() > 1:
                words_per_tag['Overlaped'].append(word)
            else:
                words_per_tag[tag].append(word)

    # words_per_tag = get_words_per_tag(validation_data)

    assert all_words.__len__() == np.sum([word_list.__len__() for word_list in words_per_tag.values()])

    # w2v_vectors_cache = Word2Vec.load_w2v(args['w2v_vectors_cache'])
    word2index = cPickle.load(open(get_output_path('word2index.p'), 'rb'))
    original_embeddings = cPickle.load(open(get_output_path('original_vectors.p'), 'rb'))
    trained_embeddings = cPickle.load(open(get_output_path('trained_vectors.p'), 'rb'))

    if args['plot_pca']:
        reps, tags, counts = get_vector_matrix(words_per_tag, word2index, original_embeddings)
        plot_pca(reps, tags, counts, dimensions=2, get_output_path=get_output_path, title='before', **args)
        reps, tags, counts = get_vector_matrix(words_per_tag, word2index, trained_embeddings)
        plot_pca(reps, tags, counts, dimensions=2, get_output_path=get_output_path, title='after', **args)

    for tag, words in words_per_tag.iteritems():
        print tag+'\t'+', '.join(words)

    print 'End'