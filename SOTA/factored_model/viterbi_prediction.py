__author__ = '_lqrz_'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import argparse
from itertools import chain
import cPickle
import numpy as np
import time
from collections import defaultdict
import multiprocessing as mp

from trained_models import get_factored_model_path
from data.dataset import Dataset
from data import get_hierarchical_mapping
from SOTA.language_model.language_model import Language_model
from utils.metrics import Metrics

def multiprocessingWorkaround(arg, **kwarg):
    '''
    Workaround for multiprocessing
    '''

    return Search.expand_nodes(*arg, **kwarg)

class Search(object):
    tag_mapping = get_hierarchical_mapping()
    language_model = Language_model.Instance()
    # language_model = Language_model()

    __instance = None

    def __new__(cls, **kwargs):
        '''
        Singleton pattern
        '''

        if Search.__instance is None:
            Search.__instance = object.__new__(cls, kwargs)
        return Search.__instance

    def __init__(self, top_n, get_output_path, output_distribution, true_values, index2label,
                 stupid_backoff_alpha):
        self.top_n = top_n
        self.get_output_path = get_output_path
        self.output_distribution = output_distribution
        self.true_values = true_values
        self.index2label = index2label
        self.language_model.stupid_backoff_alpha = stupid_backoff_alpha

        self.detailed_tag_log_probability_function = None
        self.meta_tag_log_probability_function = None

    def convert_to_metatag(self, sequence):
        conv = []

        for tag in sequence:
            if tag == '<PAD>':
                conv.append(tag)
            else:
                conv.append(self.tag_mapping[tag])

        assert conv.__len__() == sequence.__len__()

        return conv

    def compute_candidate_probability(self, candidate_ix, candidate_prob, prev_tag, acc_prob, last_word):
        candidate_tag = self.index2label[candidate_ix]
        trigram = prev_tag[-2:] + [candidate_tag]

        trigram_meta_tags = self.convert_to_metatag(trigram)

        detail_log_prob = self.detailed_tag_log_probability_function(trigram)
        meta_log_prob = self.meta_tag_log_probability_function(trigram_meta_tags)

        if last_word:
            trigram = prev_tag[-1:] + [candidate_tag] + ['<PAD>']
            trigram_meta_tags = self.convert_to_metatag(trigram)

            detail_log_prob += self.detailed_tag_log_probability_function(trigram)
            meta_log_prob += self.meta_tag_log_probability_function(trigram_meta_tags)

        tag_log_prob = np.log(candidate_prob)

        return acc_prob + detail_log_prob + meta_log_prob + tag_log_prob

    def expand_nodes(self, word_nr, candidates_dist, prev_tag_info, last_word):
        # .argsort()[::-1][:, :self.top_n]
        prev_tag = prev_tag_info[0]
        acc_prob = prev_tag_info[1]

        return map(lambda x: self.compute_candidate_probability(x[0], x[1], x[2], x[3], x[4]),
                   zip(
                       range(candidates_dist.shape[0]),
                       candidates_dist,
                       [prev_tag]*candidates_dist.shape[0],
                       [acc_prob]*candidates_dist.shape[0],
                       [last_word]*candidates_dist.shape[0])
                   )

    def beam_search_prediction_sentence_level(self, document_sentence_tags):

        self.detailed_tag_log_probability_function = self.language_model.compute_detailed_log_prob_sent_pad
        self.meta_tag_log_probability_function = self.language_model.compute_meta_log_prob_sent_pad

        n_processes = 2
        multiproc = False
        predictions = []
        accum = 0

        pool = mp.Pool(n_processes)

        for doc_sentences in document_sentence_tags.values():
            true_doc_tags = list(chain(*doc_sentences))

            assert self.output_distribution.shape[0] == self.true_values.__len__()

            for sent_tags in doc_sentences:

                sent_len = sent_tags.__len__()
                sent_dist = self.output_distribution[accum:accum+sent_len, :]

                hypothesis = [([u'<PAD>'], .0)]    # its log-space
                for word_nr in range(sent_len):
                    last_word = False
                    candidates_calc_probs = []
                    candidates_dist = sent_dist[word_nr, :]

                    if word_nr == sent_len-1:
                        last_word = True

                    if multiproc:
                        iterab = zip([self]*hypothesis.__len__(), [word_nr]*hypothesis.__len__(), [candidates_dist]*hypothesis.__len__(), hypothesis, [last_word]*hypothesis.__len__())
                        res = pool.map(func=multiprocessingWorkaround, iterable=iterab)
                        candidates_calc_probs.extend(chain(*res))
                    else:
                        for hyp in hypothesis:
                            # expand the node
                            candidates_calc_probs.extend(self.expand_nodes(word_nr, candidates_dist, hyp, last_word))
                    assert candidates_calc_probs.__len__() == hypothesis.__len__() * self.output_distribution.shape[1]
                    # prune the node level
                    top_n_candidates_idxs = np.array(candidates_calc_probs).argsort()[::-1][:self.top_n]
                    pruned_idxs = map(lambda x: np.mod(x, self.output_distribution.shape[1]), top_n_candidates_idxs)
                    pruned_hypothesis_idxs = np.divide(top_n_candidates_idxs, self.output_distribution.shape[1])
                    hypothesis = zip(map(lambda x: x[0]+[x[1]], zip(np.array(hypothesis)[:,0][pruned_hypothesis_idxs],map(lambda x: self.index2label[x], pruned_idxs))), np.array(candidates_calc_probs)[top_n_candidates_idxs])
                accum += sent_len
                predictions.extend(hypothesis[0][0][1:])

        return predictions

    def beam_search_prediction_document_level(self, document_sentence_tags):

        self.detailed_tag_log_probability_function = self.language_model.compute_detailed_log_prob_doc_pad
        self.meta_tag_log_probability_function = self.language_model.compute_meta_log_prob_doc_pad

        n_processes = 2
        multiproc = False
        predictions = []
        accum = 0

        pool = mp.Pool(n_processes)

        for doc_sentences in document_sentence_tags.values():
            true_doc_tags = list(chain(*doc_sentences))

            assert self.output_distribution.shape[0] == self.true_values.__len__()

            doc_len = true_doc_tags.__len__()
            doc_dist = self.output_distribution[accum:accum+doc_len, :]

            hypothesis = [([u'<PAD>'], .0)]    # its log-space
            for word_nr in range(doc_len):
                last_word = False
                candidates_calc_probs = []
                candidates_dist = doc_dist[word_nr, :]

                if word_nr == doc_len-1:
                    last_word = True

                if multiproc:
                    iterab = zip([self]*hypothesis.__len__(), [word_nr]*hypothesis.__len__(), [candidates_dist]*hypothesis.__len__(), hypothesis, [last_word]*hypothesis.__len__())
                    res = pool.map(func=multiprocessingWorkaround, iterable=iterab)
                    candidates_calc_probs.extend(chain(*res))
                else:
                    for hyp in hypothesis:
                        # expand the node
                        candidates_calc_probs.extend(self.expand_nodes(word_nr, candidates_dist, hyp, last_word))
                assert candidates_calc_probs.__len__() == hypothesis.__len__() * self.output_distribution.shape[1]
                # prune the node level
                top_n_candidates_idxs = np.array(candidates_calc_probs).argsort()[::-1][:self.top_n]
                pruned_idxs = map(lambda x: np.mod(x, self.output_distribution.shape[1]), top_n_candidates_idxs)
                pruned_hypothesis_idxs = np.divide(top_n_candidates_idxs, self.output_distribution.shape[1])
                hypothesis = zip(map(lambda x: x[0]+[x[1]], zip(np.array(hypothesis)[:,0][pruned_hypothesis_idxs],map(lambda x: self.index2label[x], pruned_idxs))), np.array(candidates_calc_probs)[top_n_candidates_idxs])
            accum += doc_len
            predictions.extend(hypothesis[0][0][1:])

        return predictions

def get_train_distribution(get_output_path):
    return cPickle.load(open(get_output_path('train_output_distribution.p'), 'rb'))

def get_valid_distribution(get_output_path):
    return cPickle.load(open(get_output_path('valid_output_distribution.p'), 'rb'))

def get_label_index(get_output_path):
    return cPickle.load(open(get_output_path('index2labels.p'), 'rb'))

def parse_arguments():
    parser = argparse.ArgumentParser(description='Viterbi prediction')
    parser.add_argument('--net', type=str, action='store', required=True,
                        choices=['hidden_cw'],
                        help='NNet type')

    level = parser.add_mutually_exclusive_group(required=True)
    level.add_argument('--sentence', action='store_true', default=False)
    level.add_argument('--document', action='store_true', default=False)

    arguments = parser.parse_args()

    args = dict()
    args['model'] = arguments.net
    args['sentence_level'] = arguments.sentence
    args['document_level'] = arguments.document

    return args

def determine_output_path(args):
    get_output_path = None

    #TODO: im always using the factored model path
    if args['model'] == 'hidden_cw':
        get_output_path = get_factored_model_path

    return get_output_path

if __name__ == '__main__':
    args = parse_arguments()

    get_output_path = determine_output_path(args)

    distribution = get_valid_distribution(get_output_path)

    index2label = get_label_index(get_output_path)

    _, _, _, document_sentence_tags = Dataset.get_clef_validation_dataset()

    true_values = list(chain(*(chain(*document_sentence_tags.values()))))

    ba_max_micro_results = defaultdict(int)
    ba_max_macro_results = defaultdict(int)
    bw_max_micro_results = defaultdict(int)
    bw_max_macro_results = defaultdict(int)
    best_ba = None
    best_bw = None
    # for bw in [0.0, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 0.96]:
    for ba in [0.1]:
        for bw in [200]:
            search = Search(top_n=bw,
                            get_output_path=get_output_path,
                            output_distribution=distribution,
                            true_values=true_values,
                            index2label=index2label,
                            stupid_backoff_alpha=ba)

            start = time.time()

            if args['sentence_level']:
                predictions = search.beam_search_prediction_sentence_level(document_sentence_tags)
            elif args['document_level']:
                predictions = search.beam_search_prediction_document_level(document_sentence_tags)

            print 'Elapsed time: %f' % (time.time()-start)

            assert true_values.__len__() == predictions.__len__()

            macro_results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='macro')

            micro_results = Metrics.compute_all_metrics(y_true=true_values, y_pred=predictions, average='micro')

            if macro_results['f1_score'] > bw_max_macro_results['f1_score']:
                best_bw = bw
                bw_max_macro_results = macro_results
                bw_max_micro_results = micro_results

        if bw_max_macro_results['f1_score'] > ba_max_macro_results['f1_score']:
            best_ba = ba
            ba_max_micro_results = bw_max_micro_results
            ba_max_macro_results = bw_max_macro_results

    print 'Best Macro result for ba: %f and bw: %f' % (best_ba, best_bw)
    print '##MACRO RESULTS'
    print ba_max_macro_results
    print '##MICRO RESULTS'
    print ba_max_micro_results
