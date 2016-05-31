__author__ = 'lqrz'

import cPickle
import os.path
import numpy as np

from utils.language_model import compute_and_pickle_counts
from trained_models import get_language_model_path
from utils.singleton import Singleton

@Singleton
class Language_model(object):

    probs_detailed_tags_doc_pad = None
    probs_meta_tags_doc_pad = None
    probs_detailed_tags_sent_pad = None
    probs_meta_tags_sent_pad = None

    def __init__(self, stupid_backoff_alpha=.4):

        self.stupid_backoff_alpha = stupid_backoff_alpha

        if not self.probs_detailed_tags_doc_pad \
            or not self.probs_meta_tags_doc_pad \
            or not self.probs_detailed_tags_sent_pad \
            or not self.probs_meta_tags_sent_pad:
            detailed_tags_doc_pad_path = get_language_model_path('probs_detailed_tags_doc_pad.p')
            meta_tags_doc_pad_path = get_language_model_path('probs_meta_tags_doc_pad.p')
            detailed_tags_sent_pad_path = get_language_model_path('probs_detailed_tags_sent_pad.p')
            meta_tags_sent_pad_path = get_language_model_path('probs_meta_tags_sent_pad.p')
            if not os.path.exists(detailed_tags_doc_pad_path)\
                    or not os.path.exists(meta_tags_doc_pad_path)\
                    or not os.path.exists(detailed_tags_sent_pad_path)\
                    or not os.path.exists(meta_tags_sent_pad_path):
                compute_and_pickle_counts()

            self.probs_detailed_tags_doc_pad = cPickle.load(open(detailed_tags_doc_pad_path, 'rb'))
            self.probs_meta_tags_doc_pad = cPickle.load(open(meta_tags_doc_pad_path, 'rb'))
            self.probs_detailed_tags_sent_pad = cPickle.load(open(detailed_tags_sent_pad_path, 'rb'))
            self.probs_meta_tags_sent_pad = cPickle.load(open(meta_tags_sent_pad_path, 'rb'))

    def compute_detailed_log_prob_sent_pad(self, sequence):
        return self._compute_detailed_log_prob(sequence, self.probs_detailed_tags_sent_pad)

    def compute_meta_log_prob_sent_pad(self, sequence):
        return self._compute_detailed_log_prob(sequence, self.probs_meta_tags_sent_pad)

    def compute_detailed_log_prob_doc_pad(self, sequence):
        return self._compute_detailed_log_prob(sequence, self.probs_detailed_tags_doc_pad)

    def compute_meta_log_prob_doc_pad(self, sequence):
        return self._compute_detailed_log_prob(sequence, self.probs_meta_tags_doc_pad)

    def _compute_detailed_log_prob(self, sequence, dictionary):

        if isinstance(sequence, list):
            sequence = tuple(sequence)

        assert isinstance(sequence, tuple)
        assert sequence.__len__() < 4

        # stupid backoff sequence
        backoff_sequence = None

        log_prob = .0
        if sequence.__len__() == 3:
            # trigram probability
            prob = dictionary[sequence]
            # stupid backoff
            if prob == 0:
                backoff_sequence = sequence[-2:]
                backoff_prob = self._compute_detailed_log_prob(backoff_sequence, dictionary)
                log_prob = np.log(self.stupid_backoff_alpha) + backoff_prob  # the backoff_prob is already in log-space
            else:
                log_prob = np.log(prob)
        elif sequence.__len__() == 2:
            # bigram probability
            prob = dictionary[sequence[-2:]]
            # stupid backoff
            if prob == 0:
                backoff_sequence = sequence[-1:]
                backoff_prob = self._compute_detailed_log_prob(backoff_sequence, dictionary)
                log_prob = np.log(self.stupid_backoff_alpha) + backoff_prob  # the backoff_prob is already in log-space
            else:
                log_prob = np.log(prob)
        elif sequence.__len__() == 1:
            # unigram probability
            prob = dictionary[sequence[0]]
            if prob == 0:
                # if i have never seen it, set it to a very small value
                backoff_prob = 10e-8
                log_prob = np.log(backoff_prob)
            else:
                log_prob = np.log(prob)

        #TODO: apply memoizing
        #TODO: kneser-kney smoothing

        return log_prob