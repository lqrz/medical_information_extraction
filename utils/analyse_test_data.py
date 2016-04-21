__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict

from data.dataset import Dataset


def get_test_data(testing_filename):
    training_data, _, _, _ = Dataset.get_crf_training_data_by_sentence(testing_filename,
                                                                       path=Dataset.TESTING_FEATURES_PATH + 'test',
                                                                       extension=Dataset.TESTING_FEATURES_EXTENSION)

    return training_data


def get_words_per_tag():

    words_per_tag = defaultdict(list)
    for doc_nr, doc_sentences in training_data.iteritems():
        for doc_sent in doc_sentences:
            for word_dict in doc_sent:
                words_per_tag[word_dict['tag']].append(word_dict['word'])

    return words_per_tag

if __name__ == '__main__':
    testing_filename = 'handover-set2.zip'

    training_data = get_test_data(testing_filename)

    words_per_tag = get_words_per_tag(training_data)

    for tag, words in words_per_tag.iteritems():
        print tag+'\t'+', '.join(words)

    print 'End'