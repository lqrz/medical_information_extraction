__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from data.dataset import Dataset
from itertools import chain
from nltk import FreqDist
import numpy as np
from utils.metrics import Metrics

if __name__ == '__main__':
    training_data_filename = 'handoverdata.zip'
    testing_data_filename = 'handover-set2.zip'

    _, _, _, training_document_sentence_tags = \
        Dataset.get_crf_training_data_by_sentence(training_data_filename)

    _, _, _, testing_document_sentence_tags = \
        Dataset.get_crf_training_data_by_sentence(testing_data_filename,
                                                  Dataset.TESTING_FEATURES_PATH+'test',
                                                  Dataset.TESTING_FEATURES_EXTENSION)

    training_tags = list(chain(*chain(*training_document_sentence_tags.values())))

    fd = FreqDist(training_tags)

    tag_index = fd.keys()

    total_count = np.sum(fd.values())
    probs = [count/float(total_count) for count in fd.values()]

    testing_tags = list(chain(*chain(*testing_document_sentence_tags.values())))

    random_predictions = []
    for test_tag in testing_tags:
        p = np.random.random()
        random_predictions.append(tag_index[np.where(p<np.cumsum(probs))[0][0]])

    assert random_predictions.__len__()==testing_tags.__len__()

    print 'MACRO results'
    print Metrics.compute_all_metrics(testing_tags, random_predictions, average='macro')

    print 'End'