__author__ = 'lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from itertools import chain
import cPickle
from collections import Counter

from data.dataset import Dataset
from data import get_w2v_training_data_vectors
from data import load_glove_representations

def get_data():
    _, _, train_doc_sent_words, train_doc_sent_tags = Dataset.get_clef_training_dataset()
    _, _, valid_doc_sent_words, valid_doc_sent_tags = Dataset.get_clef_validation_dataset()
    _, _, test_doc_sent_words, test_doc_sent_tags = Dataset.get_clef_testing_dataset()

    train_words = set(chain(*chain(*train_doc_sent_words.values())))
    valid_words = list(chain(*chain(*valid_doc_sent_words.values())))
    test_words = list(chain(*chain(*test_doc_sent_words.values())))

    train_tags = list(chain(*chain(*train_doc_sent_tags.values())))
    valid_tags = list(chain(*chain(*valid_doc_sent_tags.values())))
    test_tags = list(chain(*chain(*test_doc_sent_tags.values())))

    return train_words, valid_words, test_words, train_tags, valid_tags, test_tags

def get_w2v_pretrained_vectors_overlap():
    train_words, valid_words, test_words, train_tags, valid_tags, test_tags = get_data()

    vectors_path = 'googlenews_representations_train_True_valid_True_test_False.p'
    vectors = cPickle.load(open(get_w2v_training_data_vectors(vectors_path), 'rb'))

    train_no_representation, train_tag_no_representation = compute_dataset_overlap(vectors, train_words, train_tags)
    valid_no_representation, valid_tag_no_representation = compute_dataset_overlap(vectors, valid_words, valid_tags)
    test_no_representation, test_tag_no_representation = compute_dataset_overlap(vectors, test_words, test_tags)

    train_no_representation_ratio = train_no_representation.__len__() / float(train_words.__len__())
    valid_no_representation_ratio = valid_no_representation.__len__() / float(valid_words.__len__())
    test_no_representation_ratio = test_no_representation.__len__() / float(test_words.__len__())

    train_tag_no_rep_perc = [(tag, c / float(sum(train_tag_no_representation.values()))) for tag, c in
                             train_tag_no_representation.most_common(n=5)]
    valid_tag_no_rep_perc = [(tag, c / float(sum(valid_tag_no_representation.values()))) for tag, c in
                             valid_tag_no_representation.most_common(n=5)]
    test_tag_no_rep_perc = [(tag, c / float(sum(test_tag_no_representation.values()))) for tag, c in
                            test_tag_no_representation.most_common(n=5)]

    return True

def get_glove_pretrained_vectors_overlap():
    train_words, valid_words, test_words, train_tags, valid_tags, test_tags = get_data()

    vectors_path = 'glove_representations.p'
    vectors = load_glove_representations(vectors_path)

    train_no_representation, train_tag_no_representation = compute_dataset_overlap(vectors, train_words, train_tags)
    valid_no_representation, valid_tag_no_representation = compute_dataset_overlap(vectors, valid_words, valid_tags)
    test_no_representation, test_tag_no_representation = compute_dataset_overlap(vectors, test_words, test_tags)

    train_no_representation_ratio = train_no_representation.__len__() / float(train_words.__len__())
    valid_no_representation_ratio = valid_no_representation.__len__() / float(valid_words.__len__())
    test_no_representation_ratio = test_no_representation.__len__() / float(test_words.__len__())

    train_tag_no_rep_perc = [(tag, c / float(sum(train_tag_no_representation.values()))) for tag, c in
                             train_tag_no_representation.most_common(n=5)]
    valid_tag_no_rep_perc = [(tag, c / float(sum(valid_tag_no_representation.values()))) for tag, c in
                             valid_tag_no_representation.most_common(n=5)]
    test_tag_no_rep_perc = [(tag, c / float(sum(test_tag_no_representation.values()))) for tag, c in
                            test_tag_no_representation.most_common(n=5)]

    return True

def compute_dataset_overlap(vectors, dataset_words, dataset_tags):
    no_representation = []
    tag_no_representation = Counter()
    for word, tag in zip(dataset_words, dataset_tags):
        try:
            vectors[word]
        except KeyError:
            no_representation.append(word)
            tag_no_representation.update([tag])

    return no_representation, tag_no_representation

if __name__ == '__main__':

    get_w2v_pretrained_vectors_overlap()
    get_glove_pretrained_vectors_overlap()

    print 'End'