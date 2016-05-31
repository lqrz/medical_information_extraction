__author__ = 'lqrz'

from collections import Counter
from itertools import chain
import cPickle
from collections import defaultdict

from data.dataset import Dataset
from data import get_hierarchical_mapping
from trained_models import get_language_model_path


def get_training_data():
    _, _, _, document_sentence_tags = Dataset.get_clef_training_dataset()

    return document_sentence_tags

def pad_per_document(training_document_sentence_tags):

    mapping = get_hierarchical_mapping()

    # PAD per document.
    all_tags_padded = [['<PAD>']+list(chain(*doc_sent))+['<PAD>'] for doc_sent in training_document_sentence_tags.values()]
    all_meta_tags_padded = [['<PAD>']+map(lambda x: mapping[x], list(chain(*doc_sent)))+['<PAD>'] for doc_sent in training_document_sentence_tags.values()]

    return all_tags_padded, all_meta_tags_padded

def pad_per_sentence(training_document_sentence_tags):
    mapping = get_hierarchical_mapping()

    all_tags_padded = [['<PAD>'] + sent_tags + ['<PAD>'] for doc_sent in training_document_sentence_tags.values() for sent_tags in doc_sent]
    all_meta_tags_padded = [['<PAD>'] + map(lambda x: mapping[x], sent_tags) + ['<PAD>'] for doc_sent in training_document_sentence_tags.values() for sent_tags in doc_sent]

    return all_tags_padded, all_meta_tags_padded

def count_tags(all_tags_padded):
    counts = Counter()

    trigrams = []
    bigrams = []
    unigrams = []

    for doc_tags in all_tags_padded:
        doc_trigrams = [tuple(doc_tags[i:i + 3]) for i in range(doc_tags.__len__() - 2)]
        doc_bigrams = [tuple(doc_tags[i:i + 2]) for i in range(doc_tags.__len__() - 1)]
        doc_unigrams = [doc_tags[i:i+1][0] for i in range(doc_tags.__len__()-2)]

        counts.update(doc_trigrams)
        counts.update(doc_bigrams)
        counts.update(doc_unigrams)

        trigrams.extend(doc_trigrams)
        bigrams.extend(doc_bigrams)
        unigrams.extend(doc_unigrams)

    probs = defaultdict(int)
    # trigram probabilities
    for ngram in set(trigrams):
        try:
            bigram = ngram[:2]
            prob = counts[ngram] / float(counts[bigram])
        except ZeroDivisionError:
            raise Exception('Something went wrong. This cannot happen. Debug.')
        probs[ngram] = prob

    # bigram probabilities
    for ngram in set(bigrams):
        try:
            unigram = ngram[0]
            prob = counts[ngram] / float(counts[unigram])
        except ZeroDivisionError:
            raise Exception('Something went wrong. This cannot happen. Debug.')
        probs[ngram] = prob

    for word in set(unigrams):
        probs[word] = counts[word] / float(set(unigrams).__len__())

    # fix prob for sentence start token
    probs[('<PAD>')] = 1.0

    return probs

def compute_and_pickle_counts():
    train_document_sentence_tags = get_training_data()

    all_tags_padded, all_meta_tags_padded = pad_per_document(train_document_sentence_tags)

    probs_detailed = count_tags(all_tags_padded)
    probs_meta = count_tags(all_meta_tags_padded)

    cPickle.dump(probs_detailed, open(get_language_model_path('probs_detailed_tags_doc_pad.p'), 'wb'))
    cPickle.dump(probs_meta, open(get_language_model_path('probs_meta_tags_doc_pad.p'), 'wb'))

    all_tags_padded, all_meta_tags_padded = pad_per_sentence(train_document_sentence_tags)

    probs_detailed = count_tags(all_tags_padded)
    probs_meta = count_tags(all_meta_tags_padded)

    cPickle.dump(probs_detailed, open(get_language_model_path('probs_detailed_tags_sent_pad.p'), 'wb'))
    cPickle.dump(probs_meta, open(get_language_model_path('probs_meta_tags_sent_pad.p'), 'wb'))

    return True

if __name__ == '__main__':
    compute_and_pickle_counts()
    print 'End'