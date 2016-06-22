__author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from itertools import chain
from collections import Counter
import numpy as np

from data.dataset import Dataset
import get_word_tenses


def training_tag_tenses_distribution():
    distribution = dict()

    _, _, _, training_tags = Dataset.get_clef_training_dataset()

    tokens = list(chain(*chain(*training_tags.values())))

    tenses = get_word_tenses.get_training_set_tenses()

    tokens_tenses = list(chain(*tenses.values()))

    counts = Counter(zip(tokens, tokens_tenses))

    for (tag, tense), cnt in counts.iteritems():
        distribution[(tag, tense)] = float(cnt) / sum([v for (t,_),v in counts.iteritems() if t==tag])

    return distribution

def training_word_tenses_distribution():
    distribution = dict()

    _, _, training_words, _ = Dataset.get_clef_training_dataset()

    tokens = list(chain(*chain(*training_words.values())))

    tenses = get_word_tenses.get_training_set_tenses()

    tokens_tenses = list(chain(*tenses.values()))

    counts = Counter(zip(tokens, tokens_tenses))

    for (tag, tense), cnt in counts.iteritems():
        distribution[(tag, tense)] = float(cnt) / sum([v for (t,_),v in counts.iteritems() if t==tag])

    return distribution

def training_word_sentence_nr_distribution():
    distribution = dict()

    _, _, training_words, _ = Dataset.get_clef_training_dataset()

    sent_nr = []
    for doc in training_words.values():
        for i, sent in enumerate(doc):
            sent_nr.extend([i]*sent.__len__())

    tokens = list(chain(*chain(*training_words.values())))

    counts = Counter(zip(tokens, sent_nr))

    for (word, sent_nr), cnt in counts.iteritems():
        distribution[(word, sent_nr)] = float(cnt) / sum([v for (w,_),v in counts.iteritems() if w==word])

    return distribution

def training_tag_sentence_nr_distribution():
    distribution = dict()

    _, _, _, training_tags = Dataset.get_clef_training_dataset()

    sent_nr = []
    for doc in training_tags.values():
        for i, sent in enumerate(doc):
            sent_nr.extend([i] * sent.__len__())

    tokens = list(chain(*chain(*training_tags.values())))

    counts = Counter(zip(tokens, sent_nr))

    for (tag, sent_nr), cnt in counts.iteritems():
        distribution[(tag, sent_nr)] = float(cnt) / sum([v for (t, _), v in counts.iteritems() if t == tag])

    return distribution

def training_word_pos_distribution():
    distribution = dict()

    training_data, _, training_words, _ = Dataset.get_clef_training_dataset()

    pos_tags = [word_dict['features'][2] for word_dict in list(chain(*chain(*training_data.values())))]

    tokens = list(chain(*chain(*training_words.values())))

    counts = Counter(zip(tokens, pos_tags))

    for (word, pos_tag), cnt in counts.iteritems():
        distribution[(word,pos_tag)] = float(cnt) / sum([v for (w,_),v in counts.iteritems() if w==word])

    return distribution

def training_word_pos_representations():
    representations = dict()
    distribution = training_word_pos_distribution()
    training_data, _, training_words, _ = Dataset.get_clef_training_dataset()
    pos_tags = [word_dict['features'][2] for word_dict in list(chain(*chain(*training_data.values())))]
    pos2index = dict(zip(set(pos_tags), range(set(pos_tags).__len__())))
    n_pos_tags = set(pos_tags).__len__()
    unique_words = set(chain(*chain(*training_words.values())))

    for word in unique_words:
        rep = [0.] * n_pos_tags
        for pos_tag, prob in [(p,v) for (w,p),v in distribution.iteritems() if w==word]:
            rep[pos2index[pos_tag]] = prob

        assert any(rep)
        representations[word] = rep

    return representations

def training_word_ner_representations():
    representations = dict()
    distribution = training_word_ner_distribution()
    training_data, _, training_words, _ = Dataset.get_clef_training_dataset()
    ner_tags = [word_dict['features'][1] for word_dict in list(chain(*chain(*training_data.values())))]
    ner2index = dict(zip(set(ner_tags), range(set(ner_tags).__len__())))
    n_ner_tags = set(ner_tags).__len__()
    unique_words = set(chain(*chain(*training_words.values())))

    for word in unique_words:
        rep = [0.] * n_ner_tags
        for pos_tag, prob in [(p,v) for (w,p),v in distribution.iteritems() if w==word]:
            rep[ner2index[pos_tag]] = prob

        assert any(rep)
        representations[word] = rep

    return representations


def training_tag_pos_distribution():
    '''
    This one might not make much sense.
    '''

    distribution = dict()

    training_data, _, _, training_tags = Dataset.get_clef_training_dataset()

    pos_tags = [word_dict['features'][2] for word_dict in list(chain(*chain(*training_data.values())))]

    tokens = list(chain(*chain(*training_tags.values())))

    counts = Counter(zip(tokens, pos_tags))

    for (tag, pos_tag), cnt in counts.iteritems():
        distribution[(tag, pos_tag)] = float(cnt) / sum([v for (t, _), v in counts.iteritems() if t == tag])

    return distribution

def training_word_ner_distribution():
    distribution = dict()

    training_data, _, training_words, _ = Dataset.get_clef_training_dataset()

    ner_tags = [word_dict['features'][1] for word_dict in list(chain(*chain(*training_data.values())))]

    tokens = list(chain(*chain(*training_words.values())))

    counts = Counter(zip(tokens, ner_tags))

    for (word, ner_tag), cnt in counts.iteritems():
        distribution[(word, ner_tag)] = float(cnt) / sum([v for (w, _), v in counts.iteritems() if w == word])

    return distribution

def training_tag_ner_distribution():
    '''
    This one might not make much sense.
    '''

    distribution = dict()

    training_data, _, _, training_tags = Dataset.get_clef_training_dataset()

    ner_tags = [word_dict['features'][1] for word_dict in list(chain(*chain(*training_data.values())))]

    tokens = list(chain(*chain(*training_tags.values())))

    counts = Counter(zip(tokens, ner_tags))

    for (tag, ner_tag), cnt in counts.iteritems():
        distribution[(tag, ner_tag)] = float(cnt) / sum([v for (t, _), v in counts.iteritems() if t == tag])

    return distribution

if __name__ == '__main__':
    # training_tag_tenses_distribution()
    # training_word_tenses_distribution()
    # training_word_sentence_nr_distribution()
    # training_word_pos_distribution()
    # training_tag_pos_distribution()
    training_word_ner_distribution()
