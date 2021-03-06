__author__='lqrz'

from itertools import chain
from collections import Counter
from collections import defaultdict

from data.dataset import Dataset
from data import get_classification_report_labels


def get_overlap():
    _, _, document_sentence_words, document_sentence_tags = Dataset.get_clef_validation_dataset()

    words = list(chain(*chain(*document_sentence_words.values())))
    tags = list(chain(*chain(*document_sentence_tags.values())))

    word_tags = defaultdict(Counter)
    tags_words = defaultdict(list)

    for word, tag in zip(words, tags):
        word_tags[word].update([tag])
        tags_words[tag].append(word)

    assert words.__len__() == sum([sum(c.values()) for c in word_tags.values()])

    tag_overlapping_word_tag = defaultdict(dict)
    tag_overlapping_tag = defaultdict(dict)
    tag_overlap = defaultdict(lambda: defaultdict(int))

    for tag, words in tags_words.iteritems():
        word_types = set(words)
        n_word_types = word_types.__len__()
        overlap = 0
        tag_counter = Counter()
        for word in word_types:
            if list(word_tags[word]).__len__() > 1:
                overlap += 1
                overlapping_tags = [(overlap_tag, count / float(sum(word_tags[word].values()))) for overlap_tag, count
                                    in word_tags[word].most_common()]
                tag_overlapping_word_tag[tag][word] = overlapping_tags
                tag_counter.update(set(word_tags[word]) - set([tag]))
        tag_overlapping_tag[tag] = [(overlap_tag, count / float(sum(tag_counter.values()))) for overlap_tag, count in
                                    tag_counter.most_common()]

        tag_overlap[tag]['n_word_types'] = n_word_types
        tag_overlap[tag]['overlap'] = (overlap, float(overlap) / n_word_types)
        tag_overlap[tag]['unique'] = (n_word_types - overlap, float(n_word_types - overlap) / n_word_types)

    return tag_overlap, tag_overlapping_tag

if __name__ == '__main__':
    tag_overlap, tag_overlapping_tag = get_overlap()

    tag_order_to_print = get_classification_report_labels()
    for tag_to_print in tag_order_to_print:
        # print tag_to_print, tag_overlap[tag_to_print]['n_word_types'], tag_overlap[tag_to_print]['overlap'], \
        #     tag_overlap[tag_to_print]['unique'], tag_overlapping_tag[tag_to_print]
        if tag_overlap[tag_to_print]['n_word_types'] > 0:
            # print '{:6.3f}'.format(tag_overlap[tag_to_print]['unique'][1])
            print tag_to_print
            print ' '.join([tag+':'+'{:6.3f}'.format(per) for tag,per in tag_overlapping_tag[tag_to_print][:3]])
    print '...End'