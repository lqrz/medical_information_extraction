__author__ = 'lqrz'

from itertools import chain
from collections import defaultdict
from collections import Counter
import numpy as np
import pandas
from ggplot import *

from data.dataset import Dataset
from data import get_hierarchical_mapping

def get_data():
    _, _, _, train_document_sentence_tags = Dataset.get_clef_training_dataset()
    _, _, _, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

    train_document_tags = [list(chain(*doc_tags)) for doc_tags in train_document_sentence_tags.values()]
    valid_document_tags = [list(chain(*doc_tags)) for doc_tags in valid_document_sentence_tags.values()]

    return train_document_tags, valid_document_tags

def convert_to_aggregated_tags(document_tags):
    mapping = get_hierarchical_mapping()

    converted_doc_tags = [map(lambda x: mapping[x], doc) for doc in document_tags]

    return converted_doc_tags


def get_statistics(document_tags, doc_bins):
    bin_perc = np.cumsum([1. / doc_bins] * doc_bins)

    unnormalized_tag_positions_dict = defaultdict(Counter)
    normalized_tag_positions_dict = defaultdict(Counter)
    tag_position_density_list = defaultdict(lambda: defaultdict(list))
    tag_density_dict = defaultdict(lambda: defaultdict())
    tag_previous_dict = defaultdict(Counter)
    tag_next_dict = defaultdict(Counter)

    for doc_tags in document_tags:
        last_tag = None
        density = 0
        starting_position = 0
        for i, tag in enumerate(doc_tags):
            i_norm = np.min(np.where(i / float(doc_tags.__len__()) < bin_perc)[0])
            unnormalized_tag_positions_dict[tag].update([i])
            normalized_tag_positions_dict[tag].update([i_norm])
            # if last_tag:
            tag_previous_dict[tag].update([last_tag])
            if i < doc_tags.__len__() - 1:
                next_tag = doc_tags[i + 1]
            else:
                next_tag = None
            tag_next_dict[tag].update([next_tag])
            if tag == last_tag:
                density += 1
            else:
                if last_tag:
                    tag_position_density_list[last_tag][starting_position].append(density)
                last_tag = tag
                density = 1
                starting_position = i_norm

    for tag, positions in tag_position_density_list.iteritems():
        for pos, counts in positions.iteritems():
            tag_density_dict[tag][pos] = np.mean(counts)

    return unnormalized_tag_positions_dict, normalized_tag_positions_dict, tag_density_dict, \
           tag_previous_dict, tag_next_dict

if __name__ == '__main__':
    doc_bins = 5

    print '...Getting data'
    train_document_tags, valid_document_tags = get_data()

    aggregated = False
    if aggregated:
        print '...Converting to aggregated tags'
        train_document_tags = convert_to_aggregated_tags(train_document_tags)
        valid_document_tags = convert_to_aggregated_tags(valid_document_tags)

    print '...Getting training statistics'
    train_unnormalized_tag_positions_dict, train_normalized_tag_positions_dict, train_tag_density_dict, \
    train_tag_previous_dict, train_tag_next_dict = get_statistics(train_document_tags, doc_bins)

    print '...Getting validation statistics'
    valid_unnormalized_tag_positions_dict, valid_normalized_tag_positions_dict, valid_tag_density_dict, \
    valid_tag_previous_dict, valid_tag_next_dict = get_statistics(valid_document_tags, doc_bins)

    data = dict()
    for tag, counter in valid_normalized_tag_positions_dict.iteritems():
        values = []
        for i in range(doc_bins):
            values.append(counter[i])
        data[tag] = values
    data['Bin'] = range(doc_bins)

    df = pandas.DataFrame(data)

    x_label = 'Document bin'
    y_label = 'Counts'
    title = 'Tag position'
    output_filename = 'valid_tag_position'
    p = ggplot(data=pandas.melt(df, id_vars=['Bin']),
               aesthetics=aes(x='Bin',
                              y='value',
                              color='variable')) + \
        geom_line()

    ggsave(output_filename + '.png', p, dpi=100, bbox_inches='tight')

    print '...End'