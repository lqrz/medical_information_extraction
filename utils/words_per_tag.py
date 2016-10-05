__author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from collections import defaultdict
from collections import Counter
from itertools import chain
from operator import itemgetter
import numpy as np
import pandas as pd
from ggplot import *
from custom_geom_tile import custom_geom_tile

from data.dataset import Dataset
from data import get_classification_report_labels
from plot_confusion_matrix import ggsave_lqrz
from trained_models import get_analysis_folder_path

def get_training_words_per_tag():

    _, _, document_tokens, document_tags = Dataset.get_clef_training_dataset()

    return get_words_per_tag(document_tokens, document_tags)

def get_validation_words_per_tag():
    _, _, document_tokens, document_tags = Dataset.get_clef_validation_dataset()

    return get_words_per_tag(document_tokens, document_tags)

def get_testing_words_per_tag():
    _, _, document_tokens, document_tags = Dataset.get_clef_testing_dataset()

    return get_words_per_tag(document_tokens, document_tags)

def get_words_per_tag(document_tokens, document_tags):
    words_per_tag = defaultdict(list)

    all_tokens = [w.lower() for w in chain(*(chain(*document_tokens.values())))]
    all_tags = list(chain(*chain(*document_tags.values())))

    assert all_tokens.__len__() == all_tags.__len__()

    for word, tag in zip(all_tokens, all_tags):
        words_per_tag[tag].append(word)

    return words_per_tag

def get_word_per_tag_overlap(dataset_a_words_per_tag, dataset_b_words_per_tag):
    '''
    tag overlap in dataset B wrt dataset A.
    dataset A is meant to be the training dataset.
    '''

    tag_overlap = defaultdict(list)

    for tag, tokens in dataset_b_words_per_tag.iteritems():
        tag_overlap[tag] = [w for w in tokens if w in set(dataset_a_words_per_tag[tag])]

    return tag_overlap

def get_validation_tags_per_word():
    _, _, document_tokens, document_tags = Dataset.get_clef_validation_dataset()

    return get_tags_per_word(document_tokens, document_tags)

def get_testing_tags_per_word():
    _, _, document_tokens, document_tags = Dataset.get_clef_testing_dataset()

    return get_tags_per_word(document_tokens, document_tags)

def get_tags_per_word(document_tokens, document_tags):

    tags_per_word = defaultdict(list)

    all_tokens = [w.lower() for w in chain(*(chain(*document_tokens.values())))]
    all_tags = list(chain(*chain(*document_tags.values())))

    for word, tag in zip(all_tokens, all_tags):
        tags_per_word[word].append(tag)

    return tags_per_word

def plot_confusion_matrix(df_melted, output_filename):
    import rpy2.robjects as robj
    import rpy2.robjects.pandas2ri  # for dataframe conversion
    from rpy2.robjects.packages import importr

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
        library(ggplot2)

        function(df, output_filename){
            df$true  <- as.character(df$true)
            df$true  <- factor(df$true, levels=unique(df$true))
            df$prediction  <- as.character(df$prediction)
            df$prediction  <- factor(df$prediction, levels=unique(df$prediction))
            str(df)

            p <- ggplot(df, aes(x=prediction, y=true, fill=value)) +
                geom_tile(colour='gray92') +
                # scale_fill_gradient(low='gray99', high='steelblue4', guide = guide_legend(title = "Probability")) +
                scale_fill_gradient(low='white', high='steelblue', guide = guide_colourbar(title="", ticks=FALSE,
                                                                                barwidth = 0.5, barheight = 12)) +
                # scale_fill_gradient() +
                labs(x='Overlapping labels', y='Labels', title='') +
                theme(
                    panel.grid.major = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_blank(),
                    axis.ticks = element_blank(),
                    axis.text.x = element_text(angle=90, hjust=1, vjust=0.5))
                    #legend.position="none")

            print(p)

            ggsave(output_filename, plot=p, height=9, width=10, dpi=120)

            }
        """)

    plotFunc(conv_df, output_filename)
    gr.dev_off()

    return True


def training_data_empirical_distribution(tag_data):

    print '## TRAINIG DATA EMPIRICAL DISTRIBUTION'
    total = sum([cnt for _, cnt, _ in tag_data])
    acc = 0
    for tag, cnt, words in sorted(tag_data, key=itemgetter(1), reverse=True):
        acc += float(cnt) / total
        print(tag.replace('_', '\\_') + ' & ' + str(cnt) + ' & ' + '{:6.4f}'.format(
            float(cnt) / total) + ' & ' + ', '.join(words) + ' \\\\')

    return True

def dataset_tag_overlap(dataset_words_per_tag, dataset_tags_per_word, tag2index, title):
    items = []
    items_array = []
    for tag in get_classification_report_labels():

        tokens = dataset_words_per_tag[tag]

        overlap_tags = [t for token in tokens for t in dataset_tags_per_word[token] if t != tag]
        overlap_tokens = [token for token in tokens if
                          list(t for t in dataset_tags_per_word[token] if t != tag).__len__() > 0]

        tag_items = np.zeros(get_classification_report_labels().__len__())
        counts = Counter(overlap_tags)
        for t in set(overlap_tags):
            tag_items[tag2index[t]] = counts[t] / float(overlap_tags.__len__())
        items.append((tag, tag_items))
        items_array.append(tag_items)

        if tokens.__len__() == 0:
            continue

        n_overlap = overlap_tokens.__len__()
        common_tag_overlap = Counter(overlap_tags).most_common(n=3)
        overlap_perc = [(t, float(cnt) / sum(Counter(overlap_tags).values())) for t, cnt in common_tag_overlap]

        print('%s & %d & %d & %6.3f & ' % (
        tag.replace('_', '\\_'), tokens.__len__(), n_overlap, float(n_overlap) / tokens.__len__()))

        for t, perc in overlap_perc:
            print('(%s, %6.3f)' % (t.replace('_', '\\_'), perc))

        print('\\\\')

    # df = pd.DataFrame.from_items(items, columns=get_classification_report_labels())
    df = pd.DataFrame(np.matrix(items_array)[::-1], index=get_classification_report_labels()[::-1],
                      columns=get_classification_report_labels())
    df['true'] = df.index
    df_melted = pd.melt(df, id_vars=['true'], var_name='prediction')

    plot_confusion_matrix(df_melted, get_analysis_folder_path(title))

    return True

def dataset_tag_novelty(dataset_words_per_tag, reference_dataset_words_per_tag):
    for tag in get_classification_report_labels():
        n_tokens = dataset_words_per_tag[tag].__len__()
        if n_tokens == 0:
            continue

        novel = [w for w in dataset_words_per_tag[tag] if w not in set(reference_dataset_words_per_tag[tag])]

        print('%s & %d & %d & %6.3f \\\\' % (
        tag.replace('_', '\\_'), n_tokens, novel.__len__(), novel.__len__() / float(n_tokens)))

    return True

if __name__ == '__main__':

    train_words_per_tag = get_training_words_per_tag()
    valid_words_per_tag = get_validation_words_per_tag()
    test_words_per_tag = get_testing_words_per_tag()

    valid_tag_overlap = get_word_per_tag_overlap(train_words_per_tag, valid_words_per_tag)
    test_tag_overlap = get_word_per_tag_overlap(train_words_per_tag, test_words_per_tag)

    valid_tags_per_word = get_validation_tags_per_word()
    test_tags_per_word = get_testing_tags_per_word()

    tag_data = []
    for tag in get_classification_report_labels():
        tag_data.append((tag, train_words_per_tag[tag].__len__(), [w for w,_ in Counter(train_words_per_tag[tag]).most_common(n=5)]))
        # print tag + ' & ' + str(training_words_per_tag[tag].__len__()) + ' & ' + ', '.join([w for w,_ in Counter(training_words_per_tag[tag]).most_common(n=5)]) + ' \\\\'

    training_data_empirical_distribution(tag_data)

    tag2index = dict(zip(get_classification_report_labels(), range(get_classification_report_labels().__len__())))
    index2tag = dict(zip(range(get_classification_report_labels().__len__()), get_classification_report_labels()))

    print '## VALIDATION SET TAG OVERLAP'
    dataset_tag_overlap(valid_words_per_tag, valid_tags_per_word, tag2index, title='valid_tag_overlap.png')
    dataset_tag_overlap(test_words_per_tag, test_tags_per_word, tag2index, title='test_tag_overlap.png')

    print '## VALIDATION SET TAG NOVELTY'
    dataset_tag_novelty(valid_words_per_tag, train_words_per_tag)

    print '## TESTING SET TAG NOVELTY'
    dataset_tag_novelty(test_words_per_tag, train_words_per_tag)