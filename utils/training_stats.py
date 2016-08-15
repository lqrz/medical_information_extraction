__author__ = 'lqrz'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from itertools import chain
import numpy as np
import pandas as pd
import rpy2.robjects as robj
import rpy2.robjects.pandas2ri  # for dataframe conversion
from rpy2.robjects.packages import importr
from collections import Counter
from itertools import product
from collections import defaultdict

from data.dataset import Dataset
from trained_models import get_analysis_folder_path
from data import get_training_classification_report_labels
from empirical_distribution import Empirical_distribution

def get_empirical_distribution():
    ed = Empirical_distribution.Instance()
    training_dist = ed._get_training_empirical_distribution()

    return training_dist

def feature_is_capitalized():
    import re

    _, _, document_words, document_tags = Dataset.get_clef_training_dataset(lowercase=False)

    all_words = list(chain(*chain(*document_words.values())))
    all_tags = list(chain(*chain(*document_tags.values())))

    uppercased_idxs = np.where(map(lambda x: re.match(r'[A-Z].*', x), all_words))[0]
    lowercased_idxs = np.where(map(lambda x: not re.match(r'[A-Z].*', x), all_words))[0]

    tags_cnt = Counter(all_tags)
    cnt_upper = Counter(np.array(all_tags)[uppercased_idxs])
    cnt_lower = Counter(np.array(all_tags)[lowercased_idxs])

    df = pd.DataFrame({'labels': get_training_classification_report_labels(),
                       'p_cap': map(lambda x: cnt_upper[x] / float(tags_cnt[x]), get_training_classification_report_labels()),
                       'p_notcap': map(lambda x: cnt_lower[x] / float(tags_cnt[x]), get_training_classification_report_labels()),
                       'p_lab_cap': map(lambda x: cnt_upper[x] / float(sum(cnt_upper.values())),
                                    get_training_classification_report_labels()),
                       'p_lab_notcap': map(lambda x: cnt_lower[x] / float(sum(cnt_lower.values())),
                                    get_training_classification_report_labels())})

    df_melted = pd.melt(df, id_vars=['labels'], value_vars=['p_cap', 'p_lab_cap'], var_name='measure',
                        value_name='prob')

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
            library(ggplot2)

            function(df, output_filename, title){
                str(df)
                df$labels <- as.character(df$labels)
                df$labels <- factor(df$labels, levels=unique(df$labels))

                p <- ggplot(df, aes(x=labels)) +
                geom_bar(stat="identity", aes(y=prob, fill=measure), position="stack") +
                #geom_bar(stat="identity", aes(y=p_cap)) +
                labs(x='Label', y='Probability', title=title) +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                scale_fill_discrete(name="Measure", breaks=c("p_cap", "p_lab_cap"),
                    labels=c("p(cap(w)|label)", "p(label|cap(w))"))

                print(p)

                ggsave(output_filename, plot=p)

                }
            """)
    plotFunc(conv_df, get_analysis_folder_path('feature_capitalized_distribution.png'), 'isCapitalized')

    training_dist = get_empirical_distribution()

    probs = []
    for l in get_training_classification_report_labels():
        probs.extend([('p(label)', l, p) for t, p in training_dist if t == l])
        probs.append(('p(label|cap(w))', l, cnt_upper[l] / float(sum(cnt_upper.values()))))
        probs.append(('p(label|not_cap(w))', l, cnt_lower[l] / float(sum(cnt_lower.values()))))

    df = pd.DataFrame(probs, columns=['type', 'label', 'prob'])
    conv_df = robj.conversion.py2ri(df)
    plotFunc_comparison = robj.r("""
            library(ggplot2)

            function(df, output_filename, title){
                str(df)
                df$label <- as.character(df$label)
                df$label <- factor(df$label, levels=unique(df$label))

                # this is for re-ordering the facets. I want 'Prior' to appear first.
                df$type2 <- factor(df$type, levels=c('p(label)', 'p(label|cap(w))', 'p(label|not_cap(w))'))

                p <- ggplot(df, aes(x=label)) +
                    geom_bar(stat="identity", aes(y=prob)) +
                    facet_grid(type2 ~ .) +
                    labs(x='Label', y='Probability', title=title) +
                    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                    ylim(0,1)

                print(p)

                ggsave(output_filename, plot=p)

                }
            """)
    plotFunc_comparison(conv_df, get_analysis_folder_path('feature_capitalized_distribution_comparison.png'), 'isCapitalized')
    gr.dev_off()

    return True

def feature_is_digit():

    _, _, document_words, document_tags = Dataset.get_clef_training_dataset(lowercase=False)

    all_words = list(chain(*chain(*document_words.values())))
    all_tags = list(chain(*chain(*document_tags.values())))

    # uppercased_idxs = np.where(map(lambda x: re.match(r'.*[0-9].*', x), all_words))[0]
    # lowercased_idxs = np.where(map(lambda x: not re.match(r'[0-9].*', x), all_words))[0]
    uppercased_idxs = np.where(map(lambda x: x.isdigit(), all_words))[0]
    lowercased_idxs = np.where(map(lambda x: not x.isdigit(), all_words))[0]

    tags_cnt = Counter(all_tags)
    cnt_upper = Counter(np.array(all_tags)[uppercased_idxs])
    cnt_lower = Counter(np.array(all_tags)[lowercased_idxs])

    df = pd.DataFrame({'labels': get_training_classification_report_labels(),
                       'p_dig': map(lambda x: cnt_upper[x] / float(tags_cnt[x]), get_training_classification_report_labels()),
                       'p_notdig': map(lambda x: cnt_lower[x] / float(tags_cnt[x]), get_training_classification_report_labels()),
                       'p_lab_dig': map(lambda x: cnt_upper[x] / float(sum(cnt_upper.values())),
                                    get_training_classification_report_labels()),
                       'p_lab_notdig': map(lambda x: cnt_lower[x] / float(sum(cnt_lower.values())),
                                    get_training_classification_report_labels())})

    df_melted = pd.melt(df, id_vars=['labels'], value_vars=['p_dig', 'p_lab_dig'], var_name='measure',
                        value_name='prob')

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
            library(ggplot2)

            function(df, output_filename, title){
                str(df)
                df$labels <- as.character(df$labels)
                df$labels <- factor(df$labels, levels=unique(df$labels))

                p <- ggplot(df, aes(x=labels)) +
                geom_bar(stat="identity", aes(y=prob, fill=measure), position="stack") +
                #geom_bar(stat="identity", aes(y=p_dig)) +
                labs(x='Label', y='Probability', title=title) +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                scale_fill_discrete(name="Measure", breaks=c("p_dig", "p_lab_dig"),
                    labels=c("p(dig(w)|label)", "p(label|dig(w))"))

                print(p)

                ggsave(output_filename, plot=p)

                }
            """)
    plotFunc(conv_df, get_analysis_folder_path('feature_digit_distribution.png'), 'isDigit')

    training_dist = get_empirical_distribution()

    probs = []
    for l in get_training_classification_report_labels():
        probs.extend([('p(label)', l, p) for t, p in training_dist if t == l])
        probs.append(('p(label|dig(w))', l, cnt_upper[l] / float(sum(cnt_upper.values()))))
        probs.append(('p(label|not_dig(w))', l, cnt_lower[l] / float(sum(cnt_lower.values()))))

    df = pd.DataFrame(probs, columns=['type', 'label', 'prob'])
    conv_df = robj.conversion.py2ri(df)
    plotFunc_comparison = robj.r("""
            library(ggplot2)

            function(df, output_filename, title){
                str(df)
                df$label <- as.character(df$label)
                df$label <- factor(df$label, levels=unique(df$label))

                # this is for re-ordering the facets. I want 'Prior' to appear first.
                df$type2 <- factor(df$type, levels=c('p(label)', 'p(label|dig(w))', 'p(label|not_dig(w))'))

                p <- ggplot(df, aes(x=label)) +
                    geom_bar(stat="identity", aes(y=prob)) +
                    facet_grid(type2 ~ .) +
                    labs(x='Label', y='Probability', title=title) +
                    theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                    ylim(0,1)

                print(p)

                ggsave(output_filename, plot=p)

                }
            """)
    plotFunc_comparison(conv_df, get_analysis_folder_path('feature_digit_distribution_comparison.png'), 'isDigit')
    gr.dev_off()

    return True

def feature_is_stopword():
    from nltk.corpus import stopwords
    stop_words = stopwords.words('english')

    _, _, document_words, document_tags = Dataset.get_clef_training_dataset(lowercase=True)

    all_words = list(chain(*chain(*document_words.values())))
    all_tags = list(chain(*chain(*document_tags.values())))

    stopwords_idxs = np.where(map(lambda x: x in stop_words, all_words))[0]
    non_stopwords_idxs = np.where(map(lambda x: x not in stop_words, all_words))[0]

    tags_cnt = Counter(all_tags)
    cnt_stop = Counter(np.array(all_tags)[stopwords_idxs])
    cnt_nonstop = Counter(np.array(all_tags)[non_stopwords_idxs])

    df = pd.DataFrame({'labels': get_training_classification_report_labels(),
                       'p_stop': map(lambda x: cnt_stop[x] / float(tags_cnt[x]), get_training_classification_report_labels()),
                       'p_lab_stop': map(lambda x: cnt_stop[x] / float(sum(cnt_stop.values())),
                                        get_training_classification_report_labels()),
                       'p_lab_nonstop': map(lambda x: cnt_nonstop[x] / float(sum(cnt_nonstop.values())),
                                        get_training_classification_report_labels())})

    df_melted = pd.melt(df, id_vars=['labels'], value_vars=['p_stop', 'p_lab_stop'], var_name='measure',
                        value_name='prob')

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
            library(ggplot2)

            function(df, output_filename, title){
                str(df)
                df$labels <- as.character(df$labels)
                df$labels <- factor(df$labels, levels=unique(df$labels))

                p <- ggplot(df, aes(x=labels)) +
                geom_bar(stat="identity", aes(y=prob, fill=measure), position="stack") +
                #geom_bar(stat="identity", aes(y=p_stop)) +
                labs(x='Label', y='Probability', title=title) +
                theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                scale_fill_discrete(name="Measure", breaks=c("p_stop", "p_lab_stop"),
                    labels=c("p(stop(w)|label)", "p(label|stop(w))"))

                print(p)

                ggsave(output_filename, plot=p)

                }
            """)
    plotFunc(conv_df, get_analysis_folder_path('feature_stopword_distribution.png'), 'isStopword')

    training_dist = get_empirical_distribution()

    probs = []
    for l in get_training_classification_report_labels():
        probs.extend([('p(label)', l, p) for t, p in training_dist if t == l])
        probs.append(('p(label|stop(w))', l, cnt_stop[l] / float(sum(cnt_stop.values()))))
        probs.append(('p(label|not_stop(w))', l, cnt_nonstop[l] / float(sum(cnt_nonstop.values()))))

    df = pd.DataFrame(probs, columns=['type', 'label', 'prob'])
    conv_df = robj.conversion.py2ri(df)
    plotFunc_comparison = robj.r("""
                library(ggplot2)

                function(df, output_filename, title){
                    str(df)
                    df$label <- as.character(df$label)
                    df$label <- factor(df$label, levels=unique(df$label))

                    # this is for re-ordering the facets. I want 'Prior' to appear first.
                    df$type2 <- factor(df$type, levels=c('p(label)', 'p(label|stop(w))', 'p(label|not_stop(w))'))

                    p <- ggplot(df, aes(x=label)) +
                        geom_bar(stat="identity", aes(y=prob)) +
                        facet_grid(type2 ~ .) +
                        labs(x='Label', y='Probability', title=title) +
                        theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                        ylim(0,1)

                    print(p)

                    ggsave(output_filename, plot=p)

                    }
                """)
    plotFunc_comparison(conv_df, get_analysis_folder_path('feature_stopword_distribution_comparison.png'),
                        'isStopword')

    gr.dev_off()

    return True

def feature_tag_section(n_sections=6):
    data = feature_tag_section_data(n_sections)

    df = pd.DataFrame(data[::-1], columns=['label', 'section', 'prob'])

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df)
    plotFunc = robj.r("""
                library(ggplot2)

                function(df, output_filename, title){
                    df$label <- as.character(df$label)
                    df$label <- factor(df$label, levels=unique(df$label), ordered=-1)
                    df$section <- as.character(df$section)

                    p <- ggplot(df, aes(x=section, y=label, fill=prob)) +
                        geom_tile() +
                        scale_fill_gradient(low='white', high='steelblue', guide = guide_legend(title = "Probability")) +
                        labs(x='Section number', y='Labels', title='Section number distribution') +
                        theme(panel.grid.major = element_blank(),
                            panel.border = element_blank(),
                            panel.background = element_blank(),
                            axis.ticks = element_blank())

                    print(p)

                    ggsave(output_filename, plot=p, width=13)

                    }
                """)

    plotFunc(conv_df, get_analysis_folder_path('feature_section_distribution.png'), 'Section nr')
    gr.dev_off()

    return True


def feature_tag_section_data(n_sections=6):
    _, _, document_words, document_tags = Dataset.get_clef_training_dataset(lowercase=True)
    all_words = list(chain(*chain(*document_words.values())))
    all_tags = list(chain(*chain(*document_tags.values())))
    sent_nr = []

    for doc in document_words.values():
        n_tokens = list(chain(*doc)).__len__()
        lim = np.int(np.ceil(np.float(n_tokens) / n_sections))
        for j in range(n_tokens):
            sent_nr.append(j // lim)

    assert all_words.__len__() == sent_nr.__len__()

    cnt = Counter(zip(all_tags, sent_nr))
    tag_cnt = Counter(all_tags)

    data = map(lambda (x, y): (x, y + 1, cnt[(x, y)] / float(tag_cnt[x])),
               product(get_training_classification_report_labels(), range(n_sections)))

    return data


def text_heatmap():
    _, _, document_words, document_tags = Dataset.get_clef_training_dataset(lowercase=True)

    cnt = Counter(chain(*chain(*document_words.values())))

    sentences_words = document_words[0]
    sentences_tags = document_tags[0]

    max_sentence_len = np.max(np.array(sentences_words)).__len__()

    data = []
    for i, (sent_words, sent_tags) in enumerate(zip(sentences_words, sentences_tags)):
        for j in range(max_sentence_len):
            data.append(
                (i, j,
                 cnt[sent_words[j]]/float(np.sum(cnt.values())) if j < sent_words.__len__() else 0.,
                 sent_words[j] if j < sent_words.__len__() else '',
                 sent_tags[j] if j < sent_words.__len__() else '')
            )

    df = pd.DataFrame(data[::-1], columns=['sentence_nr', 'word_position', 'count', 'word', 'label'])

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df)
    plotFunc = robj.r("""
                library(ggplot2)

                function(df, output_filename){
                    #df$sentence_nr <- as.character(df$sentence_nr)
                    df$sentence_nr  <- as.character(df$sentence_nr)
                    df$sentence_nr  <- factor(df$sentence_nr , levels=unique(df$sentence_nr), ordered=-1)
                    #df$word_position <- as.character(df$word_position)

                    p <- ggplot(df, aes(x=word_position, y=sentence_nr, fill=count)) +
                        geom_tile() +
                        geom_text(aes(label=word), color='black', size=4) +
                        scale_fill_gradient(low='white', high='steelblue', guide = guide_legend(title = "Count")) +
                        labs(x='Word position', y='Sentence', title='Text heatmap representation') +
                        theme(panel.grid.major = element_blank(),
                            panel.border = element_blank(),
                            panel.background = element_blank(),
                            axis.ticks = element_blank())

                    print(p)

                    ggsave(output_filename, plot=p, width=13)

                    }
                """)

    plotFunc(conv_df, get_analysis_folder_path('document_heatmap_representation.png'))
    gr.dev_off()

    return True

def tags_repetition_within_document():
    _, _, document_words, document_tags = Dataset.get_clef_training_dataset()

    tag_repetition = defaultdict(list)
    for doc in document_tags.values():
        cnt = Counter()
        for sent in doc:
            tags = set(sent)
            for tag in tags:
                cnt[tag] += 1

        for tag, value in cnt.iteritems():
            tag_repetition[tag].append(value)

    max_n_sentences = np.max(map(len, document_tags.values()))  # max nr of sentences in a doc
    data = []
    for tag in get_training_classification_report_labels():
        n_docs = tag_repetition[tag].__len__()  # nr of docs in which it appears
        bincount = np.bincount(tag_repetition[tag], minlength=max_n_sentences)
        p_n_1 = np.sum(bincount[1]) / float(n_docs)
        p_n_2 = np.sum(bincount[2]) / float(n_docs)
        p_n_3 = np.sum(bincount[3:]) / float(n_docs)

        data.append(
            (tag, p_n_1, p_n_2, p_n_3)
        )

    df = pd.DataFrame(data[::-1], columns=['Label', 'p(n=1)', 'p(n=2)', 'p(n>2)'])

    df_melted = pd.melt(df, id_vars=['Label'], value_vars=['p(n=1)', 'p(n=2)', 'p(n>2)'], var_name='Bin', value_name='Prob')

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df_melted)
    plotFunc = robj.r("""
                library(ggplot2)

                function(df, output_filename){
                    #df$sentence_nr <- as.character(df$sentence_nr)
                    df$Label  <- as.character(df$Label)
                    df$Label  <- factor(df$Label , levels=unique(df$Label))
                    df$Bin <- factor(df$Bin, levels=unique(df$Bin))
                    str(df)

                    p <- ggplot(df, aes(x=Bin, y=Label, fill=Prob)) +
                        geom_tile() +
                        scale_fill_gradient(low='white', high='steelblue', guide = guide_legend(title = "Probability")) +
                        labs(x='Bin', y='Labels', title='Label bincount probability') +
                        theme(panel.grid.major = element_blank(),
                            panel.border = element_blank(),
                            panel.background = element_blank(),
                            axis.ticks = element_blank())

                    print(p)

                    ggsave(output_filename, plot=p, width=6)

                    }
                """)

    plotFunc(conv_df, get_analysis_folder_path('feature_bincount_probability.png'))
    gr.dev_off()

    return True

def feature_tag_in_document():
    _, _, document_words, document_tags = Dataset.get_clef_training_dataset()

    cnt = Counter()
    for doc in document_tags.values():
        tags = set(chain(*doc))
        for tag in tags:
            cnt[tag] += 1

    n_documents = float(document_tags.values().__len__())

    data = []
    for tag in get_training_classification_report_labels():
        data.append(
            (tag, cnt[tag] / n_documents)
        )

    df = pd.DataFrame(data, columns=['Label', 'prob'])

    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df)
    plotFunc = robj.r("""
                    library(ggplot2)

                    function(df, output_filename){
                        df$Label  <- as.character(df$Label)
                        df$Label  <- factor(df$Label , levels=unique(df$Label))
                        str(df)

                        p <- ggplot(df, aes(x=Label)) +
                            geom_bar(stat="identity", aes(y=prob)) +
                            labs(x='Labels', y='Probability', title='Label document probability') +
                            theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
                            ylim(0,1)

                        print(p)

                        ggsave(output_filename, plot=p, width=6)

                        }
                    """)

    plotFunc(conv_df, get_analysis_folder_path('feature_tag_document_probability.png'))
    gr.dev_off()

    return True

def feature_tag_contiguity():
    _, _, document_word, document_tags = Dataset.get_clef_training_dataset(lowercase=True)

    tag_contiguity = defaultdict(list)

    for doc in document_tags.values():
        for sent in doc:
            tags = set(sent)
            for tag in tags:
                remaining_tags = tags.difference(set([tag]))
                tag_contiguity[tag].extend(remaining_tags)

    data = []
    for tag1, tag2 in product(get_training_classification_report_labels(), get_training_classification_report_labels()):
        n_total = tag_contiguity[tag1].__len__()
        data.append(
            (tag1, tag2, Counter(tag_contiguity[tag1])[tag2]/float(n_total))
        )

    df = pd.DataFrame(data[::-1], columns=['tag1', 'tag2', 'prob'])
    gr = importr('grDevices')
    robj.pandas2ri.activate()
    conv_df = robj.conversion.py2ri(df)
    plotFunc = robj.r("""
        library(ggplot2)

        function(df, output_filename){
            #df$sentence_nr <- as.character(df$sentence_nr)
            df$tag1  <- as.character(df$tag1)
            df$tag1  <- factor(df$tag1, levels=unique(df$tag1))
            df$tag2  <- as.character(df$tag2)
            df$tag2  <- factor(df$tag2, levels=unique(df$tag2))
            str(df)

            p <- ggplot(df, aes(x=tag2, y=tag1, fill=prob)) +
                geom_tile() +
                scale_fill_gradient(low='white', high='steelblue', guide = guide_legend(title = "Probability")) +
                labs(x='Contiguous label', y='Label', title='Label contiguity') +
                theme(panel.grid.major = element_blank(),
                    panel.border = element_blank(),
                    panel.background = element_blank(),
                    axis.ticks = element_blank(),
                    axis.text.x = element_text(angle = 90, hjust = 1))

            print(p)

            ggsave(output_filename, plot=p, height=11, width=11)

            }
        """)

    plotFunc(conv_df, get_analysis_folder_path('feature_tag_contiguity.png'))
    gr.dev_off()

    return True

if __name__ == '__main__':

    # feature_is_capitalized()
    # feature_is_digit()
    # feature_is_stopword()
    feature_tag_section(n_sections=6)
    # text_heatmap()
    # tags_repetition_within_document()
    # feature_tag_in_document()
    # feature_tag_contiguity()

    print 'End'