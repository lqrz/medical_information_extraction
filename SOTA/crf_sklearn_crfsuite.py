__author__ = 'root'

import sklearn_crfsuite
from sklearn_crfsuite import metrics
import logging
import codecs
from data.dataset import Dataset
from nltk.tokenize import word_tokenize
from sklearn.cross_validation import LeaveOneOut
import numpy as np
from joblib import load, dump
import re
from collections import Counter
from data import get_w2v_model
import gensim
from functools import wraps
import argparse
import cPickle as pickle

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

logger_predictions = logging.getLogger(__name__+'file')
hndlr = logging.FileHandler('predicted_tags.log')
hndlr.setFormatter(logging.Formatter('%(message)s'))
logger_predictions.addHandler(hndlr)
logger_predictions.setLevel(logging.INFO)

class CRF:

    def __init__(self, training_data, training_texts, test_data, output_model_filename, w2v_vector_features=False,
                 w2v_features=False, kmeans_features=False, lda_features=False, zip_features=False):
        self.training_data = training_data
        self.file_texts = training_texts
        # self.file_texts = dataset.get_training_file_sentences(training_data_filename)

        self.test_data = test_data

        self.output_model_filename = output_model_filename
        self.model = None

        # use top 5 most similar word from word2vec or kmeans (it also uses the word representation)
        self.kmeans_features = kmeans_features
        self.w2v_features = w2v_features
        self.w2v_model = None
        self.w2v_vector_features = w2v_vector_features
        if self.w2v_features or self.kmeans_features or self.w2v_vector_features:
            W2V_PRETRAINED_FILENAME = 'GoogleNews-vectors-negative300.bin.gz'
            self.w2v_model = self.load_w2v(get_w2v_model(W2V_PRETRAINED_FILENAME))
            # self.w2v_model = True

        # querying word2vec is too expensive. Maintain a cache.
        self.similar_words_cache = dict(list()) # this is for word2vec similar words
        self.word_vector_cache = dict(list()) # this is for word2vec representations
        self.word_lda_topics = dict(list()) # this is for lda assigned topics

        # use the kmeans cluster as feature
        self.kmeans_model = None
        if self.kmeans_features:
            model_filename = 'kmeans.model'
            self.kmeans_model = self.load_kmeans_model(model_filename)

        # use the lda 5 most promising topics as feature
        self.lda_features = lda_features
        self.lda_model = None
        if self.lda_features:
            model_filename = 'wikipedia_lda.model'
            self.lda_model = load(model_filename)

        self.zip_features = zip_features

    def load_kmeans_model(self, model_filename):
        return load(model_filename)

    def get_sentence_labels(self, sentence, file_idx):
        # return [self.training_data[file_idx][j]['tag'] for j, sentence in enumerate(sentence.split(' '))]
        tags = []
        # for j, word in enumerate(word_tokenize(sentence)):
        for j, word in enumerate(sentence):
            if self.training_data[file_idx][j]['word'] != word:
                print 'mismatch'
            else:
                tags.append(self.training_data[file_idx][j]['tag'])

        return tags

    def get_labels(self):
        return [self.get_sentence_labels(sentence, file_idx)
                for file_idx, sentence in enumerate(self.file_texts)]

    def get_labels_from_crf_training_data(self):
        return [self.get_sentence_labels(sentence, file_idx)
                for file_idx, sentence in self.file_texts.iteritems()]

    def load_w2v(self, model_filename):
        return gensim.models.Word2Vec.load_word2vec_format(model_filename, binary=True)

    def get_custom_word_features(self, sentence, file_idx, word_idx):
        features = dict()
        previous_features = dict()
        next_features = dict()
        word_features = dict()

        features_names = ['word', 'lemma', 'ner', 'pos', 'parse_tree', 'dependents', 'governors', 'has_digit',
                          'is_capitalized']

        word = sentence[word_idx]
        word_lemma = self.training_data[file_idx][word_idx]['features'][0]
        word_ner = self.training_data[file_idx][word_idx]['features'][1]
        word_pos = self.training_data[file_idx][word_idx]['features'][2]
        word_parse_tree = self.training_data[file_idx][word_idx]['features'][3]
        word_basic_dependents = self.training_data[file_idx][word_idx]['features'][4]
        word_basic_governors = self.training_data[file_idx][word_idx]['features'][5]
        # word_unk_score = self.training_data[file_idx][word_idx]['features'][6]
        # word_phrase = self.training_data[file_idx][word_idx]['features'][7]
        # word_top_candidate_1 = self.training_data[file_idx][word_idx]['features'][8]
        # word_top_candidate_2 = self.training_data[file_idx][word_idx]['features'][9]
        # word_top_candidate_3 = self.training_data[file_idx][word_idx]['features'][10]
        # word_top_candidate_4 = self.training_data[file_idx][word_idx]['features'][11]
        # word_top_candidate_5 = self.training_data[file_idx][word_idx]['features'][12]
        # word_top_mapping = self.training_data[file_idx][word_idx]['features'][13]
        # word_medication_score = self.training_data[file_idx][word_idx]['features'][14]
        # word_location = self.training_data[file_idx][word_idx]['features'][15]
        word_tag = self.training_data[file_idx][word_idx]['tag']
        word_shape_digit = re.search('\d', word) is not None
        word_shape_capital = re.match('[A-Z]', word) is not None

        word_feature_values = [word, word_lemma, word_ner, word_pos,
                                 word_parse_tree, word_basic_dependents,
                                 word_basic_governors, word_shape_digit,
                                 word_shape_capital]

        for desc,val in zip(features_names,word_feature_values):
            word_features[desc] = val

        # features.append(word)
        # features.append(word_lemma)
        # features.append(word_ner)
        # features.append(word_pos)
        # features.append(word_parse_tree)
        # features.append(word_basic_dependents)
        # features.append(word_basic_governors)
        # features.append(word_shape_digit)
        # features.append(word_shape_capital)
        features['word'] = word
        features['word_lemma'] =word_lemma
        features['word_ner'] = word_ner
        features['word_pos'] = word_pos
        features['word_parse_tree'] = word_parse_tree
        features['word_dependents'] = word_basic_dependents
        features['word_governors'] = word_basic_governors
        features['word_has_digit'] = word_shape_digit
        features['word_is_capitalized'] = word_shape_capital
        features['prefix'] = word[:3]
        features['suffix'] = word[-3:]

        if word_idx > 0:
            previous_word = sentence[word_idx-1]
            previous_word_lemma = self.training_data[file_idx][word_idx-1]['features'][0]
            previous_word_ner = self.training_data[file_idx][word_idx-1]['features'][1]
            previous_word_pos = self.training_data[file_idx][word_idx-1]['features'][2]
            previous_word_parse_tree = self.training_data[file_idx][word_idx-1]['features'][3]
            previous_word_basic_dependents = self.training_data[file_idx][word_idx-1]['features'][4]
            previous_word_basic_governors = self.training_data[file_idx][word_idx-1]['features'][5]
            # previous_word_unk_score = self.training_data[file_idx][word_idx-1]['features'][6]
            # previous_word_phrase = self.training_data[file_idx][word_idx-1]['features'][7]
            # previous_word_top_candidate_1 = self.training_data[file_idx][word_idx-1]['features'][8]
            # previous_word_top_candidate_2 = self.training_data[file_idx][word_idx-1]['features'][9]
            # previous_word_top_candidate_3 = self.training_data[file_idx][word_idx-1]['features'][10]
            # previous_word_top_candidate_4 = self.training_data[file_idx][word_idx-1]['features'][11]
            # previous_word_top_candidate_5 = self.training_data[file_idx][word_idx-1]['features'][12]
            # previous_word_top_mapping = self.training_data[file_idx][word_idx-1]['features'][13]
            # previous_word_medication_score = self.training_data[file_idx][word_idx-1]['features'][14]
            # previous_word_location = self.training_data[file_idx][word_idx-1]['features'][15]
            previous_word_shape_digit = re.search('\d', previous_word) is not None
            previous_word_shape_capital = re.match('[A-Z]', previous_word) is not None

            previous_features_values = [previous_word, previous_word_lemma, previous_word_ner, previous_word_pos,
                                     previous_word_parse_tree, previous_word_basic_dependents,
                                     previous_word_basic_governors, previous_word_shape_digit,
                                     previous_word_shape_capital]

            for desc,val in zip(features_names,previous_features_values):
                previous_features[desc] = val

            # features.append(previous_word)
            # features.append(previous_word_lemma)
            # features.append(previous_word_ner)
            # features.append(previous_word_pos)
            # features.append(previous_word_parse_tree)
            # features.append(previous_word_basic_dependents)
            # features.append(previous_word_basic_governors)
            # features.append(previous_word_shape_digit)
            # features.append(previous_word_shape_capital)
            features['previous_word'] = previous_word
            features['previous_lemma'] = previous_word_lemma
            features['previous_ner'] = previous_word_ner
            features['previous_pos'] = previous_word_pos
            features['previous_parse_tree'] = previous_word_parse_tree
            features['previous_dependents'] = previous_word_basic_dependents
            features['previous_governors'] = previous_word_basic_governors
            features['previous_has_digit'] = previous_word_shape_digit
            features['previous_is_capitalized'] = previous_word_shape_capital

            #TODO
            if self.zip_features:
                # features.extend([x+'/'+y for x,y in zip(word_features, next_features)])
                assert len(word_features.keys()) == len(previous_features.keys())
                for feat_name in features_names:
                    try:
                        features['previous_'+feat_name+'_'+feat_name] = \
                        previous_features[feat_name] + '/' + word_features[feat_name]
                    except TypeError:
                        features['previous_'+feat_name+'_'+feat_name] = \
                        previous_features[feat_name] or word_features[feat_name]

        else:
            # features.append('BOS')
            features['BOS'] = True

        if word_idx < len(sentence)-1:
            next_word = sentence[word_idx+1]
            next_word_lemma = self.training_data[file_idx][word_idx+1]['features'][0]
            next_word_ner = self.training_data[file_idx][word_idx+1]['features'][1]
            next_word_pos = self.training_data[file_idx][word_idx+1]['features'][2]
            next_word_parse_tree = self.training_data[file_idx][word_idx+1]['features'][3]
            next_word_basic_dependents = self.training_data[file_idx][word_idx+1]['features'][4]
            next_word_basic_governors = self.training_data[file_idx][word_idx+1]['features'][5]
            # next_word_unk_score = self.training_data[file_idx][word_idx+1]['features'][6]
            # next_word_phrase = self.training_data[file_idx][word_idx+1]['features'][7]
            # next_word_top_candidate_1 = self.training_data[file_idx][word_idx+1]['features'][8]
            # next_word_top_candidate_2 = self.training_data[file_idx][word_idx+1]['features'][9]
            # next_word_top_candidate_3 = self.training_data[file_idx][word_idx+1]['features'][10]
            # next_word_top_candidate_4 = self.training_data[file_idx][word_idx+1]['features'][11]
            # next_word_top_candidate_5 = self.training_data[file_idx][word_idx+1]['features'][12]
            # next_word_top_mapping = self.training_data[file_idx][word_idx+1]['features'][13]
            # next_word_medication_score = self.training_data[file_idx][word_idx+1]['features'][14]
            # next_word_location = self.training_data[file_idx][word_idx+1]['features'][15]
            next_word_shape_digit = re.search('\d', next_word) is not None
            next_word_shape_capital = re.match('[A-Z]', next_word) is not None

            next_feature_values = [next_word, next_word_lemma, next_word_ner, next_word_pos,
                                     next_word_parse_tree, next_word_basic_dependents,
                                     next_word_basic_governors, next_word_shape_digit,
                                     next_word_shape_capital]

            for desc,val in zip(features_names,next_feature_values):
                next_features[desc] = val

            # features.append(next_word)
            # features.append(next_word_lemma)
            # features.append(next_word_ner)
            # features.append(next_word_pos)
            # features.append(next_word_parse_tree)
            # features.append(next_word_basic_dependents)
            # features.append(next_word_basic_governors)
            # features.append(next_word_shape_digit)
            # features.append(next_word_shape_capital)
            features['next_word'] = next_word
            features['next_lemma'] = next_word_lemma
            features['next_ner'] = next_word_ner
            features['next_pos'] = next_word_pos
            features['next_parse_tree'] = next_word_parse_tree
            features['next_dependents'] = next_word_basic_dependents
            features['next_governors'] = next_word_basic_governors
            features['next_has_digit'] = next_word_shape_digit
            features['next_is_capitalized'] = next_word_shape_capital

            #TODO
            if self.zip_features:
                # features.extend([x+'/'+y for x,y in zip(word_features, next_features)])
                assert len(word_features.keys()) == len(next_features.keys())
                for feat_name in features_names:
                    try:
                        features[feat_name+'_'+'next_'+feat_name] = \
                        word_features[feat_name] + '/' + next_features[feat_name]
                    except TypeError:
                        features[feat_name+'_'+'next_'+feat_name] = \
                        word_features[feat_name] or next_features[feat_name]

        else:
            # features.append('EOS')
            features['EOS'] = True

        # word2vec features
        if self.w2v_features and self.w2v_model:
            similar_words = self.get_similar_w2v_words(word, self.similar_words_cache, topn=5)
            for j,sim_word in enumerate(similar_words):
                features['w2v_similar_word_'+str(j)] = sim_word
            # features.extend(similar_words)

        # kmeans features
        if (self.kmeans_features and self.kmeans_model) and self.w2v_model:
            cluster = self.get_kmeans_cluster(word)
            features['kmeans_cluster'] = cluster
            # features.append(str(cluster))

        # lda features
        if self.lda_features and self.lda_model:
            topics = self.get_lda_topics(word, self.word_lda_topics, topn=5)
            for j,topic in enumerate(topics):
                features['lda_topic_'+str(j)] = topic
            # features.extend(topics)

        if (self.w2v_features and self.w2v_model):
            n_dim = self.w2v_model.syn0.shape[0]
            try:
                rep = self.get_w2v_vector(word, self.word_vector_cache)
            except KeyError:
                rep = np.zeros((n_dim,))

            for j,dim_val in rep:
                features['w2v_dim_'+str(j)] = dim_val

        return features

    def memoize(func):

        @wraps(func)
        def wrapper(self, *args, **kwds):
            similar_words = None
            word = args[0].lower()
            dictionary = args[1]
            try:
                # similar_words = self.similar_words_cache[word]
                similar_words = dictionary[word]
            except:
                similar_words = func(self, *args, **kwds)
                # self.similar_words_cache[word] = similar_words
                dictionary[word] = similar_words
            return similar_words

        return wrapper

    @memoize
    def get_similar_w2v_words(self, word, dictionary, topn=5):
        try:
            similar_words = [sim for sim,_ in self.w2v_model.most_similar(positive=[word], topn=topn)]
        except:
            similar_words = []

        return similar_words
        # return ['ej1', 'e2', 'e3']

    @memoize
    def get_w2v_vector(self, word, dictionary):
        return self.w2v_model[word]

    def get_kmeans_cluster(self, word):
        try:
            word_vector = self.get_w2v_vector(word, self.word_vector_cache)
            cluster = self.kmeans_model.predict(word_vector)[0]
        except:
            cluster = 999

        return cluster

    @memoize
    def get_lda_topics(self, word, dictionary, topn=5):
        try:
            # esto, comparado con lda_model.id2word.token2id[word] da otro resultado (erroneo). Be careful!
            # id2word = gensim.corpora.Dictionary()
            # id2word.merge_with(self.lda_model.id2word)
            # bow = id2word.doc2bow([word])
            token_id = self.lda_model.id2word.token2id[word.lower()]
            bow = [(token_id,1)]
            topics = [str(topic) for topic,_ in
                          sorted(self.lda_model.get_document_topics(bow, minimum_probability=0.),
                                 key=lambda x: x[1], reverse=True)[:5]]
        except:
            n_topics = self.lda_model.num_topics
            topics = ['999'] * n_topics

        return topics

    def get_original_paper_word_features(self, sentence, file_idx, word_idx):
        # features = []
        features = dict()

        word = sentence[word_idx]
        word_lemma = self.training_data[file_idx][word_idx]['features'][0]
        word_ner = self.training_data[file_idx][word_idx]['features'][1]
        word_pos = self.training_data[file_idx][word_idx]['features'][2]
        word_parse_tree = self.training_data[file_idx][word_idx]['features'][3]
        word_basic_dependents = self.training_data[file_idx][word_idx]['features'][4]
        word_basic_governors = self.training_data[file_idx][word_idx]['features'][5]
        word_unk_score = self.training_data[file_idx][word_idx]['features'][6]
        word_phrase = self.training_data[file_idx][word_idx]['features'][7]
        word_top_candidate_1 = self.training_data[file_idx][word_idx]['features'][8]
        word_top_candidate_2 = self.training_data[file_idx][word_idx]['features'][9]
        word_top_candidate_3 = self.training_data[file_idx][word_idx]['features'][10]
        word_top_candidate_4 = self.training_data[file_idx][word_idx]['features'][11]
        word_top_candidate_5 = self.training_data[file_idx][word_idx]['features'][12]
        word_top_mapping = self.training_data[file_idx][word_idx]['features'][13]
        word_medication_score = self.training_data[file_idx][word_idx]['features'][14]
        word_location = self.training_data[file_idx][word_idx]['features'][15]

        word_tag = self.training_data[file_idx][word_idx]['tag']

        # Unigram
        # U01:%x[0,0]
        features['word'] = word
        # U06:%x[0,1]
        features['word_lemma'] = word_lemma
        # U11:%x[0,2]
        features['word_ner'] = word_ner
        # U16:%x[0,3]
        features['word_pos'] = word_pos
        # U21:%x[0,4]
        features['word_parse_tree'] = word_parse_tree
        # U26:%x[0,5]
        features['word_dependents'] = word_basic_dependents
        # U31:%x[0,6]
        features['word_governors'] = word_basic_governors
        # U36:%x[0,8]
        features['word_phrase'] = word_phrase
        # U41:%x[0,9]
        features['word_candidate_1'] = word_top_candidate_1
        # U46:%x[0,10]
        features['word_candidate_2'] = word_top_candidate_2
        # U51:%x[0,11]
        features['word_candidate_3'] = word_top_candidate_3
        # U56:%x[0,12]
        features['word_candidate_4'] = word_top_candidate_4
        # U61:%x[0,13]
        features['word_candidate_5'] = word_top_candidate_5
        # U66:%x[0,14]
        features['word_mapping'] = word_top_mapping
        # U71:%x[0,15]
        features['word_medication_score'] = word_medication_score
        # U76:%x[0,16]
        features['word_location'] = word_location
        # U80:%x[0,1]/%x[0,2]/%x[0,3]/%x[0,5]/%x[0,6]/%x[0,7]/%x[0,8]/%x[0,9]/%x[0,10]/%x[0,11]/%x[0,12]/%x[0,13]/%x[0,14]/%x[0,15]/%x[0,16]
        features['word_all_features'] = '/'.join([word_lemma, word_ner, word_pos, word_parse_tree,
            word_basic_dependents, word_basic_governors, word_unk_score, word_phrase,
            word_top_candidate_1, word_top_candidate_2, word_top_candidate_3, word_top_candidate_4,
            word_top_candidate_5, word_top_mapping, word_medication_score, word_location])

        if word_idx > 0:
            # U00:%x[-1,0]
            previous_word = sentence[word_idx-1]
            features['previous_word'] = previous_word
            # U05:%x[-1,1]
            features['previous_lemma'] = self.training_data[file_idx][word_idx-1]['features'][0]
            # U10:%x[-1,2]
            features['previous_ner'] = self.training_data[file_idx][word_idx-1]['features'][1]
            # U15:%x[-1,3]
            features['previous_pos'] = self.training_data[file_idx][word_idx-1]['features'][2]
            # U20:%x[-1,4]
            features['previous_parse_tree'] = self.training_data[file_idx][word_idx-1]['features'][3]
            # U25:%x[-1,5]
            features['previous_dependents'] = self.training_data[file_idx][word_idx-1]['features'][4]
            # U30:%x[-1,6]
            features['previous_governors'] = self.training_data[file_idx][word_idx-1]['features'][5]
            # U35:%x[-1,8]
            features['previous_phrase'] = self.training_data[file_idx][word_idx-1]['features'][7]
            # U40:%x[-1,9]
            features['previous_candidate_1'] = self.training_data[file_idx][word_idx-1]['features'][8]
            # U45:%x[-1,10]
            features['previous_candidate_2'] = self.training_data[file_idx][word_idx-1]['features'][9]
            # U50:%x[-1,11]
            features['previous_candidate_3'] = self.training_data[file_idx][word_idx-1]['features'][10]
            # U55:%x[-1,12]
            features['previous_candidate_4'] = self.training_data[file_idx][word_idx-1]['features'][11]
            # U60:%x[-1,13]
            features['previous_candidate_5'] = self.training_data[file_idx][word_idx-1]['features'][12]
            # U65:%x[-1,14]
            features['previous_mapping'] = self.training_data[file_idx][word_idx-1]['features'][13]
            # U70:%x[-1,15]
            features['previous_medication_score'] = self.training_data[file_idx][word_idx-1]['features'][14]
            # U75:%x[-1,16]
            features['previous_location'] = self.training_data[file_idx][word_idx-1]['features'][15]

            # TODO: uncomment? or is it included in the CRF all_possible_transitions flag?
            # Bigram
            # B
            # previous_tag = self.training_data[file_idx][word_idx-1]['tag']
            # features.append(previous_tag+'/'+word_tag)

            if self.zip_features:
                # U03:%x[-1,0]/%x[0,0]
                features['previous_word_word'] = previous_word +'/'+ word

                # U08:%x[-1,1]/%x[0,1]
                features['previous_lemma_lemma'] = self.training_data[file_idx][word_idx-1]['features'][0] +'/'+ \
                    word_lemma

                # U13:%x[-1,2]/%x[0,2]
                features['previous_ner_ner'] = self.training_data[file_idx][word_idx-1]['features'][1] +'/'+ \
                    word_ner

                # U18:%x[-1,3]/%x[0,3]
                features['previous_pos_pos'] = self.training_data[file_idx][word_idx-1]['features'][2] +'/'+ \
                    word_pos

                # U23:%x[-1,4]/%x[0,4]
                features['previous_parse_tree_parse_tree'] = self.training_data[file_idx][word_idx-1]['features'][3] +'/'+ \
                    word_parse_tree

                # U28:%x[-1,5]/%x[0,5]
                features['previous_dependents_dependents'] = self.training_data[file_idx][word_idx-1]['features'][4] +'/'+ \
                    word_basic_dependents

                # U33:%x[-1,6]/%x[0,6]
                features['previous_governors_governors'] = self.training_data[file_idx][word_idx-1]['features'][5] +'/'+ \
                    word_basic_governors

                # U38:%x[-1,8]/%x[0,8]
                features['previous_phrase_phrase'] = self.training_data[file_idx][word_idx-1]['features'][7] +'/'+ \
                    word_phrase

                # U43:%x[-1,9]/%x[0,9]
                features['previous_candidate_1_candidate_1'] = self.training_data[file_idx][word_idx-1]['features'][8] +'/'+ \
                    word_top_candidate_1

                # U48:%x[-1,10]/%x[0,10]
                features['previous_candidate_2_candidate_2'] = self.training_data[file_idx][word_idx-1]['features'][9] +'/'+ \
                    word_top_candidate_2

                # U53:%x[-1,11]/%x[0,11]
                features['previous_candidate_3_candidate_3'] = self.training_data[file_idx][word_idx-1]['features'][10] +'/'+ \
                    word_top_candidate_3

                # U58:%x[-1,12]/%x[0,12]
                features['previous_candidate_4_candidate_4'] = self.training_data[file_idx][word_idx-1]['features'][11] +'/'+ \
                    word_top_candidate_4

                # U63:%x[-1,13]/%x[0,13]
                features['previous_candidate_5_candidate_5'] = self.training_data[file_idx][word_idx-1]['features'][12] +'/'+ \
                    word_top_candidate_5

                # U68:%x[-1,14]/%x[0,14]
                features['previous_mapping_mapping'] = self.training_data[file_idx][word_idx-1]['features'][13] +'/'+ \
                    word_top_mapping

                # U73:%x[-1,15]/%x[0,15]
                features['previous_medication_score_medication_score'] = self.training_data[file_idx][word_idx-1]['features'][14] +'/'+ \
                    word_medication_score

                # U78:%x[-1,16]/%x[0,16]
                features['previous_word_location_word_location'] = self.training_data[file_idx][word_idx-1]['features'][15] +'/'+ \
                    word_location

        else:
            # features['BOS'] = True
            features['BOS'] = 'True'

        if word_idx < len(sentence)-1:
            # U02:%x[1,0]
            next_word = sentence[word_idx+1]
            features['next_word'] = next_word
            # U07:%x[1,1]
            features['next_lemma'] = self.training_data[file_idx][word_idx+1]['features'][0]
            # U12:%x[1,2]
            features['next_ner'] = self.training_data[file_idx][word_idx+1]['features'][1]
            # U17:%x[1,3]
            features['next_pos'] = self.training_data[file_idx][word_idx+1]['features'][2]
            # U22:%x[1,4]
            features['next_parse_tree'] = self.training_data[file_idx][word_idx+1]['features'][3]
            # U27:%x[1,5]
            features['next_dependents'] = self.training_data[file_idx][word_idx+1]['features'][4]
            # U32:%x[1,6]
            features['next_governors'] = self.training_data[file_idx][word_idx+1]['features'][5]
            # U37:%x[1,8]
            features['next_phrase'] = self.training_data[file_idx][word_idx+1]['features'][7]
            # U42:%x[1,9]
            features['next_candidate_1'] = self.training_data[file_idx][word_idx+1]['features'][8]
            # U47:%x[1,10]
            features['next_candidate_2'] = self.training_data[file_idx][word_idx+1]['features'][9]
            # U52:%x[1,11]
            features['next_candidate_3'] = self.training_data[file_idx][word_idx+1]['features'][10]
            # U57:%x[1,12]
            features['next_candidate_4'] = self.training_data[file_idx][word_idx+1]['features'][11]
            # U62:%x[1,13]
            features['next_candidate_5'] = self.training_data[file_idx][word_idx+1]['features'][12]
            # U67:%x[1,14]
            features['next_mapping'] = self.training_data[file_idx][word_idx+1]['features'][13]
            # U72:%x[1,15]
            features['next_medication_score'] = self.training_data[file_idx][word_idx+1]['features'][14]
            # U77:%x[1,16]
            features['next_location'] = self.training_data[file_idx][word_idx+1]['features'][15]

            if self.zip_features:
                # U04:%x[0,0]/%x[1,0]
                features['word_next_word'] = word +'/'+ next_word

                # U09:%x[0,1]/%x[1,1]
                features['lemma_next_lemma'] = word_lemma +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][0]

                # U14:%x[0,2]/%x[1,2]
                features['ner_next_ner'] = word_ner +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][1]

                # U19:%x[0,3]/%x[1,3]
                features['pos_next_pos'] = word_pos +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][2]

                # U24:%x[0,4]/%x[1,4]
                features['parse_tree_next_parse_tree'] = word_parse_tree +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][3]

                # U29:%x[0,5]/%x[1,5]
                features['dependents_next_dependents'] = word_basic_dependents +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][4]

                # U34:%x[0,6]/%x[1,6]
                features['governors_next_governors'] = word_basic_governors +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][5]

                # U39:%x[0,8]/%x[1,8]
                features['phrase_next_phrase'] = word_phrase +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][7]

                # U44:%x[0,9]/%x[1,9]
                features['candidate_1_next_candidate_1'] = word_top_candidate_1 +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][8]

                # U49:%x[0,10]/%x[1,10]
                features['candidate_2_next_candidate_2'] = word_top_candidate_2 +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][9]

                # U54:%x[0,11]/%x[1,11]
                features['candidate_3_next_candidate_3'] = word_top_candidate_3 +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][10]

                # U59:%x[0,12]/%x[1,12]
                features['candidate_4_next_candidate_4'] = word_top_candidate_4 +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][11]

                # U64:%x[0,13]/%x[1,13]
                features['candidate_5_next_candidate_5'] = word_top_candidate_5 +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][12]

                # U69:%x[0,14]/%x[1,14]
                features['mapping_next_mapping'] = word_top_mapping +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][13]

                # U74:%x[0,15]/%x[1,15]
                features['medication_score_next_medication_score'] = word_medication_score +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][14]

                # U79:%x[0,16]/%x[1,16]
                features['location_next_location'] = word_location +'/'+ \
                    self.training_data[file_idx][word_idx+1]['features'][15]

        else:
            # features['EOS'] = True
            features['EOS'] = True

        return features

    def get_sentence_features(self, sentence, file_idx, feature_function):
        # return [self.training_data[file_idx][j]['features'] for j, word in enumerate(word_tokenize(sentence))]
        features = []
        # for j, word in enumerate(word_tokenize(sentence)):
        for j, word in enumerate(sentence):
            if self.training_data[file_idx][j]['word'] != word:
                print 'mismatch'
            else:
                features.append(feature_function(sentence, file_idx, j))

        return features

    def get_features(self, feature_function):
        return [self.get_sentence_features(sentence, file_idx, feature_function)
                for file_idx, file_text in self.file_texts.iteritems()
                for sentence in file_text[0].split('\n')]

    def get_features_from_crf_training_data(self, feature_function):
        return [self.get_sentence_features(sentence, file_idx, feature_function)
                for file_idx, sentence in self.file_texts.iteritems()]

    def train(self, x_train, y_train, verbose=False):
        # x_train = self.get_features()

        # crf_trainer = sklearn_crfsuite.Trainer(verbose=verbose)
        crf_trainer = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=1.0,
            c2=1e-3,
            max_iterations=50,
            all_possible_transitions=True
        )

        # for xseq, yseq in zip(x_train, y_train):
        #     crf_trainer.append(xseq, yseq)

        # crf_trainer.set_params({
        #     'algorithm' : 'lbfgs',
        #     'c1' : 1.0,
        #     'c2' : 1e-3,
        #     'max_iterations' : 50,
        #     'all_possible_transitions' : True
        # })

        self.model = crf_trainer

        crf_trainer.fit(x_train, y_train)

        # dump(crf_trainer, self.output_model_filename)

        # crf_trainer.train(self.output_model_filename)

        return

    def predict(self, x_test, y_test, feature_function):

        accuracy = 0

        # tagger = pycrfsuite.Tagger()
        # tagger.open(self.output_model_filename)

        # tagger = load(self.output_model_filename)
        tagger = self.model

        # for sent in x_train:
            # print tagger.tag(sent)
        predictions = tagger.predict(x_test)
        accuracy = sum([pred==y_test[j][i]
                        for j in range(len(predictions))
                        for i, pred in enumerate(predictions[j])])

        predicted_tags = zip([dic['word'] for dic in x_test[0]], predictions[0])

        # metrics.flat_f1_score(y_train, predictions, average='weighted')

        # return float(accuracy)/len(predictions[0]) #this calculation is ok
        # print metrics.flat_classification_report(y_train, predictions)

        return predicted_tags, metrics.flat_accuracy_score(y_test, predictions), metrics.flat_f1_score(y_test, predictions)

def print_state_features(state_features):
    for (attr, label), weight in state_features:
        print("%0.6f %-8s %s" % (weight, label, attr))

def save_predictions_to_file(predicted_labels, true_labels, logger):
    for (word,pred),true in zip(predicted_labels, true_labels[0]):
        logger.info('%s\t%s\t%s' % (word,pred,true))

    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CRF Sklearn')
    parser.add_argument('--outputfolder', default='./', type=str, help='Output folder for the model and logs')
    parser.add_argument('--w2vfeatures', action='store_true', default=False)
    parser.add_argument('--kmeans', action='store_true', default=False)
    parser.add_argument('--lda', action='store_true', default=False)
    parser.add_argument('--zipfeatures', action='store_true', default=False)
    parser.add_argument('--originalfeatures', action='store_true', default=False)
    parser.add_argument('--customfeatures', action='store_true', default=False)
    parser.add_argument('--w2vvectorfeatures', action='store_true', default=False)

    arguments = parser.parse_args()

    training_data_filename = 'handoverdata.zip'
    test_data_filename = 'handover-set2.zip'
    output_model_filename = arguments.outputfolder+ '/' + 'crf_trained.model'

    w2v_features = arguments.w2vfeatures
    kmeans = arguments.kmeans
    lda = arguments.lda
    zip_features = arguments.zipfeatures
    use_original_paper_features = arguments.originalfeatures
    use_custom_features = arguments.customfeatures
    w2v_vector_features = arguments.w2vvectorfeatures

    training_data, training_texts = Dataset.get_crf_training_data(training_data_filename)

    # test_data = Dataset.get_crf_training_data(test_data_filename)

    crf_model = CRF(training_data, training_texts, test_data=None, output_model_filename=output_model_filename,
                    w2v_vector_features=w2v_vector_features,
                    w2v_features=w2v_features, kmeans_features=kmeans, lda_features=lda, zip_features=zip_features)

    if use_original_paper_features:
        feature_function = crf_model.get_original_paper_word_features
    elif use_custom_features:
        feature_function = crf_model.get_custom_word_features

    logger.info('Extracting features with: '+feature_function.__str__())

    logger.info('Using w2v_similar_words:%s kmeans:%s lda:%s zip:%s' % (w2v_features, kmeans, lda, zip_features))

    results_accuracy = []
    results_f1 = []

    prediction_results = dict()

    loo = LeaveOneOut(training_data.__len__())
    for i, (x_idx, y_idx) in enumerate(loo):
        # if i+1 > 1:
        #    break
        logger.info('Cross validation '+str(i+1)+' (train+predict)')
        # print x_idx, y_idx

        x = crf_model.get_features_from_crf_training_data(feature_function)
        y = crf_model.get_labels_from_crf_training_data()
        x_train = np.array(x)[x_idx]
        y_train = np.array(y)[x_idx]

        crf_model.train(x_train, y_train, verbose=False)

        x_test = np.array(x)[y_idx]
        y_test = np.array(y)[y_idx]

        logger.info('Predicting file #%s' % (y_idx[0]))

        predicted_tags, accuracy, f1_score = crf_model.predict(x_test, y_test, feature_function)
        results_accuracy.append(accuracy)
        results_f1.append(f1_score)
        # print print_state_features(Counter(crf_model.model.state_features_).most_common(20))
        # print predicted_tags
        # if y_idx[0] == 0:
        #     save_predictions_to_file(predicted_tags, y_test, logger_predictions)

        prediction_results[y_idx[0]] = [(word,pred,true) for (word,pred),true in zip(predicted_tags, y_test[0])]

    logging.info('Pickling prediction results')
    pickle.dump(prediction_results, open(arguments.outputfolder+'prediction_results.p','wb'))

    print 'Accuracy: ', results_accuracy
    print 'F1: ', results_f1
    print 'Mean accuracy: ', np.mean(results_accuracy)
    print 'Mean f1: ', np.mean(results_f1)
