__author__ = 'root'

import pkg_resources
import cPickle
import os

def get_resource(file_name):
    # file_name could be a file or an entire directory

    return pkg_resources.resource_filename('data', file_name)

def get_w2v_model(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/%s' % file_name)

def get_wikipedia_file(file_name):
    return pkg_resources.resource_filename('data', 'wikipedia/%s' % file_name)

def get_w2v_training_data_vectors(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/'+file_name)

def load_w2v_vectors(filename):
    path = get_w2v_training_data_vectors(filename)
    if os.path.exists(path):
        return cPickle.load(open(path, 'rb'))
    else:
        raise Exception

def get_w2v_directory(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/%s' % file_name)

def get_google_knowled_graph_cache(file_name):
    return pkg_resources.resource_filename('data', 'google_knowledge_graph/%s' % file_name)

def get_param(file_name):
    return cPickle.load(open(pkg_resources.resource_filename('data', 'params/%s' % file_name),'rb'))

def get_classification_report_labels():
    return cPickle.load(open(pkg_resources.resource_filename('data', 'params/classification_report/labels_order.p'),'rb'))

def get_aggregated_classification_report_labels():
    return cPickle.load(open(pkg_resources.resource_filename('data', 'params/classification_report/aggregated_labels_order.p'),'rb'))

def get_hierarchical_mapping():
    return cPickle.load(open(pkg_resources.resource_filename('data', 'params/tag_mapping/mapping.p'),'rb'))

def get_wsj_treebank_filename():
    return pkg_resources.resource_filename('data', 'wsj/wsj01-21-right-branching-w-postags-m40.txt')

def get_config_path(filename):
    return pkg_resources.resource_filename('data', 'params/neural_network/%s' % filename)