__author__ = 'root'

import pkg_resources

def get_resource(file_name):
    # file_name could be a file or an entire directory

    return pkg_resources.resource_filename('data', file_name)

def get_w2v_model(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/%s' % file_name)

def get_wikipedia_file(file_name):
    return pkg_resources.resource_filename('data', 'wikipedia/%s' % file_name)

def get_w2v_training_data_vectors(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/'+file_name)

def get_w2v_directory(file_name):
    return pkg_resources.resource_filename('data', 'word2vec/%s' % file_name)

def get_google_knowled_graph_cache(file_name):
    return pkg_resources.resource_filename('data', 'google_knowledge_graph/%s' % file_name)