__author__ = 'root'

import pkg_resources
import os.path

def get_cwnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'cwnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_vector_tag_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'vector_tag/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_last_tag_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'last_tag/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_rnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'rnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path