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

def get_crpp_path(file_name):
    """
    returns path where to save crf++ related files.

    :param file_name:
    :return:
    """

    path = pkg_resources.resource_filename('trained_models', 'crfpp/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_single_mlp_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'single_mlp/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_pycrf_customfeats_folder():
    path = pkg_resources.resource_filename('trained_models', 'pycrf/custom_feats/')
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_pycrf_originalfeats_folder():
    path = pkg_resources.resource_filename('trained_models', 'pycrf/original_feats/')
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_lda_path(file_name):
    return pkg_resources.resource_filename('trained_models', 'lda/'+file_name)

def get_kmeans_path(file_name):
    return pkg_resources.resource_filename('trained_models', 'kmeans/'+file_name)