__author__ = 'root'

import pkg_resources
import os.path

def get_cwnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'cwnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_cw_rnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'cw_rnn/'+file_name)
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

def get_pycrf_customfeats_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'pycrf/custom_feats/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_pycrf_originalfeats_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'pycrf/original_feats/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_sklearncrf_customfeats_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'sklearn_crf/custom_feats/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_sklearncrf_originalfeats_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'sklearn_crf/original_feats/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_lda_path(file_name):
    return pkg_resources.resource_filename('trained_models', 'lda/'+file_name)

def get_kmeans_path(file_name):
    return pkg_resources.resource_filename('trained_models', 'kmeans/'+file_name)

def get_ensemble_forest_mlp_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'forest_mlp/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_random_forest_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'random_forest/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_gbdt_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'gbdt/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_multi_hidden_cw_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'multi_cwnn/' + filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_language_model_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'language_model/' + filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_factored_model_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'factored_model/' + filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_POS_nnet_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'POS_net/' + filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_two_cwnn_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'two_cwnn/' + filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_tf_cwnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'tf_cwnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_tf_cnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'tf_cnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_tf_multi_mlp_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'tf_multi_mlp/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path


def get_analysis_folder_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'analysis/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_pycrf_neuralcrf_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'pycrf/neural_crf/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_pycrf_neuralcrf_crfsuite_folder(filename):
    path = pkg_resources.resource_filename('trained_models', 'pycrf/neural_crf_crfsuite/%s' % filename)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_tf_hierarchical_mlp_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'tf_hierarchical_mlp/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_tf_collobert_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'tf_collobert/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_lookup_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'lookup/%s' % filename)

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path

def get_random_baseline_path(filename):
    path = pkg_resources.resource_filename('trained_models', 'random_baseline/%s' % filename)

    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path