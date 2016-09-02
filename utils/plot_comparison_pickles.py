__author__='lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import cPickle
import pandas as pd
from ggplot import *
import numpy as np

MODEL_MAPPING={
    'normal': 'RNN',
    'lstm': 'LSTM',
    'gru': 'GRU'
}

TITLE_MAPPING={
    'valid_cost': 'validation cost',
    'f1': 'validation F1 score'
}

def parse_name_rnn(name):
    params = name.split('-')[1].split('_')
    rnn_cell_type = params[0]
    minibatch_size = params[1]
    learning_rate = params[2]
    bidirectional = params[3]
    max_length = params[4]

    return rnn_cell_type, minibatch_size, learning_rate, bidirectional, max_length

def parse_name_last_tag(name):
    params = name.split('-')[1].split('_')
    window = params[0]
    using_sentence_update = params[1]
    using_token_update = params[2]
    tag_dim = params[4].replace('.p', '')

    try:
        two_layers = params[3]
    except IndexError:
        two_layers = None

    return window, using_sentence_update, using_token_update, two_layers, tag_dim

def parse_name_cnn(name):
    params = name[name.find('-') + 1:].split('_')
    window = params[0]
    minibatch = params[1]
    lr_train = params[3]
    lr_tune = params[5]
    filters = params[7]

    regions = None
    second_layer = None
    try:
        regions = params[9]
        second_layer = params[10].replace('.p', '')
    except IndexError:
        regions = params[9].replace('.p', '')

    return window, minibatch, lr_train, lr_tune, filters, regions, second_layer


def feature_mapping_rnn():
    return {
            'rnn_cell_type': {'name': 'Model', 'pos': 0},
            'minibatch_size': {'name': 'Mini-batch size', 'pos': 1},
            'learning_rate': {'name': 'Learning rate', 'pos': 2},
            'direction': {'name': 'Direction', 'pos': 3},
            'max_length': {'name': 'Grad clipping', 'pos': 4}
            }

def feature_mapping_last_tag():
    return {
            'window': {'name': 'Window size', 'pos': 0},
            'sentence_update': {'name': 'Sentence updates', 'pos': 1},
            'token_update': {'name': 'Token updates', 'pos': 2},
            'two_layers': {'name': '2nd layer', 'pos': 3},
            'tag_dim': {'name': 'Tag dim', 'pos': 4}
            }

def feature_mapping_cnn():
    return {
        'window': {'name': 'Window size', 'pos': 0},
        'minibatch': {'name': 'Minibatch', 'pos': 1},
        'lr': {'name': 'Learning rate', 'pos': [2, 3]},
        'filters': {'name': 'Filters', 'pos': 4},
        'regions': {'name': 'Region sizes', 'pos': 5},
        'two_layers': {'name': '2nd layer', 'pos': 6}
    }

def plot(data_dict, output_filename, feat_name, title):

    df = pd.DataFrame(data_dict)

    p = ggplot(pd.melt(df, id_vars=['epoch'], var_name=feat_name), aes(x='epoch', y='value', color=feat_name)) + \
        geom_line() + \
        labs(x='Epochs', y='Validation cost') + \
        ggtitle(title)

    ggsave(output_filename + '.png', p, dpi=100)

    return True

def determine_output_path(nnet):
    if nnet == 'rnn':
        from trained_models import get_tf_rnn_path
        return get_tf_rnn_path, parse_name_rnn, feature_mapping_rnn
    elif nnet == 'last_tag':
        from trained_models import get_last_tag_path
        return get_last_tag_path, parse_name_last_tag, feature_mapping_last_tag
    elif nnet == 'cnn':
        from trained_models import get_tf_cnn_path
        return get_tf_cnn_path, parse_name_cnn, feature_mapping_cnn

    return None

def determine_feat_name(parse_name_fn, name, feat_pos):
    try:
        type = parse_name_fn(name)[0]
        return MODEL_MAPPING[type] + ' ' + ' '.join(np.array(parse_name_fn(name))[feat_pos])
    except KeyError:
        return ' '.join(np.array(parse_name_fn(name))[feat_pos])

if __name__ == '__main__':

    params = []
    for i, arg in enumerate(sys.argv):
        if i == 0:
            continue
        elif i == 1:
            nnet = arg
            get_output_path, parse_name_fn, feature_mapping_fn = determine_output_path(arg)
        elif i == 2:
            feat_name = feature_mapping_fn()[arg]['name']
            feat_pos = feature_mapping_fn()[arg]['pos']
        elif i == 3:
            plot_title = TITLE_MAPPING[arg]
        else:
            params.append((arg, cPickle.load(open(get_output_path(arg), 'rb'))))

    assert get_output_path is not None

    data_dict = dict()
    for name, values in params:
        feat_value = determine_feat_name(parse_name_fn, name, feat_pos)
        data_dict[feat_value] = values
    data_dict['epoch'] = range(values.__len__())

    plot(data_dict, get_output_path(feat_name+'_comparison'), feat_name=feat_name, title=' '.join([feat_name, plot_title, 'comparison']))
