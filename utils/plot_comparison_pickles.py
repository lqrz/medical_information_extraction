__author__='lqrz'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import cPickle
import pandas as pd
from ggplot import *

from trained_models import get_tf_rnn_path

MAPPING={   'rnn_cell_type': {'name': 'Model', 'pos': 0},
            'minibatch_size': {'name': 'Mini-batch size', 'pos': 1},
            'learning_rate': {'name': 'Learning rate', 'pos': 2},
            'direction': {'name': 'Direction', 'pos': 3},
            'max_length': {'name': 'Grad clipping', 'pos': 4}
}

MODEL_MAPPING={
    'normal': 'RNN',
    'lstm': 'LSTM',
    'gru': 'GRU'
}

TITLE_MAPPING={
    'valid_cost': 'validation cost',
    'f1': 'validation F1 score'
}

def parse_name(name):
    params = name.split('-')[1].split('_')
    rnn_cell_type = params[0]
    minibatch_size = params[1]
    learning_rate = params[2]
    bidirectional = params[3]
    max_length = params[4]

    return rnn_cell_type, minibatch_size, learning_rate, bidirectional, max_length

def plot(data_dict, output_filename, feat_name, title):

    df = pd.DataFrame(data_dict)

    p = ggplot(pd.melt(df, id_vars=['epoch'], var_name=feat_name), aes(x='epoch', y='value', color=feat_name)) + \
        geom_line() + \
        labs(x='Epochs', y='Validation cost') + \
        ggtitle(title)

    ggsave(output_filename + '.png', p, dpi=100)

    return True

if __name__ == '__main__':

    params = []
    for i, arg in enumerate(sys.argv):
        if i == 0:
            continue
        elif i == 1:
            feat_name = MAPPING[arg]['name']
            feat_pos = MAPPING[arg]['pos']
        elif i == 2:
            plot_title = TITLE_MAPPING[arg]
        else:
            params.append((arg, cPickle.load(open(get_tf_rnn_path(arg), 'rb'))))

    data_dict = dict()
    for name, values in params:
        type = parse_name(name)[0]
        feat_value = MODEL_MAPPING[type] + ' ' + parse_name(name)[feat_pos]
        data_dict[feat_value] = values
    data_dict['epoch'] = range(values.__len__())

    plot(data_dict, get_tf_rnn_path(feat_name+'_comparison'), feat_name=feat_name, title=' '.join([feat_name, plot_title, 'comparison']))