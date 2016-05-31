__author__ = 'lqrz'

from itertools import chain
import argparse
import os.path
import subprocess

from data.dataset import Dataset
from trained_models import get_single_mlp_path
from trained_models import get_cwnn_path
from trained_models import get_multi_hidden_cw_path
from trained_models import get_ensemble_forest_mlp_path
from metrics import Metrics

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run conlleval script')
    parser.add_argument('--model', type=str, action='store', required=True,
                        choices=['single_cw', 'hidden_cw', 'multi_hidden_cw', 'forest_mlp'],
                        help='NNet type')

    args = dict()

    # parse arguments
    arguments = parser.parse_args()
    args['model'] = arguments.model

    return args

def get_true_values():
    _, _, _, train_document_sentence_tags = Dataset.get_clef_training_dataset()
    _, _, _, valid_document_sentence_tags = Dataset.get_clef_validation_dataset()

    train_trues = list(chain(*chain(*train_document_sentence_tags.values())))
    valid_trues = list(chain(*chain(*valid_document_sentence_tags.values())))

    return train_trues, valid_trues

def get_prediction_values(model):
    train_predictions = None
    valid_predictions = None

    if model == 'single_cw':
        # get_single_mlp_path7
        raise NotImplementedError
    elif model == 'hidden_cw':
        train_file = get_cwnn_path('train_A.txt')
        valid_file = get_cwnn_path('validation_A.txt')
    elif model == 'multi_hidden_cw':
        # get_multi_hidden_cw_path
        raise NotImplementedError
    elif model == 'forest_mlp':
        train_file = get_ensemble_forest_mlp_path('train_B.txt')
        valid_file = get_ensemble_forest_mlp_path('validation_B.txt')

    if os.path.exists(train_file) and os.path.exists(valid_file):
        train_predictions = map(lambda x: x.strip().split('\t')[2], open(train_file, 'rb').readlines())
        valid_predictions = map(lambda x: x.strip().split('\t')[2], open(valid_file, 'rb').readlines())
    else:
        raise Exception('Path does not exist')

    return train_predictions, valid_predictions

if __name__ == '__main__':

    args = parse_arguments()

    train_predictions, valid_predictions = get_prediction_values(args['model'])

    train_trues, valid_trues = get_true_values()

    assert train_trues.__len__() == train_predictions.__len__()
    assert valid_trues.__len__() == valid_predictions.__len__()

    tmp_filename = 'train_conlleval.tmp'
    tmp_file = open(tmp_filename, 'wb')
    for i, (true, pred) in enumerate(zip(train_trues, train_predictions)):
        tmp_file.write(' '.join([str(i),true,pred])+'\n')
        tmp_file.write('\n')

    tmp_file.close()

    tmp_filename = 'valid_conlleval.tmp'
    tmp_file = open(tmp_filename, 'wb')
    for i, (true, pred) in enumerate(zip(valid_trues, valid_predictions)):
        tmp_file.write(' '.join([str(i),true,pred])+'\n')
        tmp_file.write('\n')

    tmp_file.close()

    macro_train_results = Metrics.compute_all_metrics(y_true=train_trues, y_pred=train_predictions, average='macro')
    macro_valid_results = Metrics.compute_all_metrics(y_true=valid_trues, y_pred=valid_predictions, average='macro')

    print '...End'