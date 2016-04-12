__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from metrics import Metrics
from itertools import chain
import cPickle
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaluate pickled results')
    parser.add_argument('--file', type=str, required=True, help='Pickle file with predictions.')
    parser.add_argument('--index2labels', type=str, default=False, help='Pickle file with mapping.')
    parser.add_argument('--oldv', action='store_true', help='Old version pickle?.')
    arguments = parser.parse_args()
    path = arguments.file
    old_version = arguments.oldv
    index2labels_path = arguments.index2labels

    p = cPickle.load(open(path, 'rb'))

    #TODO: this is horrible. Consistency!
    if old_version:
        y_t = [true for sent in p.values() for _,_,true in sent]
        y_p = [pred for sent in p.values() for _,pred,_ in sent]
    else:
        y_t = list(chain(*[trues[0] for trues in p.values()]))
        y_p = list(chain(*[trues[1] for trues in p.values()]))

    if index2labels_path:
        index2labels = cPickle.load(open(index2labels_path, 'rb'))
        print Metrics.compute_classification_report(y_t, y_p, labels=index2labels.keys(),
                                                    target_names=index2labels.values())
    else:
        print Metrics.compute_classification_report(y_t, y_p)