__author__ = 'root'
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from metrics import Metrics
from itertools import chain
import cPickle
import argparse

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Evaliate pickled results')
    parser.add_argument('--file', type=str, required=True, help='Pickle file with predictions.')
    arguments = parser.parse_args()
    path = arguments.file

    p = cPickle.load(open(path, 'rb'))

    #TODO: this is horrible. Consistency!
    if 'pycrf' in path:
        y_t = list(chain(*[true for _,_,true in p.values()]))
        y_p = list(chain(*[pred for _,pred,_ in p.values()]))
    else:
        y_t = list(chain(*[trues[0] for trues in p.values()]))
        y_p = list(chain(*[trues[1] for trues in p.values()]))

    print '## MICRO results'
    print Metrics.compute_all_metrics(y_t,y_p,average='micro')

    print '## MACRO results'
    print Metrics.compute_all_metrics(y_t,y_p,average='macro')