__author__ = 'root'

import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

import json
import urllib
import os
from itertools import chain
import cPickle

from data.dataset import Dataset
from data import get_google_knowled_graph_cache

def get_api_key():
    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/.api_key'
    return open(path).read()

if __name__ == '__main__':
    output_filename = 'knowledge_graph_cache.p'

    api_key = get_api_key()

    print 'Retrieving datasets...'
    _, _, train_document_sentence_words, _ = Dataset.get_clef_training_dataset(lowercase=True)
    _, _, valid_document_sentence_words, _ = Dataset.get_clef_validation_dataset(lowercase=True)
    _, _, test_document_sentence_words, _ = Dataset.get_clef_testing_dataset(lowercase=True)

    unique_words = set(
        list(chain(*chain(*train_document_sentence_words.values()))) + \
        list(chain(*chain(*valid_document_sentence_words.values()))) + \
        list(chain(*chain(*test_document_sentence_words.values())))
    )

    service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
    params = {
        'limit': 10,
        'indent': True,
        'key': api_key,
    }

    results = dict()

    print 'Querying knowledge graph...'
    for word_type in unique_words:
        params['query'] = word_type
        url = service_url + '?' + urllib.urlencode(params)
        response = json.loads(urllib.urlopen(url).read())
        types = []
        try:
            types = list(chain(*[element['result']['@type'] for element in response['itemListElement']]))
        except KeyError:
            pass

        results[word_type] = types

    print 'Pickling results...'
    out_path = get_google_knowled_graph_cache(output_filename)

    cPickle.dump(results, open(out_path, 'wb'))

    print 'End.'