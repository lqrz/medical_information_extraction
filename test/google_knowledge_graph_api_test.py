"""Example of Python client calling Knowledge Graph Search API."""
import json
import urllib

import os

def get_api_key():
    path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))+'/.api_key'
    return open(path).read()

api_key = get_api_key()
query = 'Taylor Swift'
service_url = 'https://kgsearch.googleapis.com/v1/entities:search'
params = {
    'query': query,
    'limit': 10,
    'indent': True,
    'key': api_key,
}
url = service_url + '?' + urllib.urlencode(params)
response = json.loads(urllib.urlopen(url).read())
for element in response['itemListElement']:
  print element['result']['name'] + ' (' + str(element['resultScore']) + ')'