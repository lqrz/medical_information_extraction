__author__ = 'root'

import pkg_resources
import os.path

def get_cwnn_path(file_name):

    path = pkg_resources.resource_filename('trained_models', 'cwnn/'+file_name)
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

    return path