__author__ = 'root'
import pkg_resources

def get_filename(file_name):
    path = pkg_resources.resource_filename('utils', file_name)

    return path