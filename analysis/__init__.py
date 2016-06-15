import pkg_resources

def get_lookup_path(filename):
    return pkg_resources.resource_filename('analysis', 'lookup/%s' % filename)
