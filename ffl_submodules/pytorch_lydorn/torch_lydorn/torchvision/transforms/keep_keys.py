class KeepKeys(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        return {key: data[key] for key in self.keys if key in data}
