class RemoveKeys(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, data):
        return {key: data[key] for key in data.keys() if key not in self.keys}
