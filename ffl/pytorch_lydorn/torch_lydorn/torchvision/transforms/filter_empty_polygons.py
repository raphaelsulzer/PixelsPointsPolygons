class FilterEmptyPolygons(object):
    """Removes None elements of input data_list"""

    def __init__(self, key):
        self.key = key

    def _filter(self, item):
        if item[self.key] is not None:
            return len(item[self.key])
        else:
            return False

    def __call__(self, data_list):
        return [item for item in data_list if self._filter(item)]
