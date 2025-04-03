from lydorn_utils import polygon_utils


class RemoveDoubles(object):
    """Removes redundant vertices of all input polygons"""

    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon

    def __call__(self, polygons):
        return [polygon_utils.remove_doubles(polygon, epsilon=self.epsilon) for polygon in polygons]
