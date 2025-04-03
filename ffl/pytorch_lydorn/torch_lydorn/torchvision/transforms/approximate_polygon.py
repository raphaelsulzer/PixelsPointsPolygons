from skimage.measure import approximate_polygon


class ApproximatePolygon(object):
    """Simplifies polygons"""

    def __init__(self, tolerance=0.1):
        self.tolerance = tolerance

    def __call__(self, polygons):
        return [approximate_polygon(polygon, tolerance=self.tolerance) for polygon in polygons]
