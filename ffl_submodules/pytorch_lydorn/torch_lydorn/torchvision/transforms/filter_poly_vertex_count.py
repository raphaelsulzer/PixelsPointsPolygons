class FilterPolyVertexCount(object):
    """
    if min is set, only keep polygons with at least min vertices
    if max is set, only keep polygons with at most max vertices
    """

    def __init__(self, min=None, max=None):
        self.min = min
        self.max = max
        if self.min is not None and self.max is not None:
            if self.max < self.min:
                print("WARNING: min and max of FilterPolyVertexCount() are {} and {},"
                      " which creates an impossible-to_satisfy condition.".format(self.min, self.max))

    def __call__(self, polygons):
        new_polygons = []
        for polygon in polygons:
            if self.min is not None and polygon.shape[0] < self.min:
                continue
            if self.max is not None and self.max < polygon.shape[0]:
                continue
            new_polygons.append(polygon)
        return new_polygons