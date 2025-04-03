# class Map(object):
#     """Applies parameter transform to all items of input list"""
#
#     def __init__(self, transform, multi_outs=False):
#         self.transform = transform
#         if multi_outs:
#             self.__call__ = self._map_multi_outs
#         else:
#             self.__call__ = self._map_single_out
#
#     def _map_single_out(self, data_list):
#         assert type(data_list) == list, "data_list should be a list"
#         return [self.transform(item) for item in data_list]
#
#     def _map_multi_outs(self, data_list):
#         assert type(data_list) == list, "data_list should be a list"
#         return tuple(zip(*[self.transform(item) for item in data_list]))


class Map(object):
    """Applies parameter transform to all items of input list"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, data_list):
        assert type(data_list) == list, "data_list should be a list"
        return [self.transform(item) for item in data_list]
