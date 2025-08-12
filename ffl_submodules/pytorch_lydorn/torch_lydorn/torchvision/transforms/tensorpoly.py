import numpy as np

import torch


class TensorPoly(object):
    def __init__(self, pos, poly_slice, batch, batch_size, is_endpoint=None):
        """

        :param pos:
        :param poly_slice:
        :param batch: Batch index for each node
        :param is_endpoint: One value per node. If true, that node is an endpoint and is thus part of an open polyline
        """
        assert pos.shape[0] == batch.shape[0]
        self.pos = pos
        self.poly_slice = poly_slice
        self.batch = batch
        self.batch_size = batch_size
        self.is_endpoint = is_endpoint

        self.to_padded_index = None  # No pad initially
        self.to_unpadded_poly_slice = None

    @property
    def num_nodes(self):
        return self.pos.shape[0]

    def to(self, device):
        self.pos = self.pos.to(device)
        self.poly_slice = self.poly_slice.to(device)
        self.batch = self.batch.to(device)
        if self.is_endpoint is not None:
            self.is_endpoint = self.is_endpoint.to(device)
        if self.to_padded_index is not None:
            self.to_padded_index = self.to_padded_index.to(device)
        if self.to_unpadded_poly_slice is not None:
            self.to_unpadded_poly_slice = self.to_unpadded_poly_slice.to(device)


def polygons_to_tensorpoly(polygons_batch):
    """
    Parametrizes N polygons into a 1d grid to be used in 1d conv later on:
    - pos (n1+n2+..., 2) concatenation of polygons vertex positions
    - poly_slice (poly_count, 2) polygon vertex slices [start, end)

    :param polygons_batch: Batch of polygons: [[(n1, 2), (n2, 2), ...], ...]
    :return: TensorPoly(pos, poly_slice, batch, batch_size, is_endpoint)
    """
    # TODO: If there are no polygons
    batch_size = len(polygons_batch)
    is_endpoint_list = []
    batch_list = []
    polygon_list = []
    for i, polygons in enumerate(polygons_batch):
        for polygon in polygons:
            if not np.max(np.abs(polygon[0] - polygon[-1])) < 1e-6:
                # Polygon is open
                is_endpoint = np.zeros(polygon.shape[0], dtype=bool)
                is_endpoint[0] = True
                is_endpoint[-1] = True
            else:
                # Polygon is closed, remove last redundant point
                polygon = polygon[:-1, :]
                is_endpoint = np.zeros(polygon.shape[0], dtype=bool)
            batch = i * np.ones(polygon.shape[0], dtype=int)
            is_endpoint_list.append(is_endpoint)
            batch_list.append(batch)
            polygon_list.append(polygon)
    if not len(polygon_list):
        return None
    pos = np.concatenate(polygon_list, axis=0)
    is_endpoint = np.concatenate(is_endpoint_list, axis=0)
    batch = np.concatenate(batch_list, axis=0)

    slice_start = 0
    poly_slice = np.empty((len(polygon_list), 2), dtype=int)
    for i, polygon in enumerate(polygon_list):
        slice_end = slice_start + polygon.shape[0]
        poly_slice[i][0] = slice_start
        poly_slice[i][1] = slice_end
        slice_start = slice_end
    pos = torch.tensor(pos, dtype=torch.float)
    is_endpoint = torch.tensor(is_endpoint, dtype=torch.bool)
    poly_slice = torch.tensor(poly_slice, dtype=torch.long)
    batch = torch.tensor(batch, dtype=torch.long)
    tensorpoly = TensorPoly(pos=pos, poly_slice=poly_slice, batch=batch, batch_size=batch_size, is_endpoint=is_endpoint)
    return tensorpoly


def _get_to_padded_index(poly_slice, node_count, padding):
    """
    Pad each polygon with a cyclic padding scheme on both sides.
    Increases length of x by (padding[0] + padding[1])*polygon_count values.

    :param poly_slice:
    :param padding:
    :return:
    """
    assert len(poly_slice.shape) == 2, "poly_slice should have shape (poly_count, 2), not {}".format(poly_slice.shape)
    poly_count = poly_slice.shape[0]
    range_tensor = torch.arange(node_count, device=poly_slice.device)
    to_padded_index = torch.empty((node_count + (padding[0] + padding[1])*poly_count, ), dtype=torch.long, device=poly_slice.device)
    to_unpadded_poly_slice = torch.empty_like(poly_slice)
    start = 0
    for poly_i in range(poly_count):
        poly_indices = range_tensor[poly_slice[poly_i, 0]:poly_slice[poly_i, 1]]

        # Repeat poly_indices if necessary when padding exceeds polygon length
        vertex_count = poly_indices.shape[0]
        left_repeats = padding[0] // vertex_count
        left_padding_remaining = padding[0] % vertex_count
        right_repeats = padding[1] // vertex_count
        right_padding_remaining = padding[1] % vertex_count
        total_repeats = left_repeats + right_repeats
        poly_indices = poly_indices.repeat(total_repeats + 1)  # +1 includes the original polygon

        if left_padding_remaining:
            poly_indices = torch.cat([poly_indices[-left_padding_remaining:],
                                      poly_indices,
                                      poly_indices[:right_padding_remaining]])
        else:
            poly_indices = torch.cat([poly_indices, poly_indices[:right_padding_remaining]])

        end = start + poly_indices.shape[0]
        to_padded_index[start:end] = poly_indices
        to_unpadded_poly_slice[poly_i, 0] = start  # Init value
        to_unpadded_poly_slice[poly_i, 1] = end  # Init value
        start = end
    to_unpadded_poly_slice[:, 0] += padding[0]  # Shift all inited values to the right one
    to_unpadded_poly_slice[:, 1] -= padding[1]  # Shift all inited values to the right one
    return to_padded_index, to_unpadded_poly_slice


def tensorpoly_pad(tensorpoly, padding):
    to_padded_index, to_unpadded_poly_slice = _get_to_padded_index(tensorpoly.poly_slice, tensorpoly.num_nodes, padding)
    tensorpoly.to_padded_index = to_padded_index
    tensorpoly.to_unpadded_poly_slice = to_unpadded_poly_slice
    return tensorpoly


def main():
    device = "cuda"

    np.random.seed(0)
    torch.manual_seed(0)
    padding = (0, 1)

    batch_size = 2
    poly_count = 3
    vertex_min_count = 4
    vertex_max_count = 5

    polygons_batch = []
    for batch_i in range(batch_size):
        polygons = []
        for poly_i in range(poly_count):
            vertex_count = np.random.randint(vertex_min_count, vertex_max_count)
            polygon = np.random.uniform(0, 1, (vertex_count, 2))
            polygons.append(polygon)
        polygons_batch.append(polygons)
    print("polygons_batch:")
    print(polygons_batch)
    tensorpoly = polygons_to_tensorpoly(polygons_batch)
    print("batch:")
    print(tensorpoly.batch)
    print("pos:")
    print(tensorpoly.pos.shape)
    print("poly_slice:")
    print(tensorpoly.poly_slice.shape)
    print(tensorpoly.poly_slice)

    tensorpoly.to(device)

    tensorpoly = tensorpoly_pad(tensorpoly, padding)
    to_padded_index = tensorpoly.to_padded_index
    print("to_padded_index:")
    print(to_padded_index.shape)
    print(to_padded_index)


if __name__ == "__main__":
    main()
