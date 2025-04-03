import numpy as np
import time
import sklearn.datasets
import skimage.transform
import scipy.stats

from . import python_utils
from . import image_utils

# if python_utils.module_exists("matplotlib.pyplot"):
#     import matplotlib.pyplot as plt

CV2 = False
if python_utils.module_exists("cv2"):
    import cv2

    CV2 = True


# import multiprocessing
#
# import python_utils
#
# if python_utils.module_exists("joblib"):
#     from joblib import Parallel, delayed
#     JOBLIB = True
# else:
#     JOBLIB = False


# def plot_field_map(field_map):
#     from mpl_toolkits.mplot3d import Axes3D
#
#     row = np.linspace(0, 1, field_map.shape[0])
#     col = np.linspace(0, 1, field_map.shape[1])
#     rr, cc = np.meshgrid(row, col, indexing='ij')
#
#     fig = plt.figure(figsize=(18, 9))
#     ax = fig.add_subplot(121, projection='3d')
#     ax.plot_surface(rr, cc, field_map[:, :, 0], rstride=3, cstride=3, linewidth=1, antialiased=True)
#
#     ax = fig.add_subplot(122, projection='3d')
#     ax.plot_surface(rr, cc, field_map[:, :, 1], rstride=3, cstride=3, linewidth=1, antialiased=True)
#
#     plt.show()

# --- Classes --- #

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name="", init_val=0, fmt=':f'):
        self.name = name
        self.init_val = init_val
        self.fmt = fmt
        self.val = self.avg = self.init_val
        self.sum = self.count = 0

    def reset(self):
        self.val = self.avg = self.init_val
        self.sum = self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_avg(self):
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class RunningDecayingAverage(object):
    """
    Updates average with val*(1 - decay) + avg*decay
    """
    def __init__(self, decay, init_val=0):
        assert 0 < decay < 1
        self.decay = decay
        self.init_val = init_val
        self.val = self.avg = self.init_val

    def reset(self):
        self.val = self.avg = self.init_val

    def update(self, val):
        self.val = val
        self.avg = (1 - self.decay)*val + self.decay*self.avg

    def get_avg(self):
        return self.avg


class DispFieldMapsPatchCreator:
    def __init__(self, global_shape, patch_res, map_count, modes, gauss_mu_range, gauss_sig_scaling):
        self.global_shape = global_shape
        self.patch_res = patch_res
        self.map_count = map_count
        self.modes = modes
        self.gauss_mu_range = gauss_mu_range
        self.gauss_sig_scaling = gauss_sig_scaling

        self.current_patch_index = -1
        self.patch_boundingboxes = image_utils.compute_patch_boundingboxes(self.global_shape, stride=self.patch_res,
                                                                           patch_res=self.patch_res)
        self.disp_maps = None
        self.create_new_disp_maps()

    def create_new_disp_maps(self):
        print("DispFieldMapsPatchCreator.create_new_disp_maps()")
        self.disp_maps = create_displacement_field_maps(self.global_shape, self.map_count, self.modes,
                                                        self.gauss_mu_range, self.gauss_sig_scaling)

    def get_patch(self):
        self.current_patch_index += 1

        if len(self.patch_boundingboxes) <= self.current_patch_index:
            self.current_patch_index = 0
            self.create_new_disp_maps()

        patch_boundingbox = self.patch_boundingboxes[self.current_patch_index]
        patch_disp_maps = self.disp_maps[:, patch_boundingbox[0]:patch_boundingbox[2],
                          patch_boundingbox[1]:patch_boundingbox[3], :]
        return patch_disp_maps


# --- --- #

def compute_crossfield_c0c2(u, v):
    c0 = np.power(u, 2) * np.power(v, 2)
    c2 = - (np.power(u, 2) + np.power(v, 2))
    crossfield = np.stack([c0.real, c0.imag, c2.real, c2.imag], axis=-1)
    return crossfield


def compute_crossfield_uv(c0c2):
    c0 = c0c2[..., 0] + 1j * c0c2[..., 1]
    c2 = c0c2[..., 2] + 1j * c0c2[..., 3]
    sqrt_c2_squared_minus_4c0 = np.sqrt(np.power(c2, 2) - 4 * c0)
    u_squared = (c2 + sqrt_c2_squared_minus_4c0) / 2
    v_squared = (c2 - sqrt_c2_squared_minus_4c0) / 2
    u = np.sqrt(u_squared)
    v = np.sqrt(v_squared)
    return u, v


def to_homogeneous(array):
    new_array = np.ones((array.shape[0], array.shape[1] + 1), dtype=array.dtype)
    new_array[..., :-1] = array
    return new_array


def to_euclidian(array_homogeneous):
    array = array_homogeneous[:, 0:2] / array_homogeneous[:, 2:3]
    return array


def stretch(array):
    mini = np.min(array)
    maxi = np.max(array)
    if maxi - mini:
        array -= mini
        array *= 2 / (maxi - mini)
        array -= 1
    return array


def crop_center(array, out_shape):
    assert len(out_shape) == 2, "out_shape should be of length 2"
    in_shape = np.array(array.shape[:2])
    start = in_shape // 2 - (out_shape // 2)
    out_array = array[start[0]:start[0] + out_shape[0], start[1]:start[1] + out_shape[1], ...]
    return out_array


def multivariate_gaussian(pos, mu, sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    sigma_det = np.linalg.det(sigma)
    sigma_inv = np.linalg.inv(sigma)
    N = np.sqrt((2 * np.pi) ** n * sigma_det)
    # This einsum call calculates (x-mu)T.sigma-1.(x-mu) in a vectorized
    # way across all the input variables.

    # print("\tStarting to create multivariate Gaussian")
    # start = time.time()

    # print((pos - mu).shape)
    # print(sigma_inv.shape)
    try:
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu, optimize=True)
    except:
        fac = np.einsum('...k,kl,...l->...', pos - mu, sigma_inv, pos - mu)
    # print(fac.shape)

    # end = time.time()
    # print("\tFinished Gaussian in {}s".format(end - start))

    return np.exp(-fac / 2) / N


def create_multivariate_gaussian_mixture_map(shape, mode_count, mu_range, sig_scaling):
    shape = np.array(shape)
    # print("Starting to create multivariate Gaussian mixture")
    # main_start = time.time()

    dim_count = 2
    downsample_factor = 4
    dtype = np.float32

    mu_scale = mu_range[1] - mu_range[0]
    row = np.linspace(mu_range[0], mu_range[1], mu_scale * shape[0] / downsample_factor, dtype=dtype)
    col = np.linspace(mu_range[0], mu_range[1], mu_scale * shape[1] / downsample_factor, dtype=dtype)
    rr, cc = np.meshgrid(row, col, indexing='ij')
    grid = np.stack([rr, cc], axis=2)

    mus = np.random.uniform(mu_range[0], mu_range[1], (mode_count, dim_count, 2)).astype(dtype)
    # gams = np.random.rand(mode_count, dim_count, 2, 2).astype(dtype)
    signs = np.random.choice([1, -1], size=(mode_count, dim_count))

    # print("\tAdding gaussian mixtures one by one")
    # start = time.time()

    # if JOBLIB:
    #     # Parallel computing of multivariate gaussians
    #     inputs = range(8)
    #
    #     def processInput(i):
    #         size = 10 * i + 2000
    #         a = np.random.random_sample((size, size))
    #         b = np.random.random_sample((size, size))
    #         n = np.dot(a, b)
    #         return n
    #
    #     num_cores = multiprocessing.cpu_count()
    #     print("num_cores: {}".format(num_cores))
    #     # num_cores = 1
    #
    #     results = Parallel(n_jobs=num_cores)(delayed(processInput)(i) for i in inputs)
    #     for result in results:
    #         print(result.shape)
    #
    #     gaussian_mixture = np.zeros_like(grid)
    # else:
    gaussian_mixture = np.zeros_like(grid)
    for mode_index in range(mode_count):
        for dim in range(dim_count):
            sig = (sig_scaling[1] - sig_scaling[0]) * sklearn.datasets.make_spd_matrix(2) + sig_scaling[0]
            # sig = (sig_scaling[1] - sig_scaling[0]) * np.dot(gams[mode_index, dim], np.transpose(gams[mode_index, dim])) + sig_scaling[0]
            sig = sig.astype(dtype)
            multivariate_gaussian_grid = signs[mode_index, dim] * multivariate_gaussian(grid, mus[mode_index, dim], sig)
            gaussian_mixture[:, :, dim] += multivariate_gaussian_grid

    # end = time.time()
    # print("\tFinished adding gaussian mixtures in {}s".format(end - start))

    # squared_gaussian_mixture = np.square(gaussian_mixture)
    # magnitude_disp_field_map = np.sqrt(squared_gaussian_mixture[:, :, 0] + squared_gaussian_mixture[:, :, 1])
    # max_magnitude = magnitude_disp_field_map.max()

    gaussian_mixture[:, :, 0] = stretch(gaussian_mixture[:, :, 0])
    gaussian_mixture[:, :, 1] = stretch(gaussian_mixture[:, :, 1])

    # Crop
    gaussian_mixture = crop_center(gaussian_mixture, shape // downsample_factor)

    # plot_field_map(gaussian_mixture)

    # Upsample mixture
    # gaussian_mixture = skimage.transform.rescale(gaussian_mixture, downsample_factor)
    gaussian_mixture = skimage.transform.resize(gaussian_mixture, shape)

    main_end = time.time()
    # print("Finished multivariate Gaussian mixture in {}s".format(main_end - main_start))

    return gaussian_mixture


def create_displacement_field_maps(shape, map_count, modes, gauss_mu_range, gauss_sig_scaling, seed=None):
    if seed is not None:
        np.random.seed(seed)
    disp_field_maps_list = []
    for disp_field_map_index in range(map_count):
        disp_field_map_normed = create_multivariate_gaussian_mixture_map(shape,
                                                                         modes,
                                                                         gauss_mu_range,
                                                                         gauss_sig_scaling)
        disp_field_maps_list.append(disp_field_map_normed)
    disp_field_maps = np.stack(disp_field_maps_list, axis=0)

    return disp_field_maps


def get_h_mat(t, theta, scale_offset, shear, p):
    """
    Computes the homography matrix given the parameters
    See https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
    (fixed mistake in H_a)

    :param t: 2D translation vector
    :param theta: Scalar angle
    :param scale_offset: 2D scaling vector
    :param shear: 2D shearing vector
    :param p: 2D projection vector
    :return: h_mat: shape (3, 3)
    """
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    h_e = np.array([
        [cos_theta, -sin_theta, t[0]],
        [sin_theta, cos_theta, t[1]],
        [0, 0, 1],
    ])
    h_a = np.array([
        [1 + scale_offset[0], shear[1], 0],
        [shear[0], 1 + scale_offset[1], 0],
        [0, 0, 1],
    ])
    h_p = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [p[0], p[1], 1],
    ])
    h_mat = h_e @ h_a @ h_p
    return h_mat


if CV2:
    def find_homography_4pt(src, dst):
        """
        Estimates the homography that transforms src points into dst points.
        Then converts the matrix representation into the 4 points representation.

        :param src:
        :param dst:
        :return:
        """
        h_mat, _ = cv2.findHomography(src, dst)
        h_4pt = convert_h_mat_to_4pt(h_mat)
        return h_4pt


    def convert_h_mat_to_4pt(h_mat):
        src_4pt = np.array([[
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ]], dtype=np.float64)
        h_4pt = cv2.perspectiveTransform(src_4pt, h_mat)
        return h_4pt


    def convert_h_4pt_to_mat(h_4pt):
        src_4pt = np.array([
            [-1, -1],
            [1, -1],
            [1, 1],
            [-1, 1],
        ], dtype=np.float32)
        h_4pt = h_4pt.astype(np.float32)
        h_mat = cv2.getPerspectiveTransform(src_4pt, h_4pt)
        return h_mat


    def field_map_to_image(field_map):
        mag, ang = cv2.cartToPolar(field_map[..., 0], field_map[..., 1])
        hsv = np.zeros((field_map.shape[0], field_map.shape[1], 3))
        hsv[..., 0] = ang * 180 / np.pi / 2
        hsv[..., 1] = 255
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        hsv = hsv.astype(np.uint8)
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return rgb
else:
    def find_homography_4pt(src, dst):
        print("cv2 is not available, the find_homography_4pt(src, dst) function cannot work!")


    def convert_h_mat_to_4pt(h_mat):
        print("cv2 is not available, the convert_h_mat_to_4pt(h_mat) function cannot work!")


    def convert_h_4pt_to_mat(h_4pt):
        print("cv2 is not available, the convert_h_4pt_to_mat(h_4pt) function cannot work!")


    def field_map_to_image(field_map):
        print("cv2 is not available, the field_map_to_image(field_map) function cannot work!")


def circular_diff(a1, a2, range_max):
    """
    Compute difference between a1 and a2 belonging to the circular interval [0, range_max).
    For example to compute angle difference, use range_max=2*PI.
    a1 and a2 must be between range_min and range_max!
    Thus difference between 0 and range_max is 0.
    :param a1: numpy array
    :param a2: numpy array
    :param range_max:
    :return:
    """
    d = range_max / 2 - np.abs(np.abs(a1 - a2) - range_max / 2)
    return d


def invert_permutation(p):
    '''The argument p is assumed to be some permutation of 0, 1, ..., len(p)-1.
    Returns an array s, where s[i] gives the index of i in p.
    '''
    s = np.empty(p.size, p.dtype)
    s[p] = np.arange(p.size)
    return s


def region_growing_1d(array, max_range, max_skew):
    """
    :param array:
    :param max_var:
    :param max_mean_median_diff:
    :return:
    """

    def verify_predicate(region):
        """
        Region is sorted
        :param region:
        :return:
        """
        skew = scipy.stats.skew(region)
        return region[-1] - region[0] < max_range and abs(skew) < max_skew

    assert len(array.shape) == 1, "array should be 1d, not {}".format(array.shape)
    p = np.argsort(array)
    sorted_array = array[p]

    labels = np.zeros(len(sorted_array), dtype=np.long)
    region_start = 0
    region_label = 1
    labels[region_start] = region_label
    centers = []
    for i in range(1, len(sorted_array)):
        region = sorted_array[region_start:i + 1]
        if not verify_predicate(region):
            # End current region
            median = region[len(region) // 2]  # region is sorted
            centers.append(median)
            # Begin a new region
            region_start = i
            region_label += 1
        labels[i] = region_label
    centers.append(median)

    return labels[invert_permutation(p)], centers


def bilinear_interpolate(im, pos):
    # From https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    x = pos[..., 1]
    y = pos[..., 0]

    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1

    x0_clipped = np.clip(x0, 0, im.shape[1] - 1)
    x1_clipped = np.clip(x1, 0, im.shape[1] - 1)
    y0_clipped = np.clip(y0, 0, im.shape[0] - 1)
    y1_clipped = np.clip(y1, 0, im.shape[0] - 1)

    Ia = im[y0_clipped, x0_clipped]
    Ib = im[y1_clipped, x0_clipped]
    Ic = im[y0_clipped, x1_clipped]
    Id = im[y1_clipped, x1_clipped]

    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    value = (Ia.T * wa).T + (Ib.T * wb).T + (Ic.T * wc).T + (Id.T * wd).T

    return value


def main():
    import matplotlib.pyplot as plt
    # shape = (220, 220)
    # mode_count = 30
    # mu_range = [0, 1]
    # sig_scaling = [0.0, 0.002]
    # create_multivariate_gaussian_mixture_map(shape, mode_count, mu_range, sig_scaling)

    # a1 = np.array([0.0])
    # a2 = np.array([3*np.pi/4])
    # range_max = np.pi
    # d = circular_diff(a1, a2, range_max)
    # print(d)

    array = np.concatenate([np.arange(1, 1.01, 0.001), np.arange(0, np.pi / 2, np.pi / 100)])
    print(array)
    labels = region_growing_1d(array, max_range=np.pi / 10, max_skew=1)
    print(labels)

    plt.plot(array, labels, ".")
    plt.show()


if __name__ == "__main__":
    main()
