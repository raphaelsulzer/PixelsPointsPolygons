import torch

from kornia.filters.kernels import get_sobel_kernel2d, get_sobel_kernel2d_2nd_order, get_diff_kernel2d, get_diff_kernel2d_2nd_order


def get_scharr_kernel_3x3() -> torch.Tensor:
    """Utility function that returns a sobel kernel of 3x3"""
    return torch.tensor([
        [-47., 0., 47.],
        [-162., 0., 162.],
        [-47., 0., 47.],
    ])


def get_scharr_kernel2d(coord: str = "xy") -> torch.Tensor:
    assert coord == "xy" or coord == "ij"
    kernel_x: torch.Tensor = get_scharr_kernel_3x3()
    kernel_y: torch.Tensor = kernel_x.transpose(0, 1)
    if coord == "xy":
        return torch.stack([kernel_x, kernel_y])
    elif coord == "ij":
        return torch.stack([kernel_y, kernel_x])


def get_spatial_gradient_kernel2d(mode: str, order: int, coord: str = "xy") -> torch.Tensor:
    r"""Function that returns kernel for 1st or 2nd order image gradients,
    using one of the following operators: sobel, diff"""
    if mode not in ['sobel', 'diff', 'scharr']:
        raise TypeError("mode should be either sobel, diff or scharr. Got {}".format(mode))
    if order not in [1, 2]:
        raise TypeError("order should be either 1 or 2\
                         Got {}".format(order))
    if mode == 'sobel' and order == 1:
        kernel: torch.Tensor = get_sobel_kernel2d()
    elif mode == 'sobel' and order == 2:
        kernel = get_sobel_kernel2d_2nd_order()
    elif mode == 'diff' and order == 1:
        kernel = get_diff_kernel2d()
    elif mode == 'diff' and order == 2:
        kernel = get_diff_kernel2d_2nd_order()
    elif mode == 'scharr' and order == 1:
        kernel = get_scharr_kernel2d(coord)
    else:
        raise NotImplementedError("")
    return kernel


def main():
    import cv2
    k_x, k_y = cv2.getDerivKernels(kx=3, ky=3, dx=1, dy=1, ksize=-1)
    print(k_x)
    print(k_y)


if __name__ == "__main__":
    main()