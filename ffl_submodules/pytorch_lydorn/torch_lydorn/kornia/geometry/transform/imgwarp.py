import numpy as np
import torch

__all__ = [
    "get_affine_grid",
]


def get_affine_grid(tensor: torch.Tensor, angle: torch.Tensor, offset: torch.Tensor, scale: torch.Tensor=None) -> torch.Tensor:
    r"""Get rotation sample grid to rotate the image anti-clockwise.
    """
    assert len(tensor.shape) == 4, "tensor should have shape (N, C, H, W)"
    assert len(angle.shape) == 1, "tensor should have shape (N,)"
    assert len(offset.shape) == 2 and offset.shape[1] == 2, "tensor should have shape (N, 2)"
    assert tensor.shape[0] == angle.shape[0] == offset.shape[0], "tensor, angle and offset should have the same batch_size"
    rad = np.pi * angle / 180
    affine_mat = torch.zeros((rad.shape[0], 2, 3), device=rad.device)
    sin_rad = torch.sin(rad)
    cos_rad = torch.cos(rad)
    if scale is not None:
        assert len(scale.shape) == 1, "tensor should have shape (N,)"
        affine_mat[:, 0, 0] = cos_rad * scale
        affine_mat[:, 1, 1] = cos_rad * scale
    else:
        affine_mat[:, 0, 0] = cos_rad
        affine_mat[:, 1, 1] = cos_rad
    affine_mat[:, 0, 1] = - sin_rad
    affine_mat[:, 1, 0] = sin_rad
    affine_mat[:, :, 2] = offset
    # affine_grid = torch.nn.functional.affine_grid(affine_mat, tensor.shape, align_corners=False)  # uncomment when using more recent version of PyTorch
    affine_grid = torch.nn.functional.affine_grid(affine_mat, tensor.shape)
    return affine_grid


def main():
    pass


if __name__ == '__main__':
    main()
