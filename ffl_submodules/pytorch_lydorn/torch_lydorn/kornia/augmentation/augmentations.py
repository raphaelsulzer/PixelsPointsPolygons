from typing import Tuple, List, Union, cast
import torch

from kornia.geometry.transform import vflip, rotate

UnionType = Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]


def random_rotate(input: torch.Tensor) -> UnionType:
    r"""Rotate a tensor image or a batch of tensor images randomly.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    Args:
        input tensor.
    Returns:
        torch.Tensor: The rotated input
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    device: torch.device = input.device
    input = input.unsqueeze(0)
    input = input.view(-1, *input.shape[-3:])
    angle: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(-180, -180)

    rotated = rotate(input, angle)
    return rotated


def random_vflip(input: torch.Tensor, p: float = 0.5, return_transform: bool = False) -> UnionType:
    r"""Vertically flip a tensor image or a batch of tensor images randomly with a given probability.
    Input should be a tensor of shape (C, H, W) or a batch of tensors :math:`(*, C, H, W)`.
    Args:
        p (float): probability of the image being flipped. Default value is 0.5
        return_transform (bool): if ``True`` return the matrix describing the transformation applied to each
        input tensor.
    Returns:
        torch.Tensor: The vertically flipped input
        torch.Tensor: The applied transformation matrix :math: `(*, 3, 3)` if return_transform flag
                      is set to ``True``
    """

    if not torch.is_tensor(input):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not isinstance(p, float):
        raise TypeError(f"The probability should be a float number. Got {type(p)}")

    if not isinstance(return_transform, bool):
        raise TypeError(f"The return_transform flag must be a bool. Got {type(return_transform)}")

    device: torch.device = input.device
    dtype: torch.dtype = input.dtype

    input = input.unsqueeze(0)
    input = input.view(-1, *input.shape[-3:])

    probs: torch.Tensor = torch.empty(input.shape[0], device=device).uniform_(0, 1)

    to_flip: torch.Tensor = probs < p
    flipped: torch.Tensor = input.clone()

    flipped[to_flip] = vflip(input[to_flip])

    if return_transform:

        trans_mat: torch.Tensor = torch.eye(3, device=device, dtype=dtype).expand(input.shape[0], -1, -1)

        w: int = input.shape[-2]
        flip_mat: torch.Tensor = torch.tensor([[-1, 0, w],
                                               [0, 1, 0],
                                               [0, 0, 1]])

        trans_mat[to_flip] = flip_mat.to(device).to(dtype)

        return flipped, trans_mat

    return flipped
