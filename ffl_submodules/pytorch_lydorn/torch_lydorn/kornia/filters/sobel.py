import torch

from kornia.filters.kernels import normalize_kernel2d

from .kernels import get_spatial_gradient_kernel2d


class SpatialGradient(torch.nn.Module):
    r"""Computes the first order image derivative in both x and y using a Sobel or Scharr
    operator.

    Return:
        torch.Tensor: the sobel edges of the input feature map.

    Shape:
        - Input: :math:`(B, C, H, W)`
        - Output: :math:`(B, C, 2, H, W)`

    Examples:
        input = torch.rand(1, 3, 4, 4)
        output = kornia.filters.SpatialGradient()(input)  # 1x3x2x4x4
    """

    def __init__(self,
                 mode: str = 'sobel',
                 order: int = 1,
                 normalized: bool = True,
                 coord: str = "xy",
                 device: str = "cpu",
                 dtype: torch.dtype = torch.float) -> None:
        super(SpatialGradient, self).__init__()
        self.normalized: bool = normalized
        self.order: int = order
        self.mode: str = mode
        self.kernel: torch.Tensor = get_spatial_gradient_kernel2d(mode, order, coord)
        if self.normalized:
            self.kernel = normalize_kernel2d(self.kernel)
        # Pad with "replicate for spatial dims, but with zeros for channel
        self.spatial_pad = [self.kernel.size(1) // 2,
                            self.kernel.size(1) // 2,
                            self.kernel.size(2) // 2,
                            self.kernel.size(2) // 2]
        # Prepare kernel
        self.kernel: torch.Tensor = self.kernel.to(device).to(dtype).detach()
        self.kernel: torch.Tensor = self.kernel.unsqueeze(1).unsqueeze(1)
        self.kernel: torch.Tensor = self.kernel.flip(-3)
        return

    def __repr__(self) -> str:
        return self.__class__.__name__ + '('\
            'order=' + str(self.order) + ', ' + \
            'normalized=' + str(self.normalized) + ', ' + \
            'mode=' + self.mode + ')'

    def forward(self, inp: torch.Tensor) -> torch.Tensor:  # type: ignore
        if not torch.is_tensor(inp):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(inp)))
        if not len(inp.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxCxHxW. Got: {}"
                             .format(inp.shape))
        # prepare kernel
        b, c, h, w = inp.shape

        # convolve inp tensor with sobel kernel
        out_channels: int = 3 if self.order == 2 else 2
        padded_inp: torch.Tensor = torch.nn.functional.pad(inp.reshape(b * c, 1, h, w),
                                                           self.spatial_pad, 'replicate')[:, :, None]
        return torch.nn.functional.conv3d(padded_inp, self.kernel, padding=0).view(b, c, out_channels, h, w)
