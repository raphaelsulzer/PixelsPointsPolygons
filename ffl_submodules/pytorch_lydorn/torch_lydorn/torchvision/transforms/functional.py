import numbers
import numpy as np
import torch
from torchvision.transforms.functional import _is_pil_image, _is_numpy_image

from torch_lydorn.torch.utils.complex import complex_mul, complex_abs_squared

try:
    import accimage
except ImportError:
    accimage = None


__all__ = ["to_tensor", "batch_normalize"]


def _is_numpy(img):
    return isinstance(img, np.ndarray)


def to_tensor(pic):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor without typecasting and rescaling

    See ``ToTensor`` for more details.

    Args:
        pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

    Returns:
        Tensor: Converted image.
    """
    if not(_is_pil_image(pic) or _is_numpy(pic)):
        raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

    if _is_numpy(pic) and not _is_numpy_image(pic):
        raise ValueError('pic should be 2/3 dimensional. Got {} dimensions.'.format(pic.ndim))

    if isinstance(pic, np.ndarray):
        # handle numpy array
        if pic.ndim == 2:
            pic = pic[:, :, None]

        img = torch.from_numpy(pic.transpose((2, 0, 1)))
        return img

    if accimage is not None and isinstance(pic, accimage.Image):
        nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.uint8)
        pic.copyto(nppic)
        return torch.from_numpy(nppic)

    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    return img


def batch_normalize(tensor, mean, std, inplace=False):
    """Normalize a batched tensor image with batched mean and batched standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Tensor means of size (B, C).
        std (sequence): Tensor standard deviations of size (B, C).
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    assert len(tensor.shape) == 4, \
        "tensor should have 4 dims (B, H, W, C) , not {}".format(len(tensor.shape))
    assert len(mean.shape) == len(std.shape) == 2, \
        "mean and std should have 2 dims (B, C) , not {} and {}".format(len(mean.shape), len(std.shape))
    assert tensor.shape[1] == mean.shape[1] == std.shape[1], \
        "tensor, mean and std should have the same number of channels, not {}, {} and {}".format(tensor.shape[1], mean.shape[1], std.shape[1])

    if not inplace:
        tensor = tensor.clone()

    mean = mean.to(tensor.dtype)
    std = std.to(tensor.dtype)

    tensor.sub_(mean[..., None, None]).div_(std[..., None, None])
    return tensor


def batch_denormalize(tensor, mean, std, inplace=False):
    """Denormalize a batched tensor image with batched mean and batched standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    Args:
        tensor (Tensor): Tensor image of size (B, C, H, W) to be normalized.
        mean (sequence): Tensor means of size (B, C).
        std (sequence): Tensor standard deviations of size (B, C).
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    assert len(tensor.shape) == 4, \
        "tensor should have 4 dims (B, H, W, C) , not {}".format(len(tensor.shape))
    assert len(mean.shape) == len(std.shape) == 2, \
        "mean and std should have 2 dims (B, C) , not {} and {}".format(len(mean.shape), len(std.shape))
    assert tensor.shape[1] == mean.shape[1] == std.shape[1], \
        "tensor, mean and std should have the same number of channels, not {}, {} and {}".format( tensor.shape[-1], mean.shape[-1], std.shape[-1])

    if not inplace:
        tensor = tensor.clone()

    mean = mean.to(tensor.dtype)
    std = std.to(tensor.dtype)

    tensor.mul_(std[..., None, None]).add_(mean[..., None, None])
    return tensor


def crop(tensor, top, left, height, width):
    """Crop the given Tensor batch of images.
    Args:
        tensor (B, C, H, W): Tensor to be cropped. (0,0) denotes the top left corner of the image.
        top (int): Vertical component of the top left corner of the crop box.
        left (int): Horizontal component of the top left corner of the crop box.
        height (int): Height of the crop box.
        width (int): Width of the crop box.
    Returns:
        Tensor: Cropped image.
    """
    return tensor[..., top:top+height, left:left+width]


def center_crop(tensor, output_size):
    """Crop the given tensor batch of images and resize it to desired size.

        Args:
            tensor (B, C, H, W): Tensor to be cropped.
            output_size (sequence or int): (height, width) of the crop box. If int,
                it is used for both directions
        Returns:
            Tensor: Cropped tensor.
        """
    if isinstance(output_size, numbers.Number):
        output_size = (int(output_size), int(output_size))
    tensor_height, tensor_width = tensor.shape[-2:]
    crop_height, crop_width = output_size
    crop_top = int(round((tensor_height - crop_height) / 2.))
    crop_left = int(round((tensor_width - crop_width) / 2.))
    return crop(tensor, crop_top, crop_left, crop_height, crop_width)


def rotate_anglefield(angle_field, angle):
    """
    :param angle_field: (B, 1, H, W), in radians
    :param angle_deg: (B) in degrees
    :return:
    """
    assert len(angle_field.shape) == 4, "angle_field should have shape (B, 1, H, W)"
    assert len(angle.shape) == 1, "angle should have shape (B)"
    assert angle_field.shape[0] == angle.shape[0], "angle_field and angle should have the same batch size"
    angle_field += np.pi * angle[:, None, None, None] / 180
    return angle_field


def vflip_anglefield(angle_field):
    """

    :param angle_field: (B, 1, H, W), in radians
    :return:
    """
    assert len(angle_field.shape) == 4, "angle_field should have shape (B, 1, H, W)"
    angle_field = np.pi - angle_field  # Angle is expressed in ij coordinate (it's a horizontal flip in xy)
    return angle_field


def rotate_framefield(framefield, angle):
    """
    ONly rotates values of the framefield, does not rotate the spatial domain (use already-made torch functions to do that).

    @param framefield: shape (B, 4, H, W). The 4 channels represent the c_0 and c_2 complex coefficients.
    @param angle: in degrees
    @return:
    """
    assert framefield.shape[1] == 4, f"framefield should have shape (B, 4, H, W), not {framefield.shape}"
    rad = np.pi * angle / 180
    z_4angle = torch.tensor([np.cos(4*rad), np.sin(4*rad)], dtype=framefield.dtype, device=framefield.device)
    z_2angle = torch.tensor([np.cos(2*rad), np.sin(2*rad)], dtype=framefield.dtype, device=framefield.device)
    framefield[:, :2, :, :] = complex_mul(framefield[:, :2, :, :], z_4angle[None, :, None, None], complex_dim=1)
    framefield[:, 2:, :, :] = complex_mul(framefield[:, 2:, :, :], z_2angle[None, :, None, None], complex_dim=1)
    return framefield


def vflip_framefield(framefield):
    """
    Flips the framefield vertically. This means switching the signs of the real part of u and v
    (this is because the framefield is in ij coordinates: it's a horizontal flip in xy), which translates to
    switching the signs of the imaginary parts of c_0 and c_2.

    @param framefield: shape (B, 4, H, W). The 4 channels represent the c_0 and c_2 complex coefficients.
    @return:
    """
    assert framefield.shape[1] == 4, f"framefield should have shape (B, 4, H, W), not {framefield.shape}"
    framefield[:, 1, :, :] = - framefield[:, 1, :, :]
    framefield[:, 3, :, :] = - framefield[:, 3, :, :]
    return framefield


def draw_circle(draw, center, radius, fill):
    draw.ellipse([center[0] - radius,
                  center[1] - radius,
                  center[0] + radius,
                  center[1] + radius], fill=fill, outline=None)
