import random

from .functional import to_tensor, center_crop

__all__ = ["CenterCrop", "ToTensor", "RandomBool", "ConditionApply"]


class CenterCrop(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, tensor):
        return center_crop(tensor, self.output_size)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor without typecasting and rescaling.
    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.ByteTensor of shape (C x H x W) in the range [0, 255]
    if the PIL Image belongs to one of the modes (L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK, 1)
    or if the numpy.ndarray has dtype = np.uint8
    """

    def __call__(self, pic):
        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        return to_tensor(pic)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class RandomBool(object):
    """Produce a random boolean with p probability for it to be True

    Args:
        p (float): probability
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self):
        return random.random() < self.p


class ConditionApply(object):
    """Apply a transformation if condition is met

    Args:
        transform:
    """

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, tensor, condition):
        """

        :param tensor:
        :param condition (bool): True: apply, False: do not apply
        :return:
        """
        if condition:
            tensor = self.transform(tensor)
        return tensor

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += '\n'
        format_string += '    {0}'.format(self.transform)
        format_string += '\n)'
        return format_string
