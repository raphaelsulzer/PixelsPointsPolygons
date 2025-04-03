import torch


class SampleUniform(object):

    def __init__(self, low, high):
        self.m = torch.distributions.uniform.Uniform(low, high)

    def __call__(self):
        return self.m.sample()
