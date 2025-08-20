import numpy as np
import torch

class Tokenizer:
    def __init__(self, num_classes: int, num_bins: int, width: int, height: int, max_len=256):
        self.num_classes = num_classes
        self.num_bins = num_bins
        self.width = width
        self.height = height
        self.max_len = max_len
        
        # define start, end, and pad tokens
        self.BOS_code = num_bins
        self.EOS_code = self.BOS_code + 1
        self.PAD_code = self.EOS_code + 1

        self.vocab_size = num_bins + 3 #+ num_classes

    def quantize(self, x: np.array):
        """
        x is a real number in [0, 1]
        """

        return (x * (self.num_bins - 1)).round(0).astype('int')

    def dequantize(self, x: np.array):
        """
        x is an integer between [0, num_bins-1]
        """

        return x.astype('float32') / (self.num_bins - 1)

    def __call__(self, coords: np.array, shuffle=True):

        if len(coords) > 0:
            coords[:, 0] = coords[:, 0] / self.width
            coords[:, 1] = coords[:, 1] / self.height

        coords = self.quantize(coords)[:self.max_len]

        if shuffle:
            rand_idxs = np.arange(0, len(coords))
            if self.cfg.run_type.name == 'debug':
                rand_idxs = rand_idxs[::-1]
            else:
                np.random.shuffle(rand_idxs)
            coords = coords[rand_idxs]
        else:
            rand_idxs = np.arange(0, len(coords))

        tokenized = [self.BOS_code]
        for coord in coords:
            tokens = list(coord)

            tokenized.extend(list(map(int, tokens)))
        tokenized.append(self.EOS_code)

        return tokenized, rand_idxs

    def decode(self, tokens: torch.Tensor):
        """
        tokens: torch.LongTensor with shape [L]
        """

        mask = tokens != self.PAD_code
        tokens = tokens[mask]
        tokens = tokens[1:-1]
        assert len(tokens) % 2 == 0, "Invalid tokens!"

        ## why so lengthy?
        # coords = []
        # for i in range(2, len(tokens)+1, 2):
        #     coord = tokens[i-2: i]
        #     coords.append([int(item) for item in coord])
        # coords = np.array(coords)
        
        coords = np.array(tokens).reshape(-1, 2)
        coords = self.dequantize(coords)

        if len(coords) > 0:
            coords[:, 0] = coords[:, 0] * self.width
            coords[:, 1] = coords[:, 1] * self.height

        return coords

