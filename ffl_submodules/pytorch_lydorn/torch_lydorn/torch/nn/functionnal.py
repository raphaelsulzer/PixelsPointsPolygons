import torch


def bilinear_interpolate(im, pos, batch=None):
    # From https://gist.github.com/peteflorence/a1da2c759ca1ac2b74af9a83f69ce20e
    x = pos[:, 1]
    y = pos[:, 0]
    x0 = torch.floor(x).long()
    x1 = x0 + 1

    y0 = torch.floor(y).long()
    y1 = y0 + 1

    x0_int = torch.clamp(x0, 0, im.shape[-1] - 1)
    x1_int = torch.clamp(x1, 0, im.shape[-1] - 1)
    y0_int = torch.clamp(y0, 0, im.shape[-2] - 1)
    y1_int = torch.clamp(y1, 0, im.shape[-2] - 1)

    if batch is not None:
        Ia = im[batch, :, y0_int, x0_int]
        Ib = im[batch, :, y1_int, x0_int]
        Ic = im[batch, :, y0_int, x1_int]
        Id = im[batch, :, y1_int, x1_int]
    else:
        Ia = im[..., y0_int, x0_int].t()
        Ib = im[..., y1_int, x0_int].t()
        Ic = im[..., y0_int, x1_int].t()
        Id = im[..., y1_int, x1_int].t()

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (y - y0.float())
    wc = (x - x0.float()) * (y1.float() - y)
    wd = (x - x0.float()) * (y - y0.float())

    wa = wa.unsqueeze(1)
    wb = wb.unsqueeze(1)
    wc = wc.unsqueeze(1)
    wd = wd.unsqueeze(1)

    out = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return out


def main():
    im = torch.tensor([
        [
            [0, 0.5, 0, 0],
            [0.25, 1, 0, 0],
        ],
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        [
            [2, 2, 2, 2],
            [2, 2, 2, 2],
        ]
    ], dtype=torch.float)
    im = im[:, None, ...]
    print(im.shape)
    print(im)
    pos = torch.tensor([
        [1.0, 0],
        [0.5, 0.5],
        [0.5, 0.5],
    ])
    batch = torch.tensor([0, 1, 2], dtype=torch.long)

    val = bilinear_interpolate(im, pos, batch=batch)
    print(val)


if __name__ == '__main__':
    main()
