import torch


def get_real(t, complex_dim=-1):
    return t.select(complex_dim, 0)


def get_imag(t, complex_dim=-1):
    return t.select(complex_dim, 1)


def complex_mul(t1, t2, complex_dim=-1):
    t1_real = get_real(t1, complex_dim)
    t1_imag = get_imag(t1, complex_dim)
    t2_real = get_real(t2, complex_dim)
    t2_imag = get_imag(t2, complex_dim)

    ac = t1_real * t2_real
    bd = t1_imag * t2_imag
    ad = t1_real * t2_imag
    bc = t1_imag * t2_real
    tr_real = ac - bd
    tr_imag = ad + bc

    tr = torch.stack([tr_real, tr_imag], dim=complex_dim)

    return tr


def complex_sqrt(t, complex_dim=-1):
    sqrt_t_abs = torch.sqrt(complex_abs(t, complex_dim))
    sqrt_t_arg = complex_arg(t, complex_dim) / 2
    # Overwrite t with cos(\theta / 2) + i sin(\theta / 2):
    sqrt_t = sqrt_t_abs.unsqueeze(complex_dim) * torch.stack([torch.cos(sqrt_t_arg), torch.sin(sqrt_t_arg)], dim=complex_dim)
    return sqrt_t


def complex_abs_squared(t, complex_dim=-1):
    return get_real(t, complex_dim)**2 + get_imag(t, complex_dim)**2


def complex_abs(t, complex_dim=-1):
    return torch.sqrt(complex_abs_squared(t, complex_dim=complex_dim))


def complex_arg(t, complex_dim=-1):
    return torch.atan2(get_imag(t, complex_dim), get_real(t, complex_dim))


def main():
    device = None

    t1 = torch.Tensor([
        [2, 0],
        [0, 2],
        [-1, 0],
        [0, -1],
    ]).to(device)
    t2 = torch.Tensor([
        [2, 0],
        [0, 2],
        [-1, 0],
        [0, -1],
    ]).to(device)
    complex_dim = -1

    print(t1.int())
    print(t2.int())
    t1_mul_t2 = complex_mul(t1, t2, complex_dim)
    print(t1_mul_t2.int())

    sqrt_t1_mul_t2 = complex_sqrt(t1_mul_t2)
    print(sqrt_t1_mul_t2.int())


if __name__ == "__main__":
    main()