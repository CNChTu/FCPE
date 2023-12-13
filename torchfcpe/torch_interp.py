import torch
'''
use from https://github.com/autumn-DL/TorchInterp
it use MIT license
'''


def torch_interp(x, xp, fp):
    # if not isinstance(x, torch.Tensor):
    #     x = torch.tensor(x)
    # if not isinstance(xp, torch.Tensor):
    #     xp = torch.tensor(xp)
    # if not isinstance(fp, torch.Tensor):
    #     fp = torch.tensor(fp)

    sort_idx = torch.argsort(xp)
    xp = xp[sort_idx]
    fp = fp[sort_idx]

    right_idxs = torch.searchsorted(xp, x)

    right_idxs = right_idxs.clamp(max=len(xp) - 1)

    left_idxs = (right_idxs - 1).clamp(min=0)

    x_left = xp[left_idxs]
    x_right = xp[right_idxs]
    y_left = fp[left_idxs]
    y_right = fp[right_idxs]

    interp_vals = y_left + ((x - x_left) * (y_right - y_left) / (x_right - x_left))

    interp_vals[x < xp[0]] = fp[0]
    interp_vals[x > xp[-1]] = fp[-1]

    return interp_vals


def batch_interp_with_replacement_detach(uv, f0):
    '''
    :param uv: B T
    :param f0: B T
    :return: f0 B T
    '''

    result = f0.clone()

    for i in range(uv.shape[0]):
        x = torch.where(uv[i])[-1]
        xp = torch.where(~uv[i])[-1]
        fp = f0[i][~uv[i]]

        interp_vals = torch_interp(x, xp, fp).detach()

        result[i][uv[i]] = interp_vals
    return result


def unit_text():
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print('  [UNIT_TEST] torchfcpe.torch_interp: matplotlib not found, skip plotting.')
        exit(1)

    # f0
    f0 = torch.tensor([1, 0, 3, 0, 0, 3, 4, 5, 0, 0]).float()
    uv = torch.tensor([0, 1, 0, 1, 1, 0, 0, 0, 1, 1]).bool()

    interp_f0 = batch_interp_with_replacement_detach(uv.unsqueeze(0), f0.unsqueeze(0)).squeeze(0)

    print(interp_f0)


if __name__ == '__main__':
    unit_text()
