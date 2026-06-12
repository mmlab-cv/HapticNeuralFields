import torch as t
import numpy as np


def spectrum(block: t.Tensor, mat: t.Tensor):
    spect = calc_dct(block, mat)
    return spect


def get_dct_mat(N):
    dct_mat = t.zeros((N, N))

    k = 1
    for n in range(1, N + 1):
        dct_mat[k - 1, n - 1] = np.sqrt(1 / N) * np.cos(
            (np.pi / (2 * N)) * (2 * n - 1) * (k - 1)
        )

    for k in range(2, N + 1):
        for n in range(1, N + 1):
            dct_mat[k - 1, n - 1] = np.sqrt(2 / N) * np.cos(
                (np.pi / (2 * N)) * (2 * n - 1) * (k - 1)
            )

    return dct_mat


def calc_dct(x, mat):
    x = t.unsqueeze(x, -1)
    X = t.matmul(mat, x)
    return X.squeeze(-1)
