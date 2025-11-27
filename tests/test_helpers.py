"""Shared helper functions and test problems for IVP tests."""
import numpy as np
from scipy.sparse import coo_matrix, csc_matrix


def fun_zero(t, y):
    return np.zeros_like(y)


def fun_linear(t, y):
    return np.array([-y[0] - 5 * y[1], y[0] + y[1]])


def jac_linear():
    return np.array([[-1, -5], [1, 1]])


def sol_linear(t):
    return np.vstack((-5 * np.sin(2 * t),
                      2 * np.cos(2 * t) + np.sin(2 * t)))


def fun_rational(t, y):
    return np.array([y[1] / t,
                     y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))])


def fun_rational_vectorized(t, y):
    return np.vstack((y[1] / t,
                      y[1] * (y[0] + 2 * y[1] - 1) / (t * (y[0] - 1))))


def jac_rational(t, y):
    return np.array([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def jac_rational_sparse(t, y):
    return csc_matrix([
        [0, 1 / t],
        [-2 * y[1] ** 2 / (t * (y[0] - 1) ** 2),
         (y[0] + 4 * y[1] - 1) / (t * (y[0] - 1))]
    ])


def sol_rational(t):
    return np.asarray((t / (t + 10), 10 * t / (t + 10) ** 2))


def fun_medazko(t, y):
    n = y.shape[0] // 2
    k = 100
    c = 4

    phi = 2 if t <= 5 else 0
    y = np.hstack((phi, 0, y, y[-2]))

    d = 1 / n
    j = np.arange(n) + 1
    alpha = 2 * (j * d - 1) ** 3 / c ** 2
    beta = (j * d - 1) ** 4 / c ** 2

    j_2_p1 = 2 * j + 2
    j_2_m3 = 2 * j - 2
    j_2_m1 = 2 * j
    j_2 = 2 * j + 1

    f = np.empty(2 * n)
    f[::2] = (alpha * (y[j_2_p1] - y[j_2_m3]) / (2 * d) +
              beta * (y[j_2_m3] - 2 * y[j_2_m1] + y[j_2_p1]) / d ** 2 -
              k * y[j_2_m1] * y[j_2])
    f[1::2] = -k * y[j_2] * y[j_2_m1]

    return f


def medazko_sparsity(n):
    cols = []
    rows = []

    i = np.arange(n) * 2

    cols.append(i[1:])
    rows.append(i[1:] - 2)

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i + 1)

    cols.append(i[:-1])
    rows.append(i[:-1] + 2)

    i = np.arange(n) * 2 + 1

    cols.append(i)
    rows.append(i)

    cols.append(i)
    rows.append(i - 1)

    cols = np.hstack(cols)
    rows = np.hstack(rows)

    return coo_matrix((np.ones_like(cols), (cols, rows)))


def fun_event_dense_output_LSODA(t, y):
    return y * (t - 2)


def jac_event_dense_output_LSODA(t, y):
    return t - 2


def sol_event_dense_output_LSODA(t):
    return np.exp(t ** 2 / 2 - 2 * t + np.log(0.05) - 6)


def compute_error(y, y_true, rtol, atol):
    e = (y - y_true) / (atol + rtol * np.abs(y_true))
    return np.linalg.norm(e, axis=0) / np.sqrt(e.shape[0])
