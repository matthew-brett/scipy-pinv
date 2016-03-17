import numpy as np
import numpy.linalg as npl
import scipy
import scipy.linalg as spl

import sympy as sy

sp16 = np.load('sp16.npz')
sp17 = np.load('sp17.npz')

assert np.all(sp16['X'] == sp17['X'])
assert np.all(sp16['y'] == sp17['y'])
X = sp16['X']
y = sp16['y']


def fairly_exact_pinv(a, precision=50):
    floats = np.zeros(a.shape, dtype=object)
    for index, val in np.ndenumerate(a):
        floats[index] = sy.Float(val, precision)
    sym_a = sy.Matrix(floats)
    return sy.matrix2numpy(sym_a.pinv(), dtype=float)


def max_diff(a, b):
    return np.max(np.abs(a - b))


def sum_abs_diff(a, b):
    return np.sum(np.abs(a - b))


def print_scalar(s):
    print('{:0.16f}'.format(s))


def sse(X, piX, y):
    beta = piX.dot(y)
    errors = y - X.dot(beta)
    return np.sum(errors ** 2)


s_piX = fairly_exact_pinv(X)
print_scalar(max_diff(s_piX, sp16['piX']))
print_scalar(max_diff(s_piX, sp17['piX']))
print_scalar(sum_abs_diff(s_piX, sp16['piX']))
print_scalar(sum_abs_diff(s_piX, sp17['piX']))
print('Sympy high-precision pinv SSE')
print_scalar(sse(X, s_piX, y))
print('Local numpy {} linalg pinv'.format(np.__version__))
print_scalar(sse(X, npl.pinv(X), y))
print('Local scipy {} linalg pinv'.format(scipy.__version__))
print_scalar(sse(X, spl.pinv(X), y))
print('Scipy 0.16 linalg pinv, stored')
print_scalar(sse(X, sp16['piX'], y))
print('Scipy 0.17 linalg pinv, stored')
print_scalar(sse(X, sp17['piX'], y))
