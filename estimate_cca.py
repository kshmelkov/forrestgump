#!/usr/bin/python

from fg_constants import *
import cross_validation as cv

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd


def load_regressor(name, num):
    lags = cv.REGRESSORS[name](num)
    return lags[:, :(lags.shape[1]/3)]


def trunc_svd(x, d):
    u, s, _ = svd(x, full_matrices=False)
    if d > s.shape[0]:
        d = s.shape[0]
        print 'reduced dims to %i' % d
    return u[:, :d]


def cca(x, y, d=10):
    u1 = trunc_svd(x, d)
    u2 = trunc_svd(y, d)
    return svd(u1.T.dot(u2), full_matrices=False, compute_uv=False)


if __name__ == '__main__':
    X = load_regressor('word2vec', 0)
    Y = load_regressor('audio2', 0)
    print cca(X, Y, d=10).sum()
