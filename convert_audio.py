#!/usr/bin/python

import numpy as np
from nsgt import CQ_NSGT

from os import path, makedirs
from itertools import imap
from fg_constants import *
import matplotlib.pyplot as plt
from cross_validation import load_audio

class interpolate:
    def __init__(self, cqt, Ls):
        from scipy.interpolate import interp1d
        self.intp = [interp1d(np.linspace(0, Ls, len(r)), r) for r in cqt]

    def __call__(self, x):
        try:
            len(x)
        except:
            return np.array([i(x) for i in self.intp])
        else:
            return np.array([[i(xi) for i in self.intp] for xi in x])


def constant_q(num, bins):
    rate, mono = load_audio(num)
    Ls = mono.shape[0]
    cq = CQ_NSGT(FREQ_MIN, FREQ_MAX, bins, rate, Ls)
    c = cq.forward(mono)
    # interpolate CQT to get a grid
    x = np.linspace(0, Ls, Ls/rate*10)
    grid = interpolate(imap(np.abs, c[2:-1]), Ls)(x)
    sg = np.log(np.fliplr(grid))
    return sg


def compute_all_CQ(bins):
    print 'Converting all files with %i bins' % bins
    current_dir = path.join(CQ_DIR, str(bins))

    if not path.exists(current_dir):
        makedirs(current_dir)

    for num in range(TOTAL_SEGMENTS):
        print 'Computing Constant-Q for %i segment' % num
        cq = constant_q(num, bins)
        np.save(path.join(current_dir, 'run00%i.npy' % num), cq)


if __name__ == '__main__':
    compute_all_CQ(8)
