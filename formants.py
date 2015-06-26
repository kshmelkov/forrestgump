#!/usr/bin/env python
import numpy
import math
from scipy.signal import lfilter, hamming
from scikits.talkbox import lpc

"""
Estimate formants using LPC.
"""

def get_formants(x, Fs, limit=3):
    # Get Hamming window.
    N = len(x)
    w = numpy.hamming(N)

    # Apply window and high pass filter.
    x1 = x * w
    x1 = lfilter([1], [1., 0.63], x1)

    ncoeff = 2 + Fs / 1000
    # Get LPC.
    A, e, k = lpc(x1, ncoeff)

    # Get roots.
    rts = numpy.roots(A)
    rts = [r for r in rts if numpy.imag(r) >= 0]

    # Get angles.
    angz = numpy.arctan2(numpy.imag(rts), numpy.real(rts))

    # Get frequencies.
    frqs = sorted(angz * (Fs / (2 * math.pi)))

    return frqs[:limit]

