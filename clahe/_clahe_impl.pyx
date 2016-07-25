#cython: binding=True
#cython: boundscheck=False
#cython: cdivision=True
#cython: initializedcheck=False
#cython: language_level=3
#cython: wraparound=False

# Adaptive Histogram Equalization and Its Variations, Pizer et al. (1987).

cimport numpy as np

import numpy as np


ctypedef fused dtype:
    np.npy_uint8
    np.npy_int8
    np.npy_uint16
    np.npy_int16


def clahe(np.ndarray[dtype, ndim=2] img_orig,
          size_t kx,
          size_t ky,
          double clip_limit,
          bint multiplicative=False):
    cdef:
        dtype[:, ::1] img = np.require(img_orig, requirements="C") - img_orig.min()
        size_t nx = img.shape[0], ny = img.shape[1], x0, y0, \
            x, y, hkx = kx // 2, hky = ky // 2, kxy = kx * ky, \
            val, viter, vmax = np.max(img), \
            count_clip = <size_t>(kx * ky * clip_limit), clip_sum
        size_t[::1] hist = np.zeros(vmax + 1, np.uintp)
        double clip_psum
        double[:, ::1] out = np.zeros_like(img, np.double)
    for x0 in range(nx - kx):
        hist[:] = 0
        for x in range(kx):
            for y in range(ky):
                hist[img[x0 + x, y]] += 1 # y0 = 0
        for y0 in range(ny - ky):
            # Limit contrast.
            val = img[x0 + hkx, y0 + hky]
            clip_sum = 0
            for viter in range(val):
                clip_sum += min(hist[viter], count_clip)
            clip_psum = clip_sum + min(hist[val] / 2., count_clip)
            for viter in range(val, vmax):
                clip_sum += min(hist[viter], count_clip)
            out[x0 + hkx, y0 + hky] = (
                # Multiplicative redistribution.
                clip_psum / clip_sum
                if multiplicative else
                # Additive redistribution.
                (clip_psum + (kxy - clip_sum) * (<double>val / vmax)) / kxy)
            # Update histogram.
            for x in range(kx):
                hist[img[x0 + x, y0]] -= 1
                hist[img[x0 + x, y0 + ky]] += 1
    return np.asarray(out)
