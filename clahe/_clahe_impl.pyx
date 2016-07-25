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
    np.npy_uint32
    np.npy_int32


def clahe(dtype[:, :] img_orig,
          size_t kx,
          size_t ky,
          double clip_limit,
          bint multiplicative=False):
    cdef:
        dtype[::1] orig_vals = np.unique(img_orig)
        size_t[:, ::1] img = np.searchsorted(orig_vals, img_orig).view(np.uintp)
        size_t nx = img.shape[0], ny = img.shape[1], x0, y0, \
            x, y, hkx = kx // 2, hky = ky // 2, kxy = kx * ky, \
            val, viter, \
            nvals = orig_vals.shape[0], max_val_p1 = orig_vals[nvals - 1]
        double count_clip = clip_limit * kx * ky / nvals, clip_sum, clip_psum
        size_t[::1] hist = np.zeros(nvals, np.uintp)
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
            for viter in range(val, nvals):
                clip_sum += min(hist[viter], count_clip)
            out[x0 + hkx, y0 + hky] = (
                # Multiplicative redistribution.
                clip_psum / clip_sum
                if multiplicative else
                # Additive redistribution.
                (clip_psum + (kxy - clip_sum) * (orig_vals[val] + .5) / max_val_p1) / kxy)
            # Update histogram.
            for x in range(kx):
                hist[img[x0 + x, y0]] -= 1
                hist[img[x0 + x, y0 + ky]] += 1
    return np.asarray(out)
