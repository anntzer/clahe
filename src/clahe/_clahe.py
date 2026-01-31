import warnings

import numpy as np

from . import _clahe_impl


def clahe(img, win_shape, clip_limit, *, _fast=True):
    """
    Contrast-limited adaptive histogram equalization.

    Parameters
    ----------
    img : int array
        Input image.
    win_shape : int sequence or int
        Shape of the contextual region.  Its length must match the
        dimensionality of *img*, or it can be a single int which is used as
        size over each axis.
    clip_limit : float
        Clipping limit; setting this to ``+inf`` disables clipping.
    _fast : bool, default: True
        Whether to use the fast (Cython) implementation, which is limited to
        2 or 3D inputs, or the slow (Python) implementation, which can handle
        any dimensionality.
    """
    # Bring the largest dimension to the front to optimize the in-loop
    # histogram updates.
    win_shape = np.broadcast_to(win_shape, img.ndim)
    largest_dim = np.argmax(win_shape)
    img = np.swapaxes(img, 0, largest_dim)
    if img.dtype.char in "fd":
        warnings.warn(
            "clahe with floating point input is very slow; consider "
            "quantizing the data to a small bitwidth integer dtype instead")
    win_shape = list(win_shape)
    win_shape[0], win_shape[largest_dim] = win_shape[largest_dim], win_shape[0]
    img = np.pad(
        img, [((sz - 1) // 2, sz // 2) for sz in win_shape], "reflect")
    if _fast:
        if img.ndim == 2:
            res = _clahe_impl.clahe(
                img[..., None], *win_shape, 1, clip_limit)[..., 0]
        elif img.ndim == 3:
            res = _clahe_impl.clahe(img, *win_shape, clip_limit)
        else:
            raise TypeError("Wrong dimensionality")
    else:
        res = _clahe_nd(img, win_shape, clip_limit)
    return np.swapaxes(
        res[tuple(np.s_[(sz - 1) // 2 : -(sz // 2) or None]
                  for sz in win_shape)],
        0, largest_dim)


def _clahe_nd(img, win_shape, clip_limit, multiplicative=False):
    orig_vals = np.unique(img)
    img_ord = np.searchsorted(orig_vals, img)
    nvals = len(orig_vals)
    bincount = lambda t: np.bincount(t.flat, minlength=nvals)
    buf = np.zeros(nvals + 1)
    out = np.zeros_like(img, float)

    win_shape = np.asarray(win_shape)
    win_size = np.prod(win_shape)

    count_clip = clip_limit * win_size / nvals

    # Update on the *first* dimension in the inner loop in order to get a
    # contiguous array in the 2D case.
    for nndi in np.ndindex(*np.array(img.shape - win_shape)[1:] + 1):
        img_slice = img_ord[tuple(
            [slice(None)]
            + [slice(i, i + w) for i, w in zip(nndi, win_shape[1:])])]
        hist = bincount(img_slice[:win_shape[0] - 1])
        for idx in range(img.shape[0] - win_shape[0] + 1):
            hist += bincount(img_slice[idx + win_shape[0] - 1])
            ndi = tuple(((idx,) + nndi) + (win_shape - 1) // 2)
            # Limit contrast.
            np.minimum(hist, count_clip, buf[1:])
            val = img_ord[ndi]
            np.cumsum(buf[1:], out=buf[1:])
            clip_sum = buf[-1]
            clip_psum = (buf[val] + buf[val + 1]) / 2
            out[ndi] = (
                clip_psum / clip_sum
                if multiplicative else
                (clip_psum
                 + ((win_size - clip_sum)
                    * (orig_vals[val] + .5) / (orig_vals[-1] + 1)))
                / win_size)
            hist -= bincount(img_slice[idx])

    return out
