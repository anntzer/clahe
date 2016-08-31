import numpy as np

from . import _clahe_impl

from utils import timed
@timed
def clahe(img, kshape, clip_limit):
    """Contrast-limited adaptive histogram equalization.
    """
    kx, ky = kshape
    img = np.pad(img, ((kx, kx), (ky, ky)), "reflect")
    res = _clahe_impl.clahe(img, kx, ky, clip_limit)
    # res = _clahe_nd(img, (kx, ky), clip_limit)
    return res[kx:-kx, ky:-ky]


def _clahe_nd(img, win_shape, clip_limit, multiplicative=False):
    orig_vals = np.unique(img)
    img_ord = np.searchsorted(orig_vals, img)
    nvals = len(orig_vals)
    hist = np.empty(nvals, int)
    out = np.zeros_like(img, float)

    win_shape = np.asarray(win_shape)
    win_size = np.product(win_shape)

    count_clip = clip_limit * win_size / nvals

    for nndi in np.ndindex(*(img.shape - win_shape)[:-1]):
        img_slice = (
            img_ord[tuple([slice(i, i + w)
                           for i, w in zip(nndi, win_shape[:-1])])])
        hist[:] = 0
        np.add.at(hist, img_slice[..., :win_shape[-1]], 1)
        for lasti in range(img.shape[-1] - win_shape[-1]):
            ndi = tuple((nndi + (lasti,)) + win_shape // 2)
            # Limit contrast.
            clip_hist = np.minimum(hist, count_clip)
            val = img_ord[ndi]
            clip_csum = np.cumsum(clip_hist)
            clip_sum = clip_csum[-1]
            clip_psum = clip_csum[val] - clip_hist[val] / 2
            out[ndi] = (
                clip_psum / clip_sum
                if multiplicative else
                (clip_psum
                 + (win_size - clip_sum) * (orig_vals[val] + .5) / orig_vals[-1])
                / win_size)
            np.add.at(hist, img_slice[..., lasti], -1)
            np.add.at(hist, img_slice[..., lasti + win_shape[-1]], 1)

    return out
