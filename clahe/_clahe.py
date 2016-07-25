import numpy as np

from . import _clahe_impl


def clahe(img, kshape, clip_limit):
    kx, ky = kshape
    img = np.pad(img, ((kx, kx), (ky, ky)), "reflect")
    res = _clahe_impl.clahe(img, kx, ky, clip_limit)
    return res[kx:-kx, ky:-ky]
