import clahe
import numpy as np
from numpy.testing import assert_array_max_ulp
import pytest


@pytest.mark.parametrize("win_size, img_size",
                         [((2, 2), (4, 4)),
                          ((3, 3), (4, 4)),
                          ((4, 4), (4, 4)),
                          ((1, 3, 3), (3, 3, 3)),
                          ((2, 3, 3), (3, 3, 3)),
                          ((3, 3, 3), (4, 4, 4))])
def test_consistency(win_size, img_size):
    np.random.seed(0)
    img = np.random.randint(0, 256, img_size, np.uint8)
    clip_limit = .5
    fast = clahe.clahe(img, win_size, clip_limit)
    slow = clahe.clahe(img, win_size, clip_limit, _fast=False)
    assert fast.shape == slow.shape == img.shape
    # Not clear why it's not exactly equal.
    assert_array_max_ulp(fast, slow)
