from unittest import TestCase

import clahe
import numpy as np
from numpy.testing import assert_array_max_ulp


class TestClahe(TestCase):
    def test_consistency(self):
        for win_size, img_size in [
                ((2, 2), (4, 4)),
                ((3, 3), (4, 4)),
                ((4, 4), (4, 4)),
                ((1, 3, 3), (3, 3, 3)),
                ((2, 3, 3), (3, 3, 3)),
                ((3, 3, 3), (4, 4, 4)),
        ]:
            np.random.seed(0)
            img = np.random.randint(0, 256, img_size, np.uint8)
            clip_limit = .5
            fast = clahe.clahe(img, win_size, clip_limit)
            slow = clahe.clahe(img, win_size, clip_limit, _fast=False)
            assert fast.shape == slow.shape == img.shape
            # Not clear why it's not exactly equal.
            assert_array_max_ulp(fast, slow)
