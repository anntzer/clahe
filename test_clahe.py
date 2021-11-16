from unittest import TestCase

import clahe
import numpy as np
from numpy.testing import assert_array_max_ulp


class TestClahe(TestCase):
    def test_consistency(self):
        for win_size, img_size in [
                ((2, 2), (4, 4)),
                ((3, 3), (4, 4)),
                (4, (4, 4)),
                ((1, 3, 3), (3, 3, 3)),
                ((2, 3, 3), (3, 3, 3)),
                (3, (4, 4, 4)),
        ]:
            for dtype in [
                    np.uint8, np.uint16, np.uint32, np.uint64,
                    np.int8, np.int16, np.int32, np.int64,
            ]:
                np.random.seed(0)
                info = np.iinfo(dtype)
                img = np.random.randint(
                    info.min, info.max, img_size, dtype=dtype)
                clip_limit = .5
                fast = clahe.clahe(img, win_size, clip_limit)
                slow = clahe.clahe(img, win_size, clip_limit, _fast=False)
                assert fast.shape == slow.shape == img.shape
                # Not clear why it's not exactly equal.
                assert_array_max_ulp(fast, slow, 14)
