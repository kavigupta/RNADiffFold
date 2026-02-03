"""Tests that vectorized creatmat matches creatmat_slow."""
import sys
import unittest
from pathlib import Path

import numpy as np

# Allow importing from prediction when run from repo root or RNADiffFold
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from prediction.prediction_utils import creatmat, creatmat_slow


def _random_onehot(shape, rng=None):
    """Random one-hot (..., 4) arrays: each row is A/U/C/G."""
    rng = rng or np.random.default_rng()
    n = int(np.prod(shape[:-1]))
    idx = rng.integers(0, 4, size=n)
    out = np.zeros((n, 4), dtype=np.float64)
    out[np.arange(n), idx] = 1
    return out.reshape(shape)


class TestCreatmat(unittest.TestCase):
    def test_creatmat_empty(self):
        data = np.zeros((0, 4))
        out = creatmat(data)
        out_slow = creatmat_slow(data)
        self.assertEqual(out.shape, (0, 0))
        self.assertEqual(out_slow.shape, (0, 0))

    def test_creatmat_single(self):
        for i in range(4):
            data = np.zeros((1, 4))
            data[0, i] = 1
            out = creatmat(data)
            out_slow = creatmat_slow(data)
            np.testing.assert_array_almost_equal(out, out_slow)
            self.assertEqual(out.shape, (1, 1))
            self.assertEqual(out[0, 0], 0)

    def test_creatmat_equiv_random(self):
        rng = np.random.default_rng(42)
        for L in (2, 5, 10, 20, 100):
            for _ in range(10):
                data = _random_onehot((L, 4), rng=rng)
                out = creatmat(data)
                out_slow = creatmat_slow(data)
                np.testing.assert_allclose(out, out_slow, rtol=1e-12, atol=1e-12)

    def test_creatmat_equiv_known_seq(self):
        data = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float64)
        out = creatmat(data)
        out_slow = creatmat_slow(data)
        np.testing.assert_allclose(out, out_slow, rtol=1e-12, atol=1e-12)
        self.assertGreater(out[0, 1], 0)
        self.assertGreater(out[1, 0], 0)


if __name__ == "__main__":
    unittest.main()
