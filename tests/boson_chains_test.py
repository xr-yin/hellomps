import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.models.boson_chains import *
from hellomps.networks.mps import *

class TestBosonChains(unittest.TestCase):

    def test_BoseHubburd(self):

        rng = np.random.default_rng()
        N = rng.integers(3,7)
        d = rng.integers(2,5)
        t, U, mu = rng.uniform(size=3)
        model = BoseHubburd(N, d, t, U, mu)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))

    def test_DDBoseHubbard(self):

        rng = np.random.default_rng()
        N = rng.integers(3,7)
        d = rng.integers(2,5)
        t, U, mu, F, gamma = rng.uniform(size=5)
        model = DDBoseHubburd(N, d, t, U, mu, F, gamma)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full().toarray()))

if __name__ == '__main__':
    unittest.main()