import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import *

class TestLPTN(unittest.TestCase):

    def test_to_density_matrix(self):
        
        dm = randomLPTN.to_density_matrix()
        self.assertEqual(dm.shape, (np.prod(phy_dims),)*2)
        self.assertTrue(np.allclose(dm, dm.T.conj()))

    def test_orthonormalize(self):

        # $\rho = X X^\dagger$
        # when $X$ is orthonormalized, the density matrix has trace 1.
        randomLPTN.orthonormalize('right')
        self.assertAlmostEqual(np.trace(randomLPTN.to_density_matrix()), 1, 12)

    def test_probabilties(self):
        norms = randomLPTN.probabilities()
        self.assertAlmostEqual(np.sum(norms), 1., 12)


if __name__ == '__main__':

    rng = np.random.default_rng()
    N = rng.integers(3,8)
    m_max = rng.integers(4,9)
    k_max = rng.integers(4,9)
    phy_dims = rng.integers(2, 5, size=N)
    randomLPTN = LPTN.gen_random_state(N, m_max, k_max, phy_dims)

    unittest.main()