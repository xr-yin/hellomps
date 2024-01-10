import numpy as np

import unittest
import sys
import os
from copy import deepcopy

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import *
from hellomps.networks.lptn import _load_right_bond_tensors

class TestLPTN(unittest.TestCase):

    def test_to_density_matrix(self):
        
        dm = self.psi.to_density_matrix()
        self.assertEqual(dm.shape, (np.prod(self.phy_dims),)*2)
        self.assertTrue(np.allclose(dm, dm.T.conj()))

    def test_orthonormalize(self):

        # $\rho = X X^\dagger$
        # when $X$ is orthonormalized, the density matrix has trace 1.
        self.psi.orthonormalize('right')
        self.assertAlmostEqual(np.trace(self.psi.to_density_matrix()), 1, 12)

    def test_probabilties(self):
        self.psi.orthonormalize('right')
        norms = self.psi.probabilities()
        self.assertAlmostEqual(np.sum(norms), 1., 12)

    def test_load_right_bond_tensors(self):
        psi = self.psi
        phi = deepcopy(psi)
        phi.orthonormalize('right')
        RBT = _load_right_bond_tensors(psi, phi)
        self.assertEqual(len(RBT), len(psi))
        self.assertEqual([_.shape for _ in RBT], 
                [(i,j) for i,j in zip(psi.bond_dims[1:],phi.bond_dims[1:])])

    def test_compress(self):
        self.psi.orthonormalize('right')
        phi, overlap = compress(self.psi, 1e-7, 4, 4, max_sweeps=3)
        self.assertAlmostEqual(overlap, 1)
        self.assertAlmostEqual(np.linalg.norm(self.psi.to_matrix()-phi.to_matrix()), 0.)

    def setUp(self) -> None:
        rng = np.random.default_rng()
        N = rng.integers(3,8)
        m_max = rng.integers(4,9)
        k_max = rng.integers(4,9)
        self.phy_dims = rng.integers(2, 5, size=N)
        self.psi = LPTN.gen_random_state(N, m_max, k_max, self.phy_dims)

if __name__ == '__main__':

    unittest.main()