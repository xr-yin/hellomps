import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import *
from hellomps.networks.mps import _load_right_bond_tensors

class TestMPS(unittest.TestCase):

    def test_orthonormalize(self):
        psi = self.psi
        psi.orthonormalize('left')
        for A in psi:
            A = np.transpose(A, (2,0,1))
            s = A.shape
            A = np.reshape(A, (s[0]*s[1], s[2]))
            self.assertTrue(np.allclose(A.conj().T @ A, np.eye(s[2])))
        psi.orthonormalize('right')
        for A in psi:
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))
        rng = np.random.default_rng()
        idx = rng.integers(len(psi))
        psi.orthonormalize('mixed', idx)
        for i in range(idx):
            A = np.transpose(psi[i], (2,0,1))
            s = A.shape
            A = np.reshape(A, (s[0]*s[1], s[2]))
            self.assertTrue(np.allclose(A.conj().T @ A, np.eye(s[2])))
        for i in range(idx+1,len(psi)):
            A = psi[i]
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))

    def test_inner(self):

        psi, phi = self.psi, self.phi
        psi.orthonormalize('right')
        phi.orthonormalize('left')
        res1 = inner(psi, phi)
        res2 = np.vdot(psi.as_array(), phi.as_array())
        self.assertAlmostEqual(res1, res2, 12)

    def test_load_right_bond_tensors(self):
        psi, phi = self.psi, self.phi
        RBT = _load_right_bond_tensors(psi, phi)
        self.assertEqual(len(RBT), len(psi)+1)
        self.assertEqual([_.shape for _ in RBT[:-1]], 
                [(i,j) for i,j in zip(psi.bond_dims[1:],phi.bond_dims[1:])])
        self.assertEqual(RBT[-1].shape, (1,1))
        self.assertTrue(np.allclose(inner(psi,phi), RBT[-1].ravel()))

    def test_compress(self):
        psi = self.psi
        psi.orthonormalize('right')
        phi, overlap = compress(psi, 1e-6, max(psi.bond_dims)-2, max_sweeps=2)
        self.assertAlmostEqual(inner(psi,phi), overlap)
        self.assertAlmostEqual(inner(psi,phi), 1)

    def setUp(self) -> None:
        rng = np.random.default_rng()
        N = rng.integers(5,10)
        m_max = rng.integers(11,19)
        phy_dims = rng.integers(2,5,size=N)
        self.psi = MPS.gen_random_state(N, m_max, phy_dims)
        self.phi = MPS.gen_random_state(N, m_max, phy_dims)

if __name__ == '__main__':
    unittest.main()