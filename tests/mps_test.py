import numpy as np

import unittest
import sys
sys.path.append('/home/stud/ge47jac/hellomps')

from hellomps.networks.mps import *

class TestMPS(unittest.TestCase):

    def test_orthonormalize(self):
        rng = np.random.default_rng()
        N = rng.integers(5,10)
        m_max = rng.integers(3,7)
        psi = MPS.gen_random_state(N, m_max, [3]*N)
        psi.orthonormalize('left')
        for A in psi.As:
            A = np.transpose(A, (2,0,1))
            s = A.shape
            A = np.reshape(A, (s[0]*s[1], s[2]))
            self.assertTrue(np.allclose(A.conj().T @ A, np.eye(s[2])))
        psi.orthonormalize('right')
        for A in psi.As:
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))
        rng = np.random.default_rng()
        idx = rng.integers(1,N)
        psi.orthonormalize('mixed', idx)
        for i in range(idx):
            A = np.transpose(psi.As[i], (2,0,1))
            s = A.shape
            A = np.reshape(A, (s[0]*s[1], s[2]))
            self.assertTrue(np.allclose(A.conj().T @ A, np.eye(s[2])))
        for i in range(idx+1,N):
            A = psi.As[i]
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))

    def test_inner(self):

        rng = np.random.default_rng()
        N = rng.integers(5,10)
        m_max = rng.integers(3,7)
        phy_dims = rng.integers(2,5,size=N)
        psi = MPS.gen_random_state(N, m_max, phy_dims)
        phi = MPS.gen_random_state(N, m_max, phy_dims)
        res1 = inner(psi, phi)
        res2 = np.vdot(psi.as_array(), phi.as_array())
        self.assertAlmostEqual(res1, res2, 12)

    def test_load_right_bond_tensors(self):
        N = 10
        m_max = 10
        psi = MPS.gen_random_state(N, m_max, [3]*N)
        phi = MPS.gen_random_state(N, m_max, [3]*N)
        RBT = load_right_bond_tensors(psi, phi)
        self.assertEqual(len(RBT), N+1)
        self.assertEqual([_.shape for _ in RBT[:-1]], 
                [(i,j) for i,j in zip(psi.bond_dims[1:],phi.bond_dims[1:])])
        self.assertEqual(RBT[-1].shape, (1,1))
        self.assertTrue(np.allclose(inner(psi,phi), RBT[-1].ravel()))

    def test_compress(self):
        psi = MPS.gen_random_state(10, 10, [2]*10)
        pass


if __name__ == '__main__':
    unittest.main()