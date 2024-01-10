import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mpo import *

class TestMPO(unittest.TestCase):

    def test_orthonormalize(self):
        W = self.randomMPO
        W.orthonormalize('left')
        for A in W:
            A = np.swapaxes(A, 0, 1)
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.conj().T, np.eye(s[0])))
        W.orthonormalize('right')
        for A in W:
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))
        rng = np.random.default_rng()
        idx = rng.integers(len(W))
        W.orthonormalize('mixed', idx)
        for i in range(idx):
            A = np.swapaxes(W[i], 0, 1)
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.conj().T, np.eye(s[0])))
        for i in range(idx+1,len(W)):
            A = W[i]
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))

    def test_to_matrix(self):
        W = self.randomMPO
        self.assertEqual(W.to_matrix().shape, (np.prod(W.physical_dims),)*2)

    def test_hc(self):
        W = self.randomMPO
        self.assertTrue(np.allclose(W.to_matrix().T.conj(), W.hc().to_matrix()))

    def setUp(self) -> None:
        rng = np.random.default_rng()
        N = rng.integers(5,8)
        m_max = rng.integers(3,6)
        phy_dims = rng.integers(2, 5, size=N)
        self.randomMPO = MPO.gen_random_mpo(N, m_max, phy_dims)

if __name__ == '__main__':
    unittest.main()