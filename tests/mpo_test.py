import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mpo import *

class TestMPO(unittest.TestCase):

    def test_orthonormalize(self):

        randomMPO.orthonormalize('left')
        for A in randomMPO.As:
            A = np.swapaxes(A, 0, 1)
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.conj().T, np.eye(s[0])))
        randomMPO.orthonormalize('right')
        for A in randomMPO.As:
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))
        rng = np.random.default_rng()
        idx = rng.integers(1,N)
        randomMPO.orthonormalize('mixed', idx)
        for i in range(idx):
            A = np.swapaxes(randomMPO.As[i], 0, 1)
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.conj().T, np.eye(s[0])))
        for i in range(idx+1,N):
            A = randomMPO.As[i]
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]*s[3]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))

    def test_to_matrix(self):

        self.assertEqual(randomMPO.to_matrix().shape, (np.prod(phy_dims),)*2)

    def test_hc(self):

        self.assertTrue(np.allclose(randomMPO.to_matrix().T.conj(), randomMPO.hc().to_matrix()))


if __name__ == '__main__':

    rng = np.random.default_rng()
    N = rng.integers(5,8)
    m_max = rng.integers(3,6)
    phy_dims = rng.integers(2, 5, size=N)
    randomMPO = MPO.gen_random_mpo(N, m_max, phy_dims)

    unittest.main()