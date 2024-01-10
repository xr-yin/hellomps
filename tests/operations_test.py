import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.operations import *
from hellomps.networks.mpo import MPO
from hellomps.networks.mps import MPS, inner

class TestMergeSplit(unittest.TestCase):

    def test_merge_mps(self):

        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=3)
        b_size = rng.integers(1,8,size=3)
        b_size[0] = a_size[1]
        a = rng.random(a_size)
        b = rng.random(b_size)
        c = merge(a,b)
        self.assertEqual(c.shape, (a_size[0],b_size[1],a_size[2],b_size[2]))

        b_size[0] = a_size[1] + 1
        b = rng.random(b_size)
        with self.assertRaises(ValueError):
            merge(a, b)

    def test_merge_mpo(self):

        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=4)
        b_size = rng.integers(1,8,size=4)
        b_size[0] = a_size[1]
        a = rng.random(a_size)
        b = rng.random(b_size)
        c = merge(a,b)
        self.assertEqual(c.shape, 
                    (a_size[0],b_size[1],a_size[2],b_size[2],a_size[3],b_size[3]))

        b_size[0] = a_size[1] + 1
        b = rng.random(b_size)
        with self.assertRaises(ValueError):
            merge(a, b)

    def test_split_mps(self):
        
        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=4)
        a = rng.random(a_size)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 1e-6)
            self.assertEqual(b.shape[0], a_size[0])
            self.assertEqual(b.shape[2], a_size[2])
            self.assertEqual(c.shape[1], a_size[1])
            self.assertEqual(c.shape[2], a_size[3])
            self.assertEqual(b.shape[1], c.shape[0])

    def test_split_mpo(self):
        
        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=6)
        a = rng.random(a_size)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 1e-6)
            self.assertEqual(b.shape[0], a_size[0])
            self.assertEqual(b.shape[2], a_size[2])
            self.assertEqual(c.shape[1], a_size[1])
            self.assertEqual(c.shape[2], a_size[3])
            self.assertEqual(b.shape[1], c.shape[0])
            self.assertEqual(b.shape[3], a_size[4])
            self.assertEqual(c.shape[3], a_size[5])

class TestMultiplication(unittest.TestCase):

    def test_mul(self):

        A = self.O
        N = len(A)
        m_max = max(A.bond_dims)
        phy_dims = A.physical_dims
        
        # MPO x MPS
        amps = MPS.gen_random_state(N, m_max, phy_dims)
        C = mul(A, amps)
        self.assertTrue(np.allclose(C.as_array(), A.to_matrix() @ amps.as_array()))

        # MPO x MPO
        ampo = MPO.gen_random_mpo(N, m_max, phy_dims)
        C = mul(A, ampo)
        self.assertTrue(np.allclose(C.to_matrix(), A.to_matrix() @ ampo.to_matrix()))

    def test_apply_mpo(self):

        O, psi = self.O, self.psi
        Opsi = mul(O, psi)
        phi = apply_mpo(O, psi, tol=1e-6, m_max=max(psi.bond_dims), max_sweeps=2)

        self.assertAlmostEqual(inner(Opsi, phi), np.sqrt(inner(Opsi,Opsi)*inner(phi,phi)), 9)

    def test_zip_up(self):

        O, psi = self.O, self.psi
        # test start from 'left'
        Opsi = mul(O, psi)
        zip_up(O, psi, 1e-10, start='left')
        self.assertAlmostEqual(inner(Opsi, psi), np.sqrt(inner(Opsi,Opsi)*inner(psi,psi)), 9)
        # now psi is in left canonical form
        for A in psi.As:
            A = np.transpose(A, (2,0,1))
            s = A.shape
            A = np.reshape(A, (s[0]*s[1], s[2]))
            self.assertTrue(np.allclose(A.conj().T @ A, np.eye(s[2])))
        # test start from 'right'
        Opsi = mul(O, psi)
        zip_up(O, psi, 1e-10, start='right')
        self.assertAlmostEqual(inner(Opsi, psi), np.sqrt(inner(Opsi,Opsi)*inner(psi,psi)), 9)
        # now psi is in right canonical form
        for A in psi.As:
            s = A.shape
            A = np.reshape(A, (s[0], s[1]*s[2]))
            self.assertTrue(np.allclose(A @ A.T.conj(), np.eye(s[0])))

    def setUp(self) -> None:

        rng = np.random.default_rng()
        N = rng.integers(5,9)
        m_max = rng.integers(4,7)
        phy_dims = rng.integers(2, 5, size=N)
        self.O = MPO.gen_random_mpo(N, m_max, phy_dims)
        self.O.orthonormalize('right')
        self.psi = MPS.gen_random_state(N, m_max, phy_dims)
        self.psi.orthonormalize('right')

if __name__ == '__main__':
    unittest.main()