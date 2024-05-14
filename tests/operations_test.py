import numpy as np

import unittest
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.operations import *
from hellomps.networks.mpo import MPO
from hellomps.networks.mps import MPS, inner
from hellomps.networks.lptn import LPTN

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
            b, c = split(a, mode, 0.)
            s = (b.shape[0], c.shape[1], 
                 b.shape[2], c.shape[2])
            self.assertEqual(s, a.shape)
            self.assertEqual(b.shape[1], c.shape[0])

    def test_split_mpo(self):
        
        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=6)
        a = rng.random(a_size)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 0.)
            s = (b.shape[0], c.shape[1], 
                 b.shape[2], c.shape[2], 
                 b.shape[3], c.shape[3])
            self.assertEqual(s, a.shape)
            self.assertEqual(b.shape[1], c.shape[0])

    def test_splitandmerge_mpo(self):
        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=6)
        a = rng.random(a_size)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 0., renormalize=False)
            new_a = merge(b, c)
            self.assertTrue(np.allclose(new_a, a))

    def test_splitandmerge_mps(self):
        rng = np.random.default_rng()
        a_size = rng.integers(1,8,size=4)
        a = rng.random(a_size)
        for mode in ['left', 'right', 'sqrt']:
            b, c = split(a, mode, 0., renormalize=False)
            new_a = merge(b, c)
            self.assertTrue(np.allclose(new_a, a))

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
    
    def test_apply_mpo2mps(self):

        O, psi = self.O, self.psi
        logging.info(f' O  dims: {O.bond_dims}')
        logging.info(f'psi dims: {psi.bond_dims}')
        Opsi = mul(O, psi)
        logging.info(f'Opsi dims: {Opsi.bond_dims}')
        norm = apply_mpo(O, psi, tol=1e-6, m_max=max(Opsi.bond_dims)-3, max_sweeps=2)
        logging.info(f'res dims: {psi.bond_dims}')
        self.assertAlmostEqual(norm**2, inner(Opsi, Opsi))  # why square the norm

        self.assertAlmostEqual(inner(Opsi, psi), np.sqrt(inner(Opsi,Opsi)*inner(psi,psi)), 9)

    def test_apply_mpo2mpo(self):

        O, phi = self.O, self.phi

        multibonds = np.array(phi.bond_dims)*np.array(O.bond_dims)
        logging.info(f'multiplied bond dimensions and mean: {multibonds}')
        logging.info(f'krauss dimensions: {phi.krauss_dims}')

        # regular matrix products for comparison
        ref = O.to_matrix() @ phi.to_matrix() # |v> = O|psi>

        overlap = apply_mpo(O, phi, tol=1e-6, m_max=max(multibonds), max_sweeps=2)
        logging.info(f'optimized bond dimensions: {phi.bond_dims}')
        logging.info(f'optimized krauss dimensions: {phi.krauss_dims}')

        self.assertAlmostEqual(overlap, np.linalg.norm(ref))    # <v|v> =? <v|v>**0.5

        self.assertAlmostEqual(np.linalg.norm(ref - phi.to_matrix())**2,
                               np.linalg.norm(phi.to_matrix())**2 + np.linalg.norm(ref)**2 - 2*np.real(overlap))
        
        r = ref / np.linalg.norm(ref)
        self.assertAlmostEqual(np.linalg.norm(r - phi.to_matrix()), 0.)
        self.assertAlmostEqual(np.linalg.norm(r@r.T.conj() - phi.to_density_matrix()), 0.)

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
        m_max = rng.integers(9,19)
        phy_dims = rng.integers(2, 5, size=N)

        self.O = MPO.gen_random_mpo(N, m_max, phy_dims)
        self.O.orthonormalize('right')

        self.psi = MPS.gen_random_state(N, m_max, phy_dims)
        self.psi.orthonormalize('right')

        self.phi = LPTN.gen_random_state(N, m_max, 5, phy_dims)
        self.phi.orthonormalize('right')

if __name__ == '__main__':
    unittest.main()