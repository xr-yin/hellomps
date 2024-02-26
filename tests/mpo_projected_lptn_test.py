import numpy as np
from pylops.utils import dottest

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mpo_projected_lptn import *
from hellomps.networks.lptn import LPTN
from hellomps.networks.mpo import MPO

class TestRightBondTensors(unittest.TestCase):

    def test_load(self):

        rng = np.random.default_rng()
        N = rng.integers(5,10)
        m_max = rng.integers(4,9)
        phy_dims = rng.integers(2, 5, size=N)
        O = MPO.gen_random_mpo(N, m_max, phy_dims)
        krauss_dims = rng.integers(3, 6, size=N)

        psi = self.urand(N, phy_dims, krauss_dims)
        phi = self.urand(N, phy_dims, krauss_dims)
        Rs = RightBondTensors(N)
        Rs.load(phi, psi, O)
        self.assertEqual([_.shape for _ in Rs],
                         [(i,j,k) for i,j,k in zip(phi.bond_dims[1:], O.bond_dims[1:], psi.bond_dims[1:])])
        
    def urand(self, N, phy_dims, krauss_dims):
        rng = np.random.default_rng()
        bond_dims = rng.integers(5,8,size=N+1)
        bond_dims[0] = bond_dims[-1] = 1
        As = []
        for i in range(N):
            As.append(rng.normal(size=(bond_dims[i],bond_dims[i+1],phy_dims[i],krauss_dims[i])))
        return LPTN(As)


if __name__ == '__main__':
    unittest.main()