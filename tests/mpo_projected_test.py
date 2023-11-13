import numpy as np
from pylops.utils import dottest

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mpo_projected import *
from hellomps.networks.operations import mul
from hellomps.networks.mps import MPS, inner
from hellomps.networks.mpo import MPO

class TestRightBondTensors(unittest.TestCase):

    def test_load(self):

        rng = np.random.default_rng()
        N = rng.integers(5,10)
        m_max = rng.integers(4,9)
        phy_dims = rng.integers(2, 5, size=N)
        O = MPO.gen_random_mpo(N, m_max, phy_dims)
        psi = MPS.gen_random_state(N, m_max, phy_dims)
        phi = MPS.gen_random_state(N, m_max, phy_dims)

        Rs = RightBondTensors(N)
        Rs.load(phi, psi, O)
        self.assertEqual([_.shape for _ in Rs],
                         [(i,j,k) for i,j,k in zip(phi.bond_dims[1:], O.bond_dims[1:], psi.bond_dims[1:])])


if __name__ == '__main__':
    unittest.main()