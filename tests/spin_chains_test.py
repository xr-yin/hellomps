import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.models.spin_chains import *
from hellomps.networks.mps import *

class TestSpinChains(unittest.TestCase):

    def test_TransverseIsing(self):
        N = np.random.randint(5,10)
        g = 2 * np.random.random()
        model = TransverseIsing(N, g)
        self.assertTrue(np.allclose(model.mpo.to_matrix(), model.H_full.toarray()))

        all_up = MPS.gen_polarized_spin_chain(N, polarization='+z')
        self.assertEqual(model.energy(all_up), -1*(N-1))

        all_right = MPS.gen_polarized_spin_chain(N, polarization='+x')
        self.assertAlmostEqual(model.energy(all_right), -g*N)

if __name__ == '__main__':
    unittest.main()