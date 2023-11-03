import numpy as np

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.mps_evolution import tMPS

class TestOperations(unittest.TestCase):
        
    def test_tMPS(self):

        N = 14
        model = TransverseIsing(N, g=1.5)
        psi = MPS.gen_polarized_spin_chain(N,'+z')
        print('energy at start', model.energy(psi))
        lab = tMPS(psi, model)
        lab.run(500, 0.01, tol=1e-10, m_max=30, compress_sweeps=1)
        print('energy at the end', model.energy(psi))

if __name__ == "__main__":
    unittest.main()