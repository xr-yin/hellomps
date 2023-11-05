import numpy as np
from scipy.sparse.linalg import eigsh

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.mps_evolution import tMPS, TEBD2

class TestOperations(unittest.TestCase):
        
    def test_TEBD2(self):

        psi = MPS.gen_polarized_spin_chain(N, '+z')

        # example TEBD2 imaginary time evolution
        lab = TEBD2(psi, model)
        lab.run(500, 0.01, tol=1e-10, m_max=30)
        
        self.assertAlmostEqual(model.energy(psi).item(), E, 5)

    
    def test_tMPS_zipup(self):

        psi = MPS.gen_polarized_spin_chain(N,'+z')
        
        # example tMPS imaginary time evolution
        lab = tMPS(psi, model)
        lab.run(500, 0.01, tol=1e-10)
        print("The final bond dimension in the zip-up case is", max(psi.bond_dims))

        self.assertAlmostEqual(model.energy(psi).item(), E, 5)

    def test_tMPS_variational(self):

        psi = MPS.gen_polarized_spin_chain(N,'+z')
        
        # example tMPS imaginary time evolution
        lab = tMPS(psi, model)
        lab.run(500, 0.01, tol=1e-10, m_max=30, compress_sweeps=1)

        self.assertAlmostEqual(model.energy(psi).item(), E, 5)

if __name__ == "__main__":

    N = 14
    model = TransverseIsing(N, g=1.5)

    # SciPy eigensolver for comparison
    w, v = eigsh(model.H_full, k=5, which='SA')
    E = w[0].item()

    unittest.main()