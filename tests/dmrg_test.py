import numpy as np
from scipy.sparse.linalg import eigsh

import unittest
import sys
import os
import logging
logging.basicConfig(level=logging.INFO)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.dmrg import DMRG

class TestOperations(unittest.TestCase):

    def test_DMRG(self):

        psi = MPS.gen_polarized_spin_chain(N, '+z')

        lab = DMRG(psi, model.mpo)
        lab.run_two_sites(10, 1e-7, 30)
        
        self.assertAlmostEqual(model.energy(psi).item(), E)

        bond_dims = psi.bond_dims
        logging.info(f'final bond dimensions: {bond_dims}')

        rng = np.random.default_rng()
        As = []
        for i in range(N):
            As.append(rng.random((bond_dims[i],bond_dims[i+1],2)))
        psi = MPS(As)
        lab = DMRG(psi, model.mpo)
        lab.run_one_stie(10)
        self.assertAlmostEqual(model.energy(psi).item(), E)
        self.assertTrue(psi.bond_dims, bond_dims)
        

if __name__ == "__main__":

    N = 14
    model = TransverseIsing(N, g=1.5)

    # SciPy eigensolver for comparison
    w, v = eigsh(model.H_full, k=5, which='SA')
    E = w[0].item()

    unittest.main()