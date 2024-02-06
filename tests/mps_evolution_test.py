import numpy as np
from scipy.linalg import expm
from scipy.sparse.linalg import eigsh

import unittest
import sys
import os

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.networks.operations import mul
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.mps_evolution import tMPS, TEBD2

class TestMPSEvolution(unittest.TestCase):
        
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

    def test_make_uMPO(self):
        N = 5
        model = TransverseIsing(N, g=.5)
        psi = MPS.gen_polarized_spin_chain(N,'+z')
        lab = tMPS(psi, model)

        dt = 0.2
        lab.make_uMPO(1j*dt)
        u = lab.uMPO
        uh = u.hc()
        # commutation error, should decrease as we decrease the time step
        #self.assertAlmostEqual(np.linalg.norm(u.to_matrix() - expm(-1j*dt*model.H_full.toarray())), 0.)
        ida = mul(u, uh)
        idb = mul(uh, u)
        self.assertAlmostEqual(np.linalg.norm(ida.to_matrix()-np.eye(2**N)), 0.)
        self.assertAlmostEqual(np.linalg.norm(idb.to_matrix()-np.eye(2**N)), 0.)

    def test_make_unitaries(self):
        N = 5
        model = TransverseIsing(N, g=.5)
        psi = MPS.gen_polarized_spin_chain(N,'+z')
        lab = TEBD2(psi, model)

        dt = 0.4
        lab.make_unitaries(1j*dt)
        for key in ['half', 'full']:
            for a in lab.u_duo[key]:
                self.assertTrue(np.allclose(np.tensordot(a.conj(), a, axes=([0,1],[2,3])), np.eye(4).reshape(2,2,2,2)))

if __name__ == "__main__":

    N = 14
    model = TransverseIsing(N, g=1.5)

    # SciPy eigensolver for comparison
    w, v = eigsh(model.H_full, k=5, which='SA')
    E = w[0].item()

    unittest.main()