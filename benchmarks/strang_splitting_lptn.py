"""This module demonstrates the errors in our LPTN simulations converge with quadratic of time step O(dt^2) in 
the presence of Strang splitting.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

import os
import sys
import time
import logging
from copy import deepcopy
logging.basicConfig(level=logging.WARNING)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import LPTN
from hellomps.networks.mps import MPS
from hellomps.models.scqubits import QubitCavity
from hellomps.models.spin_chains import TransverseIsing, SpinChain, Heisenberg
from hellomps.algorithms.lindblad_master import LindbladOneSite, contract_disspative_layer

def XXZ(N: int, tmax: float, dt_list: list, ax: plt.Axes):
    """we use the Heisenberg XXZ model to test if our simulation method exhibit quadratic errors

    Parameters
    ----------
    N : int
        system size
    tmax : float
        total time for simulation
    dt_list : list
        list of time steps
    ax : matplotlib.pyplot.Axes object
        plot
    
    Remarks
    ----------
    The full Liouvillian matrix has a size of n^2 x n^2, which requires a lot of computer memory if 
    system size is large, which is why we need tensor network methods for simulation.
    The quadratic error scaling can be violated if the bond dimensions are chosen too small due to the
    accumulated dicard of weights. Therefore, it is worthwhile to pick the parameters (bond dimensions, 
    time step) to balance the trotter errors and the discarded weights.
    """

    X = LPTN.gen_polarized_spin_chain(N, '+x')

    err_t = np.zeros(len(dt_list))
    time_t = np.zeros(len(dt_list))
    model = Heisenberg(N, (1., 1., 1.5), 0., 1.)

    Xt_ref = expm_multiply(tmax * model.Liouvillian, X.to_density_matrix().ravel())

    for n, dt in enumerate(dt_list):
        psi = deepcopy(X)
        lab = LindbladOneSite(psi, model)
        Nsteps = round(tmax / dt)
        print(f"Nsteps={Nsteps}")
        start = time.time()
        lab.run_detach(Nsteps, dt, 1e-9, 20, 20, max_sweeps=2)
        time_t[n] = time.time() - start
        # record error
        err_t[n] = np.linalg.norm(psi.to_density_matrix().ravel() - Xt_ref)
        print('trace:', np.trace(psi.to_density_matrix()))

    print('errors:', err_t)
    ax.plot(dt_list, err_t, 'o-', label='error')
    ax.plot(dt_list, np.array(dt_list)**2, '--', label="$\delta t^2$") # second order in dt
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel("$\delta t$")
    ax.set_ylabel("errors")
    ax.set_title(f"XXZ model evolved up to t = {tmax}")

    ax1 = ax.twinx()
    ax1.plot(dt_list, time_t, 'x-', label='time')
    ax1.set_ylabel("execution times (s)")

    ax.legend()
    ax1.legend()

def tfi_coherent():
    """errors from the coherent layer alone
    the LindbladOneSite() will reduce to a second order TEBD in absense of dissipators"""

    N = 12
    tmax = 2.

    dt_list = np.array([0.5**k for k in range(6)])  # hence the total time steps = 2**k
    err_t = np.zeros(len(dt_list))

    model = TransverseIsing(N, g=1.5)

    v = MPS.gen_polarized_spin_chain(N, '+z')
    # reference time-evolved state
    vt_ref = expm_multiply(-1j * tmax * model.H_full, v.as_array())
    rhot_ref = np.outer(vt_ref, vt_ref.conj())

    for n, dt in enumerate(dt_list):

        psi = LPTN.gen_polarized_spin_chain(N, '+z')
        lab = LindbladOneSite(psi, model)

        Nsteps = round(tmax / dt)
        logging.info(f"Nsteps={Nsteps}")

        lab.run_detach(1, dt, tol=1e-9, m_max=25, k_max=5, max_sweeps=2)
        lab.run_detach(Nsteps-1, dt, tol=1e-9, m_max=25, k_max=5, max_sweeps=1)
        print(psi.bond_dims)
        print(psi.krauss_dims)

        # record error
        err_t[n] = np.linalg.norm(psi.to_density_matrix() - rhot_ref)

    print(err_t)
    plt.loglog(dt_list, err_t, 'o-', label='variational')
    plt.loglog(dt_list, dt_list**2, '--', label="$\delta t^2$") # second order in dt
    plt.xlabel('$\delta t$')
    plt.ylabel('errors')
    plt.legend()

    plt.savefig('tfi_coherent_01')
    
def disspative_dynamics():
    """errors from the dissipative layer alone"""
    N = 7
    tmax = 1.

    dt_list = np.array([0.5**k for k in range(3)])  # hence the total time steps = 2**k
    err_t = np.zeros(len(dt_list))

    model = disspative_testmodel(N)

    x = LPTN.gen_polarized_spin_chain(N, '+z')
    # reference time-evolved state
    xt_ref = expm_multiply(tmax * model.Liouvillian(model.H_full, *model.L_full), x.to_density_matrix().ravel())

    for n, dt in enumerate(dt_list):

        psi = LPTN.gen_polarized_spin_chain(N, '+z')
        lab = LindbladOneSite(psi, model)

        Nsteps = round(tmax / dt)
        logging.info(f"Nsteps={Nsteps}")

        lab.run_detach(1, dt, tol=1e-9, m_max=25, k_max=25, max_sweeps=2)
        lab.run_detach(Nsteps-1, dt, tol=1e-9, m_max=25, k_max=25, max_sweeps=1)
        """
        lab.make_disspative_layer(dt)
        for i in range(Nsteps):
            contract_disspative_layer(lab.B_list, psi, lab.B_keys)
        """
        print(psi.bond_dims)
        print(psi.krauss_dims)

        # record error
        err_t[n] = np.linalg.norm(psi.to_density_matrix().ravel() - xt_ref)

    print(err_t)
    plt.loglog(dt_list, err_t, 'o-', label='variational')
    plt.xlabel('$\delta t$')
    plt.ylabel('errors')
    plt.legend()

    plt.savefig('random_disspative_01')

class disspative_testmodel(SpinChain):

    def __init__(self, N:int) -> None:
        rng = np.random.default_rng()
        idx = rng.integers(N)
        self._Lloc = [rng.normal(size=(2, 2)) \
                      + 1j*rng.normal(size=(2,2)) \
                      for i in range(N)]
        self._Lloc[idx] = None
        self.idx = idx
        self._N = N

    @property
    def hduo(self):
        return [np.zeros((4,4))] * (self._N-1)
    
    @property
    def Lloc(self):
        return self._Lloc

if __name__ == '__main__': 

    tmax = float(input('simulation time:') or 1.)
    num_intervals = int(input('number of time intervals:') or 3)

    # the total time steps = 2**k
    dt_list = np.array([0.5**k for k in range(num_intervals+1)])

    # convergence plots
    fig, ax = plt.subplots()
    fig.set_size_inches(8,11)
    fig.suptitle('Strang splitting of Liouville operator')

    XXZ(8, tmax, dt_list, ax)

    plt.savefig('XXZ_converge')
    plt.show()