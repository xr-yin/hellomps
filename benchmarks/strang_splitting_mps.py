"""This module demonstrates the errors in our MPS simulations converge with quadratic of time step O(dt^2) in 
the presence of Strang splitting.

TEBD2, tMPS in algorithms/ both use Strang splitting. We compare the performance of TEBD2, tMPS using zip-up 
method and tMPS using variational MPO-MPS multiplication for real time MPS evolution
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import expm_multiply

import os
import sys
import time
import logging
from copy import deepcopy
logging.basicConfig(level=logging.INFO)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.mps_evolution import tMPS, TEBD2

def main(N: int, tmax: float, num_intervals: int):

    algs = [TEBD2, tMPS, tMPS]
    dt_list = np.array([0.5**k for k in range(num_intervals+1)])  # hence the total time steps = 2**k
    err_t = np.zeros((len(algs), len(dt_list)))
    time_t = np.zeros((len(algs), len(dt_list)))

    model = TransverseIsing(N, g=1.5)

    v = MPS.gen_polarized_spin_chain(N, '+z')
    # reference time-evolved state
    vt_ref = expm_multiply(-1j * tmax * model.H_full, v.as_array())

    for n, dt in enumerate(dt_list):

        vts = [deepcopy(v) for i in range(len(algs))]
        labs = [alg(vts[i], model) for i,alg in enumerate(algs)]

        Nsteps = round(tmax / dt)
        logging.info(f"Nsteps={Nsteps}")

        default_params = [Nsteps, 1j*dt, 1e-10, 30]

        t0 = time.time()
        labs[0].run(*default_params)
        t1 = time.time()
        labs[1].run(*default_params, backend='zip-up')
        t2 = time.time()
        labs[2].run(*default_params, backend='variational')
        t3 = time.time()

        # record error
        err_t[:,n] = [np.linalg.norm(vt.as_array() - vt_ref) for vt in vts]
        # record execution time
        time_t[:,n] = [t1-t0, t2-t1, t3-t2]

    return dt_list, err_t, time_t

def plots(tmax, dt_list, err_t, time_t):

    # convergence plots
    fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(f'Unitary evolution of a Ising chain using Strang splitting')

    ax1.loglog(dt_list, err_t[0], 'o-', label='TEBD2')
    ax1.loglog(dt_list, err_t[1], 'd-', label='tMPS with zip-up')
    ax1.loglog(dt_list, err_t[2], 'v-', label='tMPS with variational')
    ax1.loglog(dt_list, dt_list**2, '--', label="$\delta t^2$") # second order in dt
    ax1.set_xlabel("$\delta t$")
    ax1.set_ylabel("errors")
    ax1.legend()

    ax2.plot(tmax / dt_list, time_t[0], 'o-', label='TEBD2')
    ax2.plot(tmax / dt_list, time_t[1], 'd-', label='tMPS with zip-up')
    ax2.plot(tmax / dt_list, time_t[2], 'v-', label='tMPS with variational')
    ax2.set_xscale('log')
    ax2.set_xlabel("Nsteps")
    ax2.set_ylabel("execution times (s)")
    ax2.legend()

    fig.set_size_inches(8,11)
    plt.subplots_adjust(hspace=0.2)
    plt.show(block=True)
    plt.savefig('mps_converge')

if __name__ == "__main__":

    N = int(input('enter system size:') or 11)
    tmax = float(input('simulation time:') or 2.)
    num_intervals = int(input('number of time intervals:') or 6)

    dt_list, err_t, time_t = main(N, tmax, num_intervals)
    plots(tmax, dt_list, err_t, time_t)