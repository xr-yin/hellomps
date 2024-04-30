import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import LPTN
from hellomps.algorithms.lindblad_master import LindbladOneSite
from hellomps.models.boson_chains import DDBoseHubburd


def runtime_plot(tmax, **params):
    """Test the runtime for our program, which should be linear in system size N

    Parameters
    ----------
    tmax : float
        total time for simulation
    """

    # input the model parameters
    U = params.get('U', 1.)
    t = params.get('t', 1.)
    F = params.get('F', 1.)
    mu = params.get('mu', 0.)
    gamma = params.get('gamma', 1.)

    d = 4
    dt = 0.5
    chi = 10

    Nsteps = round(tmax/dt)
    print('Nsteps=', Nsteps)

    fig, ax = plt.subplots()
    fig.suptitle(params)

    N_list = [4,6,8,10]
    T_list = []

    for N in N_list:

        A = np.zeros([1,1,d,1])
        A[0,0,0,0] = 1.
        psi = LPTN([A]*N)

        model = DDBoseHubburd(N, d, t, U, mu, F, gamma)
        lab = LindbladOneSite(psi, model)

        # warm_up
        lab.run_detach(3, dt, 1e-5, chi, chi, max_sweeps=1)
        print(psi.bond_dims)  # check if the bond dimensions have
        print(psi.krauss_dims)  # saturated 

        # start benchmarking
        start = time.time()
        lab.run_detach(Nsteps, dt, 1e-5, chi, chi, max_sweeps=1)
        T_list.append(time.time()-start)

    ax.plot(N_list, T_list, '-o', label='simulation')

    coeff = np.polyfit(N_list, T_list, 1)
    poly1d_fn = np.poly1d(coeff) 
    ax.plot(N_list, poly1d_fn(N_list), '--', label=f'O(N)~{coeff[1]:.1f}+{coeff[0]:.1f}N')

    ax.set_xticks(N_list)
    ax.set_xlabel('system size')
    ax.set_ylabel('runtime (s)')
    
    ax.legend()

    plt.savefig(f'runtime_even4')

if __name__ == '__main__':

    tmax = float(input('simulation time:') or 10.)

    params = {'F': 0.3, 'U': 1., 't':0.2, 'gamma':0.1}
    runtime_plot(tmax, **params)