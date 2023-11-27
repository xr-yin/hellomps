"""This module can give you some feelings for setting the simulation parameters. Here we choose two 
example observables: magnetization and bipartite entropy, and observe how their expectation values 
converge as we decrease the discrete time step and increase the maximum bond dimension. One can also 
use observables of their interests and choose the parameters as such the expectation values converge 
within the desired simulation time.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import logging
logging.basicConfig(level=logging.WARNING)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.mps import MPS
from hellomps.models.spin_chains import TransverseIsing
from hellomps.algorithms.mps_evolution import tMPS, TEBD2

def dt_converge(model, alg, backend, dt_list: list, m_max: int, tmax, tol: float, axes: tuple):
    ax1, ax2 = axes
    Nsteps = 2
    for dt in dt_list:
        lbl = f'dt={dt}, m_max={m_max}'
        print(lbl)
        ts = np.arange(0., tmax, dt*Nsteps)
        psi = MPS.gen_polarized_spin_chain(N, '+z')
        lab = alg(psi, model)
        Sz_tot = []
        S = []
        for t in ts:
            Sz_tot.append(np.sum(psi.site_expectation_value([model.sz]*N)))
            S.append(psi.entropy(idx=N//2))
            if backend:
                lab.run(Nsteps, 1j*dt, tol, m_max, backend=backend)
            else:
                lab.run(Nsteps, 1j*dt, tol, m_max)
        ax1.plot(ts, Sz_tot, label=lbl)
        ax2.plot(ts, S, label=lbl)

    ax1.set_ylabel("total $S^z$")
    ax2.set_ylabel("bipartite entropy")
    ax1.legend(loc='upper right')

def mmax_converge(model, alg, backend, mmax_list: list, dt: float, tmax, tol: float, axes: tuple):
    ax1, ax2 = axes
    Nsteps = 2
    for m_max in mmax_list: 
        lbl = f'dt={dt}, m_max={m_max}'
        print(lbl)
        ts = np.arange(0., tmax, dt*Nsteps)
        psi = MPS.gen_polarized_spin_chain(N, '+z')
        lab = alg(psi, model)
        Sz_tot = []
        S = []
        for t in ts:
            Sz_tot.append(np.sum(psi.site_expectation_value([model.sz]*N)))
            S.append(psi.entropy(idx=N//2))
            if backend:
                lab.run(Nsteps, 1j*dt, tol, m_max, backend=backend)
            else:
                lab.run(Nsteps, 1j*dt, tol, m_max)
        ax1.plot(ts, Sz_tot, label=lbl)
        ax2.plot(ts, S, label=lbl)

    ax1.set_ylabel("total $S^z$")
    ax2.set_ylabel("bipartite entropy")
    ax1.legend(loc='upper right')

def main(N, alg, backend):
    tmax = 10
    tol = 1e-10
    dt_list = [0.2, 0.1, 0.05]
    mmax_list = [10, 30, 50]

    model = TransverseIsing(N, g=1.5)

    fig, (axes1, axes2) = plt.subplots(2,2, sharex=True, figsize=(11,11))
    fig.suptitle(f"{alg.__name__} {str(backend or '')}")
    
    dt_converge(model, alg, backend, dt_list, 30, tmax, tol, axes1)
    mmax_converge(model, alg, backend, mmax_list, 0.1, tmax, tol, axes2)

    fig.tight_layout()
    plt.xlabel("time $t$")
    plt.savefig('global_quench')

if __name__ == "__main__":

    N = int(input('enter system size:') or 50)
    algs = {'1': TEBD2, '2': tMPS}
    backends = {'1': 'zip-up', '2': 'variational'}
    backend = None
    for k,v in algs.items():
        print(k,v.__name__)
    alg = algs[input('enter your choice of algorithm (1 or 2):')]
    if alg == tMPS:
        for k,v in backends.items():
            print(k,v)
        backend = backends[input('enter the choice of backend (1 or 2):')]
    main(N, alg, backend)