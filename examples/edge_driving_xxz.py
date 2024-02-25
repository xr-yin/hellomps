"""In this module, we use LindbladOneSite() to simulate an edge driving XXZ chain
and observe the local megnetization and particle current in equalibrium.
"""

import numpy as np
import matplotlib.pyplot as plt

import sys
import os
import logging
logging.basicConfig(level=logging.ERROR)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import LPTN
from hellomps.algorithms.lindblad_master import LindbladOneSite
from hellomps.models.spin_chains import Heisenberg

def relexation(N: int, delta:float, dt: float, m_max:int, k_max:int, Nsteps: int, Nmeasurements: int):
    """Compare the relaxation time for different system sizes by observing the spin profiles
    
    Parameters
    ----------
    N_list : int
        chain length of the Heisenberg XXZ model
    delta : float
        the sigma z - sigma z coupling constant, delta = 1 is the isotropic case
    dt : float
        time step
    m_max : int
        max horizontal (matrix) bond dimension
    k_max : int
        max vertial (krauss) bond dimension
    Nsteps : int
        number of steps between two measurements, thus the time interval for measurements
        is dt*Nsteps
    Nmeasurements : int
        total number of measurements
    """

    fig, (ax1, ax2) = plt.subplots(2)
    fig.set_size_inches(12,10)

    print('N =', N)
    print('total time:', Nmeasurements*Nsteps*dt)
    model = Heisenberg(N, [1.,1.,delta], 0., 1.)
    psi = LPTN.gen_polarized_spin_chain(N, '+x')
    lab = LindbladOneSite(psi, model)
    lab.run_detach(Nsteps, dt, 1e-7, m_max, k_max, max_sweeps=2)
    ax1.plot(psi.site_expectation_value([model.sz]*N), 'o-', label=f't={Nsteps*dt}')
    print(f't={Nsteps*dt}')
    for i in range(Nmeasurements-1):
        lb = f't={(i+2)*Nsteps*dt}s'
        lab.run_detach(Nsteps, dt, 1e-6, m_max, k_max)
        ax1.plot(psi.site_expectation_value([model.sz]*N), 'o-', label=lb)
        ax1.set_title(f'N={N}, $\Delta$={delta}')
        ax1.legend()
        print(lb)
    ax2.plot(lab.overlaps, label=f'm_max={m_max}, k_max={k_max}, dt={dt}')
    ax2.set_ylabel('overlap')
    ax2.legend()
    
    print('hori:', psi.bond_dims)
    print('vert:', psi.krauss_dims)

    fig.tight_layout()
    plt.xlabel('site index')
    plt.savefig(f'xxz_relaxation_N={N}_01')
    # TODO
    # save model

def magnetization(Nmeasurements=4, start='flat'):
    """we study the steady spin profile in three different regime. 
    (a) easy-plane regime (delta < 1) 
        flat spin profile --> ballistic transport
    (b) isotropic regime (delta = 1)
        cosine spin rpfile independent of the coupling constant (gamma), system
        exhibits long range spin-spin correlation
    (c) easy-axis regime (delta > 1)
        spin profile has a kinked-shape

    analytical results: Tomaz Prosen, PRL 107, 137201 (2011)    
    
    Parameters
    ----------
    Nmeasurements : int
        number of measurements
    start : str
        inital state for the simulation
        'flat' --> x-polarized state 
        'step' --> bipartite system with half spins up and half spins down
        'random' --> random state
    """
    N = 20
    dt = 0.25
    ns = 60

    delta_list = [0.5, 1.0, 1.5]
    mag = np.zeros([len(delta_list),N])
    fig, axes = plt.subplots(ncols=Nmeasurements)
    fig.set_size_inches(20,8)

    for i, delta in enumerate(delta_list):
        print('delta:', delta)
        model = Heisenberg(N, (1., 1., delta), 0., 1.)

        if start == 'flat':
            psi = LPTN.gen_polarized_spin_chain(N, '+x')
        elif start == 'step':
            A = np.zeros([1,1,2,1])
            B = np.zeros([1,1,2,1])
            A[0,0,0,0] = 1.
            B[0,0,1,0] = 1.
            psi = LPTN([A]*(N//2) + [B]*(N//2))
        elif start == 'random':
            psi = LPTN.gen_random_state(N, m_max=2, k_max=2, phy_dims=[2]*N)
        else:
            raise ValueError('start can only be flat, step or random')
        
        lab = LindbladOneSite(psi, model)

        for j, ax in enumerate(axes):
            lab.run_detach(ns, dt, 1e-7, m_max=15, k_max=15, max_sweeps=1)
            mag[i] = psi.site_expectation_value([model.sz]*N)
            ax.plot(np.arange(N), mag[i], 'o-', label=f'$\Delta$={delta}')
            ax.set_ylabel('<$\sigma_j^z$>')
            ax.set_xlabel('site')
            ax.set_ylim(-1., 1.)
            ax.set_title(f't={ns*dt*(j+1)}s')
            ax.legend()
            print(f't={ns*dt*(j+1)}s')

    fig.suptitle(f'N={N}')
    fig.tight_layout()
    plt.savefig(f'xxz_magnetization_N={N}_02')

def spin_current():
    """we study the system-size-dependence of the steady spin current in three different regimes
    (a) easy-plane regime (delta < 1) 
        spin current does not depend on system size --> ballistic transport
    (b) isotropic regime (delta = 1)
        spin current scales as 1/n^2, n is the system size
    (c) easy-axis regime (delta > 1)
        spin current decays exponentially in system size --> insulator

    analytical results: Tomaz Prosen, PRL 107, 137201 (2011)
    """

    N_list = np.arange(10,50,10, dtype=int)
    max_bonds = np.arange(5,45,10, dtype=int)
    dt = 0.25

    current = np.zeros([2, len(N_list)])

    for i, delta in enumerate([0.5, 1.0]):
        print('delta=', delta)
        for j, N in enumerate(N_list):
            print('N=', N)
            model = Heisenberg(N, (1., 1., delta), 0., 1.)
            psi = LPTN.gen_polarized_spin_chain(N, '+x')
            lab = LindbladOneSite(psi, model)
            lab.run_detach(60, dt, 1e-8, max_bonds[j], max_bonds[j], max_sweeps=2)
            lab.run_detach(int(8*N/delta), dt, 1e-8, max_bonds[j], max_bonds[j], max_sweeps=1)
            current[i][j] = model.current(psi)[N//2]
            print('hori bonds:', psi.bond_dims)
            print('vert bonds:', psi.krauss_dims)
        plt.plot(N_list, current[i], 'o-', label=f'delta={delta}')
        plt.yscale('log')

    plt.ylabel('I')
    plt.xlabel('chain length')
    plt.title('Current')
    plt.legend()
    plt.savefig('xxz_spin_current_01')
    print(current)

if __name__ == "__main__":

    relexation(N=40, delta=1., dt=0.25, m_max=25, k_max=25, Nsteps=60, Nmeasurements=6)
    magnetization()
    spin_current()