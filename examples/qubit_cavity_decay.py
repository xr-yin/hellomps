"""In this module, we use LindbladOneSite() to simualate the decay of occupation number 
of a symmetric coupled qubit-cavity model.
"""

import numpy as np
import matplotlib.pyplot as plt

import os
import sys
import time
import logging
from copy import deepcopy

logging.basicConfig(level=logging.WARNING)

hellompspath = os.path.dirname(os.path.abspath(os.getcwd()))
sys.path.append(os.path.join(hellompspath, "hellomps"))

from hellomps.networks.lptn import LPTN
from hellomps.algorithms.lindblad_master import LindbladOneSite
from hellomps.algorithms.lptn_evolution import LindbladTwoSite
from hellomps.models.scqubits import QubitCavity

def main():

    # initial state with N(0)=3 in the right cavity and all the others empty
    S1 = np.zeros([1,1,2,1])
    S2 = np.zeros([1,1,2,1])
    C1 = np.zeros([1,1,4,1])
    C2 = np.zeros([1,1,4,1])

    S1[0,0,1,0] = S2[0,0,1,0] = C1[0,0,0,0] = C2[0,0,3,0] = 1.

    X = LPTN([S1,C1,C2,S2])
    gamma = 0.05
    model = QubitCavity(alphas=[-0.48]*2, omegas=[1.]*2, gamma=gamma)

    dt = 0.25
    Nsteps = 4 # take measurement every Nsteps
    ts = np.arange(0,10,dt*Nsteps)
    algs = [LindbladOneSite, LindbladTwoSite]
    data = np.zeros([len(algs),len(ts),4])
    fig, axes = plt.subplots(2, sharex=True, figsize=(11,11))

    for j, (alg, ax) in enumerate(zip(algs, axes)):

        Xt = deepcopy(X)
        lab = alg(Xt,model)

        if j == 0:
            start = time.time()
            for i in range(len(ts)):
                data[j,i,:] = model.occupations(Xt)
                lab.run_detach(Nsteps,dt,1e-3,20,20,max_sweeps=1)
            print('time taken by LindbladOneSite():', time.time()-start)
        else:
            start = time.time()
            for i in range(len(ts)):
                data[j,i,:] = model.occupations(Xt)
                lab.run(Nsteps,dt,1e-3,20,20)    
            print('time taken by LindbladTwoSite():', time.time()-start)        

        for i, label in enumerate(['S1','C1','C2','S2']):
            ax.plot(ts,data[j,:,i], label=label)
        ax.plot(ts, np.sum(data,axis=2)[j], label='total excitation')
        ax.plot(ts, 3*np.exp(-gamma*ts), '--', label='interpolation')
        ax.set_yscale('log')
        ax.set_ylabel('<$n_j$>')
        ax.set_title('simulation with ' + alg.__name__)
        ax.legend()

    fig.tight_layout()
    plt.xlabel('time (s)')
    plt.savefig('decays')

if __name__ == "__main__":
    main()