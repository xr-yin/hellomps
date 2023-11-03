#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------time_evolution_algorithms----------

__author__='Xianrui Yin'

import numpy as np
from scipy.linalg import expm, qr, rq

import gc
import logging
logging.basicConfig(level=logging.ERROR)

from ..networks.mps import MPS
from ..networks.mpo import MPO
from ..networks.operations import *

__all__ = ['TEBD2', 'tMPS']

class TEBD2(object):
    '''
    Second order 'TEBD' algorithm for 1D Hamiltonian with only nearest-neighbour coupling
    
    Parameters:
        psi: MPS to be evolved
        hduo: local Hamiltonian terms involving only two nearest neighbors

    Attributes:
        psi, hduo, dims

    Remark:
    This is not TEBD in strict sense because we are not truncating in the mixed
    canonical form. Rather, this looks like a hybridation of TEBD and tMPS.
    '''
    def __init__(self, psi:MPS, model) -> None:
        self.psi = psi
        self.dims = psi.physical_dims
        self.hduo = model.hduo   # list of two-local terms
        self.dt = None
        self._bot()
    
    def run(self, Nsteps, dt:float, tol:float, m_max:int):
        Nbonds = len(self.psi)-1
        if dt != self.dt:
            self.make_unitaries(dt)
            self.dt = dt
        # apply a half step at odd-even sites
        for i in range(1, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize(mode='right')
        # ---(even-odd )*(Nsteps-1)---       
        for n in range(Nsteps-1):
            for k in [0, 1]:  # even-odd and odd-even
                for i in range(k, Nbonds, 2):
                    self.update_local_two_sites(i, tol, m_max)
            self.psi.orthonormalize(mode='right')
        # apply a full step at even bonds
        for i in range(0, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max)
        # apply a half step at odd-even sites
        for i in range(1, Nbonds, 2):
            self.update_local_two_sites(i, tol, m_max, half=True)
        self.psi.orthonormalize(mode='right')

    def update_local_two_sites(self, i, tol, m_max, half=False):
        # construct theta matrix
        j = i+1
        theta = merge(self.psi.As[i], self.psi.As[j])  # i, j, k1, k2
        # apply U
        if half:
            theta = np.tensordot(theta, self.u_duo['half'][i], axes=([2, 3], [2, 3]))
        else:
            theta = np.tensordot(theta, self.u_duo['full'][i], axes=([2, 3], [2, 3]))
        # split and truncate
        self.psi.As[i], self.psi.As[j] = split(theta, "sqrt", tol, m_max)
        
    def make_unitaries(self, dt:float):
        '''
        calculate the two-site time evolution operator
        note that real dt means imaginary time evolution
        '''
        u_duo = {'half': [], 'full': []}
        for idx, hh in enumerate(self.hduo):
            dk1, dk2 = self.dims[idx], self.dims[idx+1]
            u_duo['full'].append(np.reshape(expm(-hh*dt), (dk1,dk2,dk1,dk2)))
            u_duo['half'].append(np.reshape(expm(-hh*dt/2), (dk1,dk2,dk1,dk2)))
        self.u_duo = u_duo
        print('Unitaries prepared')

    def _bot(self):
        assert len(self.hduo) == len(self.psi) -1

class TimedependentTEBD(TEBD2):
    def __init__(self, psi: MPS, hduo) -> None:
        super().__init__(psi, hduo)
        pass


class tMPS(object):
    """
    Apply the unitary time evolution operator in MPO form, compress the 
    MPS after a few updates.
    """
    def __init__(self, psi:MPS, model) -> None:
        self.psi = psi
        self.dims = psi.physical_dims
        self.hduo = model.hduo   # list of two-local terms
        self.dt = None

    def make_uMPO(self, dt:float):
        """
        Here we adopt the strong splitting exp(i*H*t)~exp(i*H_e*t/2)exp(i*H_o*t)exp(i*H_e*t/2)
        """
        if dt != self.dt:
            self.dt = dt
            N = len(self.psi)
            half_e = [np.eye(self.dims[i])[None,None,:,:] for i in range(N)]
            full_o = [np.eye(self.dims[i])[None,None,:,:] for i in range(N)]
            for k, ls in enumerate([half_e, full_o]):  # even k = 0, odd k = 1
                for i in range(k,N-1,2):
                    j = i+1
                    di, dj = self.dims[i], self.dims[j]
                    u2site = expm(-self.hduo[i]*(k+1)*dt/2)
                    u2site = np.reshape(u2site, (1,1,di,dj,di,dj))  # mL,mR,i,j,i*,j*
                    ls[i], ls[j] = split(u2site, mode='sqrt', tol=0.)
            logging.debug([_.shape for _ in full_o])
            half_e = MPO(half_e)
            full_o = MPO(full_o)
            self.uMPO = mul(half_e, mul(full_o, half_e))
            #del half_e
            #del full_o
            #gc.collect()

    def run(self, Nsteps:int, dt:float, tol:float, m_max=None, compress_sweeps=2):
        self.make_uMPO(dt)
        for i in range(Nsteps):
            zip_up(self.uMPO, self.psi, tol)
            #apply_mpo(self.uMPO, self.psi, tol, m_max, max_sweeps=compress_sweeps, overwrite=True)


class TDVPOneSite(object):
    '''
    Time evolution algorthim class based on time dependent variational principle

    Parameters:
        psi: MPS to be evolved
        dt: time step for a full sweep
        M: total number of steps  t=M*dt

    Attributes:
        psi:
        dt:

    Methods:
        sweep:
    '''
    def __init__(self, psi:MPS, dt:float, M:int):
        self.psi = psi
        self.dt = dt /2
        self.M = M


    def train(self):
        t = 0
        '''symmetric one-site''' 
        for i in range(self.M):
            ''' left to right sweep '''
            # initialize a right canonical form
            self.psi.orthonormalize(mode='right')
            self.sweep(t)
            t+=self.dt

            
    def sweep(self,t):
        '''symmetric one-site''' 

        self.psi.orthonormalize(mode='right')
        for idx in range(1,len(self.psi)):
            # compute one-site effective Hamiltonian H(n)


            self.psi.As[idx]  # evolve A_C(n) forward in time


            # QR decompose evolved A_C(n) into A_L(n) @ C_tilde(n)
            A_L, C_tilde = qr()


            # compute zero-site effective Hamiltonian K(n)
            pass

            C_tilde  # evolve C_tilde backwards in time

            
            t += self.dt
            # right to left sweep
        for idx in range(len(self.psi)-1, 0, -1):

            t += self.dt
            pass

class TimedependentTDVP(TDVPOneSite):
    pass


if __name__ == "__main__":
    pass