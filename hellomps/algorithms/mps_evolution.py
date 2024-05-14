#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#----------Time Evolution Algorithms for (pure) MPS----------

__author__='Xianrui Yin'

import numpy as np
from scipy.linalg import qr, rq
from scipy.sparse.linalg import expm, eigsh

import os
import logging
logging.info(f'number of threads in use:{os.environ.get("OMP_NUM_THREADS")}')

from ..networks.mps import MPS
from ..networks.mpo import MPO
from ..networks.operations import *
from ..networks.mpo_projected import *

__all__ = ['TEBD2', 'tMPS']

class TEBD2(object):
    r'''Second order 'TEBD' algorithm for 1D Hamiltonian with only nearest-neighbour 
    coupling
    
    Parameters
    ----------
    psi : MPS
        initial MPS state to be evolved, modification is done in place
    model : None
        1D model object, must have property `hduo`

    Attributes
    ----------
    psi : MPS
        see above
    hduo : list
        a list containing two-local Hamiltonian terms involving only nearest neighbors
    dims : list
        a list containing local Hilbert space dimensions of each site

    Methods
    ----------
    run()
        time evolve the MPS

    Remark
    ----------
    This is not TEBD in strict sense because we are not truncating in the mixed
    canonical form. Rather, this looks like a hybridation of TEBD and tMPS.
    '''
    def __init__(self, psi:MPS, model) -> None:
        self.psi = psi
        self.dims = psi.physical_dims
        self.hduo = model.hduo   # list of two-local terms
        self.dt = None
        self._bot()
    
    def run(self, Nsteps:int, dt:float, tol:float, m_max:int):
        """
        Parameters
        ----------
        Nsteps : int
            total time steps
        dt : float
            time step size, real (imaginary) dt implies imaginary (real) time evolution
        tol : float
            largest discarded singular value during a SVD
        m_max : int
            allowed maximum MPS bond dimension

        Return
        ----------
        None
        """
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
        '''calculate the two-site time evolution operator
        Note that real dt means imaginary time evolution
        '''
        u_duo = {'half': [], 'full': []}
        for idx, hh in enumerate(self.hduo):
            dk1, dk2 = self.dims[idx], self.dims[idx+1]
            u_duo['full'].append(np.reshape(expm(-hh*dt), (dk1,dk2,dk1,dk2)))
            u_duo['half'].append(np.reshape(expm(-hh*dt/2), (dk1,dk2,dk1,dk2)))
        self.u_duo = u_duo
        logging.info('Unitaries prepared')

    def _bot(self):
        assert len(self.hduo) == len(self.psi) -1

class TimedependentTEBD(TEBD2):
    def __init__(self, psi: MPS, hduo) -> None:
        super().__init__(psi, hduo)
        raise NotImplementedError('Not implemented yet')


class tMPS(object):
    """
    Apply the unitary time evolution operator in MPO form, compress the MPS after 
    a few updates.

    Parameters
    ----------
    psi : MPS
        initial MPS state to be evolved, modification is done in place
    model : None
        1D model object, must have property `hduo`

    Attributes
    ----------
    psi : MPS
        see above
    hduo : list
        a list containing two-local Hamiltonian terms involving only nearest neighbors
    dims : list
        a list containing local Hilbert space dimensions of each site

    Methods
    ----------
    run(backend='zip-up', compress_sweeps=2)
        time evolve the MPS
    """
    def __init__(self, psi:MPS, model) -> None:
        self.psi = psi
        self.dims = psi.physical_dims
        self.hduo = model.hduo   # list of two-local terms
        self.dt = None

    def make_uMPO(self, dt:float):
        """
        Here we adopt the strong splitting exp(-i*H*t)~exp(-i*H_e*t/2)exp(-i*H_o*t)exp(-i*H_e*t/2)
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
                    ls[i], ls[j] = split(u2site, mode='sqrt', tol=0., renormalize=False)
            half_e = MPO(half_e)
            full_o = MPO(full_o)
            self.uMPO = mul(half_e, mul(full_o, half_e))

    def run(self, Nsteps:int, dt:float, tol:float, m_max=None, backend='zip-up', compress_sweeps=2):
        """
        Parameters
        ----------
        Nsteps : int
            total time steps
        dt : float
            time step size, real (imaginary) dt implies imaginary (real) time evolution
        tol : float
            largest discarded singular value during a SVD
        m_max : int
            allowed maximum MPS bond dimension
        backend : str
            'zip-up' or 'variational', used for MPO-MPS multiplication
        compress_sweeps : int
            when backend=='variational', the number of compress sweeps that will be performed

        Notes
        ----------
        When the MPS bond dimension is alreay large, and the MPO in question is closed to the 
        identity, 'variational' backend is recommended. Therefore, one might first start out 
        with 'zip-up' and switch to 'variational' at some point when the MPS bond dimensions 
        have grown to a certain value.        
        """
        self.make_uMPO(dt)
        if backend == 'zip-up':
            self.psi.orthonormalize('right')
            for i in range(Nsteps//2):
                zip_up(self.uMPO, self.psi, tol, start='left')
                zip_up(self.uMPO, self.psi, tol, start='right')
            if Nsteps % 2:
                zip_up(self.uMPO, self.psi, tol, start='left')
        elif backend == 'variational':
            for i in range(Nsteps):
                apply_mpo(self.uMPO, self.psi, tol, m_max, max_sweeps=compress_sweeps)
        else:
            raise ValueError('backend can only be zip-up or variational.')


class TDVP(object):
    '''
    Time dependent variational principle.

    Parameters
    ----------
    psi : MPS 
        the initial MPS to be evolved, modification is done in place
    H : MPO
        model Hamiltonian in `MPO` format

    Attributes
    ----------
    same as parameters

    Methods
    ----------
    run_one_site()
    run_two_site()
    '''
    def __init__(self, psi:MPS, H: MPO) -> None:
        assert len(psi) == len(H)
        self.psi = psi
        self.H = H
            
    def run_one_stie(self, dt: float, Nsteps: int, iter: int):
        """Symmetric one site update

        Parameters
        ----------
        dt : float
            time step
        Nsteps : int
            total number of steps, one sweep goes from left to right and back from right to left.
        """
        N = len(self.psi)
        self.psi.orthonormalize('right')
        Ls = LeftBondTensors(N)
        Rs = RightBondTensors(N)
        Rs.load(self.psi, self.psi, self.H)
        for n in range(Nsteps):
            # each of the first and last tensor are only attended once during a back and forth sweep
            for i in range(N-1): # sweep from left to right
                # effective one-stie Hamiltonian
                eff_H = ProjOneSite(Ls[i], Rs[i], self.H[i])
                eigvals, eigvecs = eigsh(eff_H, k=1, which='SA', v0=self.psi[i], return_eigenvectors=True)
                # effective zero-site Hamiltonian
                eff_K = ProjZeroSite(Ls[i+1], Rs[i])
                # TODO: implement expm()
                self.psi[i] = np.reshape(self.psi[i], eff_H.dims)

                self.psi[i], self.psi[i+1] = qr(self.psi[i])
                Ls.update(i+1, self.psi[i].conj(), self.psi[i], self.H[i])
            for i in range(N-1,0,-1):
                eff_H = ProjOneSite(Ls[i], Rs[i], self.H[i])
                _, self.psi[i] = eigsh(eff_H, k=iter, which='SA', v0=self.psi[i], return_eigenvectors=True)
                self.psi[i] = np.reshape(self.psi[i], eff_H.dims)
                self.psi[i-1], self.psi[i] = rq_step(self.psi[i-1], self.psi[i])
                Rs.update(i-1, self.psi[i].conj(), self.psi[i], self.H[i])

class TimedependentTDVP(TDVP):

    def __init__(self, psi: MPS, H: MPO) -> None:
        super().__init__(psi, H)
        raise NotImplementedError('not implemented yet')