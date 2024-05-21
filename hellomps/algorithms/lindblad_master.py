#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Algorithms for evolving a mixed state in locally purified tensor network (LPTN) 
representation according to the Lindblad master equation."""

__author__='Xianrui Yin'

import numpy as np
from scipy.linalg import expm

import logging
from copy import deepcopy

from .mps_evolution import tMPS
from ..networks.mpo import MPO
from ..networks.lptn import LPTN, compress
from ..networks.operations import merge, split, mul
from ..networks.mpo_projected_lptn import *

__all__ = ['LindbladOneSite', 'applyMPKO', 'contract_coherent_layer', 'contract_dissipative_layer']

class LindbladOneSite(tMPS):
    r"""Handles the case when the Lindblad jump operators only act on single sites.

    Parameters
    ----------
    psi : LPTN
        the LPTN to be evolved, modification is done in place
    model : None
        1D model object, must have the following two properties: `hduo` and `Lloc`.

    Attributes
    ----------
    psi : LPTN
        see above
    hduo : list
        a list containing two-local Hamiltonian terms involving only nearest neighbors
    dims : list
        a list containing local Hilbert space dimensions of each site
    Lloc : list 
        a list conataing local Lindblad jump operators
    dt : float
        time step
    overlaps : list
        a list containing the optimized overlaps between the updated LPTN and the exact LPTN
        at each time step

    Methods
    ----------
    run_detach()
        primal method for simulating the Lindblad master equation with one-stie dissipators,
        using a variational approach to update the purification network at every time step.
    run_attach()
        a simpler simulating method, contracting the matrix product krauss operator and the 
        purification network at every time step. Warning: long runtime, still in test phase.
    make_coherent_layer()
        calculate the unitary time evolution operator at a MPO (MPU) from the local Hamiltonian
        terms
    make_dissipative_layer()
        calculate the Krauss representation of the local quantum channels
    """
    def __init__(self, psi: LPTN, model) -> None:
        self.Lloc = model.Lloc # list containing all (local) jump operators
        super().__init__(psi, model)

    def run_detach(self, Nsteps, dt: float, tol: float, m_max: int, k_max: int, max_sweeps=1):
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo)
        if dt != self.dt:
            self.make_coherent_layer(dt)
            self.make_dissipative_layer(dt)
            self.dt = dt
        for i in range(Nsteps):
            # apply uMPO[0]
            lp = contract_coherent_layer(self.uMPO[0], self.psi, tol, m_max, max_sweeps)
            # now in right canonical form
            logging.debug(f'overlap={lp}')
            # apply bMPO
            contract_dissipative_layer(self.B_list, self.psi, self.B_keys)
            # now STILL in right canonical form because application of Kraus operators preserve
            # canonical forms
            truncate_krauss_sweep(self.psi, tol, k_max)
            # now in left canonical form
            # apply uMPO[1]
            lp = contract_coherent_layer(self.uMPO[1], self.psi, tol, m_max, max_sweeps)
            # now in right canonical form
            logging.debug(f'overlap={lp}')
    
    def run_attach(self, Nsteps, dt: float, tol: float, m_max: int, k_max: int): 
        Nbonds = len(self.psi)-1
        assert Nbonds == len(self.hduo)
        if dt != self.dt:
            self.make_full_layer(dt)
            self.dt = dt
        self.psi.orthonormalize('right')
        for i in range(Nsteps//2):
            # zip-up
            applyMPKO(self.full_list, self.psi, tol, start='left')
            applyMPKO(self.full_list, self.psi, tol, start='right')
            # followed by variational compress
            compress(self.psi, tol, m_max, k_max)
            # now psi is almost right canonical
            # optional: call disentangle (externally)
        if Nsteps % 2:
            applyMPKO(self.full_list, self.psi, tol=tol, start='left')
                
    def make_dissipative_layer(self, dt: float):
        """
        calculate the Kraus operators from the Lindblad jump operators, the resulting
        Kraus operators have the following shape
                   0|
                :.......:
                :   B   :--:
                :.......:  |
                   1|     2|

        0 : output (dim=d)
        1 : input  (dim=d)
        2 : Kraus  (dim<=d^2)
        """
        B_list = []
        B_keys = []
        for i, L in enumerate(self.Lloc):
            d = self.dims[i]
            if L is not None:
                B_keys.append(1)
                # calculate the dissipative part in superoperator form
                D = np.kron(L,L.conj()) \
                - 0.5*(np.kron(L.conj().T@L, np.eye(d)) + np.kron(np.eye(d), L.T@L.conj()))
                eDt = expm(D*dt)
                eDt = np.reshape(eDt, (d,d,d,d))
                eDt = np.transpose(eDt, (0,2,1,3))
                eDt = np.reshape(eDt, (d*d,d*d))
                assert np.allclose(eDt, eDt.conj().T)
                B = _cholesky(eDt)
                B = np.reshape(B, (d,d,-1))
                B_list.append(B)
            else:
                B_keys.append(0)
                B_list.append(np.eye(d)[:,:,None])
        self.B_list = B_list
        self.B_keys = B_keys

    def make_coherent_layer(self, dt:float):
        """
        Here we adopt the strong splitting exp(-i*H*t)~exp(-i*H_e*t/2)exp(-i*H_o*t)exp(-i*H_e*t/2)
        """
        N = len(self.psi)
        half_e = [np.eye(self.dims[i])[None,None,:,:] for i in range(N)]
        half_o = [np.eye(self.dims[i])[None,None,:,:] for i in range(N)]
        for k, ls in enumerate([half_e, half_o]):  # even k = 0, odd k = 1
            for i in range(k,N-1,2):
                j = i+1
                di, dj = self.dims[i], self.dims[j]
                u2site = expm(-1j*self.hduo[i]*dt/2) # imaginary unit included!
                u2site = np.reshape(u2site, (1,1,di,dj,di,dj))  # mL,mR,i,j,i*,j*
                ls[i], ls[j] = split(u2site, mode='sqrt', tol=0., renormalize=False)
        half_e = MPO(half_e)
        half_o = MPO(half_o)
        self.uMPO = [mul(half_e, half_o), mul(half_o, half_e)]
        del half_e
        del half_o

    def make_full_layer(self, dt: float):
        """            
                        |k      output
                    ----a----
                        |k*, 3  input
                        |k , 0  output
                        b ---:
                        |k*  |  input
                        |k      output
                    ----c----
                        |k*, 3  input
        """
        self.make_coherent_layer(dt)
        self.make_dissipative_layer(dt)
        Os = []
        for a, b, c in zip(self.uMPO[0], self.B_list, self.uMPO[1]):
            a0, a1 = a.shape[:2]
            c0, c1 = c.shape[:2]
            b0, b1, b2  = b.shape  # b0 = b1 = physical dimension
            O = np.tensordot(a, b, axes=(3,0)) # 5 legs mL,mR,d,d*,k
            O = np.tensordot(O, c, axes=(3,2))
            O = np.transpose(O, (0,4,1,5,2,6,3))
            O = np.reshape(O, (a0*c0,a1*c1,b0,b1,b2))
            Os.append(O) # mL, mR, d, d*, k
        self.full_list = Os

    def _bot(self):
        super()._bot()
        assert len(self.Lloc) == len(self.psi)

def _cholesky(a):
    """stablized cholesky decomposition of matrix a"""
    eigvals, eigvecs = np.linalg.eigh(a)
    mask = abs(eigvals) > 1e-13
    eigvals = eigvals[mask]
    eigvecs = eigvecs[:,mask]
    assert min(eigvals) > 0
    B = eigvecs*np.sqrt(eigvals)
    logging.info(f'error during cholesky decomposition: {np.linalg.norm(a-B@B.T.conj())}')
    return B

def applyMPKO(O: list, psi: LPTN, tol: float, m_max=None, k_max=None, start='left'):
    r"""Zip-up method for contracting a MPKO with a LPTN

    Parameters:
        O : list
            the operator, in this case a matrix product Kraus operator
        psi : LPTN
            the operand, in this case a locally purified TN, modified in place
        tol : float
            largest discarded singular value in every truncation step
        m_max : int or None
            largest matrix bond dimension allowed, default is None
        k_max : int or None
            largest kraus bond dimension allowed, default is None
        start : str
            if 'left', the contraction (zipping) is performed from left
            to right. This should be used when the inital state is right
            canonical. For `start`='right', it is the other way around.
    """
    assert len(O) == len(psi)
    N = len(psi)
    if start == 'left':
        M = np.tensordot(psi[0], O[0], axes=([0,2],[0,3]))  # ndim=5
        M = M[None,:,:,:,:] # s, m, k, w, d, k'  ndim=6
        for i in range(N-1):
            M = np.transpose(M, (0,4,2,5,1,3)) # s,d,k,k',m,w
            M = np.reshape(M, M.shape[:2]+(-1,)+M.shape[-2:])  # s,d,k,m,w
            s, d, k, m, w = M.shape # s,d,k,m,w
            M = np.reshape(M, (s*d*k, m*w)) # (s,d,k), (m,w)
            u, svals, vt = np.linalg.svd(M, full_matrices=False)
            svals = svals / np.linalg.norm(svals)
            pivot = min(np.sum(svals>tol), m_max) if m_max else np.sum(svals>tol)
            psi[i] = np.reshape(u[:,:pivot], (s, d, k, -1)) # s, d, k, s'
            psi[i] = np.transpose(psi[i], (0,3,1,2))  # s, s', d, k
            #truncate_krauss(psi[i], tol, k_max)
            M = np.reshape(np.diag(svals[:pivot]) @ vt[:pivot,:], (-1, m, w))
            M = np.tensordot(M, psi[i+1], axes=(1,0))
            M = np.tensordot(M, O[i+1], axes=([1,3],[0,3]))
        M = M[:,:,:,0,:,:] # mL, mR, k, d, k'
        M = np.swapaxes(M, 2 ,3) # mL, mR, d, k, k'
        M = np.reshape(M, M.shape[:-2]+(-1,)) # mL, mR, d, k
        psi[N-1] = M / np.linalg.norm(M)
        # psi is now in the left canonical form
    elif start == 'right':
        M = np.tensordot(psi[-1], O[-1], axes=([1,2],[1,3])) # ndim=5
        M = M[:,:,None,:,:] # m, k, s, w, d, k'  ndim=6
        M = np.transpose(M, (0,3,2,4,1,5)) # m, w, s, d, k, k'
        M = np.reshape(M, M.shape[:-2]+(-1,)) # ndim=5
        for i in range(N-1, 0, -1):
            m, w, s, d, k = M.shape
            M = np.reshape(M, (m*w, s*d*k))
            u, svals, vt = np.linalg.svd(M, full_matrices=False)
            svals = svals / np.linalg.norm(svals)
            pivot = min(np.sum(svals>tol), m_max) if m_max else np.sum(svals>tol)
            psi[i] = np.reshape(vt[:pivot,:], (-1, s, d, k)) # s', s, d, k
            #truncate_krauss(psi[i], tol, k_max)
            M = np.reshape(u[:,:pivot]*svals[:pivot], (m, w, -1))
            M = np.tensordot(psi[i-1], M, axes=(1,0))
            M = np.tensordot(M, O[i-1], axes=([1,3],[3,1]))
            M = np.transpose(M, (0,3,2,4,1,5)) # m, w, s, d, k, k'
            M = np.reshape(M, M.shape[:-2]+(-1,)) # ndim=5
        M = M[:,0,:,:]
        psi[0] = M / np.linalg.norm(M)
        # psi is now in the right canonical form
    else:
        raise ValueError('start can only be left or right.')

def contract_coherent_layer(O: MPO, psi: LPTN, tol: float, m_max: int, max_sweeps=1):
    """Varationally calculate the product of a MPO and a LPTN.

    psi is modified in place, the result is the product O|psi>        
                    |k   output
                ----O----
    O |psi> =       |k*, 3
                    |k , 2
                ---psi----
                    |

    Parameters
    ----------
    O : MPO
        the operator
    psi : LPTN 
        the operand
    tol : float
        largest discarded singular value in each truncation step
    m_max : int 
        largest bond dimension allowed, default is None
    max_sweeps : int
        maximum number of optimization sweeps

    Return
    ----------
    overlap : float
        the optimized inner product <phi|O|psi> where |phi> ~ O|psi>
        The name 'overlap' is only appropriate when |psi> is a unit 
        vector and O is unitary. In this case, <phi|O|psi> evaluates 
        to 1 whenever phi is a good approximation. In general cases, 
        'norm square' would be a better term since <phi|O|psi> ~ 
        <phi|phi>.
        
    """
    N = len(psi)
    phi = deepcopy(psi) # assume psi is only changed slightly, useful
    phi.orthonormalize('right') # when O is a time evolution operator 
    Rs = RightBondTensors(N)  # for a small time step
    Ls = LeftBondTensors(N)
    Rs.load(phi, psi, O)
    for n in range(max_sweeps):
        for i in range(N-1): # sweep from left to right
            j = i+1
            x = merge(psi[i], psi[j])
            eff_O = ProjTwoSite(Ls[i], Rs[j], O[i], O[j])
            x = eff_O._matvec(x)
            # split the result tensor
            phi[i], phi[j] = split(x, 'right', tol, m_max)
            # update the left bond tensor LBT[j]
            Ls.update(j, phi[i].conj(), psi[i], O[i])
        for j in range(N-1,0,-1): # sweep from right to left
            i = j-1
            x = merge(psi[i], psi[j])
            # contracting left block LBT[i]
            eff_O = ProjTwoSite(Ls[i], Rs[j], O[i], O[j])
            x = eff_O._matvec(x)
            # split the result tensor
            phi[i], phi[j] = split(x, 'left', tol, m_max)
            # update the right bond tensor RBT[i]
            Rs.update(i, phi[j].conj(), psi[j], O[j])
        
        overlap = np.tensordot(phi[0].conj(), Rs[0], axes=(1,0))
        overlap = np.tensordot(overlap, O[0], axes=([1,3],[2,1]))
        overlap = np.tensordot(overlap, psi[0], axes=([2,4,1],[1,2,3]))
        overlap = overlap.ravel()
        # overlap should be REAL no matter O is unitary or not
        # because <phi|O|psi> ~ <psi|O*O|psi> = <v|v> is real
        # whenever |phi> ~ O|psi>
        # here * means hermitian conjugate and |v> = O|psi>
        logging.info(f'n={n}, overlap={np.real_if_close(overlap).item()}')
    psi.As = phi.As
    del phi
    del Rs
    del Ls
    return np.real_if_close(overlap).item()

def contract_dissipative_layer(O: list, psi: LPTN, keys: list):
    """Contract the dissipative layer of Kraus operators with the LPTN

    Parameters
    ----------
    O : list
        list containing the local one-site Kraus operator
    psi : LPTN
        the operand
    keys : list
        a binary list
    tol : float
        largest discarded singular value in each truncation step
    k_max : int 
        largest Kraus dimension allowed

    Return
    ----------
    phi : LPTN
        result of the product O|psi>
        
                    |k   0 output
                    O -----2 Kraus
    O |psi> =       |k*, 1
                    |k , 2
                ---psi----
                    |
    """
    assert len(psi) == len(O) 
    assert len(psi) == len(keys)
    for i in range(len(psi)):
        if keys[i]:
            psi[i] = np.tensordot(psi[i], O[i], axes=(2,1))
            psi[i] = np.swapaxes(psi[i], 2, 3)
            psi[i] = np.reshape(psi[i], psi[i].shape[:-2]+(-1,))

def truncate_krauss(x, tol, k_max):
    di, dj, dd, dk = x.shape
    u, svals, _ = np.linalg.svd(x.reshape(-1,dk), full_matrices=False)
    svals1 = svals / np.linalg.norm(svals)
    pivot = min(np.sum(svals1>tol), k_max) if k_max else np.sum(svals1>tol)
    svals = svals[:pivot] 
    #/ np.linalg.norm(svals[:pivot])
    return np.reshape(u[:,:pivot]*svals[:pivot], (di, dj, dd, -1)) # s, d, k, s'

def truncate_krauss_sweep(As:list, tol:float, k_max:int):
    """psi must be in right canonical form when passed in"""
    for i in range(len(As)-1):
        di, dj, dd, dk = As[i].shape
        u, svals, _ = np.linalg.svd(As[i].reshape(-1,dk), full_matrices=False)
        pivot = min(np.sum(svals**2>tol), k_max)
        svals = svals[:pivot] / np.linalg.norm(svals[:pivot])
        As[i] = np.reshape(u[:,:pivot]*svals[:pivot], (di, dj, dd, -1)) # s, d, k, s'

        dk = As[i].shape[-1]    # reduced Kraus dim
        As[i], r = np.linalg.qr(np.reshape(As[i].transpose(0,2,3,1), (-1, dj)))
        # qr decomposition might change dj
        As[i] = np.transpose(As[i].reshape(di,dd,dk,-1), (0,3,1,2))
        
        As[i+1] = np.tensordot(r, As[i+1], axes=1)

    i = -1
    di, dj, dd, dk = As[i].shape
    u, svals, _ = np.linalg.svd(As[i].reshape(-1,dk), full_matrices=False)
    pivot = min(np.sum(svals**2>tol), k_max)
    svals = svals[:pivot] / np.linalg.norm(svals[:pivot])
    As[i] = np.reshape(u[:,:pivot]*svals[:pivot], (di, dj, dd, -1)) # s, d, k, s'

    dk = As[i].shape[-1]    # reduced Kraus dim
    As[i], r = np.linalg.qr(np.reshape(As[i].transpose(0,2,3,1), (-1, dj)))
    # qr decomposition might change dj
    As[i] = np.transpose(As[i].reshape(di,dd,dk,-1), (0,3,1,2))