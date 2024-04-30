#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np

from copy import deepcopy

from .mps import MPS
from .mpo import MPO
from .operations import merge, split, mul, qr_step

__all__ = ['LPTN', 'compress']

class LPTN(MPO):
    """Locally purified tensor networks
    
    Parameters
    ----------
        As : list
            list of local rank-4 tensors, each tensor has the following shape
                k
                |
            i---A---j       (i, j, k, l)
                |
                l
    
    Methods
    ----------
    site_expectation_value(idx=None)
        compute the expectation value of local (one-site) observables
    bond_expectation_value(idx=None)
        compute the expectation value of local two-site observables
    to_density_matrix()
    probabilities()
    """
    def __init__(self, As) -> None:
        super().__init__(As)

    @classmethod
    def gen_polarized_spin_chain(cls, N: int, polarization: str):
        if polarization not in ['+z','-z','+x']:
            raise ValueError('Only support polarization +z, -z or +x')
        A = np.zeros([1,1,2,1])
        if polarization == '+z':
            A[0,0,0,0] = 1.
        elif polarization == '-z':
            A[0,0,1,0] = 1.
        else:
            A[0,0,0,0] = A[0,0,1,0] = 0.5**0.5
        return cls([A]*N)
    
    @classmethod
    def gen_random_state(cls,N:int,m_max:int, k_max:int, phy_dims:list):
        assert len(phy_dims) == N
        rng = np.random.default_rng()
        bond_dims = rng.integers(1, m_max, size=N+1)
        krauss_dims = rng.integers(1, k_max, size=N)
        bond_dims[0] = bond_dims[-1] = 1
        As = []
        for i in range(N):
            As.append(rng.random((bond_dims[i],bond_dims[i+1],phy_dims[i],krauss_dims[i])))
        return cls(As)
    
    @property
    def krauss_dims(self):
        return [A.shape[3] for A in self]

    def site_expectation_value(self, op, idx=None):
        """
        Parameters
        ----------
        op : list or NDArray
            list of local operators or a single local operator
        idx : int or None
            if `idx` is None, `op` must be a list of ordered local operators for 
            every physical site.
            if `idx` is an integer, `op` must be a single operator for this specific 
            physical site.
        """
        if isinstance(op, list):
            assert len(op) == self._N
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                amp = self[i]   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=(2,1)) # apply local operator
                res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
                exp.append(np.real_if_close(res))
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            amp = self[-1]
            opc = np.tensordot(amp, op[-1], axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
            exp.append(np.real_if_close(res))
            self.As = cache
            return exp
        else:
            #assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = self[idx]   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
            self.As = cache
            return np.real_if_close(res)
        
    def bond_expectation_value(self, op, idx=None):
        """
        Parameters
        ----------
        op : list or NDArray
            list of local two-site operators or a single local two-site operator
        idx : int or None
            if `idx` is None, `op` must be a list of ordered two-local operators 
            for every pair of neighbouring site.
            if `idx` is an integer, `op` must be a single two-local operator for 
            this specific pair of neighbouring site.
        """
        if isinstance(op, list):
            assert len(op) == self._N-1
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                j = i+1
                amp = merge(self[i], self[j])   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=([2,3],[2,3])) # apply local operator
                res = np.tensordot(amp.conj().transpose(0,1,4,5,2,3), opc, axes=6)
                exp.append(np.real_if_close(res))
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            self.As = cache
            return exp
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = merge(self[idx],self[idx+1])   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=([2,3],[2,3])) # apply local operator
            res = np.tensordot(amp.conj().transpose(0,1,4,5,2,3), opc, axes=6)
            self.As = cache
            return np.real_if_close(res)
        
    def measure(self, op_list, idx=None):
        """Perform measurements successively on each site

        When only one operator is measured, one can call site_expectation_value().
        When there are more to be measured, this function will be faster bacause 
        here we only go through the entire system once and the measurements are 
        handled together .

        Parameters
        ----------
        op_list : list
            list of local operators
        idx : int or None
            if `idx` is None, the measurements are done on every site in order
            if `idx` is an integer, the measurements are done on the specified 
            site only

        Return
        ----------
        exp : (nested) list
            if `idx` is None, a nested list of same length as op_list will be 
            returned, with each sublist of length N, containing the expectation
            values
            if `idx` is an integer, a list of same length as op_list will be returned, 
            containing the expectation values
        """
        if idx:
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = self[idx]   # amplitude in the Schmidt basis
            res = []
            for op in op_list:
                opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
                res.append(np.real_if_close(np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)))
            self.As = cache
            return res
        else:
            cache = self.As
            exp = [[] for op in op_list]    # a nested list to store the measurement results
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                amp = self[i]   # amplitude in the Schmidt basis
                for j, op in enumerate(op_list):
                    opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
                    res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
                    exp[j].append(np.real_if_close(res))
                self[i], self[i+1] = qr_step(self[i], self[i+1]) # move the orthogonality center
            amp = self[-1]
            for j, op in enumerate(op_list):
                opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
                res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
                exp[j].append(np.real_if_close(res))
            self.As = cache
            return exp

    def to_density_matrix(self):
        r"""density matrix for the locally purified tensor network
        \rho = X  X^\dagger
        """
        return mul(self, self.hc()).to_matrix()
    
    def probabilities(self):
        """probability amplitudes of each state in the ensemble
        """
        Nstates = np.prod([self.krauss_dims])
        norms = np.zeros(Nstates)
        kspace = np.arange(Nstates).reshape(self.krauss_dims)
        for i in range(Nstates):
            loc = np.where(kspace==i)
            As = []
            assert len(self) == len(loc)
            for A, idx in zip(self, loc):
                As.append(A[:,:,:,idx])
            psi = MPS(As)
            norms[i] = np.linalg.norm(psi.as_array())**2        
        return norms
    
    def rho2trace(self):
        """tr(rho**2) = trace(rho*rho) = ||rho||_F ^2"""
        return np.linalg.norm(self.to_density_matrix())**2

def compress(psi:LPTN, tol:float, m_max:int, max_sweeps=2):
    """variationally compress a LPTN by optimizing the trace norm |X'-X|, where X' is 
    the guess state

    Parameters
    ----------
    psi : MPS
        the MPS to be compressed
    tol : float
        the largest truncated singular value
    m_max : int
        maximum bond dimension
    k_max : int
        maximun kraus dimension
    max_sweeps : int
        maximum optimization sweeps
    disentangle : Bool
        if True, fastDisentangle() is used on top

    Return
    ----------
    phi : MPS
        the compressed MPS, in the mixed canonial form centered on the 0-th site
        almost right canonical as one would say
    """
    N = len(psi)
    phi = deepcopy(psi)  # overwrite set to False, first copy then orthonormalize
    phi.orthonormalize('left')
    # peform a SVD sweep from the right to left
    for i in range(N-1,0,-1):
        di, dj, dd, dk = phi[i].shape
        phi[i] = np.reshape(phi[i], (di, dj*dd*dk))
        u, s, vt = np.linalg.svd(phi[i], full_matrices=False)
        mask = s>10*tol
        phi[i] = vt[mask,:].reshape(-1,dj,dd,dk)
        phi[i-1] = np.tensordot(u[:,mask]*s[mask], phi[i-1], axes=(0,1)).swapaxes(1,0)
    # now we arrive at a right canonical LPTN
    RBT = _load_right_bond_tensors(psi,phi)
    LBT = [np.ones((1,1))] * N
    for n in range(max_sweeps):
        for i in range(N-1): # sweep from left to right
            j = i+1
            temp = merge(psi[i],psi[j])
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            temp = np.tensordot(RBT[j], temp, axes=(0,1)).swapaxes(0,1)
            phi[i], phi[j] = split(temp, 'right', tol, m_max)
            # compute left bond tensor L[j]
            LBT[j] = np.tensordot(psi[i], LBT[i], axes=(0,0))
            LBT[j] = np.tensordot(LBT[j], phi[i].conj(), axes=([1,2,3],[2,3,0]))
        for j in range(N-1,0,-1):  # sweep from right to left
            i = j-1
            temp = merge(psi[i],psi[j])
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            temp = np.tensordot(RBT[j], temp, axes=(0,1)).swapaxes(0,1)
            phi[i], phi[j] = split(temp, 'left', tol, m_max)
            # compute right bond tensor R[i]
            RBT[i] = np.tensordot(psi[j], RBT[j], axes=(1,0))
            RBT[i] = np.tensordot(RBT[i], phi[j].conj(), axes=([1,2,3],[2,3,1]))
        overlap = np.tensordot(psi[0], RBT[0], axes=(1,0))
        overlap = np.tensordot(overlap, phi[0].conj(), axes=([1,2,3],[2,3,1]))
    return phi, overlap.item()

def _load_right_bond_tensors(psi:LPTN, phi:LPTN):
    """Calculate the right bond tensors while contracting two LPTNs.
    RBT[i] is to the right of the MPS[i].

    Parameters
    ----------
    psi : LPTN
        used as ket
    phi : LPTN 
        used as bra

    Return
    ----------
    RBT : list
        list of length N containing the right bond tensors, RBT[N-1] is trivial
    """
    assert len(psi) == len(phi)
    N = len(psi)
    RBT = [np.ones((1,1))] * N
    for i in range(N-1,0,-1):
        RBT[i-1] = np.tensordot(psi[i], RBT[i], axes=(1,0))
        RBT[i-1] = np.tensordot(RBT[i-1], phi[i].conj(), axes=([1,2,3],[2,3,1]))
    return RBT