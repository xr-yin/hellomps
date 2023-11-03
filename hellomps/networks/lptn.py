#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np
from scipy.linalg import norm

from .mps import MPS
from .mpo import MPO
from .operations import merge, mul

class LPTN(MPO):
    """
    class for locally purified tensor networks
    
    Parameters:
        As: list of local rank-4 tensors, each tensor has the following shape
                k
                |
            i---A---j            (i, j, k, l)
                |
                l
    """
    def __init__(self, As) -> None:
        super().__init__(As)

    @classmethod
    def gen_polarized_spin_chain(cls, N: int, polarization: str):
        if polarization not in ['+z','-z','+x']:
            raise ValueError('Only support polarization +z, -z or +x')
        A = np.zeros([1,1,2,1])
        if polarization == '+z':
            A[0,0,0] = 1.
        elif polarization == '-z':
            A[0,0,1] = 1.
        else:
            A[0,0,0] = A[0,0,1] = 0.5**0.5
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
        return [A.shape[3] for A in self.As]

    def site_expectation_value(self, op, idx=None):
        """
        if `idx` is None, `op` must be a list of ordered local operators for every physical site
        if `idx` is an integer, `op` must be a single operator for this specific physical site
        """
        if isinstance(op, list):
            assert len(op) == self._N
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                amp = self.As[i]   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=(2,1)) # apply local operator
                res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
                exp.append(np.real_if_close(res))
                self.As[i], self.As[i+1] = self._qr_step(self.As[i], self.As[i+1]) # move the orthogonality center
            amp = self.As[-1]
            opc = np.tensordot(amp, op[-1], axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
            exp.append(np.real_if_close(res))
            self.As =cache
            return exp
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = self.As[idx]   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=(2,1)) # apply local operator
            res = np.tensordot(amp.conj(), opc.swapaxes(2,3), axes=4)
            self.As = cache
            return np.real_if_close(res)
        
    def bond_expectation_value(self, op, idx=None):
        """
        if `idx` is None, `op` must be a list of ordered two-local operators for every pair of neighbouring site
        if `idx` is an integer, `op` must be a single two-local operator for this specific pair of neighbouring site
        """
        if isinstance(op, list):
            assert len(op) == self._N-1
            cache = self.As
            exp = []
            self.orthonormalize(mode='right')
            for i in range(self._N-1):
                j = i+1
                amp = merge(self.As[i], self.As[j])   # amplitude in the Schmidt basis
                opc = np.tensordot(amp, op[i], axes=([2,3],[2,3])) # apply local operator
                res = np.tensordot(amp.conj().transpose(0,1,4,5,2,3), opc, axes=6)
                exp.append(np.real_if_close(res))
                self.As[i], self.As[i+1] = self._qr_step(self.As[i], self.As[i+1]) # move the orthogonality center
            self.As = cache
            return exp
        else:
            assert isinstance(idx, int)
            cache = self.As
            self.orthonormalize(mode='mixed', center_idx = idx)
            amp = merge(self.As[idx],self.As[idx+1])   # amplitude in the Schmidt basis
            opc = np.tensordot(amp, op, axes=([2,3],[2,3])) # apply local operator
            res = np.tensordot(amp.conj().transpose(0,1,4,5,2,3), opc, axes=6)
            self.As = cache
            return np.real_if_close(res)
            #return res

    @property
    def to_density_matrix(self):
        return mul(self.hc, self).to_matrix()