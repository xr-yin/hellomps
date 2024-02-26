#!/usr/bin/env python3
# -*- coding: utf-8 -*-


__author__='Xianrui Yin'

import numpy as np

import math
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
            list of local two-site operators or a single local two-site operator
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
            self.As =cache
            return exp
        else:
            assert isinstance(idx, int)
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
            #return res

    def to_density_matrix(self):
        r"""density matrix for the locally purified tensor network
        \rho = X  X^\dagger
        """
        A = self.hc()
        B = self
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

def compress(psi:LPTN, tol:float, m_max:int, k_max:int, max_sweeps=2, disentangle=False):
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
            # dientangle
            phi[i], phi[j] = split(temp, 'right', tol, m_max)
            # compute left bond tensor L[j]
            LBT[j] = np.tensordot(psi[i], LBT[i], axes=(0,0))
            LBT[j] = np.tensordot(LBT[j], phi[i].conj(), axes=([1,2,3],[2,3,0]))
        for j in range(N-1,0,-1):  # sweep from right to left
            i = j-1
            temp = merge(psi[i],psi[j])
            temp = np.tensordot(LBT[i], temp, axes=(0,0))
            temp = np.tensordot(RBT[j], temp, axes=(0,1)).swapaxes(0,1)
            # dientangle
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

"""
The following was taken from K. Slagle, SciPost Phys. 11, 056 (2021)
but is currently not used in our library.
"""
def fastDisentangle(chi1, chi2, A, transposeQ=None):
    """Calculate a unitary tensor with shape (chi1, chi2, chi1*chi2)
    that approximately disentangles the input tensor 'A'.
    'A' must have the shape (chi1*chi2, chi3, chi4) where either
    chi2 <= ceil(chi4 / ceil(chi1/chi3)) or chi1 <= ceil(chi3 / ceil(chi2/chi4)).
    If these ratios are integers, then chi1*chi2 <= chi3*chi4 is sufficient.
    chi1 <= chi3 and chi2 <= chi4 is also sufficient.
    
    example: fastDisentangle(2, 3, randomComplex(6,5,7))
    """
    A = np.asarray(A)
    rand = randomComplex if np.iscomplexobj(A) else randomReal
    n,chi3,chi4 = A.shape
    if n != chi1*chi2:
        raise ValueError("fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4).")
    
    # implementing Appendix B in https://arxiv.org/pdf/2104.08283
    if chi1 > chi3:
        chi4to3 = math.ceil(chi1 / chi3)
        chi4new = math.ceil(chi4 / chi4to3)
        if not chi2 <= chi4new:
            raise ValueError("""
                fastDisentangle: The input array must have the shape (chi1*chi2, chi3, chi4) where either
                chi2 <= ceil(chi4 / ceil(chi1/chi3)) or
                chi1 <= ceil(chi3 / ceil(chi2/chi4)).""")
        _,_,V1  = np.linalg.svd(np.reshape(A, (n*chi3, chi4))) # 1
        V1      = V1.conj().T
        V       = np.pad(V1, ((0,0), (0, chi4to3*chi4new - chi4))) # 2
        V       = np.reshape(V, (chi4, chi4to3, chi4new))
        Anew    = np.reshape(np.tensordot(A, V, 1), (n, chi3*chi4to3, chi4new)) # 3
        return fastDisentangle(chi1, chi2, Anew, False)
    
    if chi2 > chi4:
        return np.transpose(fastDisentangle(chi2, chi1, np.transpose(A, (0,2,1))), (1,0,2))
    
    # implementing Algorithm 1 in https://arxiv.org/pdf/2104.08283
    r = rand(n) # 1
    alpha3, _, alpha4 = np.linalg.svd(np.tensordot(r, A, 1), full_matrices=False) # 2
    alpha3, alpha4 = np.conj(alpha3[:,0]), np.conj(alpha4[0,:])
    V3  = np.mat(np.linalg.svd(np.tensordot(A, alpha4,   1  ), full_matrices=False)[2][:chi1]).H # 3
    V4  = np.mat(np.linalg.svd(np.tensordot(A, alpha3, (1,0)), full_matrices=False)[2][:chi2]).H # 4
    B   = np.einsum("kab,ai,bj -> kij", A, V3, V4, optimize=True) # 5
    if transposeQ is None:
        transposeQ = chi1 > chi2
    Bdg = np.transpose(np.conj(B), (2,1,0) if transposeQ else (1,2,0))
    U   = np.reshape(orthogonalize(np.reshape(Bdg, (chi1*chi2, n))), Bdg.shape) # 6
    if transposeQ:
        U = np.swapaxes(U, 0, 1)
    return U

def entanglement(UA):
    """Compute the entanglement entropy of UA = np.tensordot(U, A, 1)"""
    # defined in equation (10) of https://arxiv.org/pdf/2104.08283
    UA = np.asarray(UA)
    chi1,chi2,chi3,chi4 = UA.shape
    lambdas  = np.linalg.svd(np.reshape(np.swapaxes(UA, 1, 2), (chi1*chi3, chi2*chi4)), compute_uv=False)
    ps  = lambdas*lambdas
    ps /= np.sum(ps)
    return max(0., -np.dot(ps, np.log(np.maximum(ps, np.finfo(ps.dtype).tiny))))

def randomReal(*ns):
    """Gaussian random array with dimensions ns"""
    return np.random.normal(size=ns)

def randomComplex(*ns):
    """Gaussian random array with dimensions ns"""
    return np.reshape(np.random.normal(scale=1/np.sqrt(2), size=(*ns,2)).view(np.complex128), ns)

def orthogonalize(M, reorthogonalizeQ=True):
    """Gram-Schmidt orthonormalization of the rows of M.
       Inserts random vectors in the case of linearly dependent rows."""
    M = np.array(M)
    rand = randomComplex if np.iscomplexobj(M) else randomReal
    epsMin = np.sqrt(np.finfo(M.dtype).eps) # once eps<epsMin, we add random vectors if needed
    eps = 0.5*np.sqrt(epsMin) # only accept new orthogonal vectors if their relative norm is at last eps after orthogonalization
    m,n = M.shape
    assert m <= n
    norms = np.linalg.norm(M, axis=1)
    maxNorm = np.max(norms)
    orthogQ = np.zeros(m, 'bool')
    allOrthog = False
    while not allOrthog:
        allOrthog = True
        eps1 = max(eps, epsMin)
        for i in range(m):
            if not orthogQ[i]:
                if norms[i] > eps1 * maxNorm:
                    Mi = M[i]
                    for j in range(m):
                        if orthogQ[j]:
                            Mi = Mi - M[j] * (np.conj(M[j]) @ Mi)
                    norm = np.linalg.norm(Mi)
                    if norm > eps1 * norms[i]:
                        M[i] = Mi / norm
                        orthogQ[i] = True
                        continue
                # M[i] was a linear combination of the other vectors
                if eps < epsMin:
                    M[i]  = rand(n)
                    M[i] *= maxNorm/np.linalg.norm(M[i])
                    norms[i] = maxNorm
                allOrthog = False
        eps = eps*eps
    if reorthogonalizeQ and np.linalg.norm(M * np.mat(M).H - np.eye(m)) > np.finfo(M.dtype).eps ** 0.75:
        return orthogonalize(M, False)
    return M 