#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__='Xianrui Yin'

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from numpy.linalg import norm

from .spin_chains import SpinChain
from .boson_chains import BosonChain
from ..networks.mps import MPS

__all__ = ['QubitCavity']

class QubitCavity(BosonChain,SpinChain):
    """
    Two qubits each coupled to a cavity, the two cavities are also coupled together.
    Sketch of the system
           alpha1    alpha2    alpha1
        S1 ------ C1 ------ C2 ------ S2

    where S stands for Spin qubit, C stands for Cavity.

    As default, we truncate the local Hilbert space dimension of cavities to 4, i.e. at most 3 excitations 
    and design the system to be symmetric. We may as well modify these assumptions.

    Parameters:
        alphas: a list containing coupling strengths between spins and cavities, cavity and cavity
        omegas: a list containing resonate frequencies of the cavities and the spins
        gamma: coupling strength between the system and the environment
    """
    def __init__(self, alphas: list, omegas:list, gamma:float) -> None:
        super().__init__(N=4, d=4)  # calls __init__() in the BosonChain class
        # in the following , self.num also refers to the class BosonChain
        self.alphas = alphas
        self.omegas = omegas
        self.gamma = gamma

    @property
    def hduo(self):
        a1, a2 = self.alphas
        wc, ws = self.omegas
        S1C1 = a1*(np.kron(self.splus,self.bn) + np.kron(self.sminus,self.bt))\
              + wc*np.kron(self.cid,self.num) + ws*np.kron(self.sz,self.bid)
        C2S2 = a2*(np.kron(self.bn,self.splus) + np.kron(self.bt,self.sminus))\
              + wc*np.kron(self.num,self.cid) + ws*np.kron(self.bid,self.sz)
        C1C2 = np.kron(self.bt, self.bn) + np.kron(self.bn, self.bt)
        return [S1C1, C1C2, C2S2]

    @property    
    def Lloc(self):
        r = [self.sminus,self.bn,self.bn,self.sminus]
        return [np.sqrt(self.gamma)*item for item in r]
    
    @property
    def ham_full(self):
        return sparse.kron(self.hduo[0],sparse.eye(4*2)) + sparse.kron(sparse.eye(2),sparse.kron(self.hduo[1],sparse.eye(2))) \
            + sparse.kron(sparse.eye(4*2),self.hduo[2])
    
    @property
    def L_full(self):
        Ll = self.Lloc
        res = sparse.kron(Ll[0], sparse.eye(4*4*2)) \
            + sparse.kron(sparse.eye(2), sparse.kron(Ll[1], sparse.eye(4*2))) \
            + sparse.kron(sparse.eye(2*4), sparse.kron(Ll[2], sparse.eye(2))) \
            + sparse.kron(sparse.eye(2*4*4), Ll[3])
        return res
    
    @property
    def Liouvillian(self):
        """
        Time evolution operator in Fock-Liouvillian space, mathematical expression
        ``
        """
        H = self.ham_full
        L = self.L_full
        assert H.shape == L.shape
        d = H.shape[0]
        Ls = sparse.kron(L,L.conj()) \
            - 0.5*(sparse.kron(L.conj().T@L, sparse.eye(d)) + sparse.kron(sparse.eye(d), L.T@L.conj())) \
            - 1j*sparse.kron(H,sparse.eye(d)) + 1j*sparse.kron(sparse.eye(d),H.T)
        return Ls
    
    def occupation(self,psi:MPS):
        return psi.site_expectation_value([(self.sz+self.cid)/2,self.num,self.num,(self.sz+self.cid)/2])

class hamiltonian(object):
    """
    class for 1D hamiltonian
    """
    def __init__(self, dim, L) -> None:
        self.L = L
        self.id = sparse.eye(dim)
        self.bt = sparse.diags([m**0.5 for m in range(1,dim)], -1)
        self.bn = sparse.diags([m**0.5 for m in range(1,dim)], +1)
        self.n = sparse.diags(range(dim))
        self.q = self.bt + self.bn

    def stretch(self, idx, op, format="csr"):
        op_list = [self.id]*self.L  # = [Id, Id, Id ...] with L entries
        op_list[idx] = op
        full = op_list[0]
        for op_i in op_list[1:]:
            full = sparse.kron(full, op_i, format=format)
        return full

class diagonal(hamiltonian):
    def __init__(self,dim,L):
        super().__init__(dim,L)

    def __call__(self,idx,omega,alpha):
        op = omega*self.n + alpha/2*self.n@(self.n-self.id)
        return self.stretch(idx, op, format="csr")
    
class off_diagonal(hamiltonian):
    def __init(self,dim,L):
        super().__init__(dim,L)

    def __call__(self,couple,g):
        i, j = couple
        return g*self.stretch(i,self.q,format="csc")@self.stretch(j,self.q,format="csc")

def envelope(t, T_rf, T_d):
    lam = np.pi / (2*T_rf)
    del_T = T_d-T_rf
    if t<0:
        raise ValueError('t must be greater than 0')
    elif t < T_rf:
        return np.sin(lam*t)
    elif t <= del_T:
        return 1.
    elif t <= T_d:
        return np.cos(lam*(t-del_T))
    else:
        return 0.

def flux(t, phi0=0.15, delta=0.075, freq_drive=1.088, T_rf=13., T_d=139.6):
# external flux in the unit of 2pi
    freq_drive *= 2*np.pi
    return phi0 + delta*envelope(t, T_rf, T_d)*np.cos(freq_drive*t)

def frequency(phi, Esum, d, E_C=5.529):
    phi *= 2*np.pi
    return (2*E_C*Esum)**0.5 * (np.cos(phi/2)**2 + d**2*np.sin(phi/2)**2)**0.25 / (2*np.pi)