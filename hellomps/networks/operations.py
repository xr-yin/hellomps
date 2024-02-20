import numpy as np
from scipy.linalg import qr, rq, norm
from copy import deepcopy
#from numba import jit

from ..networks.mpo_projected import *

__all__ = ['qr_step', 'rq_step', 'split', 'merge', 'mul', 'apply_mpo', 'zip_up']


def qr_step(ls,rs):
    r"""Move the orthogonality center one site to the right.
    
    Given two neighboring MPS (MPO) tensors as following,
          2,k         2,k
           |           |
        ---ls---    ---rs--- 
        0,i  1,j    0,i  1,j

    \        2,k           2,k       \
    \         |             |        \
    \     ----ls----    ----rs----   \
    \     0,i |  1,j    0,i | 1,j    \
    \        3,l           3,l       \
    compute the QR decompostion of ls, and multiply r with rs.

    Parameters
    ----------
    ls : ndarray, ndim==3 (or 4)
        local MPS tensor on the left, to be QR decomposed
    rs : ndarray, ndim==3 (or 4)
        local MPS tensor on the right

    Return
    --------
    ls_new : ndarray, ndim==3 (or 4)
        left orthonormal MPS tensor
    rs_new : ndarray, ndim==3 (or 4)
        new orthogonality cneter
    """
    if ls.ndim == 3:
        di, dj, dk = ls.shape
        ls = ls.swapaxes(1,2).reshape(-1,dj) # stick i,k together, first need to switch j,k
        # compute QR decomposition of the left matrix
        ls_new, _r = qr(ls, overwrite_a=True, mode='economic') 
        ls_new = ls_new.reshape(di,dk,-1).swapaxes(1,2)
    elif ls.ndim == 4:
        di, dj, dk, dl = ls.shape
        ls = ls.swapaxes(1,3).reshape(-1,dj) # stick i,k,l together, first need to switch j,k
        # compute QR decomposition of the left matrix
        ls_new, _r = qr(ls, overwrite_a=True, mode='economic') 
        ls_new = ls_new.reshape(di,dl,dk,-1).swapaxes(3,1)
    else:
        raise ValueError('the inputs must be both rank-3 or rank-4 tensors.')
    # multiply matrix R into the right matrix
    rs_new = np.tensordot(_r, rs, axes=1)
    return ls_new, rs_new

def rq_step(ls,rs):
    r"""Move the orthogonality center one site to the left.
    
    Given two neighboring MPS tensors as following,
          2,k         2,k
           |           |
        ---ls---    ---rs--- 
        0,i  1,j    0,i  1,j

    \        2,k           2,k       \
    \         |             |        \
    \     ----ls----    ----rs----   \
    \     0,i |  1,j    0,i | 1,j    \
    \        3,l           3,l       \
    compute the QR decompostion of ls, and multiply r with rs.

    Parameters
    ----------
    ls : ndarray, ndim==3 (or 4)
        local MPS tensor on the left
    rs : ndarray, ndim==3 (or 4)
        local MPS tensor on the right, to be RQ decomposed

    Return
    ----------
    ls_new : ndarray, ndim==3 (or 4)
        new orthogonality cneter
    rs_new : ndarray, ndim==3 (or 4)
        right orthonormal MPS tensor
    """
    if rs.ndim == 3:
        di, dj, dk = rs.shape
        rs = rs.reshape(di,-1)
        # compute RQ decomposition of the right matrix
        _r, rs_new = rq(rs, overwrite_a=True, mode='economic')
        rs_new = rs_new.reshape(-1,dj,dk)
        # multiply matrix R into the left matrix
        ls_new = np.tensordot(ls, _r, axes=(1,0)).transpose(0,2,1)
    elif rs.ndim == 4:
        di, dj, dk, dl = rs.shape
        rs = rs.reshape(di,-1)
        # compute RQ decomposition of the right matrix
        _r, rs_new = rq(rs, overwrite_a=True, mode='economic')
        rs_new = rs_new.reshape(-1,dj,dk,dl)
        # multiply matrix R into the left matrix
        ls_new = np.tensordot(ls, _r, axes=(1,0)).transpose(0,3,1,2)    
    return ls_new, rs_new

def orthonormalizer(self, mode:str, center_idx=None):
    r"""
    Transforming the MPS (MPO) into canaonical forms by doing successive QR decompositions.

    Parameters
    ----------
    mode : str
        'right', 'left', 'mixed'. When choosing 'mixed,' the corresponding index of the
        orthogonality center must be given
    center_idx : int
        the index of the orthogonality center

    Return
    ----------
    None

    Notes
    ----------
    scipy.linalg.qr, which we use here, only accepts 2-d arrays (matrices) as inputs to 
    be decomposed. Therefore, one must first combine the physical and matrix leg by doing 
    a reshape, before calling qr().
    
    On the other hand, numpy.linalg.qr can take in (N>2)-d arrays, which are regarded 
    as stacks of matrices residing on the last 2 dimensions. Consequently, one can call 
    qr() with the original tensors. In this regard, [physical, left bond, right bond] 
    indexing is preferred.
    """

    if mode == 'right':
        for i in range(self._N-1, 0,-1):
                self[i-1], self[i] = rq_step(self[i-1], self[i])
        #self[0], norm = rq_step(np.ones([1,1,1], self[0]))
        self[0] /= norm(self[0].squeeze())
    elif mode == 'left':
        for i in range(self._N - 1):
            self[i], self[i+1] = qr_step(self[i], self[i+1])
        #self[-1], norm = qr_step(self[-1], np.ones([1,1,1]))
        #norm = norm.ravel()
        self[-1] /= norm(self[-1].squeeze())
    elif mode == 'mixed':
        #assert isinstance(center_idx, int)
        assert center_idx >= 0
        assert center_idx < self._N
        for i in range(center_idx):
            self[i], self[i+1] = qr_step(self[i], self[i+1])
        for i in range(self._N-1,center_idx,-1):
            self[i-1], self[i] = rq_step(self[i-1], self[i])
        self[center_idx] /= norm(self[center_idx].squeeze())
    else:
            raise ValueError(
                'Mode argument should be one of left, right or mixed')

def split(theta, mode:str, tol:float, m_max=None, renormalize=True):
    '''
    split a local 4-tensor into two parts by doing a SVD 
    and discard the impertinent singular values
    '''
    if mode not in ["left", "right", "sqrt"]:
        raise ValueError('unknown mode')
    if theta.ndim == 4:
        di, dj, dk1, dk2 = theta.shape
        theta = np.swapaxes(theta, 1, 2)
        theta = np.reshape(theta, (di*dk1, dj*dk2))
        toggle = 'MPS'
    elif theta.ndim == 6:
        di, dj, dk1, dk2, dl1, dl2 = theta.shape
        theta = np.transpose(theta, (0,2,4,1,3,5))
        theta = np.reshape(theta, (di*dk1*dl1, dj*dk2*dl2))
        toggle = 'MPO'
    else:
        raise ValueError('Theta must have rank-4 (for MPS) or rank-6 (for MPO)')
    theta1, s, theta2 = np.linalg.svd(theta, full_matrices=False)
    s1 = s / np.linalg.norm(s)
    pivot = min(np.sum(s1>tol), m_max) if m_max else np.sum(s1>tol) # sum over the mask
    theta1, s, theta2 = theta1[:,:pivot], s[:pivot], theta2[:pivot,:]
    if renormalize:
        s = s / np.linalg.norm(s)
    if mode == 'left':
        theta1 *= s
    elif mode == 'right':
        theta2 = np.diag(s) @ theta2
    else:
        _s = np.sqrt(s)
        theta1 *= _s
        theta2 = np.diag(_s) @ theta2
    if toggle == 'MPS':
        theta1 = np.reshape(theta1, (di,dk1,-1))
        theta1 = np.swapaxes(theta1, 1, 2)
        theta2 = np.reshape(theta2, (-1,dj,dk2))
    else:
        theta1 = np.reshape(theta1, (di,dk1,dl1,-1))
        theta1 = np.transpose(theta1, (0,3,1,2))
        theta2 = np.reshape(theta2, (-1,dj,dk2,dl2))
    return theta1, theta2

def merge(theta1, theta2):
    '''
    merge two local tensors into one, the result has the following shape

            k1          k2             
            |           |
      i---theta1--//--theta2---j  
            |           |      
           (l1)        (l2)
    
                k1    k2
                |     |
          i------theta------j
                |     |
               (l1)  (l2)
    '''
    theta = np.tensordot(theta1, theta2, axes=(1,0))
    if theta1.ndim == 3:
        theta = np.swapaxes(theta, 2, 1)
    elif theta1.ndim == 4:
        theta = np.transpose(theta, (0,3,1,4,2,5)) # i,j,k1,k2,l1,l2
    return theta

def mul(A, B):
    """
    Calculate the product of a MPO and a MPS (or another MPO) by direct 
    contraction. The dimensions of the bonds will simply multiply. This 
    When B is a MPS, you should consider using apply_mpo() or zip-up() 
    instead for achieving optimzed bond dimension.

    Parameters:
        A: a MPO
        B: a MPO or a MPS

    Return:
        A x B: MPO (resp. MPS)
        
                    |k   output
                ----A----
    A x B   =       |k*, 3
                    |k , 2
                ----B----
                (   |k*  input)
    """
    
    from .mpo import MPO
    from .mps import MPS
    Os = []
    for a, b in zip(A.As, B.As): 
        a0, a1, a2 = a.shape[:3]
        b0, b1 = b.shape[:2]
        O = np.tensordot(a, b, axes=(3,2))
        O = np.swapaxes(O, 1, 3)
        O = np.reshape(O, (a0*b0, a2, a1*b1, -1))
        O = np.swapaxes(O, 1, 2)
        Os.append(O)
    return MPS([O[:,:,:,0] for O in Os]) if isinstance(B, MPS) else MPO(Os)

def apply_mpo(O, psi, tol:float, m_max:int, max_sweeps:int, overwrite=False):
    """Varationally calculate the product of a MPO and a MPS.

    Parameters
    ----------
    O : MPO
        the operator
    psi : MPS 
        the operand
    tol : float
        largest discarded singular value in each truncation step
    m_max : int 
        largest bond dimension allowed, default is None
    max_sweeps : int
        maximum number of optimization sweeps
    overwrite : Bool
        if True, psi will be overwritten with the result of the product

    Return
    ----------
    phi : LPTN or None
        if overwrite=True, return None, otherwise the result of the product
        will be returned
        
                    |k   output
                ----O----
    O |psi> =       |k*, 3
                    |k , 2
                ---psi----
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
            x = eff_O._matvec(x, vectorize=False)
            # split the result tensor
            phi[i], phi[j] = split(x, 'right', tol, m_max)
            # update the left bond tensor LBT[j]
            Ls.update(j, phi[i].conj(), psi[i], O[i])
        for j in range(N-1,0,-1): # sweep from right to left
            i = j-1
            x = merge(psi[i], psi[j])
            # contracting left block LBT[i]
            eff_O = ProjTwoSite(Ls[i], Rs[j], O[i], O[j])
            x = eff_O._matvec(x, vectorize=False)
            # split the result tensor
            phi[i], phi[j] = split(x, 'left', tol, m_max)
            # update the right bond tensor RBT[i]
            Rs.update(i, phi[j].conj(), psi[j], O[j])
        #logging.info('TODO: show norm')
    if overwrite:
        psi.As = phi.As
        del Ls
        del Rs
        del phi
    else:
        return phi

def zip_up(O, psi, tol, m_max=None, start='left'):
    r"""Zip-up method for contracting a MPO with a MPS

    Parameters
    ----------
    O : MPO
        the operator
    psi : MPS
        the operand, modified in place
    tol : float
        largest discarded singular value in each truncation step
    m_max : int or None
        largest bond dimension allowed, default is None
    start : str
        if 'left', the contraction (zipping) is performed from left
        to right. This should be used when the inital state is right
        canonical. For `start`='right', it is the other way around.

    Return
    ----------
    None
    """
    assert len(O) == len(psi)
    N = len(psi)
    if start == 'left':
        M = np.tensordot(psi.As[0], O.As[0], axes=([0,2],[0,3]))
        M = M[None,:,:,:] # s, m, w, k
        for i in range(N-1):
            M = np.transpose(M, (0,3,1,2)) # s,k,m,w
            s, k, m, w = M.shape
            M = np.reshape(M, (s*k, m*w)) # (s,k), (m,w)
            u, svals, vt = np.linalg.svd(M, full_matrices=False)
            svals = svals / np.linalg.norm(svals)
            pivot = min(np.sum(svals>tol), m_max) if m_max else np.sum(svals>tol)
            psi[i] = np.reshape(u[:,:pivot], (s, k, -1)) # s, k, s'
            psi[i] = np.swapaxes(psi.As[i], 1, 2)  # s, s', k
            M = np.reshape(np.diag(svals[:pivot]) @ vt[:pivot,:], (-1, m, w))
            M = np.tensordot(M, psi[i+1], axes=(1,0))
            M = np.tensordot(M, O[i+1], axes=([1,3],[0,3]))
        M = M[:,:,0,:]
        psi[N-1] = M / np.linalg.norm(M)
        # psi is now in the left canonical form
    elif start == 'right':
        M = np.tensordot(psi.As[-1], O.As[-1], axes=([1,2],[1,3]))
        M = M[:,:,None,:] # m, w, s, k        
        for i in range(N-1, 0, -1):
            m, w, s, k = M.shape
            M = np.reshape(M, (m*w, s*k))
            u, svals, vt = np.linalg.svd(M, full_matrices=False)
            svals = svals / np.linalg.norm(svals)
            pivot = min(np.sum(svals>tol), m_max) if m_max else np.sum(svals>tol)
            psi[i] = np.reshape(vt[:pivot,:], (-1, s, k)) # s', s, k
            M = np.reshape(u[:,:pivot] * svals[:pivot], (m, w, -1))
            M = np.tensordot(psi[i-1], M, axes=(1,0))
            M = np.tensordot(M, O[i-1], axes=([1,2],[3,1]))
            M = np.transpose(M, (0,2,1,3))
        M = M[:,0,:,:]
        psi[0] = M / np.linalg.norm(M)
        # psi is now in the right canonical form
    else:
        raise ValueError('start can only be left or right.')

if __name__ == "__main__":
    pass