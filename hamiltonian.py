import numpy as np

class eff_ham_mpo(object):
    '''
    effective model Hamiltonian for architecture I -- Eq.31
    '''


    '''
    --------------device parameters in units of 2*pi---------------

             qubit0----qubit1----qubit2
                    g0        g1
    '''
    def __init__(self, omega=[5.100, 8.100, 6.200], alpha=[-0.310/2, -0.235/2, -0.285/2], g_eff=[0.146, 0.164]) -> None:
        # bosonic creation operator
        self.bt = np.diag([1., np.sqrt(2)], k=-1)
        # bosonic creation operator
        self.bn = bt.transpose()
        # number operator
        self.n = np.kron(bt, bn)
        assert self.n == np.diag([0, 1., 2])
        # charge operator
        self.q = bt + bn
        # identity
        self.I = np.eye(3)
        # null matrix
        self.O = np.zeros((3,3))
        # qubit frequencies: E_1 - E_0
        self.omega = omega
        # anharmonicity: (E_2 - E_1)-(E_1 - E_0)
        self.alpha = alpha
        # effective coupling strength
        self.g_eff = g_eff
    
    onsite = [w*n + a*n@(a-I) for w,a in zip(omega, alpha)]
    inter = [g*q for g in g_eff]

    W0 = np.array([[onsite[0], inter[0], I]])
    W1 = np.array([
        [I, O, O],
        [q, O, O],
        [onsite[1], inter[q], I]
    ])
    W2 = np.array([
        [I],
        [q],
        [onsite[2]]
    ])