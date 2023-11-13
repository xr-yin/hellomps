HelloMPS
==========
A developing project for simulating dynamics of 1D quantum systems using tensor networks.

The main project is named ``hellomps/`` is organized as follows: 
``networks/`` contains commonly used tensor network architectures, ``MPS`` for pure states and ``LPTN`` for mixed states.`operations.py` integrates a variety of tensor networks operations, which are essential to the simulation algorithms. `mpo_projected.py` offers the abstract interface between projected MPO and iterative solvers, which is the central unit to variational optimization problems, such as DMRG.

``models/`` provides models used for simulation, such as transverse field Ising model, Bose-Hubburd model, etc. This part is still quite basic as for now.

``algorithms/`` implements ground state serach algorithms for MPS, real and imaginary time evolution algorithms for both MPS and LPTN.

References
----------
1. Sebastian Paeckel, Thomas Köhler, Andreas Swoboda, Salvatore R. Manmana, Ulrich Schollwöck, Claudius Hubig,
`Time-evolution methods for matrix-product states <https://doi.org/10.1016/j.aop.2019.167998>`__
2. Werner, A. H. and Jaschke, D. and Silvi, P. and Kliesch, M. and Calarco, T. and Eisert, J. and Montangero, S.
`Positive Tensor Network Approach for Simulating Open Quantum Many-Body Systems <https://link.aps.org/doi/10.1103/PhysRevLett.116.237201>`__
3. Jutho Haegeman, Christian Lubich, Ivan Oseledets, Bart Vandereycken, and Frank Verstraete
`Unifying time evolution and optimization with matrix product states <https://journals.aps.org/prb/abstract/10.1103/PhysRevB.94.165116>`__
