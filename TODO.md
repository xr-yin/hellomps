Near term:
1. Check the individual tensor in uMPO, if they are close to identities.
2. Incoporate more codes into `operation.py`, e.g. orthonormalize(), site_expectation_value(), ...
3. Disable caching when calculating expectation values and entropies?
4. Use a MPO approach in comparsion to brick-wall approach for `TEBD2`.
4.1 apply directly the two site unitaries without first splitting and assembling into a big unitary MPO. Compare
performances.
5. Inspect the norm profile of the LPTN.
6. How to incoporate int and numpy.int64? ---> Pass
7. Check optimal contraction scheme in compress(MPS). ---> Pass
8. How to set the tolerence while splitting a two-site tensor in our guess state during compression? Test on 3 regimes?
9. Combine codes for varational optimization. By creating a seperate class?
10. Create a class holding multiplication?

Middle term:
1. Write a Lanzcos method for (Hermitian) matrix exponential. (Might not matter much for small matrices.)
2. Implement quantum number conservation.

Long term:
1. Add support for parallel computation