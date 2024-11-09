# GPU_ETOM-2DES

GPU: Use GPU to conduct the caululation. In this project, we use CUDA.

ETOM: Effective Thermal Ocillator Model

2DES: 2 Dimensional Electronic Spectroscopy

Simulate 2DES of multilevel excitonic system using heirachical equation of motion (HEOM) with ETOM. It is far more efficient than traditional HEOM.

The code uses HEOM to calculate numerically exact quantum dynamics of multilevel quantum systems. The density matrix of a system is propagated using RK4 propagator. Moreover, the code handles multiple laser pulses and consider its pulse width which enables it to simulate electronic four-wave mixing signals such as three-pulse photon-echo peakshift measurements.

For how to setup and process the jobs, see INSTALL file.

Module: GPU_2DES: Calculates 2DES for multilevel excitonic system including Monte-Carlo Gaussian static disorders.
