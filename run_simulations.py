"""
Generates initial conditions and runs jAxions simulation for the kinetic misalignment axion scenario.
Uses kin_mis_utils for physics and pyaxions for simulation.
Saves output files in a directory named by simulation parameters.
"""

# Modules and functions
import os
import numpy as np
from kin_mis_utils import genspec_kinetic_mis
from pyaxions import simgen as sg

# Simulation parameters
N = 128 # Points/dimension
L = 3 # Box length in L1 units
R = 1.0 # Scale factor in R1 units
msa = 1.0 # String resolution
tauf = 3.0 # End of simulation 

# Axion parameters
fAGeV_theta1_vheta1_dict = {
    1e11: [(5.40, 102.70)]
}

for fAGeV, theta1_vheta1_list in fAGeV_theta1_vheta1_dict.items():
    for theta1, vheta1 in theta1_vheta1_list:

        print('>>> f = {:.2e} GeV <<<'.format(fAGeV))

        # Initial conditions
        k, m, v = genspec_kinetic_mis(N, L, R, theta1, vheta1)
    
        print(r'Initial velocity of zero mode of conformal field: {:.2f}.'.format(v[0]))
        print('Initial conditions created.')
    
        rank, jax = sg.simgen(N = N, L = L, msa = msa, zRANKS = 2, ctf = tauf, ic = 'spax', ict = 'spax', prec = 'single', 
                              dev = 'gpu', lap = 2, fftplan = 64, fA = fAGeV, vqcd = 'vqcdC', xtr = ' --ftype axion --mode0 1', 
                              cti = 1, meas = 0, spmask = 1, spKGV = 15, rmask = 4.0, p2Dmap = True, p2DmapPE = True,
                              slc = N//2, nologmpi = True, verbose = 0, sIter = 5, verb = False, dump = 1)
    
        sg.runsim(JAX = jax, RANK = rank, THR = 16//rank, USA = '', VERB = False, BONDEN = True)
    
        os.system('mv axion.log.0 log-run.txt initialspectrum.dat out')
        os.system('mv out out_N%d_L%d_fA%.0e_theta%.2f_vheta%.2f'%(N, L, fAGeV, theta1, vheta1))