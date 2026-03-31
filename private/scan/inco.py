# produce initial spectrum data

import numpy as np
import sys

def genspec_kinetic_mis(N, L, theta1, vheta1, hom = False):
    """
    Creates the IC for jAxions spax option in 'initialspectrum.dat' for the kinetic misalignment scenario.

    Args:
        N (int): Number of points per spatial dimension.
        L (float): Physical size of simulation box in L1 units.
        theta1 (float): Axion field value at η₁.
        vheta1 (float): Axion velocity/H1 value at η₁.
        hom (float, optional): If set to True, no fluctuations added. Defaults to False.

    Returns:
        (tuple):
            - k (ndarray): Array of Fourier wavenumbers corresponding to modes.
            - m (ndarray): Fourier amplitudes |FT{Ψ}| of the conformal axion field.
            - v (ndarray): Fourier amplitudes |FT{Ψ'}| of the conformal axion velocity field.
    """
    
    k0 = 2*np.pi/L # Smallest non-zero wavenumber
    k = k0*np.arange(2*N) # Full range of FFT wavenumbers

    m0 = theta1
    v0 = vheta1 + theta1

    # |FT{Ψ}| for jAxions
    m = np.empty_like(k)
    m[0] = m0 # Zero mode
    if hom:
        m[1:] = 0 * m[1:]
    else:
        m[1:] = 4.5 * 10**-5 * (L*k[1:])**(-3/2) * vheta1

    # |FT{Ψ'}| for jAxions
    v = np.zeros_like(m) 
    v[0] = v0 # Zero mode

    # Save to file
    xy = np.column_stack((m, v))
    np.savetxt('initialspectrum.dat', xy, delimiter = ' ', fmt = '%.8e %.8e') 
    
    return k, m, v

if len(sys.argv) == 5:
    N = int(sys.argv[1])
    L = float(sys.argv[2])
    theta1 = float(sys.argv[3])
    vheta1 = float(sys.argv[4])
else:
    print('usage:')
    print('python3 inco.py N L theta1 vheta1')
    sys.exit()

k, m, v = genspec_kinetic_mis(N, L, theta1, vheta1)
