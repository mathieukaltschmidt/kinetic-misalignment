"""
Module for computing axion cosmology observables in the kinetic misalignment scenario.

Functions:
    - evolution_kinetic_mis: Solves the axion field evolution with temperature-dependent potential.
    - genspec_kinetic_mis: Generates initial condition.
    - find_params: Finds pairs of initial field and velocity that gives the correct dark matter abundance.
                   Starts from vheta1 = 0 and walks forward.
    - find_all_axitons: Identifies axitons from a 2D energy density projection map.
"""

# Modules
import os
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.integrate import odeint
from scipy.optimize import minimize, minimize_scalar
from scipy.ndimage import maximum_filter
from pyaxions import jaxions as pa

def evolution_kinetic_mis(fAGeV, theta1, vheta1, tautab):
    """
    Solves the axion field equation of motion for a kinetic misalignment scenario within the jAxions cosmological framework
    and computes the resulting axion relic abundance.

    Args:
        fAGeV (float): Axion decay constant in GeV.
        theta1 (float): Axion field value at η₁.
        vheta1 (float): Axion velocity value at η₁.
        tautab (array_like): Array of normalized conformal time τ = η/η₁ over which to solve the EOM.

    Returns:
        (dict):
            - 'fAGeV': Axion decay constant in GeV. 
            - 'mAMeV': Axion mass at zero temperature in MeV.
            - 'T1MeV': Temperature T₁ in MeV.
            - 'H1MeV': Hubble parameter at T₁ in MeV.
            - 'R1': Scale factor at T₁.
            - 'TMeV_trap': Temperature at which axion becomes trapped (if occurs), else None.
            - 'HMeV_trap': Hubble parameter at which axion becomes trapped (if occurs), else None.
            - 'R_trap': Scale factor at which axion becomes trapped (if occurs), else None.
            - 'mAMeV_trap': Axion mass at which axion becomes trapped in MeV (if occurs), else None.
            - 'TMeVtab': Array of temperatures in MeV corresponding to tautab.
            - 'thetatab': Field values corresponding to tautab.
            - 'vhetatab': Velocity values corresponding to tautab.
            - 'Omegah2tab': Array of axion relic density Ωh² corresponding to tautab.
    """

    # ----- Constants -----
    GN = 1/(1.22e19*1e3)**2 # Newton’s gravitational constant in MeV⁻²

    # ----- Degrees of Freedom -----
    log10TMeV = np.array([0.00, 0.50, 1.00, 1.25, 1.60, 2.00, 2.15, 2.20, 2.40, 2.50, 3.00, 
                          4.00, 4.30, 4.60, 5.00, 5.45]) # log10 of temperature in MeV 
    
    gtab0 = np.array([10.71, 1.00228, 10.74, 1.00029, 10.76, 1.00048, 11.09, 1.00505,
                      13.68, 1.02159, 17.61, 1.02324, 24.07, 1.05423, 29.84, 1.07578,
                      47.83, 1.06118, 53.04, 1.04690, 73.48, 1.01778, 83.10, 1.00123,
                      85.56, 1.00389, 91.97, 1.00887, 102.17, 1.00750, 104.98, 1.00023])
    
    gtab1 = np.reshape(gtab0, (16, 2))
    
    grho = gtab1[:, 0] # Energy density dof
    gs = gtab1[:, 0]/gtab1[:, 1] # Entropy density dof

    # Interpolations
    igrho = CubicSpline(log10TMeV, grho)
    igs = CubicSpline(log10TMeV, gs)

    # ----- jAxions Cosmology -----
    jd = os.environ['JAXIONS_DIR']
    
    # Conformal time, scale factor, temperature in MeV, second derivative of scale factor, 
    # topological susceptibility in GeV⁴, log₁₀(χ/H²)/2
    eta, R, T, Rpp, chi, hlogchiH2 = np.loadtxt(jd + '/jaxions/include/cosmos/jaxi-cosmo.txt', skiprows = 2, unpack = True)
    leta = np.log10(eta)
    
    # Interpolations
    iT = CubicSpline(leta, np.log10(T))
    ichi = CubicSpline(leta, np.log10(chi))
    iR = CubicSpline(leta, np.log10(R))
    iRpp = CubicSpline(leta, Rpp)

    # Temperature, susceptibility, scale factor, and second derivative functions
    def T_func(eta):
        return 10**(iT(np.log10(eta)))
    def chi_func(eta):
        return 10**(ichi(np.log10(eta)))
    def R_func(eta):
        return 10**(iR(np.log10(eta)))
    def Rpp_func(eta):
        return iRpp(np.log10(eta))

    # ----- Find η₁ -----
    # Axion mass becomes dynamically relevant: χ(η₁)/H(η₁)² ≈ fA²
    def findeta1(fAGeV, hlogchiH2, leta):
        lfA = np.log10(fAGeV)
        lfA0, lfA2 = hlogchiH2[0], hlogchiH2[-1]
        leta0, leta2 = leta[0], leta[-1]
        sR = CubicSpline(leta, hlogchiH2)
        err = 1
        while (np.abs(err) > 1e-5):
            lslope = (lfA2-lfA0)/(leta2-leta0)
            leta1 = leta0 + (lfA-lfA0)/lslope
            lfA1 = sR(leta1)
            err = 1.0 - lfA1/lfA
            if (np.abs(lfA2-lfA) < np.abs(lfA0-lfA)):
                leta0, lfA0 = leta1, lfA1
            else:
                leta2, lfA2 = leta1, lfA1
        return 10**leta1

    eta1 = findeta1(fAGeV, hlogchiH2, leta)
    T1MeV = T_func(eta1)
    R1 = R_func(eta1)
    g1 = igrho(np.log10(T1MeV))
    # Hubble parameter at T₁ in MeV: H₁ = sqrt[(8πGN/3)*(π²/30)*g(T₁)*T₁⁴]
    H1MeV = np.sqrt(8 * np.pi * GN / 3 * np.pi**2 / 30 * g1 * T1MeV**4) 

    # ----- Axion Mass -----
    mAMeV = np.sqrt(chi[-1])/fAGeV * 10**3 # at T ≈ 0: mA = sqrt(χ(T→0))/fA

    # ----- Equation of Motion -----
    # Normalize χ(T)
    def chi_norm_func(tau):
        return chi_func(tau * eta1)/(H1MeV * 1e-3 * fAGeV)**2
        
    def dpsidtau(V, tau, eta1):        
        psi, psip = V
        chiT = chi_norm_func(tau)
        Rpp = Rpp_func(tau*eta1)
        RR = R_func(tau*eta1)/R_func(eta1) # RR = R/R(η₁)
    
        return [psip, Rpp*psi - chiT*RR**3 * np.sin(psi/RR)]
    
    def solveom(theta1, vheta1, eta1, tautab):
        tau1 = tautab[0] # Should be 1
        psip1 = vheta1 + theta1 # Conformal field
        psi1 = theta1 * tau1 - vheta1*(1-tau1) # Velocity of conformal field
        sol = odeint(dpsidtau, [psi1, psip1], tautab, args = (eta1,))
    
        return sol, tautab

    sol, tau = solveom(theta1, vheta1, eta1, tautab)

    Rtab = R_func(tau*eta1)/R1 # Normalized scale factor

    thetatab = sol[:, 0]/Rtab
    vhetatab = (sol[:, 1]/Rtab - sol[:, 0]/Rtab**2) # Derivative w.r.t. conformal time
    
    # ----- Trapping -----
    TMeVtab = np.array([T_func(tau * eta1) for tau in tautab]) # Temperatures corresponding to tautab values
    chinormtab = np.array([chi_norm_func(tau) for tau in tautab])
     
    KE = 1/2 * (vhetatab/Rtab)**2 # Kinetic energy term: (1/2) * θ̇^2
    PE = chinormtab * (1-np.cos(thetatab)) # Potential energy term: χ(T) * V(θ)
    energy_diff = KE - PE
    
    idx = np.where(np.diff(np.sign(energy_diff)))[0]
    if idx.size:
        TMeV_trap = TMeVtab[idx[0]]
        R_trap = R_func(tautab[idx[0]]*eta1)
        g_trap = igrho(np.log10(TMeV_trap))
        HMeV_trap = np.sqrt(8 * np.pi * GN / 3 * np.pi**2 / 30 * g_trap * TMeV_trap**4) 
        mAMeV_trap = np.sqrt(chi_func(tautab[idx[0]]*eta1))/fAGeV * 10**3
    else:
        TMeV_trap, R_trap, HMeV_trap, mAMeV_trap = None, None, None, None
        
    # ----- Axion Number Density -----
    # dimensionless number density na = (kinetic energy + potential energy) / sqrt(χ)
    # kinetic energy term: (1/2) * (θ̇)²
    # potential energy term: χ(T) * V(θ)
    na_t = (1/2 * (vhetatab/Rtab)**2 + chinormtab * (1-np.cos(thetatab))) / np.sqrt(chinormtab)

    R_t3 = (2.34e-4 * 1e-6 / TMeVtab)**3 * 3.91 / igs(np.log10(TMeVtab)) # Expansion factor of universe R³
    
    naR3_t = na_t * R_t3 # Comoving number density

    # ----- Axion Relic Density -----    
    rhocritMeV4 = 1.053e-5 * 1e3 * (1.97e-5 * 1e-6)**3 # Convert critical density from GeV/cm³ to MeV⁴ units
    constant = H1MeV * 1e-3 * fAGeV**2 * 10**12 / rhocritMeV4 # Convert to MeV⁴
    Omegah2tab = mAMeV * naR3_t  * constant # axion mass * comoving number density / critical density

    return {"fAGeV": fAGeV, "mAMeV": mAMeV, 
            "T1MeV": T1MeV, "H1MeV": H1MeV, "R1": R1,
            "TMeV_trap": TMeV_trap, "HMeV_trap": HMeV_trap, "R_trap": R_trap, "mAMeV_trap": mAMeV_trap,
            "TMeVtab": TMeVtab, "thetatab": thetatab, "vhetatab": vhetatab, "Omegah2tab": Omegah2tab}

def genspec_kinetic_mis(N, L, R, theta1, vheta1, hom = False):
    """
    Creates the IC for jAxions spax option in 'initialspectrum.dat' for the kinetic misalignment scenario.

    Args:
        N (int): Number of points per spatial dimension.
        L (float): Physical size of simulation box in L1 units.
        R (float): Scale factor at initial time in terms of R1.
        theta1 (float): Axion field value at η₁.
        vheta1 (float): Axion velocity value at η₁.
        hom (float, optional): If set to True, no fluctuations added. Defaults to False.

    Returns:
        (tuple):
            - k (ndarray): Array of Fourier wavenumbers corresponding to modes.
            - m (ndarray): Fourier amplitudes |FT{Ψ}| of the conformal axion field.
            - v (ndarray): Fourier amplitudes |FT{Ψ'}| of the conformal axion velocity field.
    """
    
    k0 = 2*np.pi/L # Smallest non-zero wavenumber
    k = k0*np.arange(2*N) # Full range of FFT wavenumbers

    m0 = theta1 * R
    v0 = R**2 * vheta1 + theta1

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

def find_params(fAGeV, tautab, vheta1_range, target = 0.12, tol = 0.005):
    """
    Finds values of theta1 and vheta1 such that the resulting dark matter abundance matches a target value.
    Starts from vheta1 = 0 and walks forward, adjusting theta1 using optimization.

    Parameters:
        fAGeV (float): Axion decay constant in GeV.
        tautab (array_like): Array of normalized conformal time τ = η/η₁ over which to solve the EOM.
        vheta1_range (tuple): vheta1 range and step.
        target (float, optional): Target dark matter abundance value to match. Defaults to 0.12.
        tol (float, optional): Tolerance for |omega - target| to consider a match. Defaults to 0.005.

    Returns:
        matches (list of tuples): [(theta1, vheta1, |omega - target|)] for all successful matches.
    """

    # Computes absolute difference between the computed abundance and target value
    def compute_omega(theta1, vheta1):
        result = evolution_kinetic_mis(fAGeV, theta1, vheta1, tautab)
        omega = np.mean(result['Omegah2tab'][-3:])
        return abs(omega - target)
    
    def find_all_theta_minima(vheta1, tol, sample_points = 100):
        theta1_min, theta1_max = 0, 2 * np.pi # Bounds of theta1
        theta1s = np.linspace(theta1_min, theta1_max, sample_points)
        errs = np.array([compute_omega(theta1, vheta1) for theta1 in theta1s]) # Differences to target
        
        derrs = np.diff(errs) / np.diff(theta1s) # Approximate derivatives
        sign_changes = (derrs[:-1] < 0) & (derrs[1:] > 0)
        candidate_indices = list(np.where(sign_changes)[0]) # Possible minima

        # --- Check near lower boundary ---
        if errs[0] < errs[1]:
            candidate_indices.insert(0, 0)
    
        # --- Check near upper boundary ---
        if errs[-1] < errs[-2]:
            candidate_indices.append(len(theta1s) - 2)

        matches = []
        for idx in candidate_indices:
            # Ensure left index is not negative
            left_idx = max(idx - 2, 0)
            right_idx = min(idx + 2, len(theta1s) - 1)
        
            left = theta1s[left_idx]
            right = theta1s[right_idx]
        
            # Swap if bounds are reversed
            if left > right:
                left, right = right, left

            res = minimize_scalar(compute_omega, bounds = (left, right), 
                                  args = (vheta1,), method = 'bounded', options = {'xatol': 1e-12}) # Optimize to find correct minima
            if res.success and res.fun < tol:
                matches.append((res.x, vheta1, res.fun))
        return matches

    matches = []

    vheta1_min, vheta1_max, vheta1_step = vheta1_range
    vheta1_vals = np.arange(vheta1_min, vheta1_max, vheta1_step)
    
    for vheta1 in vheta1_vals:
        found_minima = find_all_theta_minima(vheta1, tol)
        if found_minima:
            matches.extend(found_minima)
        else:
            return matches # Overabundance stop

    return matches

def find_all_axitons(mf, local_size, threshold_frac, flat = True):
    """
    Identify axitons as local maxima in a 2D energy density projection map above a given threshold.

    Args:
        mf (h5py.File): Measurement file.
        local_size (int): Size of the neighborhood for local maxima detection.
        threshold_frac (float): Fraction of the global maximum used as intensity threshold.
        flat (bool, optional): If True, return flattened indices; otherwise return (x, y) coordinates. Defaults to False.

    Returns:
        (np.ndarray): Array of axiton positions (flat indices or coordinates).
    """
    
    p2d = pa.gm(mf, '2DmapP')

    # Find local maxima of size local_size
    local_max = maximum_filter(p2d, size = local_size) == p2d

    # Set threshold
    threshold = threshold_frac * np.max(p2d)

    # Axiton mask
    axitons = (p2d > threshold) & local_max

    total_axitons = np.sum(axitons)
    print('Found a total of %d axitons.'%total_axitons)

    axiton_coordinates = np.argwhere(axitons) # Return 2D coordinate (x,y)
    flat_indices = np.ravel_multi_index(axiton_coordinates.T, np.shape(p2d))

    return flat_indices if flat else axiton_coordinates