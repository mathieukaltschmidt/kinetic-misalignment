import numpy as np
import sys
import os
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit
from pyaxions import jaxions as pa
from pyaxions import spectrum as sp
import warnings
warnings.filterwarnings('ignore')

# function to compute the numerical factor of the DM abundance for given f_A
# factor * R^3 * nA should give OmegaA*h^2
def factorOmega(fAGeV):
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
    #ichi = CubicSpline(leta, np.log10(chi))
    iR = CubicSpline(leta, np.log10(R))
    #iRpp = CubicSpline(leta, Rpp)

    # Temperature, susceptibility, scale factor, and second derivative functions
    def T_func(eta):
        return 10**(iT(np.log10(eta)))
    #def chi_func(eta):
    #    return 10**(ichi(np.log10(eta)))
    def R_func(eta):
        return 10**(iR(np.log10(eta)))
    #def Rpp_func(eta):
    #    return iRpp(np.log10(eta))

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
    gsr1 = igrho(np.log10(T1MeV))
    gss1 = igs(np.log10(T1MeV))
    GN = 1/(1.22093e19*1e3)**2 # Newton’s gravitational constant in 1/MeV^2
    H1MeV = np.sqrt(8 * np.pi * GN / 3 * np.pi**2 / 30 * gsr1 * T1MeV**4) # Hubble parameter at T1 in MeV
    mAMeV = np.sqrt(chi[-1])/fAGeV * 10**3 # mA at T0 in MeV
    
    gss0 = 3.971
    T0MeV = 2.34864e-10 # temperature today in MeV
    Omegagammah2 = 2.4728e-5 # photon density parameter today
    fA = fAGeV*10**3 # fA in MeV
    s1 = 2*np.pi**2*gss1*T1MeV**3/45 # entropy density at T1
    return (2/3)*gss0*Omegagammah2*(mAMeV/T0MeV)*H1MeV*fA**2/s1

if len(sys.argv) == 8:
    path = sys.argv[1]
    fAGeV = float(sys.argv[2])
    ii = int(sys.argv[3])
    jj = int(sys.argv[4])
    theta1 = float(sys.argv[5])
    vheta1 = float(sys.argv[6])
    path_outdir = sys.argv[7]
    if path[-1] != '/':
        path += '/'
    if path_outdir[-1] != '/':
        path_outdir += '/'
else:
    print('calculates axion number and various features for the kinetic misalignment scenario from jaxions measurement data at [path]')
    print('assuming theta1-vheta1 scan specified by indices (i,j) with i=index(theta1) and j=index(vheta1)')
    print('results are printed in results.dat and results_spectrum.dat files at [path_outdir]')
    print('data arrays of the evolution of axion abundance, energy, spectra, etc. are also saved as pickle files at [path]/out_analysis')
    print('usage:')
    print('python3 analysis.py [path] fAGeV index(t) index(v) theta1 vheta1 [path_outdir]')
    print('example:')
    print('python3 analysis.py ./ 1e+9 50 136 1.570796 1000 ./')
    sys.exit()

# -----------------------------------------------------------------------------------------------------
# Parameters not specified by the command line (modify them here if necessary)
#
# print progress massages (for debugging purposes)
verbose = False
# 
# Output data arrays; if True, data arrays of the evolution of axion abundance, energy, spectra, etc. are also saved as pickle files at [path]/out_analysis
outarrays = True
#
# List specifying the spectrum data output times
cts_spout = np.linspace(2.4,5,27)  # choose to output spactrum data at ct = 2.4,2.5,2.6,...,5.0
#
# Output map figures; if True, produce pdf files of theta and energy fluctuation maps and energy projection maps
# usetex option enables to print labels in tex format (depends on environments)
printmaps = False
usetex = False
# 
# If vheta1 is too large, initial fluctuations are so large that it misidentifies ct_sigmamax.
# To avoid this, we exclude the data at the initial time for the evaluation of the sigmamax if vheta1 becomes larger than the value specified below.
vheta1_crit = 1e+4
# 
# Fit range for extrapolation of the DM abundance
tstart = 4.0 
tend = 5.0
# -----------------------------------------------------------------------------------------------------

if verbose:
    print('reading data ...',end='',flush=True)

mf = pa.findmfiles(path)
if mf.size==0:
    print('(i,j)=({:d},{:d}): mfiles not found! Simulation failed?'.format(ii,jj),file=sys.stderr)
    sys.exit()

espCK = sp.espevol(mf,'espCK_0')
espG = sp.espevol(mf,'espG_0')
espS = sp.espevol(mf,'espS_0')
kl = espCK.avek
nm = espCK.nm
k_below = espCK.k_below # indices below Nyquist frequency

L = pa.gm(mf[0],'L')
N = pa.gm(mf[0],'sizeN')

ct = pa.gml(mf,'ct')
R = pa.gml(mf,'R')
T = pa.gml(mf,'T')
mA = pa.gml(mf,'massA')
eA = pa.gml(mf,'eA')
eAK = pa.gml(mf,'eAK')
eAG = pa.gml(mf,'eAG')
eAV = pa.gml(mf,'eAV')
bins = pa.gml(mf,'binthetaB')
binmax = pa.gml(mf,'binthetaBmax')
binmin = pa.gml(mf,'binthetaBmin')
binlen = pa.gml(mf,'binthetaBlen')

mAa = mA*R*L/N

psp = pa.gml(mf,'pspmasked_6.00')

eAMask = pa.gml(mf,'eAAxitmask6.00')
nmp = pa.gml(mf,'enmpAxitmask6.00')
eAm = (eA*N**3 - np.nan_to_num(eAMask)*nmp)/(N**3 - nmp) # masked energy density

if verbose:
    print('done',flush=True)

# theta and its fluctuations
if verbose:
    print('calculating mean theta and fluctuation ...',end='',flush=True)
meantheta = []
meantheta2 = []
fracout = []
for i in range(len(bins)):
    bw = (binmax[i]-binmin[i])/binlen[i]
    Nb = np.sum(bins[i])
    thetat = np.array([binmin[i]+bw*(ib+0.5) for ib in range(binlen[i])])
    meanthetat = np.sum(thetat*bins[i])/Nb
    bbins = bins[i][np.where(np.abs(thetat - meanthetat) > np.pi)]
    fo = np.sum(bbins)/Nb # fraction of points where |theta - <theta>| > pi
    meantheta.append(meanthetat)
    meantheta2.append(np.sum(thetat**2*bins[i])/Nb)
    fracout.append(fo)
theta_mean = np.array(meantheta)
theta_std = np.sqrt(np.array(meantheta2) - theta_mean**2)
fracout = np.array(fracout)
if verbose:
    print('done',flush=True)

# axion number from spectrum sum
if verbose:
    print('calculating axion number ...',end='',flush=True)
nat = []
nact = []
for it in range(len(ct)):
    w = np.sqrt(kl**2/R[it]**2 + mA[it]**2)
    wc = np.sqrt(kl**2/R[it]**2 + (1-theta_std[it]**2)*mA[it]**2)
    nat.append([(espCK.esp[it]/w).sum()/L**3/R[it]**4,(espG.esp[it]/w).sum()/L**3/R[it]**4,(espS.esp[it]/w).sum()/L**3/R[it]**4])
    nact.append([(espCK.esp[it]/wc).sum()/L**3/R[it]**4,(espG.esp[it]/wc).sum()/L**3/R[it]**4,(espS.esp[it]/wc).sum()/L**3/R[it]**4])
nat = np.array(nat)
nact = np.nan_to_num(np.array(nact)) # wc may become nan if theta_std is large

# approximate axion number from energy/mass
nae = eA/mA

# converting into Omega*h^2
factor = factorOmega(fAGeV)
Oa = factor*R[:,None]**3*nat
Oac = factor*R[:,None]**3*nact
Oae = factor*R**3*nae
if verbose:
    print('done',flush=True)

# fit
if verbose:
    print('fit and extrapolation ...',end='',flush=True)
def fitn(tdata,ndata,tstart=4.0,tend=5.0):
    def func(t,a1,a2):
        return 1/(1+a1*(1-1/t**a2))
    mask = tdata >= tstart
    tm = tdata[mask]
    nm = ndata[mask]
    iend = np.abs(tm - tend).argmin() # fit in the range tstart <= t < tend
    x = tm[:iend]/tm[0]
    y = nm[:iend]/nm[0]
    p, pv = curve_fit(func, x, y, p0=[1,7], maxfev = 20000)
    # return data of the best-fit curve
    nfit = nm[0]*func(tm/tm[0],*p)
    return p,pv,tm,nfit,nm[0]
Oatot = Oa[:,0] + Oa[:,1] + Oa[:,2] # K+G+V
Oactot = Oac[:,0] + Oac[:,1] + Oac[:,2] # K+G+V
peres,pveres,tf,Oaef,fprefactore = fitn(ct,Oae,tstart,tend)
try:
    pres,pvres,tf,Oaf,fprefactor = fitn(ct,Oatot,tstart,tend)
except:
    pres = np.full_like(peres,np.nan)
    pvres = np.full_like(pveres,np.nan)
    Oaf = np.full_like(Oaef,np.nan)
    fprefactor = np.nan
    print('fit OmegaA failed! (theta1 = %f, vheta1 = %f)'%(theta1,vheta1))
try:
    pcres,pvcres,tf,Oacf,fprefactorc = fitn(ct,Oactot,tstart,tend)
except:
    pcres = np.full_like(peres,np.nan)
    pvcres = np.full_like(pveres,np.nan)
    Oacf = np.full_like(Oaef,np.nan)
    fprefactorc = np.nan
    print('fit OmegaA_c failed! (theta1 = %f, vheta1 = %f)'%(theta1,vheta1))

# extrapolation and its error
Oaext = fprefactor/(1+pres[0])
Oacext = fprefactorc/(1+pcres[0])
Oaeext = fprefactore/(1+peres[0])
# we simply estimate the error from variance of a1 only (df/da2 vanishes in the limit t -> infinity)
Oaext_err = fprefactor*np.sqrt(pvres[0,0])/(1+pres[0])**2
Oacext_err = fprefactorc*np.sqrt(pvcres[0,0])/(1+pcres[0])**2
Oaeext_err = fprefactore*np.sqrt(pveres[0,0])/(1+peres[0])**2
if verbose:
    print('done',flush=True)

# determine trapping time from energy including fluctuations
if verbose:
    print('estimating trapping/stopping/sigma_max times ...',end='',flush=True)
itrap = np.where(np.diff(np.sign(eAK - eAV)))[0]
if itrap.size:
    ct_trap = ct[itrap[0]]
else:
    print('Warning: trapping time is not found (u. energy sum)!')
    ct_trap = np.nan

# determine trapping time from zero mode contribution only
eAK0 = espCK.esp[:,0]/R**4/L**3
eAG0 = espG.esp[:,0]/R**4/L**3
eAV0 = espS.esp[:,0]/R**4/L**3
itrap0 = np.where(np.diff(np.sign(eAK0 - eAV0)))[0]
if itrap.size:
    ct_trap0 = ct[itrap0[0]]
else:
    print('Warning: trapping time is not found (u. zero mode only)!')
    ct_trap0 = np.nan

# determine stopping time
istop = np.where(np.diff(np.sign(np.gradient(theta_mean))))[0]
if istop.size:
    ct_stop, R_stop, mA_stop = ct[istop[0]], R[istop[0]], mA[istop[0]]
    theta_stop = theta_mean[istop[0]]
    theta_std_stop = theta_std[istop[0]]
else:
    ct_stop, theta_stop, theta_std_stop = np.nan, np.nan, np.nan
    print('Warning: stopping time is not found!')

# determine time at maximum theta fluctuation
if vheta1 > vheta1_crit:
    imaxt = np.argmax(theta_std[1:])
    ct_sigmamax, R_sigmamax, mA_sigmamax = ct[1:][imaxt], R[1:][imaxt], mA[1:][imaxt]
    theta_sigmamax = theta_mean[1:][imaxt]
    theta_std_sigmamax = theta_std[1:][imaxt]
else:
    imaxt = np.argmax(theta_std)
    ct_sigmamax, R_sigmamax, mA_sigmamax = ct[imaxt], R[imaxt], mA[imaxt]
    theta_sigmamax = theta_mean[imaxt]
    theta_std_sigmamax = theta_std[imaxt]

# determine dilution times
fzero = (eAK0+eAG0+eAV0)/(eAK+eAG+eAV)
if50 = np.where(fzero < 0.5)[0]
if if50.size:
    ct_f50 = ct[if50[0]]
else:
    ct_f50 = np.nan
if10 = np.where(fzero < 0.1)[0]
if if10.size:
    ct_f10 = ct[if10[0]]
else:
    ct_f10 = np.nan

if verbose:
    print('done',flush=True)

if verbose:
    print('analyzing spectrum features ...',end='',flush=True)

# identify spectrum peaks
ANpeak = []
kNpeak = []
APpeak = []
kPpeak = []
NspK = []
NspG = []
NspV = []
Delta2 = []
for i in range(len(psp)):
    NaK = kl**3*espCK.esp[i]/(2*np.pi**2)/np.sqrt(mA[i]**2*R[i]**2 + kl**2)/nm/vheta1
    NaG = kl**3*espG.esp[i]/(2*np.pi**2)/np.sqrt(mA[i]**2*R[i]**2 + kl**2)/nm/vheta1
    NaV = kl**3*espS.esp[i]/(2*np.pi**2)/np.sqrt(mA[i]**2*R[i]**2 + kl**2)/nm/vheta1
    Na = NaK + NaG + NaV
    D2 = kl**3*psp[i]/nm/(np.pi**2)/eAm[i]**2
    # we exclude k above the Nyquist frequency when identifying peaks
    Nab = Na[k_below]
    D2b = D2[k_below]
    klb = kl[k_below]
    iNpeak = np.argmax(Nab)
    iPpeak = np.argmax(D2b)
    ANpeak.append(Nab[iNpeak])
    kNpeak.append(klb[iNpeak])
    APpeak.append(D2b[iPpeak])
    kPpeak.append(klb[iPpeak])
    NspK.append(NaK)
    NspG.append(NaG)
    NspV.append(NaV)
    Delta2.append(D2)
ANpeak = np.array(ANpeak)
kNpeak = np.array(kNpeak)
APpeak = np.array(APpeak)
kPpeak = np.array(kPpeak)
NspK = np.array(NspK)
NspG = np.array(NspG)
NspV = np.array(NspV)
Delta2 = np.array(Delta2)

# identify peak values at ct_sigmamax
ANpeak_sigmamax, APpeak_sigmamax = ANpeak[imaxt], APpeak[imaxt]
kNpeak_sigmamax, kPpeak_sigmamax = kNpeak[imaxt], kPpeak[imaxt]

# find maximum and minimum amplitudes at ct >= ct_sigmamax
imaxN = np.argmax(ANpeak[ct >= ct_sigmamax])
ct_Npeakmax = ct[ct >= ct_sigmamax][imaxN]
ANpeakmax, kNpeakmax, AP_Npeakmax, kP_Npeakmax = ANpeak[ct >= ct_sigmamax][imaxN], kNpeak[ct >= ct_sigmamax][imaxN], APpeak[ct >= ct_sigmamax][imaxN], kPpeak[ct >= ct_sigmamax][imaxN]
iminN = np.argmin(ANpeak[ct >= ct_sigmamax])
ct_Npeakmin = ct[ct >= ct_sigmamax][iminN]
ANpeakmin, kNpeakmin, AP_Npeakmin, kP_Npeakmin = ANpeak[ct >= ct_sigmamax][iminN], kNpeak[ct >= ct_sigmamax][iminN], APpeak[ct >= ct_sigmamax][iminN], kPpeak[ct >= ct_sigmamax][iminN]
imaxP = np.argmax(APpeak[ct >= ct_sigmamax])
ct_Ppeakmax = ct[ct >= ct_sigmamax][imaxP]
APpeakmax, kPpeakmax, AN_Ppeakmax, kN_Ppeakmax = APpeak[ct >= ct_sigmamax][imaxP], kPpeak[ct >= ct_sigmamax][imaxP], ANpeak[ct >= ct_sigmamax][imaxP], kNpeak[ct >= ct_sigmamax][imaxP]
iminP = np.argmin(APpeak[ct >= ct_sigmamax])
ct_Ppeakmin = ct[ct >= ct_sigmamax][iminP]
APpeakmin, kPpeakmin, AN_Ppeakmin, kN_Ppeakmin = APpeak[ct >= ct_sigmamax][iminP], kPpeak[ct >= ct_sigmamax][iminP], ANpeak[ct >= ct_sigmamax][iminP], kNpeak[ct >= ct_sigmamax][iminP]

if verbose:
    print('done',flush=True)

# produce theta fluctuation and energy maps
if printmaps:
    if verbose:
        print('producing theta fluctuation and energy maps ...',end='',flush=True)
    theta_maps = []
    E_maps = []
    PE_maps = []
    ct_maps = []
    mfmap = [mt for mt in mf if pa.gm(mt,'map?')]
    for mt in mfmap:
        theta_maps.append(pa.gm(mt,'maptheta'))
        E_maps.append(pa.gm(mt,'2Dmape'))
        PE_maps.append(pa.gm(mt,'2DmapP'))
        ct_maps.append(pa.gm(mt,'ct'))
    theta_maps = np.array(theta_maps)
    E_maps = np.array(E_maps)
    PE_maps = np.array(PE_maps)
    ct_maps = np.array(ct_maps)

    try:
        os.makedirs(path+'pics')
    except FileExistsError:
        pass

    import matplotlib.pyplot as plt
    if usetex:
        from matplotlib import rc
        rc('text', usetex=True)
    for mid in np.arange(len(ct_maps)):
        plt.clf()
        plt.figure(figsize=(6,6))
        theta_avg = np.mean(theta_maps[mid])
        fluc_maps = theta_maps[mid] - theta_avg
        plt.imshow(fluc_maps, cmap='viridis')
        if usetex:
            plt.colorbar().set_label(r'$\theta - \bar{\theta}$',fontsize=12)
            plt.xlabel(r'$x$',fontsize=12)  
            plt.ylabel(r'$y$',fontsize=12)
            plt.title(r'$\theta\ \mathrm{fluctuation\ map}\ (\tau = %.2f)$'%ct_maps[mid],fontsize = 12)
        else:
            plt.colorbar().set_label('theta - <theta>',fontsize=12)
            plt.xlabel('x',fontsize=12)  
            plt.ylabel('y',fontsize=12)
            plt.title('theta fluctuation map (tau = %.2f)'%ct_maps[mid],fontsize = 12)
        plt.savefig(path+'pics/flucmap_ct%.2f'%ct_maps[mid]+'.pdf',format='pdf')
        plt.close()
        plt.clf()
        plt.figure(figsize=(6,6))
        E_avg = np.mean(E_maps[mid])
        Efluc_maps = E_maps[mid]/E_avg - 1
        plt.imshow(Efluc_maps, cmap='viridis')
        if usetex:
            plt.colorbar().set_label(r'$(\rho-\bar{\rho})/\bar{\rho}$',fontsize=12)
            plt.xlabel(r'$x$',fontsize=12)  
            plt.ylabel(r'$y$',fontsize=12)
            plt.title(r'$\mathrm{Energy\ density\ fluctuation\ map}\ (\tau = %.2f)$'%ct_maps[mid],fontsize = 12)
        else:
            plt.colorbar().set_label('(rho-<rho>)/<rho>',fontsize=12)
            plt.xlabel('x',fontsize=12)  
            plt.ylabel('y',fontsize=12)
            plt.title('Energy density fluctuation map (tau = %.2f)'%ct_maps[mid],fontsize = 12)
        plt.savefig(path+'pics/Eflucmap_ct%.2f'%ct_maps[mid]+'.pdf',format='pdf')
        plt.close()
        plt.clf()
        plt.figure(figsize=(6,6))
        plt.imshow(np.log(PE_maps[mid]), cmap='viridis')
        if usetex:
            plt.xlabel(r'$x$',fontsize=12)  
            plt.ylabel(r'$y$',fontsize=12)
            plt.title(r'$\mathrm{Energy\ density\ projection\ map}\ (\tau = %.2f)$'%ct_maps[mid],fontsize = 12)
        else:
            plt.xlabel('x',fontsize=12)  
            plt.ylabel('y',fontsize=12)
            plt.title('Energy density projection map (tau = %.2f)'%ct_maps[mid],fontsize = 12)
        plt.savefig(path+'pics/PEmap_ct%.2f'%ct_maps[mid]+'.pdf',format='pdf')
        plt.close()
    if verbose:
        print('done',flush=True)

# output arrays
if outarrays:
    if verbose:
        print('output arrays ...',end='',flush=True)
    try:
        os.makedirs(path+'out_analysis')
    except FileExistsError:
        pass
    outarrayname = path+'out_analysis/ml'
    sp.sdata(ct,outarrayname,'ct')
    sp.sdata(R,outarrayname,'R')
    sp.sdata(T,outarrayname,'T')
    sp.sdata(mA,outarrayname,'mA')
    sp.sdata(mAa,outarrayname,'mAa')
    sp.sdata(eAK,outarrayname,'eAK')
    sp.sdata(eAG,outarrayname,'eAG')
    sp.sdata(eAV,outarrayname,'eAV')
    sp.sdata(eAK0,outarrayname,'eAK0') # kinetic energy (zero mode contribution only)
    sp.sdata(eAG0,outarrayname,'eAG0') # gradient energy (zero mode contribution only)
    sp.sdata(eAV0,outarrayname,'eAV0') # potential energy (zero mode contribution only)
    sp.sdata(eAm,outarrayname,'eAm') # masked energy
    sp.sdata(theta_mean,outarrayname,'theta_mean')
    sp.sdata(theta_std,outarrayname,'theta_std')
    sp.sdata(fracout,outarrayname,'f_out')
    sp.sdata(fracout*L**3*R**3/(np.pi/mA)**3,outarrayname,'V_out') # Volume of the simulation where |theta - <theta>| > pi in units of typical axiton size ~ (pi/mA)^3
    sp.sdata(Oa[:,0],outarrayname,'OmegaAh2_K')
    sp.sdata(Oa[:,1],outarrayname,'OmegaAh2_G')
    sp.sdata(Oa[:,2],outarrayname,'OmegaAh2_V')
    sp.sdata(Oac[:,0],outarrayname,'OmegaAh2_c_K')
    sp.sdata(Oac[:,1],outarrayname,'OmegaAh2_c_G')
    sp.sdata(Oac[:,2],outarrayname,'OmegaAh2_c_V')
    sp.sdata(Oae,outarrayname,'OmegaAh2_est')
    sp.sdata(tf,outarrayname,'ct_fit')
    sp.sdata(Oaf,outarrayname,'OmegaAh2_fit')
    sp.sdata(Oacf,outarrayname,'OmegaAh2_c_fit')
    sp.sdata(Oaef,outarrayname,'OmegaAh2_est_fit')
    sp.sdata(ANpeak,outarrayname,'Amp_Npeak')
    sp.sdata(APpeak,outarrayname,'Amp_Ppeak')
    sp.sdata(kNpeak,outarrayname,'k_Npeak')
    sp.sdata(kPpeak,outarrayname,'k_Ppeak')
    # output spectra
    id_sample = np.abs(ct[:,None]-cts_spout[None,:]).argmin(axis=0) # find indices closest to specified times
    sp.sdata(NspK[id_sample],outarrayname,'Nsp_K')
    sp.sdata(NspG[id_sample],outarrayname,'Nsp_G')
    sp.sdata(NspV[id_sample],outarrayname,'Nsp_V')
    sp.sdata(Delta2[id_sample],outarrayname,'Delta2')
    sp.sdata(kl,outarrayname,'k')
    sp.sdata(ct[id_sample],outarrayname,'ct_sp')
    sp.sdata(R[id_sample],outarrayname,'R_sp')
    sp.sdata(T[id_sample],outarrayname,'T_sp')
    sp.sdata(mA[id_sample],outarrayname,'mA_sp')
    # output map data (unnecessary)
    #if printmaps:
    #    sp.sdata(theta_maps,outarrayname,'map_theta')
    #    sp.sdata(E_maps,outarrayname,'map_E')
    #    sp.sdata(PE_maps,outarrayname,'map_PE')
    #    sp.sdata(ct_maps,outarrayname,'map_ct')
    if verbose:
        print('done',flush=True)

# output results
if verbose:
    print('writing results into files ...',end='',flush=True)

try:
    os.makedirs(path_outdir)
except FileExistsError:
    pass
is_new = not os.path.exists(path_outdir+'results.dat')
with open(path_outdir+'results.dat','a') as f:
    if is_new:
        f.write('# Results for the field evolution and DM abundance \n')
        f.write('# \n')
        #f.write('# 1. fAGeV  2. theta1  3. vheta1  4. sizeN  5. sizeL  ')
        f.write('# 1. index(t)  2. index(v)  3. theta1  4. vheta1  5. fAGeV \n')
        f.write('# 6. ct_trap (u. e sum)  7. ct_trap (u. zero mode)  8. ct_stop  9. theta_s  10. sigmatheta_s \n')
        f.write('# 11. ct_sigmamax  12. theta_sigmamax  13. sigmathetamax \n')
        f.write('# 14. ct_dilution (50%)  15. ct_dilution (10%)  16. rho0/rhotot \n')
        f.write('# 17. Omegah2  18. Omegah2_corrected  19. Omegah2_estimate  20. Omegah2(ext)  21. Omegah2_corrected(ext)  22. Omegah2_estimate(ext) \n')
        f.write('# 23. Omegah2(ext)_err  24. Omegah2_corrected(ext)_err  25. Omegah2_estimate(ext)_err \n')
        f.write('# \n')
    #f.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(fAGeV,theta1,vheta1,N,L))
    f.write('{:d} {:d} {:.6g} {:.6g} {:.6g} '.format(ii,jj,theta1,vheta1,fAGeV))
    f.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(ct_trap,ct_trap0,ct_stop,theta_stop,theta_std_stop,ct_sigmamax,theta_sigmamax,theta_std_sigmamax))
    f.write('{:.6g} {:.6g} {:.6g} '.format(ct_f50,ct_f10,fzero[-1]))
    f.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(Oatot[-1],Oactot[-1],Oae[-1],Oaext,Oacext,Oaeext))
    f.write('{:.6g} {:.6g} {:.6g}\n'.format(Oaext_err,Oacext_err,Oaeext_err))

# output spectrum features in another file
is_new_sp = not os.path.exists(path_outdir+'results_spectrum.dat')
with open(path_outdir+'results_spectrum.dat','a') as fsp:
    if is_new_sp:
        fsp.write('# Results for spectrum features \n')
        fsp.write('# \n')
        fsp.write('# 1. index(t)  2. index(v)  3. theta1  4. vheta1  5. fAGeV \n')
        fsp.write('# 6. k_Npeak(t_sigmamax)  7. Npeak(t_sigmamax)  8. k_Delta2peak(t_sigmamax)  9. Delta2peak(t_sigmamax) \n')
        fsp.write('# 10. k_Npeak(t_end)  11. Npeak(t_end)  12. k_Delta2peak(t_end)  13. Delta2peak(t_end) \n')
        fsp.write('# 14. ct_Npeakmax  15. k_Npeak(t_Npeakmax)  16. Npeak(t_Npeakmax)  17. k_Delta2peak(t_Npeakmax)  18. Delta2peak(t_Npeakmax) \n')
        fsp.write('# 19. ct_Npeakmin  20. k_Npeak(t_Npeakmin)  21. Npeak(t_Npeakmin)  22. k_Delta2peak(t_Npeakmin)  23. Delta2peak(t_Npeakmin) \n')
        fsp.write('# 24. ct_Delta2peakmax  25. k_Npeak(t_Delta2peakmax)  26. Npeak(t_Delta2peakmax)  27. k_Delta2peak(t_Delta2peakmax)  28. Delta2peak(t_Delta2peakmax) \n')
        fsp.write('# 29. ct_Delta2peakmin  30. k_Npeak(t_Delta2peakmin)  31. Npeak(t_Delta2peakmin)  32. k_Delta2peak(t_Delta2peakmin)  33. Delta2peak(t_Delta2peakmin) \n')
        fsp.write('# 34. R_stop  35. mA_stop  36. R_sigmamax  37. mA_sigmamax \n')
        fsp.write('# \n')
        fsp.write('# All ks are in ADM units. The comoving axion number N is in units of f_a^2*vheta_1 (PQ charge density). \n')
        fsp.write('# \n')
    fsp.write('{:d} {:d} {:.6g} {:.6g} {:.6g} '.format(ii,jj,theta1,vheta1,fAGeV))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} '.format(kNpeak_sigmamax,ANpeak_sigmamax,kPpeak_sigmamax,APpeak_sigmamax))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} '.format(kNpeak[-1],ANpeak[-1],kPpeak[-1],APpeak[-1]))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(ct_Npeakmax,kNpeakmax,ANpeakmax,kP_Npeakmax,AP_Npeakmax))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(ct_Npeakmin,kNpeakmin,ANpeakmin,kP_Npeakmin,AP_Npeakmin))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(ct_Ppeakmax,kN_Ppeakmax,AN_Ppeakmax,kPpeakmax,APpeakmax))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} '.format(ct_Ppeakmin,kN_Ppeakmin,AN_Ppeakmin,kPpeakmin,APpeakmin))
    fsp.write('{:.6g} {:.6g} {:.6g} {:.6g}\n'.format(R_stop,mA_stop,R_sigmamax,mA_sigmamax))

# output fit results in auxilialy file
is_new_a = not os.path.exists(path_outdir+'aux.dat')
with open(path_outdir+'aux.dat','a') as fa:
    if is_new_a:
        fa.write('# 1. index(t)  2. index(v)  3. theta1  4. vheta1  5. fAGeV \n')
        fa.write('# 6. a1  7. a2  8. a1(corrected)  9. a2(corrected)  10. a1(estimate)  11. a2(estimate) \n')
        fa.write('# \n')
    fa.write('{:d} {:d} {:.6g} {:.6g} {:.6g} '.format(ii,jj,theta1,vheta1,fAGeV))
    fa.write('{:.6g} {:.6g} {:.6g} {:.6g} {:.6g} {:.6g}\n'.format(pres[0],pres[1],pcres[0],pcres[1],peres[0],peres[1]))

if verbose:
    print('done',flush=True)
    print('')

# print summary
print('-----------------------------------------------------------------------')
print('')
print(' Analysis done')
print('')
print(' f_A [GeV]                       {:.3e}'.format(fAGeV))
print(' sim index (i,j)                 ({:d},{:d})'.format(ii,jj))
print(' theta1                          {:.6g}'.format(theta1))
print(' vheta1                          {:.6g}'.format(vheta1))
print(' m_A*a (at t_end)                {:.6g}'.format(mAa[-1]))
print('')
print(' t_trap (using energy sum)       {:.6g}'.format(ct_trap))
print(' t_trap (using zero mode only)   {:.6g}'.format(ct_trap0))
print(' t_stop                          {:.6g}'.format(ct_stop))
print(' t_sigmatheta_max                {:.6g}'.format(ct_sigmamax))
print(' t_dilution (50%)                {:.6g}'.format(ct_f50))
print(' t_dilution (10%)                {:.6g}'.format(ct_f10))
print(' t_Delta2_peak_max               {:.6g}'.format(ct_Ppeakmax))
print('')
print(' theta_stop                      {:.6g}'.format(theta_stop))
print(' theta_stop (mod 2pi)            {:.6g}'.format(np.fmod(theta_stop,2*np.pi)))
print(' sigmatheta_max                  {:.6g}'.format(theta_std_sigmamax))
print(' rho_0/rho_tot at t_end          {:.6g}'.format(fzero[-1]))
print('')
print(' k_* for N (at t_sigmamax)       {:.6g}'.format(kNpeak_sigmamax/R_sigmamax/mA_sigmamax))
print(' k_* for Delta^2 (at t_sigmamax) {:.6g}'.format(kPpeak_sigmamax/R_sigmamax/mA_sigmamax))
print(' k_* for N (at t_end)            {:.6g}'.format(kNpeak[-1]/R_sigmamax/mA_sigmamax))
print(' k_* for Delta^2 (at t_end)      {:.6g}'.format(kPpeak[-1]/R_sigmamax/mA_sigmamax))
print(' Delta^2_peak (at t_end)         {:.6g}'.format(APpeak[-1]))
print('')
print(' OmegaA*h^2 (at t_end)')
print('   spectrum sum                  {:.6f}'.format(Oatot[-1]))
print('   spectrum sum with m_eff       {:.6f}'.format(Oactot[-1]))
print('   estimate from energy/mass     {:.6f}'.format(Oae[-1]))
print('')
print(' OmegaA*h^2 (extrapolated)')
print('   spectrum sum                  {:.6f} +- {:.6f}'.format(Oaext,Oaext_err))
print('   spectrum sum with m_eff       {:.6f} +- {:.6f}'.format(Oacext,Oacext_err))
print('   estimate from energy/mass     {:.6f} +- {:.6f}'.format(Oaeext,Oaeext_err))
print('')
print('-----------------------------------------------------------------------')


