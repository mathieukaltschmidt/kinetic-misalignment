"""
Generates Slurm job script to run jAxions simulations at MPCDF clusters.
** MODIFIED VERSION **
"""

# Modules
import os
import numpy as np

def simgen_mpcdf(name,itheta=0,jobname='',mem=0,nodes=2,ntaskspernode=2,cpuspertask=36,time='24:00:00',output='stdout.%j',error='stderr.%j',mailtype='ALL',mailuser='skeni@rzg.mpg.de',
            N = 256, prec = 'single', dev = 'cpu', fftplan = 64, steps = 1000000, wDz = 1.0, sst0 = 10, lap = 1,
            nqcd = 7.0, fA = -1, msa = 1.0, lamb = -1.0, ctf = 128., L = 256.0, ind3 = 1.0, notheta = False, wkb = -1., gam = 0.0, dwgam = 1.0,
            vqcd = 'vqcdC', vpq = 0, mink = False, xtr = '', prep = False, ic = 'lola', logi = 0.0, cti = -1.11,
            index = -100, ict = 'lola', dump = 10, meas = 0, p3D = 0, spmask = 1, rmask = 1.5, redmp = -1.0, wTime = -1.0,
            spKGV = 15, printmask = False, ng0calib = 1.25, cummask = 0,
            p2Dmap = False, p2DmapE = False, p2DmapPE = False, p2DmapPE2 = False, p2DmapYZ = False, slc = -1, strmeas = -1,
            nologmpi = True, verbose = 1,
            verb = False, **kwargs):

    theta1 = 2*np.pi*itheta/200
    
    zRANKS = nodes*ntaskspernode
    f = open(name,'w')

    pres = ['#!/bin/bash -l\n\n',
        '#SBATCH -D ./\n',
        '#SBATCH --time='+time+'\n',
        '#SBATCH --nodes=%d'%nodes+'\n',
        '#SBATCH --ntasks-per-node=%d'%ntaskspernode+'\n',
        '#SBATCH --cpus-per-task=%d'%cpuspertask+'\n',
        '#SBATCH --output '+output+'\n',
        '#SBATCH --error '+error+'\n',
        '#SBATCH --mail-type='+mailtype+'\n',
        '#SBATCH --mail-user='+mailuser+'\n']
    if mem:
        pres.append('#SBATCH --mem=%d'%mem+'\n')
    if jobname:
        pres.append('#SBATCH -J '+jobname+'\n')
    pres.append('\n')
    pres.append('SECONDS=0\n')
    pres.append('\n')

    # my modules
    pres.append('sh /u/skeni/loadmodules.sh\n')

    # or specify modules manually
    #pres.append('module purge\n')
    #pres.append('module load impi/2021.6 cuda/11.4\n')

    pres.append('\n')
    pres.append('export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}\n')
    pres.append('\n')

    f.write(''.join(pres))

    GRID,SIMU,PHYS,VQCD,INCO,OUTP = simgen(N=N,zRANKS=zRANKS,prec=prec,dev=dev,fftplan=fftplan,steps=steps,wDz=wDz,sst0=sst0,lap=lap,
            nqcd=nqcd,fA=fA,msa=msa,lamb=lamb,ctf=ctf,L=L,ind3=ind3,notheta=notheta,wkb=wkb,gam=gam,dwgam=dwgam,
            vqcd=vqcd,vpq=vpq,mink=mink,xtr=xtr,prep=prep,ic=ic,logi=logi,cti=cti,
            index=index,ict=ict,dump=dump,meas=meas,p3D=p3D,spmask=spmask,rmask=rmask,redmp=redmp,wTime=wTime,
            spKGV=spKGV,printmask=printmask,ng0calib=ng0calib,cummask=cummask,
            p2Dmap=p2Dmap,p2DmapE=p2DmapE,p2DmapPE=p2DmapPE,p2DmapPE2=p2DmapPE2,p2DmapYZ=p2DmapYZ,slc=slc,strmeas=strmeas,
            nologmpi=nologmpi,verbose=verbose,
            verb=verb,**kwargs)

    f.write('RANKS=%d\n'%zRANKS)
    f.write('GRID="'+GRID+'"\n')
    f.write('SIMU="'+SIMU+'"\n')
    f.write('PHYS="'+PHYS+'"\n')
    f.write('VQCD="'+VQCD+'"\n')
    f.write('INCO="'+INCO+'"\n')
    f.write('OUTP="'+OUTP+'"\n')
    f.write('\n')
    f.write('echo "vaxion3d  " $GRID\n')
    f.write('echo "          " $SIMU\n')
    f.write('echo "          " $PHYS\n')
    f.write('echo "          " $VQCD\n')
    f.write('echo "          " $INCO\n')
    f.write('echo "          " $OUTP\n')
    f.write('\n')
    f.write('ii=%d\n'%itheta)
    f.write('thetai=%f\n'%theta1)
    f.write('\n')   
    f.write('VSTART=$(printf "%f" 2e+2)\n') # decrease L from LSTART to LEND as vheta changes from VSTART to VEND
    f.write('VEND=$(printf "%f" 1e+4)\n')
    f.write('LSTART=0.2\n')
    f.write('LEND=0.05\n')
    f.write('CA=$(echo "($LSTART-$LEND)/(l($VSTART)-l($VEND))" | bc -l)\n')
    f.write('CB=$(echo "($LEND*l($VSTART)-$LSTART*l($VEND))/(l($VSTART)-l($VEND))" | bc -l)\n')
    f.write('\n')
    f.write('for jj in $(seq 0 204) ; do\n')
    #f.write('for jj in 0 34 68 102 136 170 204 ; do\n')
    f.write('vhetai=$(echo "e((6*$jj/204.0-1)*l(10))" | bc -l)\n')
    f.write('if [ "$(echo "$vhetai < $VSTART" | bc -l)" -eq 1 ]; then\n')
    f.write('L=$LSTART\n')
    f.write('elif [ "$(echo "$vhetai > $VEND" | bc -l)" -eq 1 ]; then\n')
    f.write('L=$LEND\n')
    f.write('else\n')
    f.write('L=$(echo "$CA*l($vhetai)+$CB" | bc -l)\n')
    f.write('fi\n')
    f.write('echo\n')
    f.write('echo "========================== ($ii,$jj): theta1 = $thetai, vheta1 = $vhetai, L = $L =========================="\n')
    f.write('nmdir="$(printf "i%03dj%03d" "$ii" "$jj")"\n')
    f.write('mkdir -p "$nmdir"\n')
    f.write('cp measfile.dat inco.py analysis.py "$nmdir"\n')
    f.write('cd "$nmdir"\n')
    f.write('python3 inco.py %d $L $thetai $vhetai\n'%(N))
    f.write('srun -n $RANKS -c $OMP_NUM_THREADS vaxion3d $GRID $SIMU $PHYS $VQCD $INCO $OUTP --lsize $L 2>&1 | tee log.txt\n')
    f.write('python3 analysis.py ./ %d $ii $jj $thetai $vhetai ../\n'%(fA))
    f.write('rm -r ./out/m\n')
    f.write('rm ./out/sample.txt\n')
    f.write('if [ $jj -ne 0 ]; then\n') # remove axion.log.0 file except for jj=0 
    f.write('rm ./axion.log.0\n')
    f.write('fi\n')
    f.write('cd ..\n')
    f.write('done\n')
    f.write('\n')
    f.write('echo\n')
    f.write('echo\n')
    f.write('\n')
    f.write('Ttime=$SECONDS\n')
    f.write('Ttimem=$(echo "$Ttime / 60.0" | bc -l)\n')
    f.write('Ttimeh=$(echo "$Ttime / 3600.0" | bc -l)\n')
    f.write('echo "time (sec) : $Ttime"\n')
    f.write('echo "time (min) : $Ttimem"\n')
    f.write('echo "time (h) : $Ttimeh"\n')

    f.close()

# Inherited from jaxions/scripts/pyaxions/simgen.py but returns GRID,SIMU,PHYS,VQCD,INCO,OUTP options separately
def simgen (N=256,zRANKS=1,prec='single',dev='cpu', fftplan = 64, lowmem=False,prop='rkn4', spec=False, fspec=False, steps=1000000,wDz=1.0,sst0=10,lap=1,
            nqcd=7.0, fA = -1, msa=1.0,lamb=-1.0,ctf=128.,L=256.0, ind3=1.0,notheta=False,wkb=-1.,gam=0.0,dwgam=1.0,
            vqcd='vqcdC',vpq=0, mink = False, xtr='',prep=False,ic='lola',logi=0.0,cti=-1.11,
            index=-100,ict='lola',dump=10,meas=0,p3D=0,spmask=1,rmask=1.5,redmp=-1.0,wTime=-1.0,
            spKGV=15,printmask=False,ng0calib=1.25,cummask=0,
            p2Dmap=False,p2DmapE=False,p2DmapPE=False,p2DmapPE2=False, p2DmapYZ=False, slc=-1, strmeas =-1,
            nologmpi=True,verbose=1,
            verb=False,**kwargs):
    """
    simgen creates a string of command line flags to select options for vaxion3d

    returns int (number of ranks), string (flags)

    options:

    N         int    256      number of grid points in x,y directions
    zRANKS    int    1        number of MPI processes along z direction
    prec      str    single   single or double
    dev       str    cpu      cpu or gpu (use --measCPU)
    lowmem    bool   False
    prop      str    rkn4     propagator/time integrator
    spec      str    False    spectral propagator 1 (only CPU atm)
    fspec     str    False    spectral propagator 2 (only CPU atm)
    steps     int    10000    number of time steps (max if --wDz is used)
    wDz       float  1.0      time interval set to dt = wDz/w_max
    sst0      int    10       time steps without strings before switch to theta
    lap       int    1        number of neighbours in Laplacian
    fftplan   int    64       specify FFT plan to speed up initialisation
    nqcd      float  7.0      index of ct-dependence of Topological Susceptibility
    fA        int    -1       if fA > 0 is given, the code uses the --qcd qcd option with fA as specified
    msa       float  1.0      use PRS strings with ms = msa/(dx R); overrides lambda!
    lamb      float  -1.0     use Physical strings with SI lambda; make >0; is overridden by --msa
    ctf       float  128.     final conformal time of simulations
    L         float  256.0    Physical length of x,y directions in ADM units
    ind3      float  1.0      coefficient multiplying axion mass^2
    notheta   bool   False    False/True will switch/not switch to theta-only simulations after strings
    wkb       float  -1.0     the final fields are wkb'ed to this c-time
    gam       float  0.0      damping
    dwgam     float  1.0      rho damping activated after artificial DW destruction
    vqcd      str             vqcd type: 'vqcdC','vqcdV','vqcd0','vqcdL','N2'
    vpq       str    0        2 for vPQ2
    prep      bool   False
    ic        str    lola     type of initial conditions
    logi      float  0.0      use this log ms/H to set the initial c-time
    cti       float  -1.11    initial c-time
    index     int    -100     will read axion.int initial conditions to continue the simulation
    ict
    dump
    meas
    p3D
    spmask
    rmask
    redmp
    wTime
    spKGV
    printmask
    ng0calib
    cummask
    p2Dmap
    p2DmapE
    p2DmapPE
    p2DmapPE2
    p2DmapYZ
    slc
    strmeas
    nologmpi
    verbose
    verb
    kwargs
    """
    ####################################################
    GRID=" --size %d --depth %d --zgrid %d"%(N,N//zRANKS,zRANKS)
    ####################################################
    SIMU=" --prec %s --device %s --prop %s --steps %d --wDz %f --sst0 %d --fftplan %d"%(prec,dev,prop,steps,wDz,sst0,fftplan)

    if not (spec or fspec):
        SIMU += ' --lap %d'%lap

    if lowmem:
        SIMU += ' --lowmem'
    if dev == 'gpu':
        SIMU += ' --measCPU'

    if spec:
        SIMU += ' --spec'
        if verb:
            print('Using spec propagator.')
    if fspec:
        SIMU += ' --fspec'
        if verb:
            print('Using fspec propagator.')
    ####################################################
    VQCP=''
    if vqcd in ['vqcdC','vqcdV','vqcd0','vqcdL','N2']:
        VQCP += ' --%s'%vqcd
    else :
        if verb:
            print('Warning: VQCD not recognised!')
    if vpq == 2:
        VQCP += ' --vPQ2'
    if vqcd == 'vqcd0' and ind3 > 0:
        ind3 = 0.0
        if verb:
            print('Warning: vqcd0 -> ind3 reset to %f'%ind3)
    ####################################################
    if lamb>0:
        tension = ' --llcf %f'%lamb
    else :
        tension = ' --msa %f'%msa
    PHYS="%s --lsize %f --zf %f --ind3 %f"%(tension,L,ctf,ind3)
    noth='';wkbs='';gams='';dwgams='';mnk=''
    if fA > 0:
        qcd = ' --qcd qcd --fA %d'%fA
    else: qcd = ' --qcd %f'%nqcd
    if notheta:
        noth = ' --notheta '
    if wkb > 0:
        wkbs = ' --wkb %f'%wkb
    if gam > 0:
        gams = ' --gam %f'%gam
    if dwgam > 0:
        dwgams = ' --dwgam %f'%dwgam
    if mink:
        mnk = ' --mink'
        if verb:
            print('Minkowski!')
    PHYS += qcd+noth+wkbs+gams+dwgams+mnk+xtr
    #################################################### IC condition 1 by 1
    if index >= 0:
        # READ CONF
        INCO = ' --index %d'%index
    else:
        if cti == -1.11:
            it = ' --logi %f'%logi
        else :
            it = ' --zi %f'%cti

        INCO= it + INCOgen(ict,verb,**kwargs)
    ####################################################
    OUT0=''
    if p2Dmap:
        OUT0+=' --p2Dmap'
    if p2DmapE:
        OUT0+=' --p2DmapE'
    if p2DmapPE and not p2DmapPE2:
        OUT0+=' --p2DmapPE'
    if p2DmapPE2 and not p2DmapPE:
        OUT0+=' --p2DmapPE2'
    if p2DmapYZ:
        OUT0+=' --p2DmapYZ'
    if slc >= 0:
        OUT0+=' --sliceprint %d'%slc
    if strmeas >= 0:
        OUT0+=' --strmeas %d'%strmeas

    OUT1=" --dump %d --meas %d --p3D %d "%(dump,meas,p3D)
    if redmp > 0:
        OUT1 += ' --redmp %d'%redmp

    OUTM=" --spmask %d --rmask %s --spKGV %d"%(spmask,str(rmask),spKGV)
    if printmask:
        OUTM += ' --printmask'
    if ng0calib != 1.25:
        OUTM += ' --ng0calib %f'%ng0calib
    if cummask != 0:
        OUTM += ' --cummask %d'%cummask

    OUT2=" --verbose %d"%(verbose)
    if wTime > 0:
        OUT2 += '  --wTime %d'%wTime

    if nologmpi:
        OUT2 +=' --nologmpi'
    OUT=OUT0+OUT1+OUTM+OUT2
    if verb==True:
        print('PHYS =',PHYS)
        print('GRID =', GRID)
        print('SIMU = ',SIMU)
        print('VQCD =',VQCP)
        print('INCO =',INCO)
        print('OUT =',OUT)
    #return zRANKS, PHYS+GRID+SIMU+VQCP+INCO+OUT
    return GRID, SIMU, PHYS, VQCP, INCO, OUT

def INCOgen(ict,verb=False,**kwargs):
    def fif(ka,ja,xic):
        if ka in kwargs:
            xic += ' --%s '%ja + str(kwargs[ka])
        else:
            if verb:
                print('%s missing in kwargs: jaxion defaults will be used'%ka)
        return xic

    INCO = ''
    if ict == 'lola':
        INCO = ' --ctype %s'%ict
        if 'lola_string_multiplier' in kwargs:
            lr = 1
            if 'lolarandom' in kwargs:
                if kwargs['lolarandom']:
                    lr=2
            INCO += ' --sIter %d --kcr %f'%(lr,kwargs['lola_string_multiplier'])
    if ict == 'spax':
        INCO = ' --ctype %s'%ict
    if ict == 'smooth':
        INCO = ' --ctype %s'%ict
        if 'smvar' in kwargs:
            INCO += ' --smvar %s'%(kwargs['smvar'])
        INCO = fif('mode0','mode0',INCO)
        INCO = fif('kMax','kMax',INCO)
        INCO = fif('kcr','kcr',INCO)
    if ict == 'cole':
        INCO = ' --ctype %s'%ict
        INCO = fif('kMax','kMax',INCO)
    if ict == 'tkachev':
        INCO = ' --ctype %s'%ict
        INCO = fif('kMax','kMax',INCO)
        INCO = fif('kcr','kcr',INCO)

    if ict == 'string':
        INCO = ' --ctype %s'%ict
        INCO = fif('sIter','sIter',INCO)

    if 'kickalpha' in kwargs:
        INCO += ' --kickalpha '+str(kwargs['kickalpha'])
    if 'extrav' in kwargs:
        INCO += ' --extrav '+str(kwargs['extrav'])
    if 'nncore' in kwargs:
        INCO += ' --nncore'
    PREP = ''
    if 'preprop' in kwargs:
        PREP += ' --preprop'
        # preprequires damping?
        PREP = fif('prepcoe','prepcoe',PREP)
        PREP = fif('lz2e','lz2e',PREP)
        PREP = fif('prevqcdtype','prevqcdtype',PREP)
        PREP = fif('pregam','pregam',PREP)
        if 'icstudy' in kwargs:
            if kwargs['icstudy']:
                PREP += ' --icstudy'
    return INCO+PREP
    