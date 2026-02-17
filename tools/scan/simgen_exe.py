from simgen_mpcdf import simgen_mpcdf
import sys

if (len(sys.argv) == 2):
    i = int(sys.argv[1])
else:
    print('usage:')
    print('python3 simgen_exe.py i')

filename = 'i%03d'%i

# specify your e-mail address
mail = ''
#mail = 'skeni@rzg.mpg.de'

simgen_mpcdf(filename,itheta=i,jobname=filename,nodes=2,ntaskspernode=8,cpuspertask=9,time='8:00:00',mailuser=mail,
            N = 256, wDz = 1.5, lap = 3, fA = 1e+9, ctf = 5.0, L = 0.1,
            vqcd = 'vqcdC', cti = 1.0, ict = 'spax', xtr = ' --ftype axion --mode0 1 --edens_sigma_threshold 100', 
            dwgam = -1, dump = 10, spmask = 513, rmask = 6.0, spKGV = 26, verbose = 2)
