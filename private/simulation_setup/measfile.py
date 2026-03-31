from pyaxions import jaxions as pa
import numpy as np

a = pa.mli()

a.N     = 128
a.L     = 3.0
a.msa   = 1.0
a.ctend = 3.0

# Simple measurements
a.clear()
a.me_adds("energy")
a.me_adds("bin theta")
a.me_adds("bin logtheta")
a.me_adds("plot2D")
a.nmt_prt()
a.addset(100, 1.0, a.ctend, 0, 'lin')

# Spectrum measurements
a.clear()
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.me_addmask("AXIT")
a.me_addmask("AXITV")
a.nmt_prt()
a.addset(10, 1.0, a.ctend, 0, 'lin')

a.give()

# 'MEAS_NOTHING': 0,
# 'MEAS_BINTHETA': 1,
# 'MEAS_BINRHO': 2,
# 'MEAS_BINLOGTHETA2': 4,
# 'MEAS_BINDELTA': 8,
# 'MEAS_EMPTY': 16,
# 'MEAS_STRING': 32,
# 'MEAS_STRINGMAP': 64,
# 'MEAS_STRINGCOO': 128,
# 'MEAS_ENERGY': 256,
# 'MEAS_ENERGY3DMAP': 512,
# 'MEAS_REDENE3DMAP': 1024,
# 'MEAS_2DMAP': 2048,
# 'MEAS_3DMAP': 4096,
# 'MEAS_MASK': 8192,
# 'MEAS_PSP_A': 16384,
# 'MEAS_PSP_S': 32768,
# 'MEAS_NSP_A': 65536,
# 'MEAS_NSP_S': 131072,
# 'MEAS_NNSPEC': 262144
