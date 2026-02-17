from pyaxions import jaxions as pa
import numpy as np

a = pa.mli()

a.N     = 256
a.ctend = 5.0

a.clear()
a.me_adds("energy")
a.me_adds("bin theta")
a.me_adds("NSP_A")
a.me_adds("PSP_A")
a.me_addmask("FLAT")
a.me_addmask("AXIT")
a.me_addnrt("CK")
a.me_addnrt("G")
a.me_addnrt("S")
a.nmt_prt()
a.addset(200, 1.0, a.ctend, 0, 'lin')

# produce map data at ct = 1,2,3,4,5 if necessary (set printmaps = True in analysis.py)
#a.clear()
#a.me_addmap("XYM")
#a.me_addmap("XYV")
#a.me_addmap("E")
#a.me_addmap("PE")
#a.addset(4, 1.0, a.ctend, 0, 'lin')

a.give()
