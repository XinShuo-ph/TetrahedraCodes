# compute density PDF using CIC, as in Quijote github codes
# run this as [Rsmooth] [simno] [snapno] 
import argparse
from mpi4py import MPI
import numpy as np
import sys,os
import readgadget,readfof
import redshift_space_library as RSL
import Pk_library as PKL
import MAS_library as MASL
import smoothing_library as SL
from time import perf_counter as time
from tetrahedrafunc import *
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial_ZA/" + sys.argv[2] + "/"

Rsmooth=float(sys.argv[1])
snapno_final = int(sys.argv[3])

Ndim=512
bigboxL=1e3
celllen=bigboxL/Ndim
rhoavg=1/(celllen**3)

snapno = snapno_final
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
zmap={-1:127,0:3,1:2,2:1,3:0.5,4:0}
currentz = zmap[snapno]
# read final
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading")
# read positions
pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h

grid      = 1024
MAS       = 'CIC'
threads   = 2
Filter    = 'Top-Hat' #'Gaussian'
BoxSize   = 1e3
delta = np.zeros((grid,grid,grid), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, MAS)
delta /= np.mean(delta, dtype=np.float64)

# smooth the overdensity field
smoothing = Rsmooth
W_k = SL.FT_filter(BoxSize, smoothing, grid, Filter, threads)
delta_smoothed = SL.field_smoothing(delta, W_k, threads)

bins = np.logspace(-1,1,200) # denser bins
pdf, mean = np.histogram(delta_smoothed, bins=bins)
mean = 0.5*(mean[1:] + mean[:-1])
pdf = pdf*1.0/grid**3

# save results to file
fpdf = snapdir + "PDFCIC_R%.1f_%03d.txt"%(Rsmooth,snapno)
np.savetxt(fpdf, np.transpose([mean, pdf]), delimiter='\t')