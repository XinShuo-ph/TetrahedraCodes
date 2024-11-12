# compute final PDF for GadgetData I generated, not using tets but using standard CIC for final particles
# run this as getfinalPDF_Gadget_CIC.py [Rsmooth] [simname] [snapno] 

import numpy as np
import sys
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
import readgadget
from tetrahedrafunc import *
import Pk_library as PKL
import MAS_library as MASL
import smoothing_library as SL

mycosmospar = {'H0': 70.3,
 'Ob0': 0.045, 
 'Om0': 0.276,
 'flat': True,
 'ns': 0.961,
 'sigma8': 0.811}
mycosmos =  cosmology.setCosmology("mycosmos", **mycosmospar)
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"

Rsmooth = float(sys.argv[1])
simulation_name = sys.argv[2]
snapno = int(sys.argv[3])
snapdir=oakdir+"GadgetData/"+simulation_name +"/"

#read final
snapshot = snapdir + 'snapshot_%03d'%snapno
hd = readgadget.header(snapshot)
currentz = hd.redshift
Ndim = int(round(np.cbrt(hd.npart[1])))
BoxSize = hd.boxsize
celllen = BoxSize/Ndim
rhoavg=1/(celllen**3)
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading %d"%snapno)
pos = readgadget.read_block(snapshot, "POS ", ptype)
ids = readgadget.read_block(snapshot, "ID  ", ptype)
print("finished reading")
indexes = np.argsort(ids)
Loc_sorted = pos[indexes]

# calculate the overdensity field
delta = np.zeros((Ndim,Ndim,Ndim), dtype=np.float32)
MASL.MA(pos, delta, BoxSize, 'CIC')
delta /= np.mean(delta, dtype=np.float64)
# smooth the overdensity field
threads = 2
W_k = SL.FT_filter(BoxSize, Rsmooth, Ndim, 'Top-Hat', threads)
delta_smoothed = SL.field_smoothing(delta, W_k, threads)

bins = np.logspace(-1,1,200) # same denser bins as I used for Quijote
pdf, mean = np.histogram(delta_smoothed, bins=bins)
mean = 0.5*(mean[1:] + mean[:-1])
pdf = pdf*1.0/Ndim**3

# save results to file
fpdf = snapdir + "PDFCIC_R%.1f_%03d.txt"%(Rsmooth,snapno)
np.savetxt(fpdf, np.transpose([mean, pdf]), delimiter='\t')