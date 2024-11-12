# use camb to compute correlation function
import camb
import camb
import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
import readgadget
from tetrahedrafunc import *
nthreads=32
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial/10000/"
​
QuijoteFid = {'H0': 67.11,
 'Ob0': 0.049, 
 'Om0': 0.3175,
 'flat': True,
 'ns': 0.9624,
 'sigma8': 0.834}
Qcosmos =  cosmology.setCosmology("QuijoteFid", **QuijoteFid)

cambpar = oakdir+'/QuijoteData/Linear_Pk/fiducial/CAMB_TABLES/CAMB_params_Planck.ini'
pars = camb.read_ini(cambpar)
pars.set_matter_power(redshifts=[127], kmax=2.0)
# compute the correlation function using pars
results = camb.get_results(pars)
kh, z, pk = results.get_matter_power_spectrum(minkh=1e-4, maxkh=1, npoints = 200)
# take correlation function
r, xi = results.get_correlation_function()

