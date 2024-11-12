# compute final PDF for GadgetData I generated
# run this as getfinalPDF_Gadget.py [tetsize] [simname] [snapno] 

import numpy as np
import sys
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
from scipy.interpolate import interp1d
import readgadget
from tetrahedrafunc import *

mycosmospar = {'H0': 70.3,
 'Ob0': 0.045, 
 'Om0': 0.276,
 'flat': True,
 'ns': 0.961,
 'sigma8': 0.811}
mycosmos =  cosmology.setCosmology("mycosmos", **mycosmospar)
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"

mytetsize = int(sys.argv[1])
simulation_name = sys.argv[2]
snapno = int(sys.argv[3])
snapdir=oakdir+"GadgetData/"+simulation_name +"/"


#read final
snapshot = snapdir + 'snapshot_%03d'%snapno
hd = readgadget.header(snapshot)
currentz = hd.redshift
Ndim = int(round(np.cbrt(hd.npart[1])))
bigboxL = hd.boxsize
celllen = bigboxL/Ndim
rhoavg=1/(celllen**3)
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading %d"%snapno)
pos = readgadget.read_block(snapshot, "POS ", ptype)
ids = readgadget.read_block(snapshot, "ID  ", ptype)
print("finished reading")
indexes = np.argsort(ids)
Loc_sorted = pos[indexes]


# Lagrangian grid points
x = celllen*(np.arange(Ndim)) # not "x = celllen*(0.5+np.arange(Ndim))"
xg = np.meshgrid(x,x,x, indexing='ij')
xl = np.ravel(xg[0])
yl = np.ravel(xg[1])
zl = np.ravel(xg[2])
gridPoints=np.vstack([xl,yl,zl]).transpose()
x=None
xl=None
yl=None
zl=None
#take care of the periodic boundary
delta=Loc_sorted - gridPoints
Loc_shifted=np.copy(Loc_sorted)
for dim in range(3):
    too_small = delta[:,dim] < -bigboxL/2
    too_big = delta[:,dim] > bigboxL/2
    Loc_shifted[too_big,dim] -= bigboxL
    Loc_shifted[too_small,dim] += bigboxL
delta=None
#displacement field Phi
Phi = Loc_shifted - gridPoints
#reshape into a 3d mesh so as to refer xid,yid,zid more easily
grid3d=np.stack(xg,axis=3)
Phi3d=np.reshape(Phi,(Ndim,Ndim,Ndim,3))
Loc3d=Loc_shifted.reshape(Ndim,Ndim,Ndim,3) 
# get vol and eigenvalues
print('volume')
vol = get_tet_volumes_anysize(Ndim-mytetsize,Loc3d,mytetsize)
print("got vol")
print('eigen')
eigenvalues_2idx = get_tet_eigenvalues_2idx_anysize(Ndim-mytetsize,grid3d,Phi3d,mytetsize,printlog=True)
print('solved eigenvalues')
print("got vol")
volflat=vol.reshape(((Ndim-mytetsize)**3,6))
print("got flattened vol")
dens_vol = (mytetsize**3)/np.abs( volflat.ravel() )/6/rhoavg
print("got dens_vol")
eigenvalues=eigenvalues_2idx.reshape((6*(Ndim-mytetsize)**3,3))
eigenvalues_2idx=None
# take the real part and sort
Re_eigenvalues = np.real(eigenvalues)
Re_eigenvalues.sort(axis=1) # sort so that lamdba1<lambda2<lambda2
print("got Re_eigenvalues and sorted")
dens_eigen=np.abs(1/((1+eigenvalues).prod(axis=1)))
print("got dens_eigen")
#take distribution
bin_range=5*np.std(Re_eigenvalues)
bin_disp=bin_range/100
tmp1=np.histogram(Re_eigenvalues[:,0],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp2=np.histogram(Re_eigenvalues[:,1],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp3=np.histogram(Re_eigenvalues[:,2],bins = np.arange(-bin_range,bin_range,bin_disp))
lambda_range=(tmp1[1][0:-1]+tmp1[1][1:len(tmp1[1])])/2
lambda1dist=tmp1[0]/np.trapz(tmp1[0],lambda_range) # normalized
lambda2dist=tmp2[0]/np.trapz(tmp2[0],lambda_range)
lambda3dist=tmp3[0]/np.trapz(tmp3[0],lambda_range) 
print('got eigenvalue distribution')
if lambda_range[0]>-1:
    dm,rhobins=np.histogram(dens_eigen,bins=np.logspace(-1,1,200),density=True)
else:
    dm,rhobins=np.histogram(dens_eigen,bins=np.power(10.0,np.arange( -2,7, 0.02 )),density=True)
print('got dens_eigen distribution')
rhoxaxis=(rhobins[0:dm.shape[0]]+rhobins[1:dm.shape[0]+1])/2
FinalPDF_eigen = [rhoxaxis,dm]
if lambda_range[0]>-1:
    dm,rhobins=np.histogram(dens_vol,bins=np.logspace(-1,1,200),density=True)
else:
    dm,rhobins=np.histogram(dens_vol,bins=np.power(10.0,np.arange( -2,7, 0.02 )),density=True)
print('got dens_vol distribution')
rhoxaxis=(rhobins[0:dm.shape[0]]+rhobins[1:dm.shape[0]+1])/2
FinalPDF_vol = [rhoxaxis,dm]
# save lambda_range, lambda1dist, lambda2dist, lambda3dist, FinalPDF_eigen, FinalPDF_vol
np.save(snapdir+"lambda_range_%03d_tet%d.npy"%(snapno,mytetsize),lambda_range)
np.save(snapdir+"lambda1dist_%03d_tet%d.npy"%(snapno,mytetsize),lambda1dist)
np.save(snapdir+"lambda2dist_%03d_tet%d.npy"%(snapno,mytetsize),lambda2dist)
np.save(snapdir+"lambda3dist_%03d_tet%d.npy"%(snapno,mytetsize),lambda3dist)
np.save(snapdir+"FinalPDF_eigen_%03d_tet%d.npy"%(snapno,mytetsize),FinalPDF_eigen)
np.save(snapdir+"FinalPDF_vol_%03d_tet%d.npy"%(snapno,mytetsize),FinalPDF_vol)
