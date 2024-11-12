# joining geteig.py and getPDFsigma_fromeig.py
# still run this as getPDFsigma.py [tetsize] [simno] [snapno]  
import numpy as np
import sys
import readgadget
from tetrahedrafunc import *
from colossus.cosmology import cosmology
from time import perf_counter as time
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial_ZA/" + sys.argv[2] + "/"

QuijoteFid = {'H0': 67.11,
 'Ob0': 0.049, 
 'Om0': 0.3175,
 'flat': True,
 'ns': 0.9624,
 'sigma8': 0.834}
mycosmos =  cosmology.setCosmology("QuijoteFid", **QuijoteFid)

mytetsize=int(sys.argv[1])
snapno_final = int(sys.argv[3])

Ndim=512
bigboxL=1e3
celllen=bigboxL/Ndim
rhoavg=1/(celllen**3)



# initial
snapno = -1
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
zmap={-1:127,0:3,1:2,2:1,3:0.5,4:0}
currentz = zmap[snapno]
initz = currentz
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading")
# read positions, velocities and IDs of the particles
pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
print("finished reading")
ids_std = changeID(ids, Ndim= 512, Ndim_bybox=8)
ids=None
indexes=np.argsort(ids_std)
ids_std=None
Loc_sorted=pos[indexes]
indexes=None
pos=None
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
volflat=vol.reshape(((Ndim-mytetsize)**3,6))
print("got flattened vol")
dens_vol = (mytetsize**3)/np.abs( volflat.ravel() )/6/rhoavg
print("got dens_vol")
init_eigenvalues=eigenvalues_2idx.reshape((6*(Ndim-mytetsize)**3,3))
eigenvalues_2idx=None
init_dens_eigen=np.abs(1/((1+init_eigenvalues).prod(axis=1)))
print("got dens_eigen")
initsigma = np.sqrt(np.cov(dens_vol, aweights=1/dens_vol))
print("got sigma")
initsigma_eig = np.sqrt(np.cov(init_dens_eigen, aweights=1/init_dens_eigen))
print("got sigma_eig")
# take the real part and sort
init_Re_eigenvalues = np.real(init_eigenvalues)
init_Re_eigenvalues.sort(axis=1) # sort so that lamdba1<lambda2<lambda2
print("got Re_eigenvalues and sorted")
#take marginalized distribution
bin_range=5*np.std(init_Re_eigenvalues)
bin_disp=bin_range/100
tmp1=np.histogram(init_Re_eigenvalues[:,0],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp2=np.histogram(init_Re_eigenvalues[:,1],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp3=np.histogram(init_Re_eigenvalues[:,2],bins = np.arange(-bin_range,bin_range,bin_disp))
initlambda_range=(tmp1[1][0:-1]+tmp1[1][1:len(tmp1[1])])/2
initlambda1dist=tmp1[0]/np.trapz(tmp1[0],initlambda_range) # normalized
initlambda2dist=tmp2[0]/np.trapz(tmp2[0],initlambda_range)
initlambda3dist=tmp3[0]/np.trapz(tmp3[0],initlambda_range) 
print('got distribution')
# save initsigma, initsigma_eig, initlambda_range, initlambda1dist, initlambda2dist, initlambda3dist
np.save(snapdir+"initsigma_tet%d.npy"%mytetsize,initsigma)
np.save(snapdir+"initsigma_eig_tet%d.npy"%mytetsize,initsigma_eig)
np.save(snapdir+"initlambda_range_tet%d.npy"%mytetsize,initlambda_range)
np.save(snapdir+"initlambda1dist_tet%d.npy"%mytetsize,initlambda1dist)
np.save(snapdir+"initlambda2dist_tet%d.npy"%mytetsize,initlambda2dist)
np.save(snapdir+"initlambda3dist_tet%d.npy"%mytetsize,initlambda3dist)


# ZA
snapno = snapno_final
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
zmap={-1:127,0:3,1:2,2:1,3:0.5,4:0}
currentz = zmap[snapno]
ZAeigenvalues=init_eigenvalues * mycosmos.growthFactor(currentz)/mycosmos.growthFactor(initz)
ZARe_eigenvalues = np.real(ZAeigenvalues)
ZARe_eigenvalues.sort(axis=1) # sort so that lamdba1<lambda2<lambda2
ZAdens_eigen=np.abs(1/((1-ZAeigenvalues).prod(axis=1)))
print("got ZA eigenvalues")
#get ZA eigenvalue distribution
bin_range=5*np.std(ZARe_eigenvalues)
bin_disp=bin_range/100
tmp1=np.histogram(ZARe_eigenvalues[:,0],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp2=np.histogram(ZARe_eigenvalues[:,1],bins = np.arange(-bin_range,bin_range,bin_disp))
tmp3=np.histogram(ZARe_eigenvalues[:,2],bins = np.arange(-bin_range,bin_range,bin_disp))
ZAlambda_range=(tmp1[1][0:-1]+tmp1[1][1:len(tmp1[1])])/2
ZAlambda1dist=tmp1[0]/np.trapz(tmp1[0],ZAlambda_range) # normalized
ZAlambda2dist=tmp2[0]/np.trapz(tmp2[0],ZAlambda_range)
ZAlambda3dist=tmp3[0]/np.trapz(tmp3[0],ZAlambda_range) 
print('got ZA eigenvalue distribution')
# ZA PDF
dm,rhobins=np.histogram(ZAdens_eigen,bins=np.power(10.0,np.arange( -2,7, 0.02 )),density=True)
print('got distribution')
rhoxaxis=(rhobins[0:dm.shape[0]]+rhobins[1:dm.shape[0]+1])/2
ZAPDF = [rhoxaxis,dm]
# save ZAlambda_range, ZAlambda1dist, ZAlambda2dist, ZAlambda3dist, ZAPDF
np.save(snapdir+"ZAlambda_range_%03d_tet%d.npy"%(snapno,mytetsize),ZAlambda_range)
np.save(snapdir+"ZAlambda1dist_%03d_tet%d.npy"%(snapno,mytetsize),ZAlambda1dist)
np.save(snapdir+"ZAlambda2dist_%03d_tet%d.npy"%(snapno,mytetsize),ZAlambda2dist)
np.save(snapdir+"ZAlambda3dist_%03d_tet%d.npy"%(snapno,mytetsize),ZAlambda3dist)
np.save(snapdir+"ZAPDF_%03d_tet%d.npy"%(snapno,mytetsize),ZAPDF)


# free up unused variables
init_Re_eigenvalues=None
init_dens_eigen=None
dens_vol=None
init_eigenvalues=None
vol=None
volflat=None
grid3d=None
Phi3d=None
Loc3d=None
Loc_shifted=None
gridPoints=None
xg=None
Phi=None

# final
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading")
# read positions, velocities and IDs of the particles
pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
print("finished reading")
ids_std = changeID(ids, Ndim= 512, Ndim_bybox=8)
ids=None
indexes=np.argsort(ids_std)
ids_std=None
Loc_sorted=pos[indexes]
indexes=None
pos=None

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
dm,rhobins=np.histogram(dens_eigen,bins=np.power(10.0,np.arange( -2,7, 0.02 )),density=True)
print('got dens_eigen distribution')
rhoxaxis=(rhobins[0:dm.shape[0]]+rhobins[1:dm.shape[0]+1])/2
FinalPDF_eigen = [rhoxaxis,dm]
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
