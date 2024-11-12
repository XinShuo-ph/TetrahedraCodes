# compute the initlal sigma and also all the ZA eigenvalues for Gadget simulation I made
# run this as getinitsigma_Gadget.py [tetsize] [simname]
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
snapdir=oakdir+"GadgetData/"+simulation_name +"/"


#read initial
snapshot = oakdir + 'MUSIC/ics/ic_'+simulation_name+'.dat'
hd = readgadget.header(snapshot)
initz = hd.redshift
Ndim = int(round(np.cbrt(hd.npart[1])))
bigboxL = hd.boxsize
celllen = bigboxL/Ndim
rhoavg=1/(celllen**3)
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading")
pos = readgadget.read_block(snapshot, "POS ", ptype)
ids = readgadget.read_block(snapshot, "ID  ", ptype)
print("finished reading")
indexes = np.argsort(ids)
Loc_sorted = pos[indexes]
# Lagrangian grid points
x = celllen*(0.5+np.arange(Ndim))
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
#displacement field Psi
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


#times
ts = np.loadtxt(oakdir+'GadgetData/output_times.txt')
zs = 1/ts -1
zs = np.append(zs,0)

# ZA for all output times
for snapno in range(63,-1,-1):
    currentz = zs[snapno]
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
    if ZAlambda_range[0]>-1:
        dm,rhobins=np.histogram(ZAdens_eigen,bins=np.logspace(-1,1,200),density=True)
    else:
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
