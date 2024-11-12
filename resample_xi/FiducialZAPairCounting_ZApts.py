# run this as python FiducialZAPairCounting.py [tetsize] [simulation idx] [snapno]
from Corrfunc.theory.xi import xi
import numpy as np
import sys
from colossus.cosmology import cosmology
import readgadget
from tetrahedrafunc import *
from time import perf_counter as time
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial_ZA/"+ sys.argv[2] +"/"

QuijoteFid = {'H0': 67.11,
 'Ob0': 0.049, 
 'Om0': 0.3175,
 'flat': True,
 'ns': 0.9624,
 'sigma8': 0.834}
mycosmos =  cosmology.setCosmology("QuijoteFid", **QuijoteFid)

mytetsize=int(sys.argv[1])
nthreads = 32
xshift,yshift,zshift=0,0,0

#read initial
snapno = -1
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
    volout = snapdir+"vol_ics_tet%d.npy"%mytetsize
    eigout = snapdir+"eigenvalues_2idx_ics_tet%d.npy"%mytetsize
    initz=127
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
    volout = snapdir+"vol_%03d_tet%d.npy"%(snapno,mytetsize)
    eigout = snapdir+"eigenvalues_2idx_%03d_tet%d.npy"%(snapno,mytetsize)
    zmap={0:3,1:2,2:1,3:0.5,4:0}
    currentz = zmap[snapno]
ptype    = [1] #[1](CDM), [2](neutrinos) or [1,2](CDM+neutrinos)
print("reading",flush=True)
# read positions, velocities and IDs of the particles
pos = readgadget.read_block(snapshot, "POS ", ptype)/1e3 #positions in Mpc/h
ids = readgadget.read_block(snapshot, "ID  ", ptype)-1   #IDs starting from 0
print("finished reading",flush=True)

ids_std = changeID(ids, Ndim= 512, Ndim_bybox=8)
ids=None
indexes=np.argsort(ids_std)
ids_std=None
Loc_sorted=pos[indexes]
indexes=None
pos=None
Ndim= 512
bigboxL=1e3
celllen=bigboxL/Ndim
rhoavg=1/(celllen**3)

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

#displacement field Psi
initPsi = Loc_shifted - gridPoints


#read final redshift
snapno = int(sys.argv[3])
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
    volout = snapdir+"vol_ics_tet%d.npy"%mytetsize
    eigout = snapdir+"eigenvalues_2idx_ics_tet%d.npy"%mytetsize
    initz=127
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
    volout = snapdir+"vol_%03d_tet%d.npy"%(snapno,mytetsize)
    eigout = snapdir+"eigenvalues_2idx_%03d_tet%d.npy"%(snapno,mytetsize)
    zmap={0:3,1:2,2:1,3:0.5,4:0}
    currentz = zmap[snapno]

#take Zel'dovich approximation
ZAPsi = initPsi * mycosmos.growthFactor(currentz)/mycosmos.growthFactor(initz)
ZALoc_shifted = ZAPsi+gridPoints


# select particles
xx=np.arange(Ndim)[0:Ndim:mytetsize]
Ndimtet = len(xx)
xg=np.meshgrid(xx,xx,xx,indexing='ij')
xidxs=np.ravel(xg[0])+xshift
yidxs=np.ravel(xg[1])+yshift
zidxs=np.ravel(xg[2])+zshift
selected3idx=np.array([xidxs,yidxs,zidxs]).transpose()
selected = selected3idx[:,0]*Ndim**2+selected3idx[:,1]*Ndim+selected3idx[:,2]

testposZA = ZALoc_shifted[selected]%(celllen*Ndim)

bins = np.linspace(0.1,200,200)

X,Y,Z = (testposZA[:,0],testposZA[:,1],testposZA[:,2])
print("computing xi in ZA",flush=True)
tstart = time()
xi_counts = xi(celllen*Ndim, nthreads, bins, X, Y, Z)
tend = time()
print("finished in %f sec"%(tend-tstart),flush=True)
print(xi_counts)
np.save(snapdir+"xi_snap%03d_tet%d_ZApts.npy"%(snapno,mytetsize),xi_counts)