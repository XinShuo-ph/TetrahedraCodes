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

Ndim= 512
bigboxL=1e3
celllen=bigboxL/Ndim

#read final
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


# select particles
xx=np.arange(Ndim)[0:Ndim:mytetsize]
Ndimtet = len(xx)
xg=np.meshgrid(xx,xx,xx,indexing='ij')
xidxs=np.ravel(xg[0])+xshift
yidxs=np.ravel(xg[1])+yshift
zidxs=np.ravel(xg[2])+zshift
selected3idx=np.array([xidxs,yidxs,zidxs]).transpose()
selected = selected3idx[:,0]*Ndim**2+selected3idx[:,1]*Ndim+selected3idx[:,2]

testpos = Loc_sorted[selected]


#bins = np.logspace(np.log10(0.1),np.log10(300),150)
bins = np.linspace(0.1,200,200)

X,Y,Z = (testpos[:,0],testpos[:,1],testpos[:,2])
print("computing xi",flush=True)
tstart = time()
xi_counts = xi(celllen*Ndim, nthreads, bins, X, Y, Z)
tend = time()
print("finished in %f sec"%(tend-tstart),flush=True)
print(xi_counts)
np.save(snapdir+"xi_snap%03d_tet%d.npy"%(snapno,mytetsize),xi_counts)
