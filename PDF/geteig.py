# run this as geteig.py [tetsize] [simno] [snapno]
import numpy as np
import sys
import readgadget
from tetrahedrafunc import *
from time import perf_counter as time
oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial_ZA/" + sys.argv[2] + "/"

mytetsize=int(sys.argv[1])
snapno = int(sys.argv[3])
if snapno == -1 : # ICs
    snapshot = snapdir+"ICs/ics"
    volout = snapdir+"vol_ics_tet%d.npy"%mytetsize
    eigout = snapdir+"eigenvalues_2idx_ics_tet%d.npy"%mytetsize
else:
    snapshot = snapdir+"snapdir_%03d/snap_%03d"%(snapno,snapno)
    volout = snapdir+"vol_%03d_tet%d.npy"%(snapno,mytetsize)
    eigout = snapdir+"eigenvalues_2idx_%03d_tet%d.npy"%(snapno,mytetsize)
    

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


#displacement field Phi
Phi = Loc_shifted - gridPoints

#reshape into a 3d mesh so as to refer xid,yid,zid more easily
grid3d=np.stack(xg,axis=3)
Phi3d=np.reshape(Phi,(Ndim,Ndim,Ndim,3))
Loc3d=Loc_shifted.reshape(Ndim,Ndim,Ndim,3) 

# get vol
print('volume')
vol = get_tet_volumes_anysize(Ndim-mytetsize,Loc3d,mytetsize)
print("got vol")
#cen = get_tet_centroids_anysize(Ndim-mytetsize,Loc3d,mytetsize)
#volflat=vol.reshape(((Ndim-mytetsize)**3,6))
#vol=None
#mvol = np.fabs(volflat )/(celllen**3) # every particle in ICs has volume of 1 this way 
np.save(volout,vol)

#compute eigenvalues at particle positions, averaged over adjacent tetrahedra
print('eigen')
eigenvalues_2idx = get_tet_eigenvalues_2idx_anysize(Ndim-mytetsize,grid3d,Phi3d,mytetsize,printlog=True)
print('solved eigenvalues')

np.save(eigout,eigenvalues_2idx)