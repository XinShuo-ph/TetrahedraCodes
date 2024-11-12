# run this as python FiducialZAPairCounting.py [tetsize] [simulation idx] [snapno] [xshift] [yshift] [zshift]
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
xshift,yshift,zshift=int(sys.argv[4]),int(sys.argv[5]),int(sys.argv[6])

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
Lbox = bigboxL
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

Loc_select = ZALoc_shifted[selected]
Loc3d_select=Loc_select.reshape(Ndimtet,Ndimtet,Ndimtet,3)

# resample particles
# pad a layer of vertices so as to take care of the boundary-crossing tets
Loc3d_select_padded = np.zeros((Ndimtet+1,Ndimtet+1,Ndimtet+1,3))
tmp = np.meshgrid(range(Ndimtet+1),range(Ndimtet+1),range(Ndimtet+1),indexing='ij')
xidxs=np.ravel(tmp[0])
yidxs=np.ravel(tmp[1])
zidxs=np.ravel(tmp[2])
tmp=None
myindexes=np.array([xidxs,yidxs,zidxs]).transpose()
xidxs=None
yidxs=None
zidxs=None
Loc3d_select_padded[myindexes[:,0],myindexes[:,1],myindexes[:,2]] = Loc3d_select[myindexes[:,0]%Ndimtet,myindexes[:,1]%Ndimtet,myindexes[:,2]%Ndimtet]
# shift the outer most layer
Loc3d_select_padded[Ndimtet,:,:,0] += Lbox
Loc3d_select_padded[:,Ndimtet,:,1] += Lbox
Loc3d_select_padded[:,:,Ndimtet,2] += Lbox
print("padded")

# sample particles in tets and pair counting
newp = get_tet_centroids_anysize(Ndimtet,Loc3d_select_padded,1)
newp = newp%Lbox
newp = np.float32(newp)
print("interpolated")

bins = np.linspace(1,200,50)
X,Y,Z = (newp[:,0],newp[:,1],newp[:,2])
print("computing xi")
tstart=time()
xi_counts = xi(celllen*Ndim, nthreads, bins, X, Y, Z)
tend=time()
print("nthreads=%d, finished in %f"%(nthreads,tend-tstart))

np.save(snapdir+"xiCenresampled_snap%03d_tet%d_%d%d%d_ZApts.npy"%(snapno,mytetsize,xshift,yshift,zshift),xi_counts)
