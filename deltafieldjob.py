import numpy as np
import sys
import h5py
import matplotlib.pyplot as plt
from colossus.cosmology import cosmology
#from scipy.interpolate import interp1d
from tetrahedrafunc import *
import readgadget 
from scipy.spatial import cKDTree
import os

mycosmospar = {'H0': 70.3,
 'Ob0': 0.045, 
 'Om0': 0.276,
 'flat': True,
 'ns': 0.961,
 'sigma8': 0.811}
mycosmos =  cosmology.setCosmology("mycosmos", **mycosmospar)

oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
testsnapdir=oakdir+"test20220708/"


mytetsize=int(sys.argv[1])
fileidx = int(sys.argv[2])
denseN = int(sys.argv[3])
batchidx_start = int(sys.argv[4])
batchidx_end = int(sys.argv[5])
xshift = int(sys.argv[6])
yshift = int(sys.argv[7])
zshift = int(sys.argv[8])

outputdir = testsnapdir + "outputs_snap%03d_tet%d_N%d_x%d_y%d_z%d_batch%d/"%(fileidx,mytetsize,denseN,xshift,yshift,zshift,batchidx_start)
plotdir = outputdir+"plots/"
os.system("mkdir "+outputdir+"\n")
os.system("mkdir "+plotdir+"\n")



mydata = np.ravel( np.load(testsnapdir+"snapshot_%03d.npy"%fileidx,allow_pickle=True) )[0]
Loc_sorted  = mydata["Loc_sorted"]
Loc_shifted = mydata["Loc_shifted"]
Ndim        = mydata["Ndim"]
currentz    = mydata["currentz"]
celllen     = mydata["celllen"]

rhoavg = 1/(celllen**3)


Loc3d=Loc_shifted.reshape(Ndim,Ndim,Ndim,3) 

# get vol
vol = get_tet_volumes_anysize(Ndim-mytetsize,Loc3d,mytetsize)

# make sparse grids (spaced by mytetsize)
xx=np.arange(Ndim)[0:Ndim:mytetsize]
xg=np.meshgrid(xx,xx,xx,indexing='ij')
Ndimtet=len(xx)

#establish connectivity list using my information of initial tetrahedra
conn_all_3idx = get_tet_conn(Ndimtet-1)
conn_all = conn_all_3idx[:,:,0]*Ndimtet**2 + conn_all_3idx[:,:,1] * Ndimtet + conn_all_3idx[:,:,2]


# select sparse grids (spaced by mytetsize)
xx=np.arange(Ndim)[0:Ndim:mytetsize]
xg=np.meshgrid(xx,xx,xx,indexing='ij')
xidxs=np.ravel(xg[0])+xshift
yidxs=np.ravel(xg[1])+yshift
zidxs=np.ravel(xg[2])+zshift
selected3idx=np.array([xidxs,yidxs,zidxs]).transpose()
selected = selected3idx[:,0]*Ndim**2+selected3idx[:,1]*Ndim+selected3idx[:,2]
# the "selected" points establish the tessellation
Loc_select=Loc_shifted[selected]

# dense grid is where we paint the density fields

x = celllen*Ndim/denseN*(0.5+np.arange(denseN))
xg = np.meshgrid(x,x,x, indexing='ij')
xl = np.ravel(xg[0])
yl = np.ravel(xg[1])
zl = np.ravel(xg[2])
denseGrid=np.vstack([xl,yl,zl]).transpose()

# cleanup correlation_draft

deltafield = np.zeros(denseN**3)-1
testvol0=[]
testvol1=[]
# loop over tetrahedra, paint points in each tetrahedron with tetrahedron density
cnt=0
# loop over a batch of tetrahedra from batchidx_start to batchidx_end
for i in np.arange(batchidx_end-batchidx_start+1,dtype=int)+batchidx_start:
    current_tet=conn_all[i]
    threshold = int((batchidx_end-batchidx_start)/500)
    if cnt%threshold == 0:
        np.save(outputdir+"cnt_%d.npy"%cnt,np.array([]))
    cnt=cnt+1
    # find the four vertices of the tetrahedra
    vets = Loc_select[current_tet,:]
    #print(vets)
    # preselection. Find those in the bounding box of the tetrahedron
    min_x,min_y,min_z = np.min(vets,axis=0)
    max_x,max_y,max_z = np.max(vets,axis=0)
    mask_x = (denseGrid[:, 0] >= min_x) & (denseGrid[:, 0] <= max_x)
    mask_y = (denseGrid[:, 1] >= min_y) & (denseGrid[:, 1] <= max_y)
    mask_z = (denseGrid[:, 2] >= min_z) & (denseGrid[:, 2] <= max_z)
    mask = mask_x & mask_y & mask_z
    #print("start np.where")
    inbox_indices = np.where(mask)[0]
    #print(len(inbox_indices))
    # then for those points inbox, find whether they are in the tetrahedron
    insidepts=InTetraList(denseGrid[inbox_indices],vets[0],vets[1],vets[2],vets[3])
    #print(len(insidepts))
    # as a sheck, let's output dens_vol of this tetrahedron 
    # and also compute the vol right now using the vertices
    # take care of the idxs
    tet6id = i%6
    tetxid = int(int(i/6) / (Ndimtet-1)**2)
    tetyid = int(( int(i/6) - tetxid*(Ndimtet-1)**2 ) / (Ndimtet-1))
    tetzid = int(i/6) % (Ndimtet-1)
    currentvol = vol[tetxid*mytetsize+xshift,tetyid*mytetsize+yshift,tetzid*mytetsize+zshift,tet6id]
    testvol0.append(currentvol)
    # compute volume using vertices
    testvol1.append( np.sum( np.cross(vets[1]-vets[3], vets[2]-vets[3])* (vets[0]-vets[3]))/6 )
    # paint the density
    currentrho = (mytetsize**3)/6 / np.abs(currentvol)
    deltafield[inbox_indices[insidepts]] += currentrho/rhoavg
    
os.system("rm "+outputdir+"cnt*.npy\n")
# save everything
testvol0=np.array(testvol0)
testvol1=np.array(testvol1)
np.save(outputdir + "testvol0.npy",testvol0)
np.save(outputdir + "testvol1.npy",testvol1)
np.save(outputdir + "deltafield.npy",deltafield)