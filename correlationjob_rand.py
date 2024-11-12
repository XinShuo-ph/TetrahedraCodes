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

#rlist = np.power(10.0,np.arange( -1,1.41, 0.1 )) # 25 radius points to estimate \xi
rlist = np.append(np.power(10.0,np.arange( -1,1.41, 0.1 )),np.power(10.0,np.arange( 1.5,1.86, 0.05 )))

mytetsize=int(sys.argv[1])
fileidx = int(sys.argv[2])
denseN = int(sys.argv[3])
xshift = int(sys.argv[4])
yshift = int(sys.argv[5])
zshift = int(sys.argv[6])
rid   = int(sys.argv[7])
sampleNcen = int(sys.argv[8])
sampleNsph = int(sys.argv[9])

sampleradius = rlist[rid]

outputdir = testsnapdir + "outputs_snap%03d_tet%d_N%d_x%d_y%d_z%d_batch%d/"%(fileidx,mytetsize,denseN,xshift,yshift,zshift,0)
plotdir = outputdir+"plots/"
corrdir = outputdir+"xi_rid%d_cen%d_sph%d/"%(rid,sampleNcen,sampleNsph)
os.system("mkdir "+ corrdir)

# read the delta fields
Ndim=128
xx=np.arange(Ndim)[0:Ndim:mytetsize]
Ndimtet=len(xx)
Nconn = 6 * (Ndimtet-1)**3
Nbatchs = int(Nconn/(200000 * (150/denseN)**3))+1
Nperbatch = int(Nconn/Nbatchs)
celllen = 250.0 /Ndim
# dense grid is where we paint the density fields
x = celllen*Ndim/denseN*(0.5+np.arange(denseN))
xg = np.meshgrid(x,x,x, indexing='ij')
xl = np.ravel(xg[0])
yl = np.ravel(xg[1])
zl = np.ravel(xg[2])
denseGrid=np.vstack([xl,yl,zl]).transpose()
deltafield = np.zeros(denseN**3)
for idx in range(Nbatchs):
    batchidx_start = idx * Nperbatch
    batchidx_end = (idx+1) * Nperbatch-1
    if idx == Nbatchs-1:
        batchidx_end = Nconn-1
    dfdir = testsnapdir + "outputs_snap%03d_tet%d_N%d_x%d_y%d_z%d_batch%d/"%(fileidx,mytetsize,denseN,xshift,yshift,zshift,batchidx_start)
    dfbatch = np.load(dfdir+"deltafield.npy")
    deltafield = deltafield + dfbatch +1
deltafield = deltafield -1

# Use points that have values
treefield=deltafield[np.where(deltafield!=-1)[0]]
treepts = denseGrid[np.where(deltafield!=-1)[0]]
# to build KDTree which we will use to interpolate fields
mytree = cKDTree(treepts)

# estimate correlation
sampleliml=50.0
samplelimr=200.0
# make random sample of centers
samplePcen = sampleliml + sampleradius + (samplelimr-sampleliml-2*sampleradius) * np.random.random(size=(sampleNcen**3,3))
# make a large Fibonacci sphere, randomly choose points
Nmysph = 10000
mysph =  fibonacci_sphere(Nmysph)
samplexi = [] # sample of the pair correlation function \xi (r)
cnt =0 
for centerP in samplePcen:
    if cnt%1000 ==0:
        np.save(corrdir+"randcnt_%d.npy"%(cnt),np.array([]))
    cnt+=1
    deltacen = treefield[mytree.query(centerP)[1]]
    # make a large Fibonacci sphere, randomly choose points
    samplePsph = mysph [ np.random.randint(Nmysph,size=sampleNsph) ]
    sphereP = centerP + sampleradius*samplePsph
    deltasph = treefield[mytree.query(sphereP)[1]]
    samplexi = np.append(samplexi,deltacen*deltasph)


os.system("rm "+corrdir+"randcnt*.npy\n")
np.save(corrdir+"randsamplexi.npy",samplexi)
np.save(corrdir+"randsampleradius.npy",sampleradius)
np.save(corrdir+"randmeanxi.npy",np.mean(samplexi))