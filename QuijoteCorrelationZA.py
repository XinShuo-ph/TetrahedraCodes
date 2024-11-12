import numpy as np
import sys
from colossus.cosmology import cosmology
from tetrahedrafunc import *
from scipy.interpolate import interp1d
import readgadget
from time import perf_counter as time
from scipy.spatial import cKDTree
import os

oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial/10000/"

QuijoteFid = {'H0': 67.11,
 'Ob0': 0.049, 
 'Om0': 0.3175,
 'flat': True,
 'ns': 0.9624,
 'sigma8': 0.834}
mycosmos =  cosmology.setCosmology("QuijoteFid", **QuijoteFid)

rlist = np.linspace(0.1,200,40)
mytetsize=int(sys.argv[1])
snapno = int(sys.argv[2])
ngrid = int(sys.argv[3])
xshift = int(sys.argv[4])
yshift = int(sys.argv[5])
zshift = int(sys.argv[6])
rid   = int(sys.argv[7])
sampleNcen = int(sys.argv[8])
sampleNsph = int(sys.argv[9])
sampleradius = rlist[rid]
lbox = 1e3

zmap={-1:127,0:3,1:2,2:1,3:0.5,4:0}
currentz = zmap[snapno]
    
outputdir = snapdir + "correlationZA_snap%03d_tet%d_N%d_x%d_y%d_z%d/"%(snapno,mytetsize,ngrid,xshift,yshift,zshift)
os.system("mkdir "+ outputdir)
corrdir = outputdir+"xi_rid%d_cen%d_sph%d/"%(rid,sampleNcen,sampleNsph)
os.system("mkdir "+ corrdir)

# read delta field
print("reading field")
rhos = np.load(snapdir+"fieldsZA_tet%d_z%f_shift%d%d%d.npy"%(mytetsize,currentz,xshift,yshift,zshift))
deltafield = np.ravel(rhos/np.average(rhos)-1)

x = lbox/ngrid*(0.5+np.arange(ngrid))
xg = np.meshgrid(x,x,x, indexing='ij')
xl = np.ravel(xg[0])
yl = np.ravel(xg[1])
zl = np.ravel(xg[2])
gridpts=np.vstack([xl,yl,zl]).transpose()

print("making tree")
mytree = cKDTree(gridpts)

# estimate correlation
print("estimating correlation")
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
        print(cnt)
        np.save(corrdir+"randcnt_%d.npy"%(cnt),np.array([]))
    cnt+=1
    deltacen = deltafield[mytree.query(centerP)[1]]
    # make a large Fibonacci sphere, randomly choose points
    samplePsph = mysph [ np.random.randint(Nmysph,size=sampleNsph) ]
    sphereP = centerP + sampleradius*samplePsph
    deltasph = deltafield[mytree.query(sphereP)[1]]
    samplexi = np.append(samplexi,deltacen*deltasph)


os.system("rm "+corrdir+"randcnt*.npy\n")
np.save(corrdir+"randsamplexi.npy",samplexi)
np.save(corrdir+"randsampleradius.npy",sampleradius)
np.save(corrdir+"randmeanxi.npy",np.mean(samplexi))