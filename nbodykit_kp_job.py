import numpy as np
import sys
from nbodykit.source.mesh import ArrayMesh
from nbodykit.algorithms import FFTPower

oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial/10000/"
Lbox = 1e3

mytetsize = int(sys.argv[1])
currentz= int(sys.argv[2])
xshift,yshift,zshift= int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])

print("reading")
rhosZA = np.load(snapdir+"fieldsZA_tet%d_ngrid%d_z%f_shift%d%d%d.npy"%(mytetsize,int(sys.argv[6]),currentz,xshift,yshift,zshift))
print("read")
deltaZA = rhosZA/np.average(rhosZA) - 1
ZAmesh = ArrayMesh(deltaZA,Lbox)
print("got mesh")
ZApkobj = FFTPower(ZAmesh,'1d')
print("got pk")
ZApk = [ZApkobj.power['k'],ZApkobj.power['power']]
np.save(snapdir+"pkZA_tet%d_ngrid%d_z%f_shift%d%d%d.npy"%(mytetsize,int(sys.argv[6]),currentz,xshift,yshift,zshift),ZApk)

print("reading")
rhos = np.load(snapdir+"fields_tet%d_ngrid%d_z%f_shift%d%d%d.npy"%(mytetsize,int(sys.argv[6]),currentz,xshift,yshift,zshift))
print("read")
delta = rhos/np.average(rhos) - 1
mesh = ArrayMesh(delta,Lbox)
print("got mesh")
pkobj = FFTPower(mesh,'1d')
print("got pk")
pk = [pkobj.power['k'],pkobj.power['power']]
np.save(snapdir+"pk_tet%d_ngrid%d_z%f_shift%d%d%d.npy"%(mytetsize,int(sys.argv[6]),currentz,xshift,yshift,zshift),pk)
