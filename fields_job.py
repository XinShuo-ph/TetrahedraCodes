import PSI as psi
# import helpers as hlp
import numpy as np
# import matplotlib.pyplot as plt
import sys

oakdir="/oak/stanford/orgs/kipac/users/xinshuo/"
snapdir=oakdir+"QuijoteData/Snapshots/fiducial/10000/"


mytetsize = int(sys.argv[1])
currentz= int(sys.argv[2])
xshift,yshift,zshift= int(sys.argv[3]),int(sys.argv[4]),int(sys.argv[5])
ngrid = 3*(int(sys.argv[6]),)

#remake the mesh
print("reading")
mypos = np.load(snapdir+"pos_tet%d_z%f_shift%d%d%d.npy"%(mytetsize,currentz,xshift,yshift,zshift))
conn = np.load(snapdir+"conn_tet%d.npy"%mytetsize)
print(mypos)
vel = np.random.sample(mypos.shape)
mass = np.ones(conn.shape[0])
pbox = ((0.0,0.0,0.0),(1e3,1e3,1e3))
mesh = psi.Mesh(loader='array', posar=mypos, velar=vel, massar=mass, connar=conn, 
    box=pbox)


# create the Grid, specifying the resolution and projection window

win = (mesh.boxmin, mesh.boxmax)
grid = psi.Grid(type='cart', n=ngrid, window=win)
print(mesh.connectivity)
# call PSI.voxels()
psi.voxels(grid=grid, mesh=mesh, mode='density')
#psi.voxels(grid=grid, mesh=mesh, mode='annihilation')

# check the total mass
# show a picture
elemmass = np.sum(mesh.mass)
voxmass = np.sum(grid.fields["m"])
err = np.abs(1.0-voxmass/elemmass)

np.save(snapdir+"fields_tet%d_ngrid%d_z%f_shift%d%d%d.npy"%(mytetsize,int(sys.argv[6]),currentz,xshift,yshift,zshift),grid.fields["m"])

# print the error and show the figure
print('Global error = %.10e' % err)
#hlp.makeFigs(grid.fields['m'], log=True, title='test2',colors=plt.cm.jet)
