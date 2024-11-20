import numpy as np
#20221106: cleaned up and collect some of the useful subroutines

# change particle ids from Quijote simulation's to the standard ids
def changeID(ids,Ndim=512,Ndim_bybox=8):
    small_Ndim=Ndim/Ndim_bybox
    smallid= ((ids)%(small_Ndim**3)).astype(int)
    BoxID= ((ids)/(small_Ndim**3)).astype(int)
    BoxXID = (BoxID/(Ndim_bybox**2)).astype(int)
    BoxYID = ((BoxID-(Ndim_bybox**2)*BoxXID)/(Ndim_bybox)).astype(int)
    BoxZID = (BoxID % Ndim_bybox).astype(int)
    smallxid = ( smallid / (small_Ndim**2) ).astype(int)
    smallyid = ( (smallid - (small_Ndim**2)*smallxid )/ small_Ndim ).astype(int)
    smallzid = (smallid % small_Ndim).astype(int)
    xid = BoxXID*small_Ndim + smallxid
    yid = BoxYID*small_Ndim + smallyid
    zid = BoxZID*small_Ndim + smallzid
    ids_std = xid*(Ndim**2) + yid*Ndim + zid
    return ids_std

# given mypoints, find those inside the tetrahedron with vertices P0,P1,P2,P3
def InTetraList(mypoints,P0,P1,P2,P3,get_barycentric=False):
    if mypoints.shape[0]==0:
        return []
    mypoints_app=np.append(mypoints,np.ones((mypoints.shape[0],1)),axis=1)
    P0_app=np.append(P0,1)
    P1_app=np.append(P1,1)
    P2_app=np.append(P2,1)
    P3_app=np.append(P3,1)
    P0_app=np.repeat([P0_app ],mypoints.shape[0],axis=0)
    P1_app=np.repeat([P1_app ],mypoints.shape[0],axis=0)
    P2_app=np.repeat([P2_app ],mypoints.shape[0],axis=0)
    P3_app=np.repeat([P3_app ],mypoints.shape[0],axis=0)
    
    D0=np.linalg.det(np.stack((P0_app,P1_app,P2_app,P3_app),axis=1))
    D1=np.linalg.det(np.stack((mypoints_app,P1_app,P2_app,P3_app),axis=1))
    D2=np.linalg.det(np.stack((P0_app,mypoints_app,P2_app,P3_app),axis=1))
    D3=np.linalg.det(np.stack((P0_app,P1_app,mypoints_app,P3_app),axis=1))
    D4=np.linalg.det(np.stack((P0_app,P1_app,P2_app,mypoints_app),axis=1))
    
    insidelist=np.intersect1d( np.where(np.sign(D4)==np.sign(D0))[0] ,
               np.intersect1d( np.where(np.sign(D3)==np.sign(D0))[0] ,
               np.intersect1d( np.where(np.sign(D2)==np.sign(D0))[0] , np.where(np.sign(D1)==np.sign(D0))[0] ) ) )
    if get_barycentric:
        barycentric = np.stack([ D1/D0,D2/D0,D3/D0,D4/D0 ],axis=1)
        return [insidelist,barycentric[insidelist]]
    else:
        return insidelist

def get_tet_conn(Ndim):
    """ compute the connectivity list of all tetrahedra"""
    #note that p3d here is array of indices, not location of particles
    #e.g. the selected3idx variable in the draft reversemap_usetet_draft 
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,7,3,1), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    conn_all = np.zeros((Np*Ntetpp,4,3),dtype="int")
    newxg=np.meshgrid(np.arange(Ndim+1),np.arange(Ndim+1),np.arange(Ndim+1),indexing='ij')
    p3d=np.stack(newxg,axis=3)
    for m in range(Ntetpp):   # 6 tets
        off = vert[conn[m]]
        vet1 =(p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        vet2 =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] ).reshape((Np,3))
        vet3 =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] ).reshape((Np,3))
        vet4 =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] ).reshape((Np,3))   
        conn_all[m::Ntetpp,:,:] = np.stack([vet1,vet2,vet3,vet4],axis=1)
    return conn_all


def get_tet_eigenvalues_anysize(Ndim,grid3d,Phi3d,tetsize):
    ''' estimate eigenvalues of dPhi/dq at tetrahedra centers, by displacement fields on 4 vertices. 
        grid3d: particle's Lagrangian grid coordinates, of shape (Ndim+tetsize,Ndim+tetsize,Nim+tetsize,3)
        Phi3d: displacment field, of shape (Ndim+tetsize,Ndim+tetsize,Nim+tetsize,3)
        return eigenvalues: 3 eigenvalues of dPhi/dq at tetrahedra centers, of shape (Np*Ntetpp,3)  '''
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,7,3,1), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    dPhidq=np.zeros((Np*Ntetpp,3,3))
    for m in range(Ntetpp):
        off = tetsize*(vert[conn[m]])
        P=np.zeros((Np,3,4)) # coordinates of 4 vertices
        myPhi=np.zeros((Np,3,4)) # displacement fields on 4 vertices
        P[:,:,0]    =np.reshape(grid3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :],(Np,3))
        P[:,:,1]    =np.reshape(grid3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :],(Np,3))
        P[:,:,2]    =np.reshape(grid3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :],(Np,3))
        P[:,:,3]    =np.reshape(grid3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :],(Np,3))
        myPhi[:,:,0]=np.reshape(Phi3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] ,(Np,3))
        myPhi[:,:,1]=np.reshape(Phi3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] ,(Np,3))
        myPhi[:,:,2]=np.reshape(Phi3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] ,(Np,3))
        myPhi[:,:,3]=np.reshape(Phi3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ,(Np,3))
        center=(P[:,:,0]+P[:,:,1]+P[:,:,2]+P[:,:,3])/4
        centerPhi=(myPhi[:,:,0]+myPhi[:,:,1]+myPhi[:,:,2]+myPhi[:,:,3])/4   
        # dq from center to P[1],P[2],P[3]
        dq = np.zeros((Np,3,3))
        for i in [1,2,3]:
            for k in [1,2,3]:
                dq[:,i-1,k-1] = P[:,i-1,k] - center[:,i-1]
        # dPhi from center to P[1],P[2],P[3]
        deltaPhi = np.zeros((Np,3,3))
        for j in [1,2,3]:
            for k in [1,2,3]:
                deltaPhi [:,j-1,k-1] = myPhi[:,j-1,k] - centerPhi [:,j-1]
        dPhidq[m::Ntetpp,:,:] = np.matmul (deltaPhi,np.linalg.inv(dq)  )
    eigenvalues, eigenvectors = np.linalg.eig(dPhidq)
    return eigenvalues

def get_tet_eigenvalues_2idx_anysize(Ndim,grid3d,Phi3d,tetsize,printlog=False):
    ''' estimate eigenvalues of dPhi/dq at tetrahedra centers, by displacement fields on 4 vertices. 
        grid3d: particle's Lagrangian grid coordinates, of shape (Ndim+1,Ndim+1,Nim+1,3)
        Phi3d: displacment field, of shape (Ndim+1,Ndim+1,Nim+1,3)
        return eigenvalues: 3 eigenvalues of dPhi/dq at tetrahedra centers, of shape (Np*Ntetpp,3)  '''
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,7,3,1), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    dPhidq=np.zeros((Np,Ntetpp,3,3))
    if printlog:
        print("computing dPhidq.")
    for m in range(Ntetpp):        
        if printlog:
            print("m=%d"%m)
        off = tetsize*(vert[conn[m]])
        if printlog:
            print("get particles")
        P=np.zeros((Np,3,4)) # coordinates of 4 vertices
        myPhi=np.zeros((Np,3,4)) # displacement fields on 4 vertices
        P[:,:,0]    =np.reshape(grid3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :],(Np,3))
        P[:,:,1]    =np.reshape(grid3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :],(Np,3))
        P[:,:,2]    =np.reshape(grid3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :],(Np,3))
        P[:,:,3]    =np.reshape(grid3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :],(Np,3))
        if printlog:
            print("get Phi")
        myPhi[:,:,0]=np.reshape(Phi3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] ,(Np,3))
        myPhi[:,:,1]=np.reshape(Phi3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] ,(Np,3))
        myPhi[:,:,2]=np.reshape(Phi3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] ,(Np,3))
        myPhi[:,:,3]=np.reshape(Phi3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ,(Np,3))
        center=(P[:,:,0]+P[:,:,1]+P[:,:,2]+P[:,:,3])/4
        centerPhi=(myPhi[:,:,0]+myPhi[:,:,1]+myPhi[:,:,2]+myPhi[:,:,3])/4   
        # dq from center to P[1],P[2],P[3]
        if printlog:
            print("get dq")
        dq = np.zeros((Np,3,3))
        for i in [1,2,3]:
            for k in [1,2,3]:
                dq[:,i-1,k-1] = P[:,i-1,k] - center[:,i-1]
        # dPhi from center to P[1],P[2],P[3]
        if printlog:
            print("get dPhi")
        deltaPhi = np.zeros((Np,3,3))
        for j in [1,2,3]:
            for k in [1,2,3]:
                deltaPhi [:,j-1,k-1] = myPhi[:,j-1,k] - centerPhi [:,j-1]
        if printlog:
            print("get dPhidq")
        dPhidq[:,m,:,:] = np.matmul (deltaPhi,np.linalg.inv(dq)  )
        #release memory
        P=None
        myPhi=None
        center=None
        centerPhi=None
        dq=None
        deltaPhi=None
    if printlog:
        print("get eigenvalues")
    eigenvalues, eigenvectors = np.linalg.eig(dPhidq)
    if printlog:
        print("finished")
    return eigenvalues


# By Tom
def get_tet_volumes_anysize(Ndim,p3d,tetsize):
    """A fast function to compute the volumes of all tetrahedra"""
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,7,3,1), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    vol = np.zeros((Ndim,Ndim,Ndim, Ntetpp))
    for m in range(Ntetpp):   # 6 tets
        off = tetsize*(vert[conn[m]])
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))
        b = np.cross(b,c)
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :] ).reshape((Np,3))        
        vol[:,:,:,m] = (-np.sum(a*b,axis=1)/6.).reshape((Ndim,Ndim,Ndim))
    return vol

def get_tet_centroids_anysize(Ndim,p3d,tetsize):
    """ A fast function to compute the centroids of all tetrahedra """
    vert = np.array(( (0,0,0), (1,0,0), (1,1,0), (0,1,0), (0,0,1), (1,0,1), (1,1,1), (0,1,1) ))
    conn = np.array( ( (4,0,7,1), (1,0,7,3), (5,1,4,7), (2,7,3,1), (1,5,6,7), (2,6,7,1) ))
    Ntetpp = len(conn)
    Np = Ndim*Ndim*Ndim
    cen = np.zeros((Np*Ntetpp,3))
    for m in range(Ntetpp):   # 6 tets
        off = tetsize*(vert[conn[m]])
        orig = p3d[off[3][0]:(Ndim+off[3][0]),off[3][1]:(Ndim+off[3][1]),off[3][2]:(Ndim+off[3][2]), :]
        b =  ( p3d[off[1][0]:(Ndim+off[1][0]),off[1][1]:(Ndim+off[1][1]),off[1][2]:(Ndim+off[1][2]), :] \
            - orig ).reshape((Np,3))
        c =  ( p3d[off[2][0]:(Ndim+off[2][0]),off[2][1]:(Ndim+off[2][1]),off[2][2]:(Ndim+off[2][2]), :] \
            - orig ).reshape((Np,3))
        a =  ( p3d[off[0][0]:(Ndim+off[0][0]),off[0][1]:(Ndim+off[0][1]),off[0][2]:(Ndim+off[0][2]), :] \
            - orig).reshape((Np,3))   
        cen[m::Ntetpp,:] = orig.reshape(Np,3) + ((a+b+c)/4.)
    return cen

# By Phil
# %% Imports 
from scipy.special import erf
from scipy.special import erfc
from math import sqrt,exp
from scipy.integrate import quad
import random


# %% Probability distributions of eigenvalues
def p_lambda1(lambda1,sigma):
    los = lambda1/sigma
    ret = sqrt(5)/(12*np.pi*sigma)*(20*los*exp(-9/2*los**2)-\
        sqrt(2*np.pi)*exp(-5/2*los**2)*(1-20*los**2)*(erf(sqrt(2)*los)+1)+\
        3*sqrt(3*np.pi)*exp(-15/4*los**2)*(erf(sqrt(3)/2*los)+1))
    return ret
p_lambda1_vec = np.vectorize(p_lambda1)

def p_lambda2(lambda2,sigma):
    los = lambda2/sigma
    ret = sqrt(15)/(2*sqrt(np.pi)*sigma)*exp(-15/4*los**2)
    return ret
p_lambda2_vec = np.vectorize(p_lambda2)

def p_lambda3(lambda3,sigma):
    los = lambda3/sigma
    ret = -sqrt(5)/(12*np.pi*sigma)*(20*los*exp(-9/2*los**2)+\
        sqrt(2*np.pi)*exp(-5/2*los**2)*(1-20*los**2)*(erfc(sqrt(2)*los))-\
        3*sqrt(3*np.pi)*exp(-15/4*los**2)*(erfc(sqrt(3)/2*los)))
    return ret
p_lambda3_vec = np.vectorize(p_lambda3)


def p_lambda123(lambda1,lambda2,lambda3,sigma):
    I1 = lambda1+lambda2+lambda3
    I2 = lambda1*lambda2+lambda2*lambda3+lambda3*lambda1
    ret = 3375.0/(8*sqrt(5)*np.pi*sigma**6)*\
        exp(-3*I1**2/sigma**2+15*I2/2/sigma**2)*\
        (lambda2-lambda1)*(lambda3-lambda2)*(lambda3-lambda1)
    return ret

def p_lambda3_l3c_delta(lambda3, l3c, delta, sigma):
    dif = delta-3*l3c
    ret = (-3/4*np.sqrt(10/np.pi)/sigma)*dif*np.exp(-5*dif**2/8/sigma**2)+0.5*(erf(dif*np.sqrt(10)/(4*sigma))+erf(dif*np.sqrt(10)/(2*sigma)))*np.heavyside(dif)
    return ret

# Eq 25 of Kofman's https://arxiv.org/abs/astro-ph/9311028
def p_rho(rho,sigma):
    def bt(n,s):
        output= s * sqrt(5) *( 0.5 + np.cos( np.pi*2.0*(n-1)/3.0   + np.arccos( 54/rho/s**3 -1.0 )/3.0 ) )
        return output
    def myintegrand(s):
        output = (9* sqrt(5**3) )/(4*np.pi* rho**2 * sigma**4)* exp(-((s-3.0)**2.0) /( 2 * sigma**2 )) *( 1+ exp(- 6*s/sigma**2 ) ) * (
            exp( -bt(1,s)**2/( 2 * sigma**2 ) )+exp(-bt(2,s)**2/( 2 * sigma**2 ))-exp(-bt(3,s)**2/( 2 * sigma**2 )))
        return output
    solution = quad(myintegrand,3/np.cbrt(rho),np.inf)
    return solution[0]
p_rho_vec = np.vectorize(p_rho)

# evenly distribution smaple points on a sphere:
# copied from https://stackoverflow.com/questions/9600801/evenly-distributing-n-points-on-a-sphere
def fibonacci_sphere(samples):

    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(samples):
        y = 1 - (i / float(samples - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append(np.array([x, y, z]))

    return np.array(points)
