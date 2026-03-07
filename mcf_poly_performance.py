"""Mesh smoothing by implicit mean curvature flow of surface polygonal
meshes with or without boundary. If the mesh has a boundary, it is kept
fixed. This script serves for measuring times.

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

This script is a part of implementation of
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'

The method uses backward Euler scheme and Laplacian on vector-valued 0-forms
that represent vertex position of the given mesh.

There are three Laplacians implemented:
1) "FU": scale–dependent umbrella operator of Fujiwara (1995)
    [doi:10.2307/2161293];
2) "AW": polygonal Laplacian of Alexa and Wardetzky (2011)
    [doi:10.1145/2010324.1964997],
3) "PV": polygonal Laplacian of Ptackova and Velho (2021)
    [doi:10.1016/j.cagd.2021.102002].

"""

import numpy as np
import math
import time
from scipy.sparse import identity, diags, block_diag, csc_matrix
from scipy.sparse.linalg import spsolve
from read_write_mesh import read_mesh_obj
import matrices as mm

float_precis = np.float32


def Fujiwara(d0, V)-> np.ndarray:
    vec_WE = np.linalg.norm(d0@V, axis=1)
    vec_WE = np.where(vec_WE < 10**(-8),1,vec_WE)
    vec_sum_WE = 0.5*np.abs(d0.transpose())@vec_WE
    sigma = 0.5*np.abs(d0.transpose())@np.ones(len(vec_WE))
    Lap = 0.5*diags(sigma/vec_sum_WE)@d0.transpose()@\
          diags(1/vec_WE)@d0
    return csc_matrix(Lap)


def AleWar(d0, R, F, H, V)-> np.ndarray:
    """Creates matrix representing Laplace operator on 0-forms"""
    W1, WV = mm.W_matrices_AW(F, V)
    Lap = WV@d0.transpose()@R@W1@R.transpose()@d0
    return csc_matrix(Lap)


def PtaVel(fv, R, A, d0, d1, F, H, V)-> np.ndarray:
    """Creates matrix representing Laplace operator on 0-forms"""
    WF, W1, WV = mm.W_matrices_PV(F, V)
    Lap = -WV@fv.transpose()@d1@A@W1@R.transpose()@d0
    return csc_matrix(Lap)


def mean_curvature_flow(Laplace, V, dt, boundaryV, interiorV):
    Lap = Laplace[interiorV,:][: , interiorV] 
    intV = V[interiorV, :]
    if len(boundaryV) > 0:
        intV += -dt*Laplace[interiorV,:][:,boundaryV]@V[boundaryV, :]
    Delta = csc_matrix(identity(len(interiorV)) + dt*Lap)
    return spsolve(Delta, intV)


if __name__ == '__main__':
##    filename = "meshes/tri_anisotropic_cili_28000_200"
##    filename = "meshes/tri_anisotropic_cili_10080_120"
    filename = "meshes/quad_cilinder_wavy_10200_120"
##    filename = "meshes/quad_cilinder_wavy_28000_200"

    Laplacian = "AW"
    print(filename + " " + Laplacian + "\n topology, Laplacian, spsolve")
    maxIter, dt = 10, 0.05
    
    V, H, F = read_mesh_obj(filename + ".obj")
    
    t0 = time.time()
    if Laplacian == "PV":
        fv, R, A, d0, d1 = mm.create_incidence_matrices(F, H, V)
    elif Laplacian == "AW":
        R, A, d0 = mm.create_incidence_matrices_AW(F, H, V)
    else:
        A, d0 = mm.create_incidence_matrices_FU(H, V)
    print(time.time() - t0)
   
    diagA = np.floor(A.diagonal()).astype(bool)
    boundary_V = H[diagA,0]
    interior_V = np.array([x for x in range(len(V)) if x not in boundary_V])

    if Laplacian == "FU":
        Laplace = Fujiwara(d0, V)
    elif Laplacian == "AW":
        Laplace = AleWar(d0, R, F, H, V)
    elif Laplacian == "PV":
        Laplace = PtaVel(fv, R, A, d0, d1, F, H, V)
    else:
        print("unknown Laplacian")

    it = 0
    filenames = []
    t_linSys = 0
    t_Lap = 0
    while it< maxIter:
        it += 1
        t1 = time.time() 
        V[interior_V] = mean_curvature_flow(Laplace, V, dt,\
                                            boundary_V, interior_V) 
        t_linSys += time.time()-t1

        t2 = time.time()
        if Laplacian == "FU":
            Laplace = Fujiwara(d0, V)
        elif Laplacian == "AW":
            Laplace = AleWar(d0, R, F, H, V)
        elif Laplacian == "PV":
            Laplace = PtaVel(fv, R, A, d0, d1, F, H, V)
        t_Lap += time.time()-t2
    
    print("{:.6f}\n{:.6f}".\
          format(t_Lap/maxIter,t_linSys/maxIter))
    print("----------")
