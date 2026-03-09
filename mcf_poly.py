"""Mesh smoothing by implicit mean curvature flow of surface polygonal
meshes with or without boundary. If the mesh has a boundary, it is kept fixed.

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License.

This script is a part of implementation of methods supporting the paper
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
from scipy.sparse import identity, diags, block_diag, csc_matrix
from scipy.sparse.linalg import spsolve
from read_write_mesh import read_mesh_obj, write_mesh_obj, rewrite_vertices
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


def mcf_poly(filename: str, Laplacian: str, maxIter: int, dt: float) -> None:
    
    V, H, F = read_mesh_obj(filename + ".obj")
    if Laplacian == "PV":
        fv, R, A, d0, d1 = mm.create_incidence_matrices(F, H, V)
    elif Laplacian == "AW":
        R, A, d0 = mm.create_incidence_matrices_AW(F, H, V)
    else:
        A, d0 = mm.create_incidence_matrices_FU(H, V)
   
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

    curvature = max(np.linalg.norm(Laplace[interior_V,:]@V, axis = 1))
    print("Maximal mean curvature:")
    print([curvature])

    it = 0
    filenames = []
    t_linSys = 0
    t_Lap = 0
    while it< maxIter:
        it += 1
        V[interior_V] = mean_curvature_flow(Laplace, V, dt,\
                                            boundary_V, interior_V) 

        if Laplacian == "FU":
            Laplace = Fujiwara(d0, V)
        elif Laplacian == "AW":
            Laplace = AleWar(d0, R, F, H, V)
        elif Laplacian == "PV":
            Laplace = PtaVel(fv, R, A, d0, d1, F, H, V)
        
        curvature = max(np.linalg.norm(Laplace[interior_V,:]@V, axis = 1))
        print([curvature])
            
        rewrite_vertices(filename, V, "_{}_dt{}_it_{}".\
                         format(Laplacian, dt, it))
        nameParts = filename.split(sep="/")
        filenames.append(nameParts[0] + "/output/" + nameParts[1] +\
                         "_{}_dt{}_it_{}".format(Laplacian, dt, it))

    return filenames
