"""Decomposition of a mesh into two submeshes

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

This script is a part of implementation of
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'
"""

import numpy as np
from matrices import create_A
from read_write_mesh import write_mesh_obj


def vertex_subset(nVRow, nVCol, start, V):
    V_sub = np.zeros(nVRow*nVCol*3).reshape(nVRow*nVCol,3)
    for i in range(nVCol):
        for j in range(nVRow):
            V_sub[i*nVRow + j,:] = V[(start + i)*nVRow + j,:]
    return V_sub

def face_subset(nF, F):
    F_sub = []
    for i in range(nF):
        F_sub.append(F[i])
    return F_sub

def hedge_subset(nH, H):
    H_sub = np.zeros(nH*2, dtype = int).reshape(nH,2)
    for i in range(nH):
        H_sub[i,:] = H[i,:]
    return H_sub

def boundary_vertices(nV, H, gamma):
    A = create_A(H)
    diagA = np.floor(A.diagonal()).astype(bool)
    boundary_V = H[diagA,0]
##    boundary_V = []
##    diagA = A.diagonal()
##    for i in range(len(H)):
##        if diagA[i] == 1:
##            boundary_V.append(H[i,0])
    boundary_V = np.array([x for x in boundary_V if x not in gamma])
    return np.array(boundary_V)

###---------------------------------------------------------

def decompose(V: np.ndarray, H: np.ndarray, F: list,\
              typ, divRowV: int, nVinRow: int, overlap = 0) -> list:

    if typ == 5: ## a type of mesh, where the the faces corresponding to the interface
                 ## of the upper part are pentagons
        nV_L = (divRowV-1)*nVinRow + 2*nVinRow
        nV_U = len(V) - nV_L + 2*nVinRow
        gamma_V_L = V[nV_L- 2*nVinRow : nV_L,:].copy()
        gamma_V_U = V[nV_L- 2*nVinRow : nV_L,:].copy()
        V_L = np.concatenate((V[0:nV_L- 2*nVinRow, :],gamma_V_L), axis = 0)
        V_U = np.concatenate((gamma_V_U, V[nV_L : , :]), axis = 0)

        nF_U = nV_U - 2*nVinRow
        nF_L = len(F) - nF_U
        F_L = F[0:nF_L]
        F_U = F[nF_L: ]
        for f in F_U:
            f -= (nV_L - 2*nVinRow)*np.ones(len(f), dtype = np.int32)

        nH_L = (nF_L - nVinRow)*4 + (nVinRow)*5
        nH_U = len(H) - nH_L
        H_L = H[0:nH_L,:]
        H_U = H[nH_L:,:] - (nV_L-2*nVinRow)*np.ones(2*nH_U,\
                                                    dtype = np.int32).reshape(nH_U,2)

        gamma_L = np.array([x for x in range(nV_L - 2*nVinRow, nV_L)])
        gamma_U = np.array([x for x in range(0, 2*nVinRow)])


        boundary_L = boundary_vertices(nV_L, H_L, gamma_L)
        boundary_U = boundary_vertices(nV_U, H_U, gamma_U)

        submeshes = [[V_L, H_L, F_L, boundary_L, gamma_L],\
                     [V_U, H_U, F_U, boundary_U, gamma_U]]
##        write_mesh_obj("U.obj", V_U, F_U)
##        write_mesh_obj("L.obj", V_L, F_L)


    elif typ == 4 or typ == 3:
        nVinColumn = int(len(V)/nVinRow)
        nVCol_L = divRowV + 1 + overlap
        nVCol_U = nVinColumn - divRowV

        V_L = vertex_subset(nVinRow, nVCol_L, 0, V)
        V_U = vertex_subset(nVinRow, nVCol_U, divRowV, V)

        if typ == 4:
            if F[0][0] == F[nVinRow-1][1]:
                nF_L = (nVCol_L-1)*(nVinRow)
                nF_U = (nVCol_U-1)*(nVinRow)
            else:
                nF_L = (nVCol_L-1)*(nVinRow - 1)
                nF_U = (nVCol_U-1)*(nVinRow - 1)
        elif typ == 3:
            if F[0][0] == F[2*nVinRow-2][1]:
                nF_L = 2*(nVCol_L-1)*(nVinRow)
                nF_U = 2*(nVCol_U-1)*(nVinRow)
            else:
                nF_L = 2*(nVCol_L-1)*(nVinRow - 1)
                nF_U = 2*(nVCol_U-1)*(nVinRow - 1)

        nH_L = nF_L*typ
        nH_U = nF_U*typ
        
        F_L = face_subset(nF_L, F)
        F_U = face_subset(nF_U, F)
        H_L = hedge_subset(nH_L, H)
        H_U = hedge_subset(nH_U, H)

        if F[0][0] == F[nVinRow-1][1] or F[0][0] == F[2*nVinRow-2][1]:
            boundary_L = np.arange(nVinRow) 
            gamma_L = np.arange(len(V_L)-nVinRow, len(V_L))
            boundary_U = np.arange(len(V_U)-nVinRow ,len(V_U))
            gamma_U = np.arange(nVinRow)            
        else:
            boundary_L = np.array([x for x in range(nVinRow)] +\
                                  [x*nVinRow for x in range(1,nVCol_L)] +\
                                  [x*nVinRow - 1 for x in range(2,nVCol_L+1)])
            gamma_L = np.array([x for x in range(len(V_L)-nVinRow+1,len(V_L)-1)])           
            boundary_U = np.array([x*nVinRow for x in range(0,nVCol_U-1)] +\
                                  [x*nVinRow - 1 for x in range(1,nVCol_U)] +\
                                  [x for x in range(len(V_U)- nVinRow ,len(V_U))])
            gamma_U = np.array([x for x in range(1,nVinRow-1)])


        submeshes = [[V_L, H_L, F_L, boundary_L, gamma_L],\
                     [V_U, H_U, F_U, boundary_U, gamma_U]]
        

    elif typ == 6 and overlap == 0:
        nV_L = divRowV*nVinRow - 1
        nV_U = len(V) - nV_L + nVinRow
        gamma_V_L = V[nV_L- nVinRow+1: nV_L-1,:].copy()
        gamma_V_U = V[nV_L- nVinRow+1: nV_L-1,:].copy()
        V_L = np.concatenate((V[0:nV_L- nVinRow+1, :],gamma_V_L,\
                              V[nV_L-1:nV_L,:]), axis = 0)
        V_U = np.concatenate((V[nV_L- nVinRow:nV_L- nVinRow+1, :],gamma_V_U,\
                              V[nV_L-1: , :]), axis = 0)

        
        nFinRow = int(nVinRow/2)
        nF_L = (divRowV-1)*nFinRow - 1
        nF_U = len(F) - nF_L
        F_L = F[0:nF_L]
        F_U = F[nF_L: ]
        for f in F_U:
            f -= (nV_L - nVinRow)*np.ones(len(f), dtype = np.int32)

        nH_L = (divRowV - 1)*(nFinRow -1)*6 + (divRowV - 2)*4
        nH_U = len(H) - nH_L
        H_L = H[0:nH_L,:]
        H_U = H[nH_L:,:] - (nV_L-nVinRow)*np.ones(2*nH_U,\
                                                  dtype = np.int32).reshape(nH_U,2)

        gamma_L = np.array([x for x in range(nV_L - nVinRow + 1, nV_L - 1)])
        gamma_U = np.array([x for x in range(1, nVinRow - 1)])


        boundary_L = boundary_vertices(nV_L, H_L, gamma_L)
        boundary_U = boundary_vertices(nV_U, H_U, gamma_U)

        submeshes = [[V_L, H_L, F_L, boundary_L, gamma_L],\
                     [V_U, H_U, F_U, boundary_U, gamma_U]]

    elif typ == 6 and overlap > 0:
        nV_L = divRowV*nVinRow - 1
        nV_U = len(V) - nV_L + (1+overlap)*nVinRow
        gamma_V_L = V[nV_L- nVinRow+1: nV_L-1,:].copy()
        gamma_V_U = V[nV_L-(1+overlap)*nVinRow+1:nV_L-(overlap)*nVinRow- 1,:].copy()
        V_L = np.concatenate((V[0:nV_L- nVinRow+1, :],gamma_V_L,\
                              V[nV_L-1:nV_L,:]), axis = 0)
        V_U = np.concatenate((V[nV_L-(1+overlap)*nVinRow:nV_L-(1+overlap)*nVinRow+1,:],\
                              gamma_V_U, V[nV_L-(overlap)*nVinRow-1: , :]), axis = 0)

        
        nFinRow = int(nVinRow/2)
        nF_L = (divRowV-1)*nFinRow - 1
        nF_U = len(F) - nF_L + overlap*nFinRow
        F_L = F[0:nF_L]
        Fu_U = F[len(F) - nF_U: ]
        F_U = []
        for f in Fu_U:
            F_U.append(f - (len(V) - nV_U)*np.ones(len(f), dtype = np.int32))

##        write_mesh_obj("U.obj", V_U, F_U)
##        write_mesh_obj("L.obj", V_L, F_L)

        nH_L = (divRowV - 1)*(nFinRow -1)*6 + (divRowV - 2)*4
        nH_U = len(H) - nH_L + (overlap)*(nFinRow -1)*6 + (overlap)*4
        H_L = H[0:nH_L,:]
        H_U = H[len(H)-nH_U:,:]-(len(V)-nV_U)*np.ones(2*nH_U,\
                                                      dtype = np.int32).reshape(nH_U,2)

        gamma_L = np.array([x for x in range(nV_L - nVinRow + 1, nV_L - 1)])
        gamma_U = np.array([x for x in range(1, nVinRow - 1)])


        boundary_L = boundary_vertices(nV_L, H_L, gamma_L)
        boundary_U = boundary_vertices(nV_U, H_U, gamma_U)

        submeshes = [[V_L, H_L, F_L, boundary_L, gamma_L],\
                     [V_U, H_U, F_U, boundary_U, gamma_U]]
    
        
    return submeshes
