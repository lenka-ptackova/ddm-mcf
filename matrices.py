"""Matrices representing building blocks of discrete Laplacians

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

This script is a part of implementation of methods supporting the paper
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'
"""

import numpy as np
from scipy.sparse import csr_matrix, block_diag, diags
from math import floor

float_precis = np.float32
int_precis = np.int32


def create_d0(H: np.ndarray, V: np.ndarray) -> np.ndarray:
    row = np.zeros(2*len(H), dtype = int_precis)
    col = np.zeros(2*len(H), dtype = int_precis)
    data = np.zeros(2*len(H), dtype = int_precis)
    for i in range(len(H)):
        row[i], col[i], data[i]  = i, H[i,0], -1
        row[i+len(H)], col[i+len(H)], data[i+len(H)]= i, H[i,1], 1
    return csr_matrix((data, (row, col)), shape = (len(H), len(V)),\
                      dtype = int_precis)


def create_d1(F: list, H: np.ndarray) -> np.ndarray:
    row = np.zeros(len(H), dtype = int_precis)
    col = np.arange(len(H), dtype = int_precis)
    data = np.ones(len(H), dtype = int_precis)
    k = 0
    for i in range(len(F)):
        for j in range(len(F[i])):
            row[k] = i
            k += 1       
    return csr_matrix((data, (row, col)), shape = (len(F), len(H)),\
                      dtype = int_precis)
       

def create_A(H: np.ndarray) -> np.ndarray:
    row, col = list(range(len(H))), list(range(len(H)))
    data = list(np.ones(len(H)))
    for i in range(len(H)):
        for j in range(i+1, len(H)):
            if H[i,0] == H[j,1] and H[i,1] == H[j,0]:
                data[i] = 0.5
                data[j] = 0.5
                
                data.append(-0.5)
                row.append(i)
                col.append(j)
                data.append(-0.5)
                row.append(j)
                col.append(i)            
    return csr_matrix((np.array(data), (row, col)), shape = (len(H), len(H)),\
                      dtype = float_precis)


def create_fv(F: list, V: np.ndarray) -> np.ndarray:
    row, col, data = [], [], []
    k = 0
    for i in range(len(F)):
        face = F[i]
        p = len(face)
        for j in range(p):
            row.append(i)
            col.append(face[j])
            data.append(1/p)
    return csr_matrix((np.array(data), (row, col)), shape = (len(F), len(V)),\
                      dtype = float_precis)


def create_R(F: list) -> np.ndarray:
    list_R = []
    for i in range(len(F)):
        p = len(F[i])
        R = np.zeros(p**2).reshape(p,p)
        for j in range(p):
            for a in range(1, floor((p-1)/2) + 1):
                R[j,(j+a)%p] += (0.5 - a/p)
                R[j,(j-a)%p] -= (0.5 - a/p)
        list_R.append(R)
    return block_diag(list_R, dtype = float_precis)


def create_incidence_matrices(F, H, V):
    fv = create_fv(F, V)
    R = create_R(F)
    A = create_A(H)
    d0 = create_d0(H, V)
    d1 = create_d1(F, H)
    return fv, R, A, d0, d1

def create_incidence_matrices_AW(F, H, V):
    R = create_R(F)
    A = create_A(H)
    d0 = create_d0(H, V)
    return R, A, d0

def create_incidence_matrices_FU(H, V):
    A = create_A(H)
    d0 = create_d0(H, V)
    return A, d0

def calculate_area(face: list[int], V: np.ndarray) -> float:
    """Calculates norm of vector area of a given face"""
    area = 0
    for i in range(len(face)):
        v0 = V[face[i]]
        v1 = V[face[(i+1)%len(face)]]
        area += 0.5*np.cross(v0,v1)
    return np.linalg.norm(area)


def W_matrices_PV(F: list[np.ndarray], V: np.ndarray) -> list[np.ndarray]:
    """Creates matrices WF, W1, WV that are building blocks for the Laplacian
    of [Ptackova & Velho 2021]"""
    vec_WF = np.zeros(len(F), dtype = float_precis)
    vec_inverse_WV = np.zeros(len(V), dtype = float_precis)
    list_W1 = []
    for i in range(len(F)):
        v_ind = F[i]
        p = len(v_ind)        
        W1_f = np.zeros(p**2, dtype = float_precis).reshape(p,p)
        vec_WF[i] = calculate_area(v_ind, V)
        for j in range(p):
            vec_inverse_WV[v_ind[j]] += vec_WF[i]/p
            for k in range(p):
                W1_f[j,k] = np.dot(V[v_ind[(j+1)%p]]-V[v_ind[j]],\
                                   V[v_ind[(k+1)%p]]-V[v_ind[k]])/vec_WF[i]
        list_W1.append(W1_f)
    vec_WV = 1/vec_inverse_WV
    return diags(vec_WF), block_diag(list_W1), diags(vec_WV)


def W_matrices_AW(F: list[np.ndarray], V: np.ndarray) -> list[np.ndarray]:
    """Creates matrices that are building blocks for the Laplacian
    of [Alexa & Wardetzky 2011]"""
    vec_inverse_WV = np.zeros(len(V), dtype = float_precis)
    list_W1 = []
    for i in range(len(F)):
        v_ind = F[i]
        p = len(v_ind)        
        W1_f = np.zeros(p**2, dtype = float_precis).reshape(p,p)
        area = calculate_area(v_ind, V)
        for j in range(p):
            vec_inverse_WV[v_ind[j]] += area/p
            for k in range(p):
                W1_f[j,k] = np.dot(V[v_ind[(j+1)%p]]-V[v_ind[j]],\
                                   V[v_ind[(k+1)%p]]-V[v_ind[k]])/area
        list_W1.append(W1_f)
    vec_WV = 1/vec_inverse_WV
    return block_diag(list_W1), diags(vec_WV)


