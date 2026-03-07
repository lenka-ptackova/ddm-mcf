"""Uploading and exportation of a mesh in the .obj file format, the methods
support any polygonal meshes

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

This script is a part of implementation of
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'
"""

import numpy as np
float_precis = np.float32
int_precis = np.int32

def vertex_position(string: str) -> np.ndarray:
    """Input: a string containing coordinates of a vertex.
    Returns an np.array of 3 floats correspoding to [x,y,z] coordinates"""
    coordinates = []
    string = string.strip('v ')
    string = string.strip(' \n')
    elements = string.split(' ')
    for el in elements:
        coordinates.append(float(el))
    return np.array(coordinates, dtype = float_precis)


def incident_vertices(string: str) -> np.ndarray:
    """Function receives a line of a face specification and extracts indices
    of vertices incident to a given face
    Input: a string starting with 'f '
    Output: a list of integers
    """
    vrts = []
    string = string.strip('f ')
    string = string.strip(' \n')
    elements = string.split(' ')   
    for el in elements:
        if "/" in el:
            vertex_index = int(el[:el.index("/")])-1
        else:
            vertex_index = int(el)-1
        vrts.append(vertex_index)     
    return np.array(vrts, dtype = int_precis)


def read_mesh_obj(filename: str) -> list:
    file = open(filename, 'r')
    V, H, F = [], [], []   
    for line in file:
        if line.startswith('v '):
            V.append(vertex_position(line))
        elif line.startswith('f '):
            face_vertices = incident_vertices(line)
            F.append(face_vertices)
            p = len(face_vertices)
            for i in range(p):
                H.append([face_vertices[i], face_vertices[(i+1)%p]])            
    return np.array(V, dtype = float_precis), np.array(H, dtype = int_precis), F



def write_mesh_obj(filename: str, V: list[np.ndarray], F: list[np.ndarray])\
    -> None:
    file = open(filename, "w")
    for v in V:
        file.write("v {:.5f} {:.5f} {:.5f}\n".format(v[0], v[1], v[2]))
    for f in F:
        file.write("f")
        for index in f:
            file.write(" " +str(index + 1))
        file.write("\n")
    file.close()
    return None


def rewrite_vertices(filename: str, V: list[np.ndarray], info = "_New") -> None:
    fileToRead = open(filename + ".obj", "r")
    newText = "# New\n"
    
    i = 0
    for line in fileToRead:
        if line.startswith('v '):
            newText += "v {:.5f} {:.5f} {:.5f}\n".format(V[i,0],V[i,1],V[i,2])
            i += 1
        else:
            newText += line
            
    fileToRead.close()
    filenameS = filename.split(sep="/")
    fileToWrite = open(filenameS[0] +"/output/" + filenameS[1] +\
                       info + ".obj", "w")
    fileToWrite.write(newText)
    fileToWrite.close()
    return None
