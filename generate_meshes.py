"""Generation of triangle, quad, and hexagonal meshes to be exported
into Wavefront .obj file

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

This script is a part of implementation of methods supporting the paper
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'
"""

from math import cos, sin, pi, exp, cosh, log
import numpy as np
from random import random
from read_write_mesh import read_mesh_obj, write_mesh_obj

xBoundary = 0
yBoundary = 0



####---------------------------------------------------------------
#### PARAMETRIC FUNCTIONS OF TYPE F(x,y) in R^3 and jittering
####---------------------------------------------------------------

def A_surface(x: float, y: float):
    return 0.5*cos(pi/2*x)*0.6*sin(pi*y)+1

def Sherk(x: float, y: float):
    return log(cos(x)/cos(y)) 

def Sherkish(x: float, y: float):
    return log(cos(x)/cos(y)) - ((pi-1)**2/4 - x**2)*((pi-1)**2/4 - y**2)

def Shrek(x: float, y: float):
    return log(cos(x)/cos(y)) +\
           (1 - x**2)*(1 - y**2) +\
           0.05*sin(2*pi*(xBoundary**2 - x**2))*sin(3*pi*(yBoundary**2 - y**2)) 
           

def T_surface(x: float, y: float):
    return 0.6*cos(pi/2*x*1.1)*sin(pi*y*1.1)

def wave_surface(x: float, y: float):
    return 0.5*cos(pi/2*x*1.1)*sin(pi*y*1.1) +\
           0.05*sin(2*pi*y)*cos(2*pi*x+4*pi*y)

def superwave_surface(x: float, y: float):
    return 0.5*cos(pi/2*x*1.1)*sin(pi*y*1.1) +\
           0.05*sin(4*pi*y)*cos(3*pi*x+4*pi*y)

def C_surface(x: float, y: float):
    z = 0
    if y <= 0:
        z = (exp(pi*y) - 1)*(1 - x**2)
    else:
        z = sin(pi*y)*(1 - x**2)
    return 0.75*z

def C1_surface(x: float, y: float):
    if y <= 1/2:
        z = (exp(pi*(y-1/2)) - 1)*(1 - 4*(x-1/2)**2)/2
    else:
        z = sin(2*pi*(y-1/2))*(1 - 4*(x-1/2)**2)/4
    return z

def P_surface(x: float, y: float):
    return sin(pi*x)*(1-y**2)

def F_surface(x: float, y: float):
    return 0.1*sin(pi*(1-y**2)*5)*\
           cos(pi*(1-x**2 + 1/10)*5)*(1-x**2)*(1-y**2)+\
           0.75*(1-x**2)*(1-y**2)

def F_tex_surface(x: float, y: float):
    return 0.025*sin(pi*5*(1 - 4*(0.5-y)**2))*\
           cos(5*pi*((1 + 1/10 - 4*(0.5-x)**2)))*\
           (1-4*(0.5-x)**2)*(1-4*(0.5-y)**2)+\
           0.5*(1-4*(0.5-x)**2)*(1-4*(0.5-y)**2)
           
def tex_surface(x: float, y: float):
    x = x + 1
    x = x/4
    y = y + 1
    y = y/2
    return 2*F_tex_surface(x,y)


def flat_surface(x, y):
    return 0


def jittering(x,y,dist):
    v = np.array([random()-0.5,random()-0.5])
    return np.array([x,y]) + dist*v/np.linalg.norm(v)


def rewrite_mesh(filename:str) -> list:
    V, H, F = read_mesh_obj(filename)
    V[:,0:2] = 2*V[:,0:2]
    V[:,0] -= 1
    V[:,1] -= 2
    for i in range(len(V)):
        V[i,2] = wave_surface(V[i,0],V[i,1])      
    write_mesh_obj("wave_"+filename, V, F) 
    return None

def move_mesh(filename:str) -> list:
    V, H, F = read_mesh_obj(filename)
    for i in range(len(V)):
        V[i,2] = V[i,2] + 0.2     
    write_mesh_obj(filename+"moved", V, F) 
    return None


####---------------------------------------------------------------
#### QUADRILATERAL, TRIANGLE, HEXAGONAL MESHES
####---------------------------------------------------------------

                  
def generate_quad_mesh(nF_per_row: int, times_columns: float = 1):  
    nV_per_row = nF_per_row +1  
    ds = 2/(nF_per_row)
    s0 = -1
    surface = wave_surface
    
    global xBoundary, yBoundary
    xBoundary = s0
    yBoundary = s0

    nF_per_column = int(nF_per_row*times_columns)
    
    filename = "quad_mesh_{}_{}.obj".format(nV_per_row*(nF_per_column + 1), nV_per_row)
    file = open(filename, "w")   
    for i in range(nF_per_column + 1):
        for j in range(nV_per_row):
            y = s0 + i*ds
            x = s0 + j*ds + 0.1*sin(3*pi*y)*sin(pi*j*ds)
            file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))      
    k = 1
    for i in range(1, nF_per_column + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {} {}\n".format(j+k-1, j+k, j + nV_per_row + k,\
                                                j + nV_per_row + k-1))            
        k += nV_per_row
    file.close()
    print(nV_per_row**2)
    return None


#====================================================================

def generate_tri_mesh(nF_per_row: int, jitt_level = 0):
    
    nV_per_row = nF_per_row +1
    ds = 2/(nF_per_row)
    s0 = -1
    surface = C_surface
    
    jitter = False
    if jitt_level == 0:
        filename = "tri_mesh_C1_{}.obj".format(nV_per_row**2)
        
    else:
        jitter = True
        noise = jitt_level*ds
        filename = "tri_C1_jit_{}_{}.obj".format(jitt_level, nV_per_row**2)
   
    file = open(filename, "w")

    if jitter == False:
        for i in range(nV_per_row):
            for j in range(nV_per_row):
                file.write("v {} {} {}\n".format(s0 + j*ds, s0 + i*ds,\
                                                 surface(s0 + j*ds, s0 + i*ds)))
    else:
        for i in range(nV_per_row):
            for j in range(nV_per_row):
                x,y = jittering(s0 + j*ds, s0 + i*ds, noise)
                file.write("v {} {} {}\n".format(x,y,surface(x,y))) 
    k = 1
    for i in range(1, nF_per_row + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {}\n".format(j+k-1, j+k, j + nV_per_row + k))
            file.write("f {} {} {}\n".format(j+k-1, j+nV_per_row+k, j+nV_per_row+k-1))
        k += nV_per_row
    file.close()
    print(nV_per_row**2)
    return None



def generate_quad_mesh(nF_per_row: int, times_columns: float = 1):  
    nV_per_row = nF_per_row +1  
    ds = 1/(nF_per_row)
    s0 = 0
    surface = F_tex_surface
    
    global xBoundary, yBoundary
    xBoundary = s0
    yBoundary = s0

    nF_per_column = int(nF_per_row*times_columns)
    
    filename = "quad_{}.obj".format(nV_per_row)
    file = open(filename, "w")

    for i in range(1,nF_per_column):
        for j in range(nV_per_row):
            if j >0 and j<nV_per_row - 1:
                x, y = jittering(s0 + j*ds, s0 + i*ds, 0.1*ds)
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))
            else:
                x, y = s0 + j*ds, s0 + i*ds
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))

    for j in range(nV_per_row):
        x, y = s0 + j*ds, s0 + (nF_per_column)*ds
        file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))
        
    k = 1
    for i in range(1, nF_per_column + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {} {}\n".format(j+k-1, j+k, j + nV_per_row + k,\
                                                j + nV_per_row + k-1))            
        k += nV_per_row
    file.close()
    print(nV_per_row**2)
    return None
#====================================================================


def generate_hex_mesh(nF_per_row: int, jitt_level = 0):
    """Generates a prevalently hexagonal mesh, i.e., all interior faces
    are hexagons, whereas some faces adjacent to boundary are quadrilaterals.
    It then prints the mesh in .obj file format into a file.
    Level of jittering means that each interior vertex will be moved
    jitt_level*edge_lenght (before projection on a given parametric surface)
    Input: number of faces per row, level of jittering (optional)
    Output: None
    """
    surface = F_tex_surface
    jitter = False  
    
    nV_per_row = 2*nF_per_row + 2
    xStep = 1/(nF_per_row +0.5)
    r = xStep/np.sqrt(3)
    nF_per_column = int(2/(np.sqrt(3)*xStep)) + 1
    x0 = 0
    y0 = 0

    nVertices = nV_per_row*(nF_per_column + 1) - 2

    if jitt_level == 0:
        filename = "hex_mesh_{}_{}.obj".format(nVertices, nV_per_row)
        
    else:
        jitter = True
        noise = jitt_level*r
        filename = "hex_jit_{}_{}_{}.obj".format(jitt_level, nVertices, nV_per_row)
    file = open(filename, "w")

    file.write("### Hexagonal mesh\n")


    ### VERTICES ###-----------------
    
    # First row of vertices
    for j in range(nV_per_row-1):     
        file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0 + j*xStep/2, 0,\
                                                     surface(x0 + j*xStep/2, 0)))
                
    Ystep = 3*r
    # Midle rows of vertices
    if jitter == False:
        for i in range(1, nF_per_column):   # Y changing
            for j in range(nV_per_row):     # X changing
                if i%2 == 0:
                    file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0 + j*xStep/2, y0 - (j%2)*r/2 + Ystep,\
                                                    surface(x0 + j*xStep/2,\
                                                              y0 - (j%2)*r/2 + Ystep)))              
                else:
                    file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0 + j*xStep/2,\
                                                     y0 + (j%2)*r/2 +Ystep - 2*r,\
                                                     surface(x0 + j*xStep/2,\
                                                               y0 + (j%2)*r/2 +Ystep - 2*r)))                               
            if i%2 == 0:
                Ystep += 3*r
                
    if jitter == True:
        for i in range(1, nF_per_column):   # Y changing
            # j = 0:
            if i%2 == 0:
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0,y0+Ystep,\
                                                             surface(x0,y0+Ystep)))
            else:
                x,y = [x0,y0 +Ystep-2*r]
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))
                
            for j in range(1,nV_per_row-1): # X changing
                if i%2 == 0:
                    x,y = jittering(x0+j*xStep/2,y0-(j%2)*r/2+Ystep,noise)
                    file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,\
                                                                 surface(x,y)))              
                else:
                    x,y = jittering(x0 + j*xStep/2,y0 +(j%2)*r/2+Ystep-2*r,noise)
                    file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,\
                                                                 surface(x,y)))
            j = nV_per_row - 1
            if i%2 == 0:
                x,y = [x0+j*xStep/2,y0-(j%2)*r/2+Ystep]
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))              
            else:
                x,y = [x0 + j*xStep/2,y0 +(j%2)*r/2+Ystep-2*r]
                file.write("v {:.5f} {:.5f} {:.5f}\n".format(x,y,surface(x,y)))

            if i%2 == 0:
                Ystep += 3*r
                
    # Last row of vertices
    if nF_per_column % 2 == 0:
        for j in range(1, nV_per_row):     
            file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0 + j*xStep/2, 1,\
                                                         surface(x0 + j*xStep/2, 1)))
            nVertices += 1
    else:
        for j in range(nV_per_row-1):     
            file.write("v {:.5f} {:.5f} {:.5f}\n".format(x0 + j*xStep/2, 1,\
                                                         surface(x0 + j*xStep/2, 1)))
            nVertices += 1


    ### FACES ###-----------------
    # First row of faces:
    for j in range(1, nV_per_row - 2, 2):
        file.write("f {} {} {} {} {} {}\n".format(j, j+1, j+2,\
                                                  j + nV_per_row + 1,\
                                                  j + nV_per_row + 0,\
                                                  j + nV_per_row - 1))
    # Not first nor last row of faces:
    k = nV_per_row
    for i in range(2, nF_per_column):
        if i%2 == 0:
            file.write("f {} {} {} {}\n".format(k, 1+k, 1+k + nV_per_row,\
                                                1+k + nV_per_row-1))
            for j in range(1, nV_per_row - 1, 2):
                file.write("f {} {} {} {} {} {}\n".format(j+k+0, j+k+1, j+k+2,\
                                                          j + nV_per_row + 2 + k,\
                                                          j + nV_per_row + 1 + k,\
                                                          j + nV_per_row - 0 + k))
        else:
            for j in range(1, nV_per_row - 2, 2):
                file.write("f {} {} {} {} {} {}\n".format(j+k-1, j+k, j+ k +1,\
                                                          j + nV_per_row + 1 + k,\
                                                          j + nV_per_row + 0 + k,\
                                                          j + nV_per_row - 1 + k))
            file.write("f {} {} {} {}\n".format(nV_per_row-2 + k, nV_per_row-1 + k,\
                                                2*nV_per_row - 1 + k,\
                                                2*nV_per_row - 2 + k))
        k += nV_per_row

    # Last row of faces:
    if nF_per_column % 2 == 0:
        for j in range(1, nV_per_row - 1, 2):
            file.write("f {} {} {} {} {} {}\n".format(j+k+0, j+k+1, j+k+2,\
                                                      j + nV_per_row + 1 + k,\
                                                      j + nV_per_row + 0 + k,\
                                                      j + nV_per_row - 1 + k))
    else:
        for j in range(1, nV_per_row - 2, 2):
            file.write("f {} {} {} {} {} {}\n".format(j+k-1, j+k, j+ k +1,\
                                                      j + nV_per_row + 1 + k,\
                                                      j + nV_per_row + 0 + k,\
                                                      j + nV_per_row - 1 + k))
    file.close()   
    return None




####---------------------------------------------------------------
#### SPECIAL QUAD MESHES - CATENOID, CILINDER, ENNEPER SURFACE, ETC.
####---------------------------------------------------------------

def generate_wavy_quad(nF_per_row: int):
    """ Generates a special quad mesh, where the underlying parametrization
    is a function of sinus of x and sinus of y ("wavy parametrization")
    and contains also texture coordinates.
    """
    nV_per_row = nF_per_row +1 
    ds = 1/(nF_per_row)
    s0 = 0

    nF_per_col = nF_per_row 
    nV_per_col = nF_per_col +1 
    dr = 1/(nF_per_col)
    r0 = 0

    surface = F_tex_surface
##    surface = superwave_surface
    filename = "quad_tex_{}.obj".format(nV_per_row*nV_per_col)
    file = open(filename, "w")
    
    for i in range(nV_per_col):
        for j in range(nV_per_row):
            x = j*ds
            y = i*ds + 0.1*sin(3*pi*x)*sin(pi*i*ds)
            file.write("v {:.5f} {:.5f} {:.5f}\n".format(x, y,\
                                                         surface(x, y)))
    for i in range(nV_per_col):
        for j in range(nV_per_row):
            x = j*ds
            y = i*ds + 0.1*sin(3*pi*x)*sin(pi*i*ds)
            file.write("vt {:.5f} {:.5f}\n".format(x, y))
    k = 1
    for i in range(1, nF_per_col + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {}/{} {}/{} {}/{} {}/{}\n".\
                       format(j+k-1,j+k-1,j+k,j+k,j+nV_per_row+k,j+nV_per_row+k,\
                              j+nV_per_row+k-1,j+nV_per_row+k-1))           
        k += nV_per_row

    file.close()
    print(nV_per_row*nV_per_col)
    return None


def generate_catenoid(nF_per_row: int):  
    nV_per_row = nF_per_row    
    ds = 2*pi/(nF_per_row)
    nF_per_column = int(nF_per_row/4)

    filename = "catenoid_{}row.obj".format(nV_per_row)
    file = open(filename, "w")

    dz = 1.2/nF_per_column
    z = 0
    c = 0.74507
    for i in range(nF_per_column+1):
        for j in range(nV_per_row):
            file.write("v {:.4f} {:.4f} {:.4f}\n".format(\
                c*cosh((-0.6 + z)/c)*cos(j*ds),\
                c*cosh((-0.6 + z)/c)*sin(j*ds), z))
        z += dz
        
    k = 1
    for i in range(1, nF_per_column + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {} {}\n".format(j+k-1, j%nV_per_row +k,\
                                                j%nV_per_row + nV_per_row + k,\
                                                j + nV_per_row + k-1))            
        k += nV_per_row

    file.close()
    print(filename)
    return None


def generate_cilinder(nF_per_row: int):  
    nV_per_row = nF_per_row    
    ds = 2*pi/(nF_per_row)
    nV_per_column = int(0.7*nF_per_row)
    filename = "quad_cilinder_wavy_{}_{}.obj".format(nV_per_row*nV_per_column,\
                                                nV_per_row)
    file = open(filename, "w")

    dz = 2*pi/nV_per_column
    z = -pi
    for i in range(nV_per_column+1):
        for j in range(nV_per_row):
            r = 0
            R = 2 - (z/pi)**2  + 0.1*cos(6*z) + 0.1*sin(8*j*ds)*sin(z) 
            file.write("v {:.6f} {:.6f} {:.6f}\n".\
                       format(R*cos(j*ds+r), R*sin(j*ds+r), z + 0.05*cos(10*j*ds)*cos(z/2)))
##                       format(cos(j*ds+r), sin(j*ds+r), z))
        z += dz        
    k = 1
    for i in range(1, nV_per_column+1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {} {}\n".format(j+k-1, j%nV_per_row +k,\
                                                j%nV_per_row + nV_per_row + k,\
                                                j + nV_per_row + k-1))            
        k += nV_per_row
    file.close()
    print(filename)
    return None


def generate_cili(nV_per_row: int):     
    nV_per_column = int(0.7*nV_per_row/2)
    nV = nV_per_column*nV_per_row + (2*nV_per_column+1)*(2*nV_per_row)
    filename = "quad_cili_{}_{}_{}.obj".format(nV, nV_per_column+1, nV_per_row)
    file = open(filename, "w")

    ### LOWER PART
    ds = 2*pi/(nV_per_row)
    dz = pi/nV_per_column
    z = -pi
    for i in range(nV_per_column):
        for j in range(nV_per_row):
            r = 0
            R = 2 - (z/pi)**2  + 0.1*cos(6*z) + 0.1*sin(8*j*ds)*sin(z) 
            file.write("v {:.6f} {:.6f} {:.6f}\n".\
                       format(R*cos(j*ds+r), R*sin(j*ds+r), z + 0.05*cos(10*j*ds)*cos(z/2)))
        z += dz

    ### UPPER PART with edge lenths approximately half the lower part edge lengths
    dz = dz/2
    ds = pi/(nV_per_row)
    for i in range(2*nV_per_column+1):
        for j in range(2*nV_per_row):
            r = 0
            R = 2 - (z/pi)**2  + 0.1*cos(6*z) + 0.1*sin(8*j*ds)*sin(z) 
            file.write("v {:.6f} {:.6f} {:.6f}\n".\
                       format(R*cos(j*ds+r), R*sin(j*ds+r), z + 0.05*cos(10*j*ds)*cos(z/2)))
        z += dz

    ### TEXTURE COORDINATES
    ds = 2*pi/(nV_per_row)
    dz = pi/nV_per_column
    z = -pi
    dx = 1/(nV_per_row-1)
    for i in range(nV_per_column):
        for j in range(nV_per_row):
            x = j*dx
            y = (z + 0.05*cos(10*j*ds)*cos(z/2))/(2*pi) + 0.5
            file.write("vt {:.5f} {:.5f}\n".format(x, y))
        z += dz
    dz = dz/2
    ds = pi/(nV_per_row)
    dx = 1/(2*nV_per_row-1)
    for i in range(2*nV_per_column+1):
        for j in range(2*nV_per_row):
            x = j*dx
            y = (z + 0.05*cos(10*j*ds)*cos(z/2))/(2*pi) + 0.5
            file.write("vt {:.5f} {:.5f}\n".format(x, y))
        z += dz


    ### FACES OF THE LOWER PART    
    k = 1
    for i in range(0, nV_per_column-1):
        for j in range(1, nV_per_row + 1):
            file.write("f {}/{} {}/{} {}/{} {}/{}\n".format(j+k-1,j+k-1, j%nV_per_row +k,\
                                                            j%nV_per_row +k,\
                                                            j%nV_per_row+nV_per_row+k,\
                                                            j%nV_per_row+nV_per_row+k,\
                                                            j + nV_per_row + k-1,\
                                                            j + nV_per_row + k-1))            
        k += nV_per_row

    ### FACES OF THE UPPER PART INCIDENT TO THE INTERFACE
    l = 0
    for j in range(1, nV_per_row + 1):
        file.write("f {}/{} {}/{} {}/{} {}/{} {}/{}\n".\
                   format(j+k-1,j+k-1, j%nV_per_row +k, j%nV_per_row +k,\
                          (l+2)%(2*nV_per_row)+nV_per_row + k,\
                          (l+2)%(2*nV_per_row)+nV_per_row + k,\
                          l+2 + nV_per_row + k -1,l+2 + nV_per_row + k -1,\
                          l+2 + nV_per_row + k -2,l+2 + nV_per_row + k -2))
        l += 2
        
    ### FACES OF THE UPPER PART
    k += nV_per_row        
    nV_per_row = 2*nV_per_row
    for i in range(1, 2*nV_per_column+1):
        for j in range(1, nV_per_row + 1):
            file.write("f {}/{} {}/{} {}/{} {}/{}\n".format(j+k-1,j+k-1, j%nV_per_row +k,\
                                                            j%nV_per_row +k,\
                                                            j%nV_per_row+nV_per_row+k,\
                                                            j%nV_per_row+nV_per_row+k,\
                                                            j + nV_per_row + k-1,\
                                                            j + nV_per_row + k-1))              
        k += nV_per_row

    file.close()
    print(filename)
    return None



def generate_quad_Enneper(nF_per_row: int):  
    nV_per_row = nF_per_row +1
    eps = 0.5
    u0 = -1
    du = 2/nF_per_row
    v0 = -1
    dv = 2/nF_per_row
    filename = "quad_Enneper_{}.obj".format(nV_per_row**2)
    file = open(filename, "w")   
    for i in range(nV_per_row):
        for j in range(nV_per_row):
            u = u0 + j*du
            v = v0 + i*dv
            file.write("v {:.5f} {:.5f} {:.5f}\n".format(u - u**3/3 + u*v**2,\
                                                         -v - u**2*v + v**3/3,\
                                                         u**2 - v**2))      
    k = 1
    for i in range(1, nF_per_row + 1):
        for j in range(1, nF_per_row + 1):
            file.write("f {} {} {} {}\n".format(j+k-1, j+k, j + nV_per_row + k,\
                                                j + nV_per_row + k-1))            
        k += nV_per_row
    file.close()
    print(nV_per_row**2)
    return None


def cili(jds,z):
    R = 2 - (z/pi)**2  + 0.1*cos(6*z) + 0.1*sin(8*jds)*sin(z)
    a = R*cos(jds)
    b = R*sin(jds)
    c = z + 0.05*cos(10*jds)*cos(z/2)
    return a, b, c
 
def generate_anisotropic_cili(nF_per_row: int):  
    nV_per_row = nF_per_row    
    ds = 2*pi/(nF_per_row)
    nV_per_column = int(0.7*nF_per_row)
    filename = "tri_anisotropic_cili_{}_{}.obj".format(nV_per_row*nV_per_column,\
                                                       nV_per_row)
    file = open(filename, "w")
 
    dz = 2*pi/nV_per_column
    z = -pi
    for i in range(nV_per_column+1):
        for j in range(nV_per_row):
            jds = j*ds + 0.5*(random()-0.5)*ds
            R = 2 - (z/pi)**2  + 0.1*cos(6*z) + 0.1*sin(8*jds)*sin(z)
            a, b, c = cili(jds,z + 0.5*cos(z/2)*(random()-0.5)*dz)
            file.write("v {:.6f} {:.6f} {:.6f}\n".\
                       format(a, b, c))
        z += dz        
    k = 1
    for i in range(1, nV_per_column+1):
        for j in range(1, nV_per_row + 1):
            if j%2 != 0 and i%2 != 0: #triangle
                file.write("f {} {} {}\n".format(j+k-1, j%nV_per_row +k,\
                                                 j + nV_per_row + k-1))
                file.write("f {} {} {}\n".format(j%nV_per_row +k,\
                                                 j%nV_per_row + nV_per_row + k,\
                                                 j + nV_per_row + k-1))
            else:
                file.write("f {} {} {}\n".format(j+k-1, j%nV_per_row +k,\
                                                 j%nV_per_row + nV_per_row + k))
                file.write("f {} {} {}\n".format(j+k-1,\
                                                 j%nV_per_row + nV_per_row + k,\
                                                 j + nV_per_row + k-1))
##            else: #pentagon
##                file.write("f {} {} {} {} {}\n".format(j+k-1, j%nV_per_row +k,\
##                                                    j%nV_per_row + nV_per_row + k,\
##                                                    j + nV_per_row + k-1,\
##                                                    j + nV_per_row + k-2))
                                               
        k += nV_per_row
    file.close()
    print(filename)
    return None

