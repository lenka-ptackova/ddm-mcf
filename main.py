"""Mesh smoothing by implicit mean curvature flow of surface polygonal
mesh with or without boundary. If the mesh has a boundary, it is kept fixed.

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 


This script is a part of implementation of methods exposed in the paper
'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes'

There are three Laplacians implemented:
1) "FU": scale–dependent umbrella operator of Fujiwara (1995)
    [doi:10.2307/2161293];
2) "AW": polygonal Laplacian of Alexa and Wardetzky (2011)
    [doi:10.1145/2010324.1964997],
3) "PV": polygonal Laplacian of Ptackova and Velho (2021)
    [doi:10.1016/j.cagd.2021.102002].

"""
from mcf_ddm_Robin import mcf_ddm_Robin
from mcf_ddm_Schwarz import mcf_ddm_Schwarz
from mcf_Schwarz_poly import mcf_Schwarz_poly
from mcf_poly import mcf_poly
from compare_meshes import compare_sets_of_meshes, plot_graphs
import numpy as np
import math


def main():
    """
    The function will automatically perform mean curvature flow of the meshes
    that appear in the article with ID 1076. It will save the flown meshes
    and graphs with pseudo-Hausdorff distances and corresponding vertex distance
    in the folder meshes/output.
    For more details about the methods, please see the article ID 1076.
    """

    ###-------------------------------------------------------
    ### In the folder meshes you can find the following meshes:
    ###-------------------------------------------------------

##    mesh_file_name = "meshes/hex_C1_15226_162"  
##    mesh_file_name = "meshes/hex_F_tex_6118_102"  
##    mesh_file_name = "meshes/hex_F_tex_jit_0.1_6118_102"
##    mesh_file_name = "meshes/hex_F_tex_jit_0.1_23834_202"
##    mesh_file_name = "meshes/hex_F_tex_scaled2_6118_102"
##    mesh_file_name = "meshes/hex_F_tex_scaled2_jit_0.1_6118_102"
##    mesh_file_name = "meshes/hex_wave_elongated_12034_102"
##    mesh_file_name = "meshes/hex_mesh_1972_42"

##    mesh_file_name = "meshes/poly_C1_scaled_6558"
##    mesh_file_name = "meshes/poly_C1_scaled_6558_L"
##    mesh_file_name = "meshes/poly_C1_scaled_6558_U"
##    mesh_file_name = "meshes/poly_F_tex_6558"
##    mesh_file_name = "meshes/poly_square_6558"   
    
##    mesh_file_name = "meshes/quad_C1_6561"
##    mesh_file_name = "meshes/quad_cili_7280_23_65" 
##    mesh_file_name = "meshes/quad_cilinder_wavy_7100_100"
##    mesh_file_name = "meshes/quad_cilinder_wavy_10200_120"
##    mesh_file_name = "meshes/quad_F_tex_3721"
##    mesh_file_name = "meshes/quad_F_tex_elongated_5151_51"
##    mesh_file_name = "meshes/quad_F_tex_wavy_3721"
##    mesh_file_name = "meshes/quad_F_tex_wavy_10201"
##    mesh_file_name = "meshes/quad_tex_superwave_10201"     
##    mesh_file_name = "meshes/quad_waves_6561"
##    mesh_file_name = "meshes/quad_waves_10201"
##    mesh_file_name = "meshes/quad_waves_elongated_wavy_11421_81"
  
##    mesh_file_name = "meshes/tri_C1_6561"
##    mesh_file_name = "meshes/tri_C1_10201"
##    mesh_file_name = "meshes/tri_C1_jit_0.1_6561"  
##    mesh_file_name = "meshes/tri_Shrek_3721"
##    mesh_file_name = "meshes/tri_waves_6561"
##    mesh_file_name = "meshes/tri_waves_jit_0.1_6561"


    ###-------------------------------------------------------
    ### Examples from the paper ###
    ###-------------------------------------------------------
    
    ### Figure 1 (with texture 'stripes.png' mapped on the surfaces in Meshlab)
    
    print("---Figure 1---")
    mesh_file_name = "meshes/quad_cili_7280_23_65"
    maxIteration = 5
    dt = 0.05
    Laplacians =  ["FU", "AW", "PV"]
    names_ddms = []
    for Laplacian in Laplacians:        
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt,\
                                 divRowV = 23, typ = 5)
        names_ddms.append(name_ddm)
        
    names_polys = []
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)

    ###-------------------------------------------------------
    ### Figure 2 (with texture 'texture_F_tex_surface.png')
        
    print("---Figure 2---")
    mesh_file_name = "meshes/poly_F_tex_6558"
    maxIteration = 1
    dt = 0.01
    Laplacians =  ["FU", "AW", "PV"]
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)


    ###-------------------------------------------------------
    ### Figure 4
        
    print("---Figure 4---")
    mesh_file_name = "meshes/poly_C1_scaled_6558"
    maxIteration = 2
    dt = 0.05
    Laplacians =  ["PV"]
    for Laplacian in Laplacians:        
        mcf_Schwarz_poly("meshes/poly_C1_scaled_6558_L",\
                         "meshes/poly_C1_scaled_6558_U", Laplacian, maxIteration, dt)
        
    for Laplacian in Laplacians:
        mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        

    ###-------------------------------------------------------
    ### Figure 6
        
    print("---Figure 6---")
    mesh_file_name = "meshes/hex_F_tex_6118_102"
    maxIteration = 10
    dt = 0.01
    Laplacians = ["FU", "AW", "PV"]
    names_ddms = []
    for Laplacian in Laplacians:        
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt)
        names_ddms.append(name_ddm)
        
    names_polys = []
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)


    ###-------------------------------------------------------
    ### Figure 7
        
    print("---Figure 7---")
    mesh_file_name = "meshes/tri_Shrek_3721"
    maxIteration = 10
    dt = 0.05
    Laplacians = ["FU", "AW", "PV"]
    names_ddms = []
    for Laplacian in Laplacians:        
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt)
        names_ddms.append(name_ddm)
        
    names_polys = []
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)


    ###-------------------------------------------------------
    ### Figure 8
        
    print("---Figure 8---")       
    mesh_file_name = "meshes/quad_cilinder_wavy_10200_120"
    maxIteration = 5
    dt = 0.05
    Laplacians =  ["FU", "PV"]
    names_ddms = []
    names_ddms2 = []
    for Laplacian in Laplacians:        
        name_ddm = mcf_ddm_Schwarz(mesh_file_name, Laplacian, maxIteration, dt)
        names_ddms.append(name_ddm)
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt)
        names_ddms2.append(name_ddm)
        
    names_polys = []
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms2[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)


    ###-------------------------------------------------------
    ### Figure 9
        
    print("---Figure 9---")       
    mesh_file_name = "meshes/tri_anisotropic_cili_10080_120"
    maxIteration = 5
    dt = 0.05
    Laplacians =  ["AW"]
    names_ddms = []
    for Laplacian in Laplacians:   
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt)
        names_ddms2.append(name_ddm)
        
    names_polys = []
    for Laplacian in Laplacians:
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)


    ###-------------------------------------------------------
    ### Figure 10 (with texture 'texture_F_tex_surface.png')
        
    print("---Figures 10---")
    mesh_file_name = "meshes/quad_F_tex_3721"
    maxIteration = 1
    dt = 0.05
    Laplacians = ["FU", "AW", "PV"]
    dividingRow = 42
    names_ddms, names_polys = [], []
    for Laplacian in Laplacians:
        name_ddm = mcf_ddm_Robin(mesh_file_name, Laplacian, maxIteration, dt, dividingRow)
        names_ddms.append(name_ddm)
        name_poly = mcf_poly(mesh_file_name, Laplacian, maxIteration, dt)
        names_polys.append(name_poly)

    for i in range(len(Laplacians)):
        file_name = compare_sets_of_meshes(names_ddms[i], names_polys[i],\
                                           np.arange(1,maxIteration+1))
        plot_graphs(file_name)

    
    ###-------------------------------------------------------

    print("---done---")
    return None
    

if __name__=="__main__": 
    main() 
