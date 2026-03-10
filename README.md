mcf-ddm: Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes
==============
This is a brief explanation of implementation of algorithms for domain decomposition methods (DDM) for mean curvature flow (MCF) on polygonal meshes, more details about the implementation can be found in each script. The code is written in the Python programming language. It is an implementation created to support the findings in the article 'Domain Decomposition for Mean Curvature Flow of Surface Polygonal Meshes' submitted for the conference GMP2026.

Copyright (C) 2025, Lenka Ptackova
This program is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License 

The package consists of various modules briefly described bellow and a folder **meshes** that contains meshes in .obj file format and two textures in .png format.

When the function `main()` in the script **main.py** is called, the examples from figures in the article are processed. The geometry of an input mesh is processed by the mean curvature flow with the given method and resulting meshes are being exported in .obj file format to the folder `meshes/output`. The maximal mean curvatures, calculated by the chosen Laplacian, are printed.

**Important**: Throughout the code, the precision is set to **np.float32** and **np.int32**. You might want to increase the precision, i.e., set parameters `float_precis` and `int_precis` to np.float64 and np.float64, resp. Concretely, the integer precision must be increased if very large meshes are processesed.

The folder ddm-mcf also contains two scripts (`mcf_poly_performance.py` and `mcf_ddm_Robin_performance.py`), which solely serve for measuring the timings of the methods, they do not export flown meshes, and must be run directly from the terminal by calling `python3 mcf_ddm_Robin_performance.py`, for example.


The implementation has the following modules:
--------------
- `compare_meshes` - contains functions that calculate vertex distances and pseudo-Hausdorff distance between given sets of meshes and produce graphs;
- `decompose_mesh` - implementation of method `decompose` and its supporting functions;
- `generate_meshes` - carries numerous functions that produce triangle, quad, and hexagonal meshes over various parametric surfaces, optionally with jittering;
- `main` - runs the examples illustrated in figures of the article;
- `matrices` - consists of functions creating matrices that are building blocks for the given discrete Laplace operators;
- `mcf_ddm_Robin` - consists of the function `mcf_ddm_Robin` and its supporting functions; the function **mcf_ddm_Robin** is an implementation of DDM with our adapted Robin transmission conditions;
- `mcf_ddm_Robin_performance` - this script serves for measing the timings of tasks for MCF with the adapted Robin transmission and must be called directly from terminal; 
- `mcf_ddm_Schwarz` - contains the function `mcf_ddm_Schwarz` for the mean curvature flow with the alternating Schwarz domain decomposition method;
- `mcf_ddm_Ventcell` - consists of the function `mcf_ddm_Ventcell` and its supporting functions; function **mcf_ddm_Ventcell** is an implementation of DDM with adapted Ventcell transmission conditions;
- `mcf_poly` - implementation of the backward Euler method for the mean curvature flow on general polygonal meshes, whose faces are any simple closed polygons;
- `mcf_poly_performance` - this script serves for measing the timings of tasks for MCF on polygonal meshes and must be called directly from terminal; 
- `mcf_Schwarz_poly` - implementation of the MCF on general polygonal meshes with domain decomposition, however it expects as input two meshes with a 1-face large overlap, unlike the other (above mentioned) DDM methods;
- `read_write_mesh` - contains functions reading from and writing to .obj file format.


An example of how to use the package:
--------------
1. After downloading all the files, unzip the file meshes.zip.
2. Run the `main.py` script to produce the examples from our article.
3. You can also generate many more meshes using the methods in `generate_meshes.py` and test the methods `mcf_ddm_Schwarz`, `mcf_ddm_Robin`, or `mcf_ddm_Ventcell` with changing parameters.
