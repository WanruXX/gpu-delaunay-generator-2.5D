============
Introduction 
============

This program, termed gDel2D, constructs the constrained Delaunay Triangulation 
of a planar straight line graph in 2D using the GPU. The algorithm uses a
combination of incremental insertion and flipping to construct a triangulation
near Delaunay, followed by inserting the constrained edges also by flipping. 
The code is written in C++ using CUDA programming model of NVIDIA. 

Note that the constraints are optional, and without constraints the algorithm simply construct the 2D Delaunay triangulation.

===================
Programming authors
===================

Cao Thanh Tung
Ashwin Nanjappa

=====
Setup
=====

gDel2D works on any NVIDIA GPU with hardware capability 1.1 onward. However, 
it works best on Fermi and higher architecture. The code has been tested on 
the NVIDIA GTX 450, GTX 460, GTX 470, GTX580 (using sm_20) on Windows OS; 
and GTX Titan, GTX 970M on Linux (using sm_30 and sm_50). 

To switch from double to single precision, simply define REAL_TYPE_FP32. 

For more details on the input and output, refer to: 
	CommonTypes.h 	(near the end)
	InputGenerator.cpp 
	DelaunayChecker.cpp. 

====
Note
====

- A visualizer is included for convenience. It requires freeglut and glew to compile. It can be disabled by simply comment out the 1st line in Main.cpp. 
The visualizer can also be removed completely by deleting anything related 
to the Visualizer class in Main.cpp (there're also some reference to the
visualizer in GpuDelaunay.cpp for debugging purpose). 

- The output is a 2-sphere, with an infinity point added for ease of navigation on the mesh. Check the code in DelaunayChecker.cpp for more details. 

- Refer to the Main.cpp file (as well as the Visualizer.cpp) for how to use the  library. 
	
===========
Compilation
===========

CMake is used to build gDel2D, as shown here:

    $ mkdir build
    $ cd build
    $ cmake ..
    $ make

Note that by default, CMake generate code for sm_30 and sm_50. Please modify 
the CMakeList.txt if needed. 

==========
References
==========

The 2D Delaunay triangulation algorithm is from: 
  A GPU accelerated algorithm for 3D Delaunay triangulation (2014).
    Thanh-Tung Cao, Ashwin Nanjappa, Mingcen Gao, Tiow-Seng Tan.
    Proc. 18th ACM SIGGRAPH Symp. Interactive 3D Graphics and Games, 47-55. 
    
  Fundamental Computational Geometry on the GPU (2014)
    Thanh-Tung Cao, PhD thesis.
    http://scholarbank.nus.sg/handle/10635/118887
    
The constraint insertion algorithm is from:
  Computing Two-dimensional Constrained Delaunay Triangulation Using 
  Graphics Hardware (2013).
    Meng Qi, Thanh-Tung Cao, Tiow-Seng Tan.
    IEEE Trans. Visualization and Computer Graphics, vol 19 (5), 736-748. 
