# GPU Delaunay Generator for 2.5D

This repository contains an implementation of GPU-based Delaunay Triangulation for 2.5D surfaces using CUDA.
The codebase is from the paper "Computing Two-dimensional Constrained Delaunay Triangulation Using Graphics Hardware" by
Meng Qi, Thanh-Tung Cao, Tiow-Seng Tan, available
at [https://www.comp.nus.edu.sg/~tants/cdt.html](https://www.comp.nus.edu.sg/~tants/cdt.html) (not on GitHub).

## Introduction

Delaunay triangulation is a fundamental algorithm in computational geometry used to partition a set of points into
triangles [Wikipedia - Delaunay Triangulation](https://en.wikipedia.org/wiki/Delaunay_triangulation). It has numerous
applications in various fields such as computer graphics, mesh generation, and geographic
information systems.

In a Delaunay triangulation, no point lies inside the circumcircle of any triangle formed by the points. This property
ensures that the triangulation is as "uniform" and "well-conditioned" as possible, making it useful for many
computational tasks.

This repository provides an efficient GPU-based implementation of Delaunay triangulation specifically tailored for 2.5D
surfaces, where each point has associated elevation data. 2.5D Delaunay triangulation can construct a Triangulated
Irregular Network (TIN) model, which is widely used in representing terrain surfaces in Geographic Information Systems (
GIS) and computational modeling,

## Features

- **GPU Acceleration:** Utilizes the computational power of GPUs to perform Delaunay triangulation efficiently using
  CUDA.
- **2.5D Support:** Handles points with elevation data, enabling triangulation of surfaces rather than just planar data.
- **Flexible Input/Output:** Generates customized points with various settings / Accepts input from files / Output in
  most widely used open-source format .geojson and .obj.

## Getting Started

### Prerequisites

- CMake version 3.18 or higher.
- CUDA-enabled GPU.
- [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) installed.
- [yaml-cpp](https://github.com/jbeder/yaml-cpp) installed.

### Build

1. Clone this repository to your local machine and navigate to the project directory:
    ```bash
    git clone https://github.com/WanruXX/gpu-delaunay-generator-2.5D.git
    cd gpu-delaunay-generator-2.5D
    ```

2. You may need to change your CUDA compiling flags according to your GPU's computational capability. You can look
   up for the compiling flags here [https://developer.nvidia.com/cuda-gpus](https://developer.nvidia.com/cuda-gpus) and
   then change [here](https://github.com/WanruXX/gpu-delaunay-generator-2.5D/blob/main/CMakeLists.txt#L10) (Unfortunately it doesn't support auto setting currently).


3. The program provides different levels of profiling and cuda error checking, which can change through [here](https://github.com/WanruXX/gpu-delaunay-generator-2.5D/blob/main/CMakeLists.txt#L42) according to your need.


4. Compile the source code using the provided makefile:

    ```bash
    mkdir build && cd build
    cmake ..
    make
    ```

### Usage

1. Set up running parameters in [config.yaml](https://github.com/WanruXX/gpu-delaunay-generator-2.5D/blob/main/conf/config.yaml).
2. Prepare your point data if you choose to input from a file, in the format of
    ```
    x1 y1 z1
    x2 y2 z2
    ...
    ```
3. Prepare your constraint data if you choose to add constrains for the triangulation, in the format of
   ```
   pt_id1 pt_id2
   pt_id3 pt_id4
   ...
   ```
4. Run the executable, providing the config file:
    ```bash
    ./delaunay-app ../conf/config.yaml
    ```

## Demo

