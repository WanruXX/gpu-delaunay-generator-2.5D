project(gflip2d)
cmake_minimum_required(VERSION 3.18)
find_package(CUDA REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)
set(CMAKE_BUILD_TYPE Release)
set(CUDA_NVCC_FLAGS
    ${CUDA_NVCC_FLAGS};
    -gencode arch=compute_61,code=sm_61)
#add_definitions(-DREAL_TYPE_FP32)
include_directories(
    src
    )
cuda_add_executable(
    ${PROJECT_NAME}
      src/DelaunayChecker.cpp
      src/HashTable.cpp
      src/RandGen.cpp
      src/InputCreator.cpp
      src/gDel2D/CPU/PredWrapper.cpp
      src/gDel2D/CPU/predicates.cpp
      src/Main.cpp
      src/json.hpp
      src/gDel2D/GPU/GpuDelaunay.cu
      src/gDel2D/GPU/ThrustWrapper.cu
      src/gDel2D/GPU/KerPredicates.cu
      src/gDel2D/GPU/KerDivision.cu
      src/gDel2D/GPU/SmallCounters.cu)
set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

#include "GL/glew.h"
#
#include "gDel2D/GpuDelaunay.h"
#include "RandGen.h"
#include "HashTable.h"

