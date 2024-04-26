cmake_minimum_required(VERSION 3.18)
project(gpu-delaunay-generator VERSION 1.0.0 LANGUAGES CXX CUDA)
if (POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif ()

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

find_package(CUDAToolkit QUIET)
find_package(CUDA REQUIRED)
set(CUDA_VERSION_STRING ${CMAKE_CUDA_COMPILER_VERSION})
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode=arch=compute_86,code=sm_86
        --extended-lambda)
set(CMAKE_VERBOSE_MAKEFILE "ON")

include_directories(include)

set(SOURCES
        src/DelaunayChecker.cpp
        src/CPU/PredWrapper.cpp
        src/GPU/DPredWrapper.cu
        src/CPU/predicates.cpp
        src/GPU/GpuDelaunay.cu
        src/GPU/ThrustWrapper.cu
        src/GPU/KerPredicates.cu
        src/GPU/KerDivision.cu
        src/GPU/SmallCounters.cu
        src/IOType.cpp
)


cuda_add_library(gpu-delaunay-generator ${SOURCES})

target_include_directories(${PROJECT_NAME}
        PRIVATE
        ${CUDA_TOOLKIT_INCLUDE})

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE_NONE=0 PROFILE_DEFAULT=1 PROFILE_DETAIL=2 PROFILE_DIAGNOSE=3 PROFILE_DEBUG=4 PROFILE_LEVEL_NUM=5)
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ERROR_CHECK_LEVEL=1 PROFILE_LEVEL=1)