cmake_minimum_required(VERSION 3.18)
project(gpu-delaunay-generator VERSION 1.0.0 LANGUAGES CXX CUDA)
if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

include(CMakePackageConfigHelpers)
include(CMakeDependentOption)
include(CheckCXXCompilerFlag)
include(GNUInstallDirs)

find_package(CUDAToolkit QUIET)
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode=arch=compute_86,code=sm_86
        --extended-lambda)
set(CMAKE_VERBOSE_MAKEFILE "ON")

include(GNUInstallDirs)
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


add_library(gpu-delaunay-generator ${SOURCES})
set_target_properties(gpu-delaunay-generator PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        ${CUDA_TOOLKIT_INCLUDE}
        ${CUDAToolkit_INCLUDE_DIRS})

message("${CUDA_LIBRARIES}")
message("${CUDAToolkit_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ERROR_CHECK_LEVEL=1)