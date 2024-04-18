cmake_minimum_required(VERSION 3.18)
project(delaunay_generator)

find_package(CUDA REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PCL QUIET)
if (NOT "${PCL_LIBRARIES}" STREQUAL "")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    add_compile_definitions(WITH_PCL)
endif ()

# change your arch=<> and code=<> according to your gpu computational capability
# https://developer.nvidia.com/cuda-gpus
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode=arch=compute_86,code=sm_86
        --extended-lambda)

set(${PROJECT_NAME}_SOURCES
        gpu-delaunay-generator/src/DelaunayChecker.cpp
        src/RandGen.cpp
        src/InputGenerator.cpp
        gpu-delaunay-generator/src/CPU/PredWrapper.cpp
        gpu-delaunay-generator/src/CPU/predicates.cpp
        gpu-delaunay-generator/src/GPU/GpuDelaunay.cu
        gpu-delaunay-generator/src/GPU/ThrustWrapper.cu
        gpu-delaunay-generator/src/GPU/KerPredicates.cu
        gpu-delaunay-generator/src/GPU/KerDivision.cu
        gpu-delaunay-generator/src/GPU/SmallCounters.cu
        src/TriangulationHandler.cpp
        gpu-delaunay-generator/src/IOType.cpp)

set(${PROJECT_NAME}_HEADERS
        gpu-delaunay-generator/inc/CommonTypes.h
        gpu-delaunay-generator/inc/DelaunayChecker.h
        gpu-delaunay-generator/inc/HashFunctors.h
        inc/InputGenerator.h
        gpu-delaunay-generator/inc/json.h
        gpu-delaunay-generator/inc/PerfTimer.h
        inc/RandGen.h
        gpu-delaunay-generator/inc/CPU/PredWrapper.h
        gpu-delaunay-generator/inc/GPU/GpuDelaunay.h
        gpu-delaunay-generator/inc/GPU/CudaWrapper.h
        gpu-delaunay-generator/inc/GPU/DPredWrapper.h
        gpu-delaunay-generator/src/GPU/DPredWrapper.cu
        gpu-delaunay-generator/inc/GPU/HostToKernel.h
        gpu-delaunay-generator/inc/GPU/KerCommon.h
        gpu-delaunay-generator/inc/GPU/KerDivision.h
        gpu-delaunay-generator/inc/GPU/KerPredicates.h
        gpu-delaunay-generator/inc/GPU/MemoryManager.h
        gpu-delaunay-generator/inc/GPU/SmallCounters.h
        gpu-delaunay-generator/inc/GPU/ThrustWrapper.h
        inc/TriangulationHandler.h
        inc/PointType.h
        gpu-delaunay-generator/inc/IOType.h)

cuda_add_executable(${PROJECT_NAME} main.cpp ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} yaml-cpp Eigen3::Eigen)
target_compile_definitions(${PROJECT_NAME}  PRIVATE CUDA_ERROR_CHECK_LEVEL=1)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DDEBUG)
endif()