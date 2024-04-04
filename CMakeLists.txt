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
        src/DelaunayChecker.cpp
        src/RandGen.cpp
        src/InputGenerator.cpp
        src/CPU/PredWrapper.cpp
        src/CPU/predicates.cpp
        src/GPU/GpuDelaunay.cu
        src/GPU/ThrustWrapper.cu
        src/GPU/KerPredicates.cu
        src/GPU/KerDivision.cu
        src/GPU/SmallCounters.cu
        src/TriangulationHandler.cpp
        src/IOType.cpp)

set(${PROJECT_NAME}_HEADERS
        inc/CommonTypes.h
        inc/DelaunayChecker.h
        inc/HashFunctors.h
        inc/InputGenerator.h
        inc/json.h
        inc/PerfTimer.h
        inc/RandGen.h
        inc/CPU/PredWrapper.h
        inc/GPU/GpuDelaunay.h
        inc/GPU/CudaWrapper.h
        inc/GPU/DPredWrapper.h
        inc/GPU/KerPredWrapper.h
        inc/GPU/HostToKernel.h
        inc/GPU/KerCommon.h
        inc/GPU/KerDivision.h
        inc/GPU/KerPredicates.h
        inc/GPU/KerShewchuk.h
        inc/GPU/MemoryManager.h
        inc/GPU/SmallCounters.h
        inc/GPU/ThrustWrapper.h
        inc/TriangulationHandler.h
        inc/PointType.h
        inc/IOType.h)

cuda_add_executable(${PROJECT_NAME} main.cpp ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} ${PCL_LIBRARIES} yaml-cpp Eigen3::Eigen)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DDEBUG)
endif()