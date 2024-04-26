cmake_minimum_required(VERSION 3.18)
project(delaunay-app)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()
find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode=arch=compute_86,code=sm_86
        --extended-lambda)

find_package(PCL QUIET)
if (NOT "${PCL_LIBRARIES}" STREQUAL "")
    include_directories(${PCL_INCLUDE_DIRS})
    link_directories(${PCL_LIBRARY_DIRS})
    add_definitions(${PCL_DEFINITIONS})
    add_compile_definitions(WITH_PCL)
endif ()

find_package(yaml-cpp REQUIRED)

add_subdirectory(gpu-delaunay-generator)

set(${PROJECT_NAME}_HEADERS
        inc/InputGenerator.h
        inc/RandGen.h
        inc/TriangulationHandler.h
        inc/PointType.h
        inc/json.h)

set(${PROJECT_NAME}_SOURCES
        src/RandGen.cpp
        src/InputGenerator.cpp
        src/TriangulationHandler.cpp)

cuda_add_executable(${PROJECT_NAME} main.cpp ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
set(CMAKE_VERBOSE_MAKEFILE "ON")

target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES} ${PCL_LIBRARIES} yaml-cpp gpu-delaunay-generator)
target_compile_definitions(${PROJECT_NAME} PRIVATE PROFILE_NONE=0 PROFILE_DEFAULT=1 PROFILE_DETAIL=2 PROFILE_DIAGNOSE=3 PROFILE_DEBUG=4 PROFILE_LEVEL_NUM=5)
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ERROR_CHECK_LEVEL=1 PROFILE_LEVEL=1)
target_include_directories(${PROJECT_NAME} PUBLIC
    "${PROJECT_BINARY_DIR}"
    "gpu-delaunay-generator"
)

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_definitions(DDEBUG)
endif()