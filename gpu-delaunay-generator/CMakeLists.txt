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
set(GDG_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/gpu-delaunay-generator"
        CACHE STRING "Path to install the CMake package to")

option(GDG_BUILD_SHARED_LIBS "Build gpu-delaunay-generator shared library" ${BUILD_SHARED_LIBS})
if (GDG_BUILD_SHARED_LIBS)
    set(gdg-type SHARED)
    set(gdg-label-postfix "shared")
else ()
    set(gdg-type STATIC)
    set(gdg-label-postfix "static")
endif ()
set(build-shared $<BOOL:${GDG_BUILD_SHARED_LIBS}>)

set(${PROJECT_NAME}_HEADERS
        inc/CommonTypes.h
        inc/DelaunayChecker.h
        inc/HashFunctors.h
        inc/json.h
        inc/PerfTimer.h
        inc/CPU/PredWrapper.h
        inc/GPU/GpuDelaunay.h
        inc/GPU/CudaWrapper.h
        inc/GPU/DPredWrapper.h
        inc/GPU/HostToKernel.h
        inc/GPU/KerCommon.h
        inc/GPU/KerDivision.h
        inc/GPU/KerPredicates.h
        inc/GPU/KerShewchuk.h
        inc/GPU/MemoryManager.h
        inc/GPU/SmallCounters.h
        inc/GPU/ThrustWrapper.h
        inc/IOType.h
        gpu-delaunay-generator.h)

set(${PROJECT_NAME}_SOURCES
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
        src/GPU/KerShewchuk.cu
)

add_library(${PROJECT_NAME} ${gdg-cpp-type} ${${PROJECT_NAME}_SOURCES} ${${PROJECT_NAME}_HEADERS})
message("${CUDA_TOOLKIT_INCLUDE}")
message("${CUDAToolkit_INCLUDE_DIRS}")
target_include_directories(${PROJECT_NAME}
        PUBLIC
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/inc>
        $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
        PRIVATE
        $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
        ${CUDA_TOOLKIT_INCLUDE}
        ${CUDAToolkit_INCLUDE_DIRS})

message("${CUDA_LIBRARIES}")
message("${CUDAToolkit_LIBRARIES}")
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ERROR_CHECK_LEVEL=1)
set_target_properties(${PROJECT_NAME} PROPERTIES
        VERSION ${PROJECT_VERSION}
        SOVERSION "${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}"
        PROJECT_LABEL "gpu-delaunay-generator ${gdg-label-postfix}"
        DEBUG_POSTFIX "${CMAKE_DEBUG_POSTFIX}")
set(EXPORT_TARGETS gpu-delaunay-generator)
configure_package_config_file(
        "${PROJECT_SOURCE_DIR}/gpu-delaunay-generator-config.cmake.in"
        "${PROJECT_BINARY_DIR}/gpu-delaunay-generator-config.cmake"
        INSTALL_DESTINATION "${GDG_INSTALL_CMAKEDIR}"
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_LIBDIR)
configure_file(${PROJECT_NAME}.pc.in ${PROJECT_NAME}.pc @ONLY)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/inc/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
        FILES_MATCHING PATTERN "*.h")


install(TARGETS ${PROJECT_NAME}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${CMAKE_BINARY_DIR}/${PROJECT_NAME}.pc
        DESTINATION ${CMAKE_INSTALL_DATAROOTDIR}/pkgconfig)