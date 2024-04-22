cmake_minimum_required(VERSION 3.18)
project(example)

if(POLICY CMP0146)
    cmake_policy(SET CMP0146 OLD)
endif()

find_package(CUDA REQUIRED)
set(CUDA_NVCC_FLAGS
        ${CUDA_NVCC_FLAGS};
        -gencode=arch=compute_86,code=sm_86
        --extended-lambda)

add_subdirectory(gpu-delaunay-generator)

cuda_add_executable(${PROJECT_NAME} example.cpp)
target_compile_definitions(${PROJECT_NAME} PRIVATE CUDA_ERROR_CHECK_LEVEL=1)
target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
target_include_directories(${PROJECT_NAME} PUBLIC
        "${PROJECT_BINARY_DIR}"
        "${PROJECT_SOURCE_DIR}/gpu-delaunay-generator"
)