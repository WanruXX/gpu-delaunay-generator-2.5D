cmake_minimum_required(VERSION 3.18)
project(example)

cuda_add_executable(${PROJECT_NAME} example.cpp)

set(CMAKE_VERBOSE_MAKEFILE "ON")
target_link_libraries(${PROJECT_NAME} gpu-delaunay-generator)