cmake_minimum_required(VERSION 2.8)

project(MemoryZeroBenchmark)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-mavx2 -g -O0")
add_executable(${PROJECT_NAME} "main.cpp")
