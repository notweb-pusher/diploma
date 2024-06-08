cmake_minimum_required(VERSION 3.17)
project(thesis)

set(CMAKE_CXX_STANDARD 20)

include_directories(${CMAKE_SOURCE_DIR}/eigen-3.4.0)
include_directories(${CMAKE_SOURCE_DIR}/EigenRand-0.5.0)

add_executable(thesis main.cpp
        Optimizer.cpp
        Layer.cpp
        NeuralNet.cpp
        mnist.h
        Except.h
        Except.cpp
)