cmake_minimum_required(VERSION 3.17)
project(Differentiation)

set(CMAKE_CXX_STANDARD 20)

add_executable(Differentiation
        src/main.cpp
        src/structures/Data.cpp src/structures/Data.h
        src/layers/DenseLayer.h
        src/layers/DuplicateDenseLayer.h
        src/layers/matmul.cpp
        src/optimiser/optimiser.cpp src/optimiser/optimiser.h
        src/network/merge.cpp src/network/merge.h
        src/misc/Reader.cpp src/misc/Reader.h
        src/misc/config.h
        src/network/network.cpp src/network/network.h
        src/layers/Layer.cpp
        src/layers/Layer.h
        src/layers/matmul.h
        src/mappings.cpp
        src/mappings.h
        src/misc/logging.h
        src/misc/logging.cpp
        src/layers/WeightClamp.cpp
        src/layers/WeightClamp.h
        src/network/ThreadData.cpp
        src/network/ThreadData.h
        src/optimiser/Adam.cpp
        src/optimiser/Adam.h
        src/optimiser/GD.cpp
        src/optimiser/GD.h
        src/activations/Activation.h
        src/activations/ReLU.cpp
        src/activations/ReLU.h
        src/activations/ClippedReLU.cpp
        src/activations/ClippedReLU.h
        src/activations/Linear.cpp
        src/activations/Linear.h
        src/activations/Sigmoid.cpp
        src/activations/Sigmoid.h
        src/loss/Loss.cpp
        src/loss/Loss.h
        src/loss/MSE.cpp
        src/loss/MSE.h
        src/verify/checkGradients.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")