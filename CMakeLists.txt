cmake_minimum_required(VERSION 3.17)
project(Differentiation)

set(CMAKE_CXX_STANDARD 20)

add_executable(Differentiation
        src/main.cpp
        src/Data.cpp            src/Data.h
        src/Function.cpp        src/Function.h
        src/DenseLayer.h
        src/DuplicateDenseLayer.h
        src/matmul.cpp
        src/optimiser.cpp       src/optimiser.h
        src/merge.cpp           src/merge.h
        src/Reader.cpp          src/Reader.h
        src/config.h
        src/network.cpp         src/network.h
        src/Layer.cpp           src/Layer.h src/matmul.h src/mappings.cpp src/mappings.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")