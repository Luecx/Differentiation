cmake_minimum_required(VERSION 3.17)
project(Differentiation)

set(CMAKE_CXX_STANDARD 20)

add_executable(Differentiation
        src/activations/Activation.h
        src/activations/ClippedReLU.h       src/activations/ClippedReLU.cpp
        src/activations/Linear.h            src/activations/Linear.cpp
        src/activations/Sigmoid.h           src/activations/Sigmoid.cpp
        src/activations/ReLU.h              src/activations/ReLU.cpp

        src/layers/DenseLayer.h
        src/layers/DuplicateDenseLayer.h
        src/layers/matmul.h                 src/layers/matmul.cpp
        src/layers/Layer.h                  src/layers/Layer.cpp

        src/loss/Loss.h                     src/loss/Loss.cpp
        src/loss/MSE.h                      src/loss/MSE.cpp

        src/network/network.h               src/network/network.cpp
        src/network/merge.h                 src/network/merge.cpp
        src/network/ThreadData.h            src/network/ThreadData.cpp

        src/optimiser/optimiser.h           src/optimiser/optimiser.cpp
        src/optimiser/Adam.h                src/optimiser/Adam.cpp
        src/optimiser/GD.h                  src/optimiser/GD.cpp

        src/main.cpp
        src/structures/Data.h               src/structures/Data.cpp
        src/structures/Input.h

        src/verify/checkGradients.h

        src/misc/config.h
        src/misc/logging.h                  src/misc/logging.cpp

        src/position/piece.h
        src/position/position.h
        src/position/bitboard.h
        src/position/piecelist.h
        src/position/positionmeta.h
        src/position/defs.h
        src/position/result.h
        src/position/fenparsing.h
        src/position/square.h
        src/dataset/dataset.h
        src/dataset/header.h
        src/dataset/reader.h
        src/dataset/writer.h
        src/dataset/io.h
        src/dataset/batchloader.h src/misc/timer.h src/misc/csv.h src/misc/csv.cpp src/misc/csv.h src/dataset/shuffle.h)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -funroll-loops")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -ffast-math")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--whole-archive -lpthread -Wl,--no-whole-archive")

set(CMAKE_EXE_LINKER_FLAGS " -static")

target_link_libraries(Differentiation -static-libgcc -static-libstdc++)