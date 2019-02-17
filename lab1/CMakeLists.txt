cmake_minimum_required(VERSION 3.13)
project(lab1_1)

set(CMAKE_CXX_STANDARD 14)

add_executable(lab1_1 main.cpp
        matrix/BigMatrix.cpp
        matrix/BigMatrix.h
        matrix/InnerMatrix.cpp
        matrix/InnerMatrix.h
        random/random.cpp
        random/random.h)

set(CMAKE_CXX_COMPILER "/opt/intel/bin/icpc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -o -Wall -Wextra -O3 -qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=vec")

