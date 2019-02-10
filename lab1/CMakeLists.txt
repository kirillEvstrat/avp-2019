cmake_minimum_required(VERSION 3.13)
project(lab1_1)

set(CMAKE_CXX_STANDARD 11)

add_executable(lab1_1 main.cpp
        matrix/OuterMatrix.cpp
        matrix/OuterMatrix.h
        matrix/InnerMatrix.cpp
        matrix/InnerMatrix.h
        random/random.cpp
        random/random.h)

set(CMAKE_CXX_COMPILER "/opt/intel/bin/icpc")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -Wall -Wextra -fast -qopt-report=5 -qopt-report-file=stdout -qopt-report-phase=vec")

