cmake_minimum_required (VERSION 3.22)
project (ACMH_Kompute)

# find_package(OpenCV REQUIRED )

# include_directories(${OpenCV_INCLUDE_DIRS})

set(-O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall -gencode arch=compute_52,code=sm_52)

if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-pedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_subdirectory(kompute)

# For compilation ...
# Specify target & source files to compile it from
add_executable(
    main
    src/main.h
    src/main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
# target_link_libraries(ACMH_Kompute
#     ${OpenCV_LIBS}
#     )
target_link_libraries(main
    kompute)
