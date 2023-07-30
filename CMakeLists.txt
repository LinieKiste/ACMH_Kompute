cmake_minimum_required (VERSION 3.7)
project (ACMH_Kompute LANGUAGES CXX)

if(false AND CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++17)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-Werror)
    add_definitions(-Wpedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

find_package(SailC++ REQUIRED)
add_subdirectory(extern/kompute)

# For compilation ...
# Specify target & source files to compile it from
add_executable(main
    src/main.h
    src/main.cpp
    )

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(main PRIVATE
    kompute
    SAIL::sail-c++
    )
