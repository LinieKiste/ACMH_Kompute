cmake_minimum_required (VERSION 3.7)
project (ACMH_Kompute LANGUAGES CXX)

if(false AND CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++17)
    add_definitions(-g)
    add_definitions(-pthread)
    add_definitions(-Wall)
    add_definitions(-Wextra)
    add_definitions(-Werror)
    add_definitions(-Wpedantic)
    add_definitions(-Wno-unused-function)
    add_definitions(-Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
endif()

add_subdirectory(extern/kompute)
find_package(OpenCV REQUIRED )

option( TRACY_ENABLE "" ON)
option( TRACY_ON_DEMAND "" ON)
add_subdirectory (extern/tracy)

# add_definitions(-g)
add_definitions(-O3)
add_definitions(-ffast-math)
add_definitions(-march=native)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native") # extend release-profile with fast-math
set(CMAKE_CXX_STANDARD 17)

# For compilation ...
# Specify target & source files to compile it from
add_executable(main
    src/main.cpp
    src/ACMH.cpp
    src/helpers.cpp
    src/sfm.cpp
    src/cache.cpp
    )

target_include_directories(main PRIVATE extern/tracy/public/tracy)

# For linking ...
# Specify target & libraries to link it with
target_link_libraries(main
    PUBLIC Tracy::TracyClient
    kompute
    ${OpenCV_LIBS}
    )
