# guard against in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
  MESSAGE(FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt.")
endif()

# c++ settings
PROJECT(master_thesis CXX)
CMAKE_MINIMUM_REQUIRED(VERSION 3.22)
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  MESSAGE(STATUS "No build type selected, default to Release")
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING "Build type (default Release)" FORCE)
endif()
SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)
SET(CMAKE_CXX_STANDARD 14)
SET(CMAKE_CXX_STANDARD_REQUIRED ON)
SET(CMAKE_CXX_FLAGS "-O3 -Wall")

# project output path
SET(EXECUTABLE_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/bin")
SET(LIBRARY_OUTPUT_PATH "${PROJECT_SOURCE_DIR}/lib")

# project search path
INCLUDE_DIRECTORIES("${PROJECT_SOURCE_DIR}/include")

# add package
FIND_PACKAGE(Threads REQUIRED)
FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(benchmark REQUIRED)
FIND_PACKAGE(OpenSSL REQUIRED)

# add build target
ADD_SUBDIRECTORY(src)
ADD_SUBDIRECTORY(test)
ADD_SUBDIRECTORY(benchmark)
