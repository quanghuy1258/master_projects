# example
AUX_SOURCE_DIRECTORY("${CMAKE_SOURCE_DIR}/benchmark/example" EXAMPLE)
ADD_EXECUTABLE(exampleBenchmark ${EXAMPLE})
TARGET_LINK_LIBRARIES(exampleBenchmark benchmark)
