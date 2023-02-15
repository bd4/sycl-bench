cmake_minimum_required(VERSION 3.15)
project(sycl-bench LANGUAGES CXX)

set(SYCL_BENCH_PER_DIM_SIZE "128" CACHE STRING "Set to e.g. 96 for consumer gpu")

include(cmake/CPM.cmake)

CPMAddPackage(
  NAME benchmark
  GITHUB_REPOSITORY google/benchmark
  GIT_TAG v1.7.1
  VERSION v1.7.1
  GIT_SHALLOW TRUE
  OPTIONS
    "BENCHMARK_ENABLE_TESTING OFF"
    "BENCHMARK_ENABLE_INSTALL OFF"
    "BENCHMARK_ENABLE_WERROR OFF"
    "CMAKE_BUILD_TYPE Release"
)

CPMAddPackage(
  NAME mdspan
  GITHUB_REPOSITORY kokkos/mdspan
  GIT_TAG stable
  VERSION stable
  OPTIONS
    "CMAKE_BUILD_TYPE Release"
)

if (SYCL_BENCH_ENABLE_GTENSOR)
  CPMAddPackage(
    NAME gtensor
    GITHUB_REPOSITORY wdmapp/gtensor
    GIT_TAG main
    VERSION main
    OPTIONS
      "CMAKE_BUILD_TYPE RelWithDebInfo"
      "GTENSOR_DEVICE sycl"
  )
endif()

add_library(common INTERFACE)
target_compile_definitions(common INTERFACE
                           SYCL_BENCH_PER_DIM_SIZE=${SYCL_BENCH_PER_DIM_SIZE})

add_subdirectory(src)
