cmake_minimum_required(VERSION 3.15)
project(sycl-bench LANGUAGES CXX)

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

add_subdirectory(src)
