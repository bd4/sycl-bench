#include <benchmark/benchmark.h>

#include <sycl/sycl.hpp>

#include <gtensor/gtensor.h>

#include "common.h"

// ======================================================================
// BM_gt_device_memcpy

template <typename T>
static void BM_gt_device_memcpy(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  for (auto _ : state) {
    gt::copy(a, b);
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_memcpy<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_memcpy<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_gt_device_assign_1d

template <typename T>
static void BM_gt_device_assign_1d(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_assign_1d<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_assign_1d<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_gt_device_assign_1d_add

template <typename T>
static void BM_gt_device_assign_1d_add(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  for (auto _ : state) {
    b = a + a;
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_assign_1d_add<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_assign_1d_add<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_gt_device_assign_1d_add_scale

template <typename T>
static void BM_gt_device_assign_1d_add_scale(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  // warmup
  a = b + 2 * b;
  gt::synchronize();

  for (auto _ : state) {
    a = b + 2 * b;
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_assign_1d_add_scale<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_assign_1d_add_scale<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_gt_device_assign_1d_add_scale_launch

template <typename T>
static void BM_gt_device_assign_1d_add_scale_launch(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  auto a = gt::empty_device<T>({N});
  auto b = gt::empty_device<T>({N});

  auto k_a = a.to_kernel();
  auto k_b = b.to_kernel();

  for (auto _ : state) {
    gt::launch<1>(
      a.shape(), GT_LAMBDA(int i) { k_a(i) = k_b(i) + 2 * k_b(i); });
    gt::synchronize();
  }
}

BENCHMARK(BM_gt_device_assign_1d_add_scale_launch<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_gt_device_assign_1d_add_scale_launch<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
