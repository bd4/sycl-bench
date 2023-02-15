#include <benchmark/benchmark.h>

#include <sycl/sycl.hpp>

#include <experimental/mdspan>

#include <functional>

#include "common.h"

namespace stdex = std::experimental;

template <typename T>
class SumScale
{
public:
  SumScale(T factor) : factor_(factor) {}

  T operator()(T a, T b) const { return a + factor_ * b; }

private:
  T factor_;
};

template <typename E, typename T>
class ScaleExpr
{
public:
  ScaleExpr(E& expr, T factor) : expr_(expr), factor_(factor) {}

  T operator()(int i) const { return factor_ * expr_(i); }

private:
  E expr_;
  T factor_;
};

template <typename F, typename E1, typename E2, typename T>
class BinaryOpExpr
{
public:
  BinaryOpExpr(F f, E1& expr1, E2& expr2) : f_(f), expr1_(expr1), expr2_(expr2)
  {}

  T operator()(int i) const { return f_(expr1_(i), expr2_(i)); }

private:
  F f_;
  E1 expr1_;
  E2 expr2_;
};

// ======================================================================
// BM_device_memcpy

template <typename T>
static void BM_device_memcpy(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  for (auto _ : state) {
    q.memcpy(b, a, N);
    q.wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_memcpy<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_memcpy<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d

template <typename T>
static void BM_device_assign_1d(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  for (auto _ : state) {
    q.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) { a[i] = b[i]; })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_1d<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d_add

template <typename T>
static void BM_device_assign_1d_add(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  for (auto _ : state) {
    q.parallel_for(sycl::range<1>(N),
                   [=](sycl::id<1> i) { a[i] = b[i] + b[i]; })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_1d_add<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d_add<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d_add_scale

template <typename T>
static void BM_device_assign_1d_add_scale(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  for (auto _ : state) {
    q.parallel_for(sycl::range<1>(N),
                   [=](sycl::id<1> i) { a[i] = b[i] + 2 * b[i]; })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_1d_add_scale<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d_add_scale<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d_functor

template <typename T>
static void BM_device_assign_1d_functor(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);
  auto* c = sycl::malloc_device<T>(N, q);

  auto op = SumScale(T(7.0));

  for (auto _ : state) {
    q.parallel_for(sycl::range<1>(N),
                   [=](sycl::id<1> i) { a[i] = op(b[i], c[i]); })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
}

BENCHMARK(BM_device_assign_1d_functor<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d_functor<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_1d_array_expr

template <typename T>
static void BM_device_assign_1d_array_expr(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);
  auto* c = sycl::malloc_device<T>(N, q);

  using span_dyn_1d =
    stdex::mdspan<T, stdex::dextents<std::size_t, 1u>, stdex::layout_left>;

  auto aspan = span_dyn_1d(a, N);
  auto bspan = span_dyn_1d(b, N);
  auto cspan = span_dyn_1d(c, N);

  auto scale_expr = ScaleExpr<span_dyn_1d, T>(cspan, T(7.0));
  auto op = std::plus<T>();

  auto expr = BinaryOpExpr<decltype(op), span_dyn_1d, decltype(scale_expr), T>(
    op, bspan, scale_expr);

  for (auto _ : state) {
    q.parallel_for(sycl::range<1>(N),
                   [=](sycl::id<1> idx) { aspan(idx[0]) = expr(idx[0]); })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);
}

BENCHMARK(BM_device_assign_1d_array_expr<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_1d_array_expr<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d

template <typename T>
static void BM_device_assign_4d(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  using span_dyn_4d =
    stdex::mdspan<T, stdex::dextents<std::size_t, 4>, stdex::layout_left>;

  auto aspan = span_dyn_4d(a, n, n, n, n);
  auto bspan = span_dyn_4d(b, n, n, n, n);

  for (auto _ : state) {
    q.parallel_for(sycl::range<3>(n * n, n, n),
                   [=](sycl::id<3> idx) {
                     int i0 = idx[2];
                     int i1 = idx[1];
                     int i2 = idx[0] % n;
                     int i3 = idx[0] / n;
                     aspan(i0, i1, i2, i3) = bspan(i0, i1, i2, i3);
                   })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_4d<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d_add

template <typename T>
static void BM_device_assign_4d_add(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  using span_dyn_4d =
    stdex::mdspan<T, stdex::dextents<std::size_t, 4>, stdex::layout_left>;

  auto aspan = span_dyn_4d(a, n, n, n, n);
  auto bspan = span_dyn_4d(b, n, n, n, n);

  for (auto _ : state) {
    q.parallel_for(sycl::range<3>(n * n, n, n),
                   [=](sycl::id<3> idx) {
                     int i0 = idx[2];
                     int i1 = idx[1];
                     int i2 = idx[0] % n;
                     int i3 = idx[0] / n;
                     aspan(i0, i1, i2, i3) =
                       bspan(i0, i1, i2, i3) + bspan(i0, i1, i2, i3);
                   })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_4d_add<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d_add<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_device_assign_4d_add_scale

template <typename T>
static void BM_device_assign_4d_add_scale(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;

  sycl::queue q;

  auto* a = sycl::malloc_device<T>(N, q);
  auto* b = sycl::malloc_device<T>(N, q);

  using span_dyn_4d =
    stdex::mdspan<T, stdex::dextents<std::size_t, 4>, stdex::layout_left>;

  auto aspan = span_dyn_4d(a, n, n, n, n);
  auto bspan = span_dyn_4d(b, n, n, n, n);

  for (auto _ : state) {
    q.parallel_for(sycl::range<3>(n * n, n, n),
                   [=](sycl::id<3> idx) {
                     int i0 = idx[2];
                     int i1 = idx[1];
                     int i2 = idx[0] % n;
                     int i3 = idx[0] / n;
                     aspan(i0, i1, i2, i3) =
                       bspan(i0, i1, i2, i3) + 2 * bspan(i0, i1, i2, i3);
                   })
      .wait();
  }

  sycl::free(a, q);
  sycl::free(b, q);
}

BENCHMARK(BM_device_assign_4d_add_scale<double>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);
BENCHMARK(BM_device_assign_4d_add_scale<float>)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)
  ->Arg(SYCL_BENCH_PER_DIM_SIZE)
  ->Unit(benchmark::kMillisecond);

/*
// ======================================================================
// BM_device_assign_4d

static void BM_device_assign_4d(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<real_t>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a;
  gt::synchronize();

  for (auto _ : state) {
    b = a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d)->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)->Arg(SYCL_BENCH_PER_DIM_SIZE)->Unit(benchmark::kMillisecond);


// ======================================================================
// BM_device_assign_1d_op

static void BM_device_assign_1d_op(benchmark::State& state)
{
  int n = state.range(0);
  int N = n * n * n * n;
  auto a = gt::zeros_device<real_t>(gt::shape(N));
  auto b = gt::empty_like(a);

  // warmup, device compile
  auto expr = a + 2 * a;
  b = expr;
  gt::synchronize();

  for (auto _ : state) {
    b = expr;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_1d_op)->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)->Arg(SYCL_BENCH_PER_DIM_SIZE)->Unit(benchmark::kMillisecond);


// ======================================================================
// BM_device_assign_4d_op

static void BM_device_assign_4d_op(benchmark::State& state)
{
  int n = state.range(0);
  auto a = gt::zeros_device<real_t>(gt::shape(n, n, n, n));
  auto b = gt::empty_like(a);

  // warmup, device compile
  b = a + 2 * a;
  gt::synchronize();

  for (auto _ : state) {
    b = a + 2 * a;
    gt::synchronize();
  }
}

BENCHMARK(BM_device_assign_4d_op)->Arg(SYCL_BENCH_PER_DIM_SIZE - 1)->Arg(SYCL_BENCH_PER_DIM_SIZE)->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_add_ij_sten

auto i_sten_6d_5(const gt::gtensor_span<const real_t, 1>& sten,
                 const gt::gtensor_span_device<const complex_t, 6>& f, int bnd)
{
  return sten(0) * f.view(_s(bnd - 2, -bnd - 2)) +
         sten(1) * f.view(_s(bnd - 1, -bnd - 1)) +
         sten(2) * f.view(_s(bnd + 0, -bnd + 0)) +
         sten(3) * f.view(_s(bnd + 1, -bnd + 1)) +
         sten(4) * f.view(_s(bnd + 2, -bnd + 2));
}

static void BM_add_ij_sten(benchmark::State& state)
{
  const int bnd = 2; // # of ghost points
  auto shape_rhs = gt::shape(70, 32, 24, 24, 32, 2);
  auto shape_dist = shape_rhs;
  shape_dist[0] += 2 * bnd;

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto dist = gt::zeros_device<complex_t>(shape_dist);
  auto kj = gt::zeros_device<complex_t>({shape_rhs[1]});
  auto sten =
    gt::gtensor<real_t, 1>({1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.});

  real_t facj = 2.;

  auto fn = [&]() {
    rhs = rhs +
          facj *
            kj.view(_newaxis, _all, _newaxis, _newaxis, _newaxis, _newaxis) *
            dist.view(_s(bnd, -bnd)) +
          i_sten_6d_5(sten, dist, bnd);
    gt::synchronize();
  };

  // warmup, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_add_ij_sten)->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_add_dgdxy

// FIXME, almost same as i_sten_6d_f
template <typename E, typename E_sten>
auto x_deriv_5(const E& f, const E_sten& sten, int bnd)
{
  return sten(0) * f.view(_s(bnd - 2, -bnd - 2)) +
         sten(1) * f.view(_s(bnd - 1, -bnd - 1)) +
         sten(2) * f.view(_s(bnd + 0, -bnd + 0)) +
         sten(3) * f.view(_s(bnd + 1, -bnd + 1)) +
         sten(4) * f.view(_s(bnd + 2, -bnd + 2));
}

template <typename E, typename E_ikj>
auto y_deriv(const E& f, const E_ikj& ikj, int bnd)
{
  return ikj.view(_newaxis, _all, _newaxis) * f.view(_s(bnd, -bnd), _all, _all);
}

static void BM_add_dgdxy(benchmark::State& state)
{
  const int bnd = 2; // # of ghost points
  auto shape_rhs = gt::shape(70, 32, 24 * 24 * 32 * 2);
  auto shape_f = shape_rhs;
  shape_f[0] += 2 * bnd;

  auto sten =
    gt::gtensor<real_t, 1>({1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.});

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto f = gt::zeros_device<complex_t>(shape_f);
  auto ikj = gt::zeros_device<complex_t>({shape_rhs[1]});
  auto p1 = gt::zeros_device<complex_t>({shape_rhs[0], shape_rhs[2]});
  auto p2 = gt::zeros_device<complex_t>({shape_rhs[0], shape_rhs[2]});

  auto dij =
    gt::empty_device<complex_t>({shape_rhs[0], shape_rhs[1], shape_rhs[2], 2});

  auto fn = [&]() {
    dij.view(_all, _all, _all, 0) = x_deriv_5(f, sten, bnd);
    dij.view(_all, _all, _all, 1) = y_deriv(f, ikj, bnd);

    rhs = rhs + p1.view(_all, _newaxis) * dij.view(_all, _all, _all, 0) +
          p2.view(_all, _newaxis) * dij.view(_all, _all, _all, 1);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_add_dgdxy)->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_add_dgdxy_fused

static void BM_add_dgdxy_fused(benchmark::State& state)
{
  const int bnd = 2; // # of ghost points
  auto shape_rhs = gt::shape(70, 32, 24 * 24 * 32 * 2);
  auto shape_f = shape_rhs;
  shape_f[0] += 2 * bnd;

  auto sten =
    gt::gtensor<real_t, 1>({1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.});

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto f = gt::zeros_device<complex_t>(shape_f);
  auto ikj = gt::zeros_device<complex_t>({shape_rhs[1]});
  auto p1 = gt::zeros_device<complex_t>({shape_rhs[0], shape_rhs[2]});
  auto p2 = gt::zeros_device<complex_t>({shape_rhs[0], shape_rhs[2]});

  auto fn = [&]() {
    auto dx_f = x_deriv_5(f, sten, bnd);
    auto dy_f = y_deriv(f, ikj, bnd);

    rhs = rhs + p1.view(_all, _newaxis) * dx_f + p2.view(_all, _newaxis) * dy_f;
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_add_dgdxy_fused)->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_add_dgdxy_6d
//
// same as above, but without collapsing dims 3-6

template <typename E, typename E_ikj>
auto y_deriv_6d(const E& f, const E_ikj& ikj, int bnd)
{
  return ikj.view(_newaxis, _all, _newaxis, _newaxis, _newaxis, _newaxis) *
         f.view(_s(bnd, -bnd));
}

static void BM_add_dgdxy_6d(benchmark::State& state)
{
  const int bnd = 2; // # of ghost points
  auto shape_rhs = gt::shape(70, 32, 24, 24, 32, 2);
  auto shape_f = shape_rhs;
  shape_f[0] += 2 * bnd;

  auto sten =
    gt::gtensor<real_t, 1>({1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.});

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto f = gt::zeros_device<complex_t>(shape_f);
  auto ikj = gt::zeros_device<complex_t>({shape_rhs[1]});
  auto p1 = gt::zeros_device<complex_t>(
    {shape_rhs[0], shape_rhs[2], shape_rhs[3], shape_rhs[4], shape_rhs[5]});
  auto p2 = gt::zeros_device<complex_t>(
    {shape_rhs[0], shape_rhs[2], shape_rhs[3], shape_rhs[4], shape_rhs[5]});

  auto dij =
    gt::empty_device<complex_t>({shape_rhs[0], shape_rhs[1], shape_rhs[2],
                                 shape_rhs[3], shape_rhs[4], shape_rhs[5], 2});

  auto fn = [&]() {
    dij.view(_all, _all, _all, _all, _all, _all, 0) = x_deriv_5(f, sten, bnd);
    dij.view(_all, _all, _all, _all, _all, _all, 1) = y_deriv_6d(f, ikj, bnd);

    rhs =
      rhs +
      p1.view(_all, _newaxis) *
        dij.view(_all, _all, _all, _all, _all, _all, 0) +
      p2.view(_all, _newaxis) * dij.view(_all, _all, _all, _all, _all, _all, 1);
    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}

BENCHMARK(BM_add_dgdxy_6d)->Unit(benchmark::kMillisecond);

// ======================================================================
// BM_add_dgdxy_fused_6d

static void BM_add_dgdxy_fused_6d(benchmark::State& state)
{
  const int bnd = 2; // # of ghost points
  auto shape_rhs = gt::shape(70, 32, 24, 24, 32, 2);
  auto shape_f = shape_rhs;
  shape_f[0] += 2 * bnd;

  auto sten =
    gt::gtensor<real_t, 1>({1. / 12., -2. / 3., 0., 2. / 3., -1 / 12.});

  auto rhs = gt::zeros_device<complex_t>(shape_rhs);
  auto f = gt::zeros_device<complex_t>(shape_f);
  auto ikj = gt::zeros_device<complex_t>({shape_rhs[1]});
  auto p1 = gt::zeros_device<complex_t>(
    {shape_rhs[0], shape_rhs[2], shape_rhs[3], shape_rhs[4], shape_rhs[5]});
  auto p2 = gt::zeros_device<complex_t>(
    {shape_rhs[0], shape_rhs[2], shape_rhs[3], shape_rhs[4], shape_rhs[5]});

  auto fn = [&]() {
    auto dx_f = x_deriv_5(f, sten, bnd);
    auto dy_f = y_deriv_6d(f, ikj, bnd);

    rhs = rhs + p1.view(_all, _newaxis) * dx_f + p2.view(_all, _newaxis) * dy_f;

    gt::synchronize();
  };

  // warm up, device compile
  fn();

  for (auto _ : state) {
    fn();
  }
}
*/

BENCHMARK_MAIN();
