// [SimdTests - Trigo]
// SIMD optimized Trigonometric functions.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./trigo.h"

#define BENCH_COUNT 5
#define BENCH_ITER 500

// ============================================================================
// [SimdTests - Trigo - Fill]
// ============================================================================

static void trigo_fill(double* data, size_t length, double vmin, double vmax) {
  double scale = (vmax - vmin) / static_cast<double>(length - 1);
  
  for (size_t i = 0; i < length; i++) {
    data[i] = vmin + static_cast<double>(i) * scale;
  }
}

// ============================================================================
// [SimdTests - Trigo - Check]
// ============================================================================

static void trigo_check(
  const char* name,
  TrigoFuncD fn,
  const double* inputs, const double* ref, size_t length) {

  double* results = static_cast<double*>(malloc(length * sizeof(double)));

  double maxError = 0.0;
  uint64_t maxUlp = 0;
  uint64_t sumUlp = 0;
  unsigned nDiffs = 0;

  fn(results, inputs, length);

  for (size_t i = 0; i < length; i++) {
    double aVal = ref[i];
    double bVal = results[i];

    if (aVal != bVal) {
      double eps = fabs(aVal - bVal);
      uint64_t ulp = SimdUtils::ulpDiff(aVal, bVal);

      sumUlp += ulp;

      if (ulp > maxUlp) maxUlp = ulp;
      if (eps > maxError) maxError = eps;

      /*
      if (ulp > 2) {
        printf("[ERROR] IMPL=%-20s (Input=%.17g)\n"
               "  ref(%.17g) !=\n"
               "  res(%.17g) (ulp=%llu | epsilon=%.17g)\n", name, inputs[i], aVal, bVal, ulp, eps);
      }
      */

      nDiffs++;
    }
  }

  printf("[CHECK] IMPL=%-20s [Differences=%u/%u MaxUlp=%llu SumUlp=%llu MaxEpsilon=%0.17g]\n", name,
    nDiffs, static_cast<unsigned int>(length), maxUlp, sumUlp, maxError);

  ::free(results);
}

// ============================================================================
// [SimdTests - Trigo - Bench]
// ============================================================================

static void trigo_bench(
  const char* name,
  TrigoFuncD func,
  const double* inputs, size_t length) {

  SimdTimer timer;
  uint32_t best = 0xFFFFFFFFU;

  // Dummy counter to prevent optimizations.
  double* outputs = static_cast<double*>(malloc(length * sizeof(double)));
  double dummy = 0.0;

  for (uint32_t z = 0; z < BENCH_COUNT; z++) {
    uint32_t alpha = 1;

    timer.start();
    for (uint32_t i = 0; i < BENCH_ITER; i++) {
      func(outputs, inputs, length);
      dummy += outputs[0];
    }
    timer.stop();

    if (timer.get() < best)
      best = timer.get();
  }

  printf("[BENCH] IMPL=%-20s [%.2u.%.3u s] {dummy=%f}\n", name, best / 1000, best % 1000, dummy);
  ::free(outputs);
}

static void trigo_do_domain(size_t count, double a, double b) {
  double* inputs = static_cast<double*>(malloc(count * sizeof(double)));
  double* ref = static_cast<double*>(malloc(count * sizeof(double)));

  printf("[TRIGO] Test %llu values in domain\n"
         "  MIN=%g\n"
         "  MAX=%g\n", (uint64_t)count, a, b);

  trigo_fill(inputs, count, a, b);
  trigo_vsin_precise(ref, inputs, count);

  trigo_check("sin (math.h)"     , trigo_vsin_math_h      , inputs, ref, count);
  trigo_check("sin (Cephes-SSE2)", trigo_vsin_cephes_sse2 , inputs, ref, count);
  trigo_check("sin (VML-SSE2)"   , trigo_vsin_vml_sse2, inputs, ref, count);

  trigo_bench("sin (math.h)"     , trigo_vsin_math_h      , inputs, count);
  trigo_bench("sin (Cephes-SSE2)", trigo_vsin_cephes_sse2 , inputs, count);
  trigo_bench("sin (VML-SSE2)"   , trigo_vsin_vml_sse2, inputs, count);

  ::free(inputs);
  ::free(ref);

  printf("\n");
}

// ============================================================================
// [SimdTests - Trigo - Main]
// ============================================================================

// Problematic inputs:
//   3.1415926535897931
int main(int argc, char* argv[]) {
  size_t count = 1000000;
  const double PI = 3.141592653589793238;

  trigo_do_domain(count, -PI, 0);
  trigo_do_domain(count, 0, PI);
  trigo_do_domain(count, -100.0, 0.0);
  trigo_do_domain(count, 0.0, 100.0);
  trigo_do_domain(count, 100.0, 10000.0);
  trigo_do_domain(count, 100000.0, 1686629712.0);
  trigo_do_domain(count, 1686629712.0, 1e100);

  return 0;
}
