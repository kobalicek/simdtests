// [SimdTests - Trigo]
// SIMD optimized Trigonometric functions.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "../3rdparty/real.h"
#include "./trigo.h"

#include <math.h>

// ============================================================================
// [Pure C Implementation]
// ============================================================================

void trigo_vsin_precise(double* dst, const double* src, size_t length) {
  size_t i = 0;
  for (;;) {
    double x = src[i];
    dst[i] = lol::sin(lol::real(x));

    if (++i >= length)
      return;

    while (src[i] == x) {
      dst[i] = dst[i - 1];
      if (++i >= length)
        return;
    }
  }
}

void trigo_vsin_math_h(double* dst, const double* src, size_t length) {
  for (size_t i = 0; i < length; i++)
    dst[i] = ::sin(src[i]);
}
