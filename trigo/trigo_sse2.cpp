// [SimdTests - Trigo]
// SIMD optimized Trigonometric functions.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSE2

#include "../simdglobals.h"
#include "./trigo.h"


// ============================================================================
// [SSE/SSE2 Implementation]
// ============================================================================

// The input must be in domain [-1686629712, 1686629712].
//
// I tried to optimize the double to int conversion by using `magic`, but
// it was actually slower than using `_mm_cvttpd_epi32()` and it didn't
// offer greater domain for `x`.
static SIMD_INLINE __m128d sin_cephes_pd(__m128d x) {
  SIMD_CONST_SQ(sign     , SIMD_UINT64_C(0x8000000000000000));
  SIMD_CONST_SQ(inv_sign , SIMD_UINT64_C(0x7FFFFFFFFFFFFFFF));
  SIMD_CONST_SI(int32_one, 1);
  SIMD_CONST_SD(4_DIV_PI , 1.27323954473516268615107010698);
  SIMD_CONST_SD(DP1      , 7.85398125648498535156e-1);
  SIMD_CONST_SD(DP2      , 3.77489470793079817668e-8);
  SIMD_CONST_SD(DP3      , 2.69515142907905952645e-15);

#define DEFINE_DATA(name, x0, x1, x2, x3, x4, x5, xm, xa, y0, y1, y2, y3, y4, y5, ym, ya) \
  SIMD_ALIGN_VAR(static const double, name[], 16) = { \
    x0, x0, x1, x1, x2, x2, x3, x3, x4, x4, x5, x5, xm, xm, xa, xa, \
    y0, x0, y1, x1, y2, x2, y3, x3, y4, x4, y5, x5, ym, xm, ya, xa, \
    x0, y0, x1, y1, x2, y2, x3, y3, x4, y4, x5, y5, xm, ym, xa, ya, \
    y0, y0, y1, y1, y2, y2, y3, y3, y4, y4, y5, y5, ym, ym, ya, ya  \
  }

  DEFINE_DATA(sincos_coeff,
    1.58962301576546568060e-10,-2.50507477628578072866e-8,
    2.75573136213857245213e-6 ,-1.98412698295895385996e-4,
    8.33333333332211858878e-3 ,-1.66666666666666307295e-1, 1.0, 0.0,

   -1.13585365213876817300e-11, 2.08757008419747316778e-9,
   -2.75573141792967388112e-7 , 2.48015872888517045348e-5,
   -1.38888888888730564116e-3 , 4.16666666666665929218e-2,-0.5, 1.0);

  __m128d y;
  __m128d sign = x;                                        // Sign bit.

  x = _mm_and_pd(x, SIMD_GET_PD(inv_sign));                // Take the absolute value.
  y = _mm_mul_pd(x, SIMD_GET_PD(4_DIV_PI));                // Integer part of `x * 4 / PI`.

  __m128i ival = _mm_cvttpd_epi32(y);                      // Extract the integer part of y.
  __m128i ione = SIMD_GET_PI(int32_one);

  ival = _mm_add_epi32(ival, ione);                        // j += 1.
  ival = _mm_andnot_si128(ione, ival);                     // j &=~1.

  y = _mm_cvtepi32_pd(ival);
  ival = _mm_unpacklo_epi32(ival, ival);

  sign = _mm_xor_pd(sign,                                  // Swap the sign bit if `j & 4`.
    _mm_castsi128_pd(_mm_slli_epi64(ival, 61)));
  sign = _mm_and_pd(sign, SIMD_GET_PD(sign));              // Keep only the sign bit.

  // Get the polynom selection mask (j & 2):
  //   1. `0x0000000000000000` => `0    <= x <= PI/4`
  //   2. `0xFFFFFFFFFFFFFFFF` => `PI/4 <  x <= PI/2`
  ival = _mm_slli_epi32(ival, 30);
  ival = _mm_srai_epi32(ival, 31);

  // Extended precision modular arithmetic:
  //   x = ((x - y * DP1) - y * DP2) - y * DP3
  x = _mm_sub_pd(x, _mm_mul_pd(y, SIMD_GET_PD(DP1)));
  x = _mm_sub_pd(x, _mm_mul_pd(y, SIMD_GET_PD(DP2)));
  x = _mm_sub_pd(x, _mm_mul_pd(y, SIMD_GET_PD(DP3)));

  // Get the polynom coefficients for each lane (sin/cos).
  __m128d poly_mask = _mm_castsi128_pd(ival);
  const __m128d* coeff = reinterpret_cast<const __m128d*>(sincos_coeff) +
    static_cast<uintptr_t>(_mm_movemask_pd(poly_mask)) * 8;

  __m128d xx = _mm_mul_pd(x, x);
  y = coeff[0];
  y = Simd128::mad(y, xx, coeff[1]);
  y = Simd128::mad(y, xx, coeff[2]);
  y = Simd128::mad(y, xx, coeff[3]);
  y = Simd128::mad(y, xx, coeff[4]);
  y = Simd128::mad(y, xx, coeff[5]);
  y = _mm_mul_pd(y, xx);

  __m128d x_or_xx = _mm_or_pd(
    _mm_and_pd(xx, poly_mask),
    _mm_andnot_pd(poly_mask, x));

  y = _mm_mul_pd(y, x_or_xx);
  y = _mm_add_pd(y, _mm_mul_pd(x_or_xx, coeff[6]));
  y = _mm_add_pd(y, coeff[7]);

  return _mm_xor_pd(y, sign);
}

static SIMD_INLINE __m128d sin_vml_pd(__m128d x) {
  SIMD_CONST_SD(1_PI , 0.318309886183790671538);

  SIMD_CONST_SD(PI4_A, 0.78539816290140151978    * -4.0);
  SIMD_CONST_SD(PI4_B, 4.9604678871439933374e-10 * -4.0);
  SIMD_CONST_SD(PI4_C, 1.1258708853173288931e-18 * -4.0);
  SIMD_CONST_SD(PI4_D, 1.7607799325916000908e-27 * -4.0);

  SIMD_CONST_SD(sin_0,-7.97255955009037868891952e-18);
  SIMD_CONST_SD(sin_1, 2.81009972710863200091251e-15);
  SIMD_CONST_SD(sin_2,-7.64712219118158833288484e-13);
  SIMD_CONST_SD(sin_3, 1.60590430605664501629054e-10);
  SIMD_CONST_SD(sin_4,-2.50521083763502045810755e-08);
  SIMD_CONST_SD(sin_5, 2.75573192239198747630416e-06);
  SIMD_CONST_SD(sin_6,-1.98412698412696162806809e-04);
  SIMD_CONST_SD(sin_7, 8.33333333333332974823815e-03);
  SIMD_CONST_SD(sin_8,-1.66666666666666657414808e-01);

  SIMD_CONST_SD(magic, 6755399441055744.0);

  __m128d y = _mm_mul_pd(x, SIMD_GET_PD(1_PI));
  __m128d q = _mm_add_pd(y, SIMD_GET_PD(magic));
  __m128d i = _mm_castsi128_pd(_mm_slli_epi64(_mm_castpd_si128(q), 63));

  q = _mm_sub_pd(q, SIMD_GET_PD(magic));
  x = Simd128::mad(q, SIMD_GET_PD(PI4_A), x);
  x = Simd128::mad(q, SIMD_GET_PD(PI4_B), x);
  x = Simd128::mad(q, SIMD_GET_PD(PI4_C), x);
  x = Simd128::mad(q, SIMD_GET_PD(PI4_D), x);

  __m128d xx = _mm_mul_pd(x, x);
  x = _mm_xor_pd(x, i);

  __m128d u = SIMD_GET_PD(sin_0);
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_1));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_2));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_3));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_4));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_5));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_6));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_7));
  u = Simd128::mad(u, xx, SIMD_GET_PD(sin_8));
  u = Simd128::mad(xx, _mm_mul_pd(u, x), x);

  return u;
}

void trigo_vsin_cephes_sse2(double* dst, const double* src, size_t length) {
  size_t i = length;

  while (i) {
    if (!SimdUtils::isAligned(dst, 16) || i == 1) {
      __m128d d = _mm_load_sd(src);
      _mm_store_sd(dst, sin_cephes_pd(d));
      
      dst++;
      src++;

      if (--i == 0)
        break;
    }

    while (i >= 2) {
      __m128d d = _mm_loadu_pd(src);
      _mm_store_pd(dst, sin_cephes_pd(d));

      dst += 2;
      src += 2;
      i -= 2;
    }
  }
}

void trigo_vsin_vml_sse2(double* dst, const double* src, size_t length) {
  size_t i = length;

  while (i) {
    if (!SimdUtils::isAligned(dst, 16) || i == 1) {
      __m128d d = _mm_load_sd(src);
      _mm_store_sd(dst, sin_vml_pd(d));
      
      dst++;
      src++;

      if (--i == 0)
        break;
    }

    while (i >= 2) {
      __m128d d = _mm_loadu_pd(src);
      _mm_store_pd(dst, sin_vml_pd(d));

      dst += 2;
      src += 2;
      i -= 2;
    }
  }
}
