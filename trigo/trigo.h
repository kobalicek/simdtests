// [SimdTests - Trigo]
// SIMD optimized Trigonometric functions.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _TRIGO_H
#define _TRIGO_H

#include "../simdglobals.h"

typedef void (*TrigoFuncD)(double* dst, const double* src, size_t length);

void trigo_vsin_precise(double* dst, const double* src, size_t length);
void trigo_vsin_math_h(double* dst, const double* src, size_t length);

void trigo_vsin_cephes_sse2(double* dst, const double* src, size_t length);
void trigo_vsin_vml_sse2(double* dst, const double* src, size_t length);

#endif // _TRIGO_H
