// [SimdTests - RGBHSV]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _RGBHSV_H
#define _RGBHSV_H

#include "../simdglobals.h"

typedef void (*ArgbAhsvFunc)(float* dst, const float* src, int length);

void ahsv_from_argb_ref(float* dst, const float* src, int length);
void argb_from_ahsv_ref(float* dst, const float* src, int length);

void ahsv_from_argb_hq(float* dst, const float* src, int length);
void argb_from_ahsv_hq(float* dst, const float* src, int length);

void ahsv_from_argb_sse2(float* dst, const float* src, int length);
void argb_from_ahsv_sse2(float* dst, const float* src, int length);

#endif // _RGBHSV_H
