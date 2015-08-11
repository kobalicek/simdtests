// [SimdRgbHsv]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Zlib - See LICENSE.md file in the package.

#ifndef _SIMDRGBHSV_H
#define _SIMDRGBHSV_H

typedef void (*ArgbAhsvFunc)(float* dst, const float* src, int length);

void ahsv_from_argb_ref(float* dst, const float* src, int length);
void argb_from_ahsv_ref(float* dst, const float* src, int length);

void ahsv_from_argb_hq(float* dst, const float* src, int length);
void argb_from_ahsv_hq(float* dst, const float* src, int length);

void ahsv_from_argb_sse2(float* dst, const float* src, int length);
void argb_from_ahsv_sse2(float* dst, const float* src, int length);

#endif // _SIMDRGBHSV_H
