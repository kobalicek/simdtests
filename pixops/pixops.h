// [SimdTests - PIXOPS]
// SIMD optimized pixel manipulation.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _SIMDDEJPEG_H
#define _SIMDDEJPEG_H

#include <stdint.h>

// ============================================================================
// [SimdTests::PixOps - CrossFade]
// ============================================================================

typedef void (*PixelOpFunc)(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha);

void pixops_crossfade_ref(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha);
void pixops_crossfade_sse2(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha);
void pixops_crossfade_ssse3(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha);

#endif // _SIMDDEJPEG_H
