// [SimdTests - DePNG]
// SIMD optimized "PNG Reverse Filter" implementation.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _DEPNG_H
#define _DEPNG_H

#include "../simdglobals.h"

// ============================================================================
// [SimdTests::DePNG - Helpers]
//
// These are some inlines that are used across the reference and the optimized
// code. The goal is to move some logic here so the implementation is not
// polluted by code that can be easily moved out.
// ============================================================================

// Sum and pack to BYTE. The compiler should omit the `0xFF` as when the result
// is stored in memory it's done automatically.
static SIMD_INLINE uint8_t depng_sum(uint32_t a, uint32_t b) {
  return static_cast<uint8_t>((a + b) & 0xFF);
}

// Unsigned division by 3 translated into a multiplication and shift. The range
// of `x` is [0, 255], inclusive. This means that we need at most 16 bits to
// have the result. In SIMD this is exploited by using PMULHUW instruction that
// will multiply and shift by 16 bits right (the constant is adjusted for that).
static SIMD_INLINE int32_t depng_udiv3(int32_t x) {
  return (x * 0xAB) >> 9;
}

// Return an absolute value of `x`. This is used exclusively by `depng_paeth_ref()`
// implementation. You can experiment by trying to make `abs()` condition-
// less, but since the `depng_paeth_opt()` is using much better approach than
// the reference implementation it probably doesn't matter at all.
static SIMD_INLINE int32_t depng_abs(int32_t x) {
  return x >= 0 ? x : -x;
}

// Reference implementation of PNG's AVG reverse filter. Please note that the
// SIMD functions (PAVGB, PAVGW) are not equal to the AVG method required by
// PNG; SSE2 instructions add `1` before the result is shifted, thus becomes
// rounded instead of truncated.
static SIMD_INLINE uint32_t depng_avg(uint32_t a, uint32_t b) {
  return (a + b) >> 1;
}

// Reference implementation of PNG's Paeth reverse filter. This implementation
// follows the specification pretty closely with only minor optimizations done.
// This implementation is found in many PNG decoders; good to test against.
static SIMD_INLINE uint32_t depng_paeth_ref(uint32_t b, uint32_t a, uint32_t c) {
  int32_t pa = static_cast<int32_t>(b) - static_cast<int32_t>(c);
  int32_t pb = static_cast<int32_t>(a) - static_cast<int32_t>(c);
  int32_t pc = pa + pb;

  pa = depng_abs(pa);
  pb = depng_abs(pb);
  pc = depng_abs(pc);

  return pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
}

// This is an optimized implementation of PNG's Paeth reference filter. This
// optimization originally comes from my previous implementation where I tried
// to simplify it to be more SIMD friendly. One interesting property of Paeth
// filter is:
//
//   Paeth(a, b, c) == Paeth(b, a, c);
//
// Actually what the filter needs is a minimum and maximum of `a` and `b`, so
// I based the implementation on getting those first. If you know `min(a, b)`
// and `max(a, b)` you can divide the interval to be checked against `c`. This
// requires division by 3, which is available above as `depng_udiv3()`.
//
// The previous implementation looked like:
//
//   static inline uint32_t Paeth(uint32_t a, uint32_t b, uint32_t c) {
//     uint32_t minAB = min(a, b);
//     uint32_t maxAB = max(a, b);
//     uint32_t divAB = depng_udiv3(maxAB - minAB);
//
//     if (c <= minAB + divAB) return maxAB;
//     if (c >= maxAB - divAB) return minAB;
//
//     return c;
//   }
//
// Although it's not bad I tried to exploit more the idea of SIMD and masking.
// The following code basically removes the need of any comparison, it relies
// on bit shifting and performs an arithmetic (not logical) shift of signs
// produced by `divAB + minAB` and `divAB - maxAB`, which are then used to mask
// out `minAB` and `maxAB`. The `minAB` and `maxAB` can be negative after `c`
// is subtracted, which will basically remove the original `c` if one of the
// two additions is unmasked. The code can unmask either zero or one addition,
// but it never unmasks both.
//
// Don't hesitate to contact the author <kobalicek.petr@gmail.com> if you need
// a further explanation of the code below, it's probably hard to understand
// without looking into the original Paeth implementation and without having a
// visualization of the Paeth function.
static SIMD_INLINE uint32_t depng_paeth_opt(uint32_t a, uint32_t b, uint32_t c) {
  int32_t minAB = static_cast<int32_t>(SimdUtils::min(a, b)) - c;
  int32_t maxAB = static_cast<int32_t>(SimdUtils::max(a, b)) - c;
  int32_t divAB = depng_udiv3(maxAB - minAB);

  return c + static_cast<uint32_t>(maxAB & ~((divAB + minAB) >> 31)) +
             static_cast<uint32_t>(minAB & ~((divAB - maxAB) >> 31)) ;
}

// ============================================================================
// [SimdTests::DePNG - PngFilterType]
// ============================================================================

enum PngFilterType {
  kPngFilterNone  = 0,
  kPngFilterSub   = 1,
  kPngFilterUp    = 2,
  kPngFilterAvg   = 3,
  kPngFilterPaeth = 4,
  kPngFilterCount = 5
};

// ============================================================================
// [SimdTests::DePNG - FilterFunc]
// ============================================================================

typedef void (*DePngFilterFunc)(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl);

void depng_filter_ref(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl);
void depng_filter_opt(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl);
void depng_filter_sse2(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl);

#endif // _DEPNG_H
