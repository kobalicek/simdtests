// [SimdTests - DeJPEG]
// SIMD optimized JPEG decoding utilities.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _SIMDTESTS_DEJPEG_H
#define _SIMDTESTS_DEJPEG_H

#include "../simdglobals.h"

// ============================================================================
// [SimdTests::DeJPEG - Utilities]
// ============================================================================

template<typename T>
static SIMD_INLINE uint8_t clampToByte(T x) {
  return x < 0 ? static_cast<uint8_t>(0) :
         x > 255 ? static_cast<uint8_t>(255) : static_cast<uint8_t>(x);
}

template<typename T>
static SIMD_INLINE uint32_t dejepeg_pack32(T a, T b, T c, T d) {
  return (static_cast<uint32_t>(a) << 24) |
         (static_cast<uint32_t>(b) << 16) |
         (static_cast<uint32_t>(c) <<  8) |
         (static_cast<uint32_t>(d)      ) ;
}

// ============================================================================
// [SimdTests::DeJPEG - DeZigZag]
// ============================================================================

// De-zig-zag translates a matrix of 8x8 coefficients into the JPEG's natural
// order (this is how it's called in libjpeg and libjpeg-turbo). This usually
// happens during the block decoding so I'm not sure how useful this function
// actually is. It's just an experiment checking how fast a massive PSHUFB
// could be.
typedef void (*DeJpegDeZigZag8x8Func)(int16_t* dst, const int16_t* src);

void dejpeg_dezigzag_ref(int16_t* dst, const int16_t* src);
void dejpeg_dezigzag_ssse3_v1(int16_t* dst, const int16_t* src);
void dejpeg_dezigzag_ssse3_v2(int16_t* dst, const int16_t* src);

// ============================================================================
// [SimdTests::DeJPEG - IDCT]
// ============================================================================

// Derived from jidctint's `jpeg_idct_islow`.
#define JPEG_IDCT_PREC 12
#define JPEG_IDCT_HALF(precision) (1 << ((precision) - 1))

#define JPEG_IDCT_SCALE(x) ((x) << JPEG_IDCT_PREC)
#define JPEG_IDCT_FIXED(x) static_cast<int>(((double)(x) * (double)(1 << JPEG_IDCT_PREC) + 0.5))

#define JPEG_IDCT_M_2_562915447 -JPEG_IDCT_FIXED(2.562915447)
#define JPEG_IDCT_M_1_961570560 -JPEG_IDCT_FIXED(1.961570560)
#define JPEG_IDCT_M_1_847759065 -JPEG_IDCT_FIXED(1.847759065)
#define JPEG_IDCT_M_0_899976223 -JPEG_IDCT_FIXED(0.899976223)
#define JPEG_IDCT_M_0_390180644 -JPEG_IDCT_FIXED(0.390180644)
#define JPEG_IDCT_P_0_298631336  JPEG_IDCT_FIXED(0.298631336)
#define JPEG_IDCT_P_0_541196100  JPEG_IDCT_FIXED(0.541196100)
#define JPEG_IDCT_P_0_765366865  JPEG_IDCT_FIXED(0.765366865)
#define JPEG_IDCT_P_1_175875602  JPEG_IDCT_FIXED(1.175875602)
#define JPEG_IDCT_P_1_501321110  JPEG_IDCT_FIXED(1.501321110)
#define JPEG_IDCT_P_2_053119869  JPEG_IDCT_FIXED(2.053119869)
#define JPEG_IDCT_P_3_072711026  JPEG_IDCT_FIXED(3.072711026)

// Keep 2 bits of extra precision for the intermediate results.
#define JPEG_IDCT_COL_NORM (JPEG_IDCT_PREC - 2)
#define JPEG_IDCT_COL_BIAS JPEG_IDCT_HALF(JPEG_IDCT_COL_NORM)

// Consume 2 bits of an intermediate results precision and 3 bits that were
// produced by `2 * sqrt(8)`. Also normalize to from `-128..127` to `0..255`.
#define JPEG_IDCT_ROW_NORM (JPEG_IDCT_PREC + 2 + 3)
#define JPEG_IDCT_ROW_BIAS (JPEG_IDCT_HALF(JPEG_IDCT_ROW_NORM) + (128 << JPEG_IDCT_ROW_NORM))

// Dequantize the coefficients given in `src` by quantization table `qTable` and
// store the result in `dst`. This function does a quantization and IDCT in one
// run.
typedef void (*DeJpegIDCTFunc)(uint8_t* dst, intptr_t dstStride, const int16_t* src, const uint16_t* qTable);

void dejpeg_idct_islow_ref(uint8_t* dst, intptr_t dstStride, const int16_t* src, const uint16_t* qTable);
void dejpeg_idct_islow_sse2(uint8_t* dst, intptr_t dstStride, const int16_t* src, const uint16_t* qTable);

// ============================================================================
// [SimdTests::DeJPEG - YCbCrToRGB32]
// ============================================================================

#define JPEG_YCBCR_PREC 12
#define JPEG_YCBCR_SCALE(x) ((x) << JPEG_YCBCR_PREC)
#define JPEG_YCBCR_FIXED(x) static_cast<int>(((double)(x) * (double)(1 << JPEG_YCBCR_PREC) + 0.5))

typedef void (*YCbCrToRgbFunc)(uint8_t* dst, const uint8_t* pY, const uint8_t* pCb, const uint8_t* pCr, uint32_t count);

void dejpeg_ycbcr_to_rgb32_ref(uint8_t* dst, const uint8_t* pY, const uint8_t* pCb, const uint8_t* pCr, uint32_t count);
void dejpeg_ycbcr_to_rgb32_sse2(uint8_t* dst, const uint8_t* pY, const uint8_t* pCb, const uint8_t* pCr, uint32_t count);

#endif // _SIMDTESTS_DEJPEG_H
