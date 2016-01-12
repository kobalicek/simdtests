// [SimdTests - DeJPEG]
// SIMD optimized JPEG decoding utilities.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./dejpeg.h"

// ============================================================================
// [DeZigZag - Ref]
// ============================================================================

static const uint8_t dejpeg_dezigzag_table[64] = {
  0 , 1 , 8 , 16, 9 , 2 , 3 , 10,
  17, 24, 32, 25, 18, 11, 4 , 5 ,
  12, 19, 26, 33, 40, 48, 41, 34,
  27, 20, 13, 6 , 7 , 14, 21, 28,
  35, 42, 49, 56, 57, 50, 43, 36,
  29, 22, 15, 23, 30, 37, 44, 51,
  58, 59, 52, 45, 38, 31, 39, 46,
  53, 60, 61, 54, 47, 55, 62, 63
};

void dejpeg_dezigzag_ref(int16_t* dst, const int16_t* src) {
  const uint8_t* table = dejpeg_dezigzag_table;

  for (unsigned int i = 0; i < 64; i++)
    dst[table[i]] = src[i];
}

// ============================================================================
// [IDCT - Ref]
// ============================================================================

#define JPEG_IDCT_IDCT_PASS(s0, s1, s2, s3, s4, s5, s6, s7) \
  int x0, x1, x2, x3;                          \
  int t0, t1, t2, t3;                          \
  int p1, p2, p3, p4, p5;                      \
                                               \
  p2 = s2;                                     \
  p3 = s6;                                     \
  p1 = (p2 + p3) * JPEG_IDCT_P_0_541196100;    \
  t2 = p3 * JPEG_IDCT_M_1_847759065 + p1;      \
  t3 = p2 * JPEG_IDCT_P_0_765366865 + p1;      \
                                               \
  p2 = s0;                                     \
  p3 = s4;                                     \
  t0 = JPEG_IDCT_SCALE(p2 + p3);               \
  t1 = JPEG_IDCT_SCALE(p2 - p3);               \
                                               \
  x0 = t0 + t3;                                \
  x3 = t0 - t3;                                \
  x1 = t1 + t2;                                \
  x2 = t1 - t2;                                \
                                               \
  t0 = s7;                                     \
  t1 = s5;                                     \
  t2 = s3;                                     \
  t3 = s1;                                     \
                                               \
  p3 = t0 + t2;                                \
  p4 = t1 + t3;                                \
  p1 = t0 + t3;                                \
  p2 = t1 + t2;                                \
  p5 = p3 + p4;                                \
                                               \
  p5 = p5 * JPEG_IDCT_P_1_175875602;           \
  t0 = t0 * JPEG_IDCT_P_0_298631336;           \
  t1 = t1 * JPEG_IDCT_P_2_053119869;           \
  t2 = t2 * JPEG_IDCT_P_3_072711026;           \
  t3 = t3 * JPEG_IDCT_P_1_501321110;           \
                                               \
  p1 = p1 * JPEG_IDCT_M_0_899976223 + p5;      \
  p2 = p2 * JPEG_IDCT_M_2_562915447 + p5;      \
  p3 = p3 * JPEG_IDCT_M_1_961570560;           \
  p4 = p4 * JPEG_IDCT_M_0_390180644;           \
                                               \
  t3 += p1 + p4;                               \
  t2 += p2 + p3;                               \
  t1 += p2 + p4;                               \
  t0 += p1 + p3;

void dejpeg_idct_islow_ref(uint8_t* dst, intptr_t dstStride, const int16_t* src, const uint16_t* qTable) {
  uint32_t i;
  int32_t* tmp;
  int32_t tmpData[64];

  for (i = 0, tmp = tmpData; i < 8; i++, src++, tmp++, qTable++) {
    // Avoid dequantizing and IDCTing zeros.
    if (src[8] == 0 && src[16] == 0 && src[24] == 0 && src[32] == 0 && src[40] == 0 && src[48] == 0 && src[56] == 0) {
      int32_t dcTerm = (static_cast<int32_t>(src[0]) * static_cast<int32_t>(qTable[0])) << (JPEG_IDCT_PREC - JPEG_IDCT_COL_NORM);
      tmp[0] = tmp[8] = tmp[16] = tmp[24] = tmp[32] = tmp[40] = tmp[48] = tmp[56] = dcTerm;
    }
    else {
      JPEG_IDCT_IDCT_PASS(
        static_cast<int32_t>(src[ 0]) * static_cast<int32_t>(qTable[ 0]),
        static_cast<int32_t>(src[ 8]) * static_cast<int32_t>(qTable[ 8]),
        static_cast<int32_t>(src[16]) * static_cast<int32_t>(qTable[16]),
        static_cast<int32_t>(src[24]) * static_cast<int32_t>(qTable[24]),
        static_cast<int32_t>(src[32]) * static_cast<int32_t>(qTable[32]),
        static_cast<int32_t>(src[40]) * static_cast<int32_t>(qTable[40]),
        static_cast<int32_t>(src[48]) * static_cast<int32_t>(qTable[48]),
        static_cast<int32_t>(src[56]) * static_cast<int32_t>(qTable[56]));

      x0 += JPEG_IDCT_COL_BIAS;
      x1 += JPEG_IDCT_COL_BIAS;
      x2 += JPEG_IDCT_COL_BIAS;
      x3 += JPEG_IDCT_COL_BIAS;

      tmp[ 0] = (x0 + t3) >> JPEG_IDCT_COL_NORM;
      tmp[56] = (x0 - t3) >> JPEG_IDCT_COL_NORM;
      tmp[ 8] = (x1 + t2) >> JPEG_IDCT_COL_NORM;
      tmp[48] = (x1 - t2) >> JPEG_IDCT_COL_NORM;
      tmp[16] = (x2 + t1) >> JPEG_IDCT_COL_NORM;
      tmp[40] = (x2 - t1) >> JPEG_IDCT_COL_NORM;
      tmp[24] = (x3 + t0) >> JPEG_IDCT_COL_NORM;
      tmp[32] = (x3 - t0) >> JPEG_IDCT_COL_NORM;
    }
  }

  for (i = 0, tmp = tmpData; i < 8; i++, dst += dstStride, tmp += 8) {
    JPEG_IDCT_IDCT_PASS(tmp[0], tmp[1], tmp[2], tmp[3], tmp[4], tmp[5], tmp[6], tmp[7])

    x0 += JPEG_IDCT_ROW_BIAS;
    x1 += JPEG_IDCT_ROW_BIAS;
    x2 += JPEG_IDCT_ROW_BIAS;
    x3 += JPEG_IDCT_ROW_BIAS;

    dst[0] = clampToByte((x0 + t3) >> JPEG_IDCT_ROW_NORM);
    dst[7] = clampToByte((x0 - t3) >> JPEG_IDCT_ROW_NORM);
    dst[1] = clampToByte((x1 + t2) >> JPEG_IDCT_ROW_NORM);
    dst[6] = clampToByte((x1 - t2) >> JPEG_IDCT_ROW_NORM);
    dst[2] = clampToByte((x2 + t1) >> JPEG_IDCT_ROW_NORM);
    dst[5] = clampToByte((x2 - t1) >> JPEG_IDCT_ROW_NORM);
    dst[3] = clampToByte((x3 + t0) >> JPEG_IDCT_ROW_NORM);
    dst[4] = clampToByte((x3 - t0) >> JPEG_IDCT_ROW_NORM);
  }
}

// ============================================================================
// [YCbCrToRGB32]
// ============================================================================

void dejpeg_ycbcr_to_rgb32_ref(uint8_t* dst, const uint8_t* pY, const uint8_t* pCb, const uint8_t* pCr, uint32_t count) {
  for (uint32_t i = 0; i < count; i++) {
    int yy = (static_cast<int>(pY[i]) << JPEG_YCBCR_PREC) + (1 << (JPEG_YCBCR_PREC - 1));
    int cr = static_cast<int>(pCr[i]) - 128;
    int cb = static_cast<int>(pCb[i]) - 128;

    int r = yy + cr * JPEG_YCBCR_FIXED(1.40200);
    int g = yy - cr * JPEG_YCBCR_FIXED(0.71414) - cb * JPEG_YCBCR_FIXED(0.34414);
    int b = yy + cb * JPEG_YCBCR_FIXED(1.77200);

    reinterpret_cast<uint32_t*>(dst)[0] = dejepeg_pack32(
      static_cast<uint8_t>(0xFF),
      clampToByte(r >> JPEG_YCBCR_PREC),
      clampToByte(g >> JPEG_YCBCR_PREC),
      clampToByte(b >> JPEG_YCBCR_PREC));
    dst += 4;
  }
}
