// [SimdTests - DeJPEG]
// SIMD optimized JPEG decoding utilities.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSE2

#include "../simdglobals.h"
#include "./dejpeg.h"

// ============================================================================
// [IDCT - SSE2]
// ============================================================================

struct DeJPEG_SSE2Consts {
  // IDCT.
  int16_t rot0_0[8], rot0_1[8];
  int16_t rot1_0[8], rot1_1[8];
  int16_t rot2_0[8], rot2_1[8];
  int16_t rot3_0[8], rot3_1[8];

  int32_t colBias[4];
  int32_t rowBias[4];

  // YCbCr.
  int32_t ycbcr_allones[4];
  int16_t ycbcr_tosigned[8];
  int32_t ycbcr_round[4];
  int16_t ycbcr_yycrMul[8];
  int16_t ycbcr_yycbMul[8];
  int16_t ycbcr_cbcrMul[8];
};

#define DATA_4X(...) { __VA_ARGS__, __VA_ARGS__, __VA_ARGS__, __VA_ARGS__ }
SIMD_ALIGN_VAR(static const DeJPEG_SSE2Consts, dejpeg_sse2_consts, 16) = {
  DATA_4X(JPEG_IDCT_P_0_541196100                          , JPEG_IDCT_P_0_541196100 + JPEG_IDCT_M_1_847759065),
  DATA_4X(JPEG_IDCT_P_0_541196100 + JPEG_IDCT_P_0_765366865, JPEG_IDCT_P_0_541196100                          ),
  DATA_4X(JPEG_IDCT_P_1_175875602 + JPEG_IDCT_M_0_899976223, JPEG_IDCT_P_1_175875602                          ),
  DATA_4X(JPEG_IDCT_P_1_175875602                          , JPEG_IDCT_P_1_175875602 + JPEG_IDCT_M_2_562915447),
  DATA_4X(JPEG_IDCT_M_1_961570560 + JPEG_IDCT_P_0_298631336, JPEG_IDCT_M_1_961570560                          ),
  DATA_4X(JPEG_IDCT_M_1_961570560                          , JPEG_IDCT_M_1_961570560 + JPEG_IDCT_P_3_072711026),
  DATA_4X(JPEG_IDCT_M_0_390180644 + JPEG_IDCT_P_2_053119869, JPEG_IDCT_M_0_390180644                          ),
  DATA_4X(JPEG_IDCT_M_0_390180644                          , JPEG_IDCT_M_0_390180644 + JPEG_IDCT_P_1_501321110),
  DATA_4X(JPEG_IDCT_COL_BIAS),
  DATA_4X(JPEG_IDCT_ROW_BIAS),

  DATA_4X(0xFFFFFFFF),
  DATA_4X(-128, -128),
  DATA_4X(1 << (JPEG_YCBCR_PREC - 1)),
  DATA_4X( JPEG_YCBCR_FIXED(1.00000),  JPEG_YCBCR_FIXED(1.40200)),
  DATA_4X( JPEG_YCBCR_FIXED(1.00000),  JPEG_YCBCR_FIXED(1.77200)),
  DATA_4X(-JPEG_YCBCR_FIXED(0.34414), -JPEG_YCBCR_FIXED(0.71414))
};
#undef DATA_4X

#define JPEG_CONST_XMM(x) (*(const __m128i*)(dejpeg_sse2_consts.x))

#define JPEG_IDCT_INTERLEAVE8_XMM(a, b) { __m128i t = a; a = _mm_unpacklo_epi8(a, b); b = _mm_unpackhi_epi8(t, b); }
#define JPEG_IDCT_INTERLEAVE16_XMM(a, b) { __m128i t = a; a = _mm_unpacklo_epi16(a, b); b = _mm_unpackhi_epi16(t, b); }

// out(0) = c0[even]*x + c0[odd]*y   (c0, x, y 16-bit, out 32-bit)
// out(1) = c1[even]*x + c1[odd]*y
#define JPEG_IDCT_ROTATE_XMM(dst0, dst1, x, y, c0, c1) \
  __m128i c0##_l = _mm_unpacklo_epi16(x, y); \
  __m128i c0##_h = _mm_unpackhi_epi16(x, y); \
  \
  __m128i dst0##_l = _mm_madd_epi16(c0##_l, JPEG_CONST_XMM(c0)); \
  __m128i dst0##_h = _mm_madd_epi16(c0##_h, JPEG_CONST_XMM(c0)); \
  __m128i dst1##_l = _mm_madd_epi16(c0##_l, JPEG_CONST_XMM(c1)); \
  __m128i dst1##_h = _mm_madd_epi16(c0##_h, JPEG_CONST_XMM(c1));

// out = in << 12  (in 16-bit, out 32-bit)
#define JPEG_IDCT_WIDEN_XMM(dst, in) \
  __m128i dst##_l = _mm_srai_epi32(_mm_unpacklo_epi16(_mm_setzero_si128(), (in)), 4); \
  __m128i dst##_h = _mm_srai_epi32(_mm_unpackhi_epi16(_mm_setzero_si128(), (in)), 4);

// wide add
#define JPEG_IDCT_WADD_XMM(dst, a, b) \
  __m128i dst##_l = _mm_add_epi32(a##_l, b##_l); \
  __m128i dst##_h = _mm_add_epi32(a##_h, b##_h);

// wide sub
#define JPEG_IDCT_WSUB_XMM(dst, a, b) \
  __m128i dst##_l = _mm_sub_epi32(a##_l, b##_l); \
  __m128i dst##_h = _mm_sub_epi32(a##_h, b##_h);

// butterfly a/b, add bias, then shift by `norm` and pack to 16-bit.
#define JPEG_IDCT_BFLY_XMM(dst0, dst1, a, b, bias, norm) { \
  __m128i abiased_l = _mm_add_epi32(a##_l, bias); \
  __m128i abiased_h = _mm_add_epi32(a##_h, bias); \
  \
  JPEG_IDCT_WADD_XMM(sum, abiased, b) \
  JPEG_IDCT_WSUB_XMM(diff, abiased, b) \
  \
  dst0 = _mm_packs_epi32(_mm_srai_epi32(sum_l, norm), _mm_srai_epi32(sum_h, norm)); \
  dst1 = _mm_packs_epi32(_mm_srai_epi32(diff_l, norm), _mm_srai_epi32(diff_h, norm)); \
}

#define JPEG_IDCT_IDCT_PASS_XMM(bias, norm) { \
  /* Even part. */ \
  JPEG_IDCT_ROTATE_XMM(t2e, t3e, row2, row6, rot0_0, rot0_1) \
  \
  __m128i sum04 = _mm_add_epi16(row0, row4); \
  __m128i dif04 = _mm_sub_epi16(row0, row4); \
  \
  JPEG_IDCT_WIDEN_XMM(t0e, sum04) \
  JPEG_IDCT_WIDEN_XMM(t1e, dif04) \
  \
  JPEG_IDCT_WADD_XMM(x0, t0e, t3e) \
  JPEG_IDCT_WSUB_XMM(x3, t0e, t3e) \
  JPEG_IDCT_WADD_XMM(x1, t1e, t2e) \
  JPEG_IDCT_WSUB_XMM(x2, t1e, t2e) \
  \
  /* Odd part. */ \
  JPEG_IDCT_ROTATE_XMM(y0o, y2o, row7, row3, rot2_0, rot2_1) \
  JPEG_IDCT_ROTATE_XMM(y1o, y3o, row5, row1, rot3_0, rot3_1) \
  __m128i sum17 = _mm_add_epi16(row1, row7); \
  __m128i sum35 = _mm_add_epi16(row3, row5); \
  JPEG_IDCT_ROTATE_XMM(y4o,y5o, sum17, sum35, rot1_0, rot1_1) \
  \
  JPEG_IDCT_WADD_XMM(x4, y0o, y4o) \
  JPEG_IDCT_WADD_XMM(x5, y1o, y5o) \
  JPEG_IDCT_WADD_XMM(x6, y2o, y5o) \
  JPEG_IDCT_WADD_XMM(x7, y3o, y4o) \
  \
  JPEG_IDCT_BFLY_XMM(row0, row7, x0, x7, bias, norm) \
  JPEG_IDCT_BFLY_XMM(row1, row6, x1, x6, bias, norm) \
  JPEG_IDCT_BFLY_XMM(row2, row5, x2, x5, bias, norm) \
  JPEG_IDCT_BFLY_XMM(row3, row4, x3, x4, bias, norm) \
}

void dejpeg_idct_islow_sse2(uint8_t* dst, intptr_t dstStride, const int16_t* src, const uint16_t* qTable) {
  // Load and dequantize.
  __m128i row0 = _mm_mullo_epi16(*(const __m128i *)(src +  0), *(const __m128i *)(qTable +  0));
  __m128i row1 = _mm_mullo_epi16(*(const __m128i *)(src +  8), *(const __m128i *)(qTable +  8));
  __m128i row2 = _mm_mullo_epi16(*(const __m128i *)(src + 16), *(const __m128i *)(qTable + 16));
  __m128i row3 = _mm_mullo_epi16(*(const __m128i *)(src + 24), *(const __m128i *)(qTable + 24));
  __m128i row4 = _mm_mullo_epi16(*(const __m128i *)(src + 32), *(const __m128i *)(qTable + 32));
  __m128i row5 = _mm_mullo_epi16(*(const __m128i *)(src + 40), *(const __m128i *)(qTable + 40));
  __m128i row6 = _mm_mullo_epi16(*(const __m128i *)(src + 48), *(const __m128i *)(qTable + 48));
  __m128i row7 = _mm_mullo_epi16(*(const __m128i *)(src + 56), *(const __m128i *)(qTable + 56));

  // IDCT columns.
  JPEG_IDCT_IDCT_PASS_XMM(JPEG_CONST_XMM(colBias), 10)

  // Transpose.
  JPEG_IDCT_INTERLEAVE16_XMM(row0, row4) // [a0a4|b0b4|c0c4|d0d4] | [e0e4|f0f4|g0g4|h0h4]
  JPEG_IDCT_INTERLEAVE16_XMM(row2, row6) // [a2a6|b2b6|c2c6|d2d6] | [e2e6|f2f6|g2g6|h2h6]
  JPEG_IDCT_INTERLEAVE16_XMM(row1, row5) // [a1a5|b1b5|c2c5|d1d5] | [e1e5|f1f5|g1g5|h1h5]
  JPEG_IDCT_INTERLEAVE16_XMM(row3, row7) // [a3a7|b3b7|c3c7|d3d7] | [e3e7|f3f7|g3g7|h3h7]

  JPEG_IDCT_INTERLEAVE16_XMM(row0, row2) // [a0a2|a4a6|b0b2|b4b6] | [c0c2|c4c6|d0d2|d4d6]
  JPEG_IDCT_INTERLEAVE16_XMM(row1, row3) // [a1a3|a5a7|b1b3|b5b7] | [c1c3|c5c7|d1d3|d5d7]
  JPEG_IDCT_INTERLEAVE16_XMM(row4, row6) // [e0e2|e4e6|f0f2|f4f6] | [g0g2|g4g6|h0h2|h4h6]
  JPEG_IDCT_INTERLEAVE16_XMM(row5, row7) // [e1e3|e5e7|f1f3|f5f7] | [g1g3|g5g7|h1h3|h5h7]

  JPEG_IDCT_INTERLEAVE16_XMM(row0, row1) // [a0a1|a2a3|a4a5|a6a7] | [b0b1|b2b3|b4b5|b6b7]
  JPEG_IDCT_INTERLEAVE16_XMM(row2, row3) // [c0c1|c2c3|c4c5|c6c7] | [d0d1|d2d3|d4d5|d6d7]
  JPEG_IDCT_INTERLEAVE16_XMM(row4, row5) // [e0e1|e2e3|e4e5|e6e7] | [f0f1|f2f3|f4f5|f6f7]
  JPEG_IDCT_INTERLEAVE16_XMM(row6, row7) // [g0g1|g2g3|g4g5|g6g7] | [h0h1|h2h3|h4h5|h6h7]

  // IDCT rows.
  JPEG_IDCT_IDCT_PASS_XMM(JPEG_CONST_XMM(rowBias), 17)

  // Pack to 8-bit integers, also saturates the result to 0..255.
  row0 = _mm_packus_epi16(row0, row1);   // [a0a1a2a3|a4a5a6a7|b0b1b2b3|b4b5b6b7]
  row2 = _mm_packus_epi16(row2, row3);   // [c0c1c2c3|c4c5c6c7|d0d1d2d3|d4d5d6d7]
  row4 = _mm_packus_epi16(row4, row5);   // [e0e1e2e3|e4e5e6e7|f0f1f2f3|f4f5f6f7]
  row6 = _mm_packus_epi16(row6, row7);   // [g0g1g2g3|g4g5g6g7|h0h1h2h3|h4h5h6h7]

  // Transpose.
  JPEG_IDCT_INTERLEAVE8_XMM(row0, row4); // [a0e0a1e1|a2e2a3e3|a4e4a5e5|a6e6a7e7] | [b0f0b1f1|b2f2b3f3|b4f4b5f5|b6f6b7f7]
  JPEG_IDCT_INTERLEAVE8_XMM(row2, row6); // [c0g0c1g1|c2g2c3g3|c4g4c5g5|c6g6c7g7] | [d0h0d1h1|d2h2d3h3|d4h4d5h5|d6h6d7h7]
  JPEG_IDCT_INTERLEAVE8_XMM(row0, row2); // [a0c0e0g0|a1c1e1g1|a2c2e2g2|a3c3e3g3] | [a4c4e4g4|a5c5e5g5|a6c6e6g6|a7c7e7g7]
  JPEG_IDCT_INTERLEAVE8_XMM(row4, row6); // [b0d0f0h0|b1d1f1h1|b2d2f2h2|b3d3f3h3| | [b4d4f4h4|b5d5f5h5|b6d6f6h6|b7d7f7h7]
  JPEG_IDCT_INTERLEAVE8_XMM(row0, row4); // [a0b0c0d0|e0f0g0h0|a1b1c1d1|e1f1g1h1] | [a2b2c2d2|e2f2g2h2|a3b3c3d3|e3f3g3h3]
  JPEG_IDCT_INTERLEAVE8_XMM(row2, row6); // [a4b4c4d4|e4f4g4h4|a5b5c5d5|e5f5g5h5] | [a6b6c6d6|e6f6g6h6|a7b7c7d7|e7f7g7h7]

  // Store.
  uint8_t* dst0 = dst;
  uint8_t* dst1 = dst + dstStride;
  intptr_t dstStride2 = dstStride * 2;

  _mm_storel_pi((__m64 *)dst0, _mm_castsi128_ps(row0)); dst0 += dstStride2;
  _mm_storeh_pi((__m64 *)dst1, _mm_castsi128_ps(row0)); dst1 += dstStride2;

  _mm_storel_pi((__m64 *)dst0, _mm_castsi128_ps(row4)); dst0 += dstStride2;
  _mm_storeh_pi((__m64 *)dst1, _mm_castsi128_ps(row4)); dst1 += dstStride2;

  _mm_storel_pi((__m64 *)dst0, _mm_castsi128_ps(row2)); dst0 += dstStride2;
  _mm_storeh_pi((__m64 *)dst1, _mm_castsi128_ps(row2)); dst1 += dstStride2;

  _mm_storel_pi((__m64 *)dst0, _mm_castsi128_ps(row6));
  _mm_storeh_pi((__m64 *)dst1, _mm_castsi128_ps(row6));
}

// ============================================================================
// [YCbCrToRGB32]
// ============================================================================

void dejpeg_ycbcr_to_rgb32_sse2(uint8_t* dst, const uint8_t* pY, const uint8_t* pCb, const uint8_t* pCr, uint32_t count) {
  uint32_t i = count;
  __m128i zero = _mm_setzero_si128();

  while (i >= 8) {
    __m128i yy = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pY));
    __m128i cb = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pCb));
    __m128i cr = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(pCr));

    yy = _mm_unpacklo_epi8(yy, zero);
    cb = _mm_add_epi16(_mm_unpacklo_epi8(cb, zero), JPEG_CONST_XMM(ycbcr_tosigned));
    cr = _mm_add_epi16(_mm_unpacklo_epi8(cr, zero), JPEG_CONST_XMM(ycbcr_tosigned));

    __m128i r_l = _mm_madd_epi16(_mm_unpacklo_epi16(yy, cr), JPEG_CONST_XMM(ycbcr_yycrMul));
    __m128i r_h = _mm_madd_epi16(_mm_unpackhi_epi16(yy, cr), JPEG_CONST_XMM(ycbcr_yycrMul));

    __m128i b_l = _mm_madd_epi16(_mm_unpacklo_epi16(yy, cb), JPEG_CONST_XMM(ycbcr_yycbMul));
    __m128i b_h = _mm_madd_epi16(_mm_unpackhi_epi16(yy, cb), JPEG_CONST_XMM(ycbcr_yycbMul));

    __m128i g_l = _mm_madd_epi16(_mm_unpacklo_epi16(cb, cr), JPEG_CONST_XMM(ycbcr_cbcrMul));
    __m128i g_h = _mm_madd_epi16(_mm_unpackhi_epi16(cb, cr), JPEG_CONST_XMM(ycbcr_cbcrMul));

    g_l = _mm_add_epi32(g_l, _mm_slli_epi32(_mm_unpacklo_epi16(yy, zero), JPEG_YCBCR_PREC));
    g_h = _mm_add_epi32(g_h, _mm_slli_epi32(_mm_unpackhi_epi16(yy, zero), JPEG_YCBCR_PREC));

    r_l = _mm_add_epi32(r_l, JPEG_CONST_XMM(ycbcr_round));
    r_h = _mm_add_epi32(r_h, JPEG_CONST_XMM(ycbcr_round));

    b_l = _mm_add_epi32(b_l, JPEG_CONST_XMM(ycbcr_round));
    b_h = _mm_add_epi32(b_h, JPEG_CONST_XMM(ycbcr_round));

    g_l = _mm_add_epi32(g_l, JPEG_CONST_XMM(ycbcr_round));
    g_h = _mm_add_epi32(g_h, JPEG_CONST_XMM(ycbcr_round));

    r_l = _mm_srai_epi32(r_l, JPEG_YCBCR_PREC);
    r_h = _mm_srai_epi32(r_h, JPEG_YCBCR_PREC);

    b_l = _mm_srai_epi32(b_l, JPEG_YCBCR_PREC);
    b_h = _mm_srai_epi32(b_h, JPEG_YCBCR_PREC);

    g_l = _mm_srai_epi32(g_l, JPEG_YCBCR_PREC);
    g_h = _mm_srai_epi32(g_h, JPEG_YCBCR_PREC);

    __m128i r = _mm_packs_epi32(r_l, r_h);
    __m128i g = _mm_packs_epi32(g_l, g_h);
    __m128i b = _mm_packs_epi32(b_l, b_h);

    r = _mm_packus_epi16(r, r);
    g = _mm_packus_epi16(g, g);
    b = _mm_packus_epi16(b, b);

    __m128i ra = _mm_unpacklo_epi8(r, JPEG_CONST_XMM(ycbcr_allones));
    __m128i bg = _mm_unpacklo_epi8(b, g);

    __m128i bgra0 = _mm_unpacklo_epi16(bg, ra);
    __m128i bgra1 = _mm_unpackhi_epi16(bg, ra);

    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst +  0), bgra0);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(dst + 16), bgra1);

    dst += 32;
    pY  += 8;
    pCb += 8;
    pCr += 8;
    i   -= 8;
  }

  while (i) {
    int yy = (static_cast<int>(pY[0]) << JPEG_YCBCR_PREC) + (1 << (JPEG_YCBCR_PREC - 1));
    int cr = static_cast<int>(pCr[0]) - 128;
    int cb = static_cast<int>(pCb[0]) - 128;

    int r = yy + cr * JPEG_YCBCR_FIXED(1.40200);
    int g = yy - cr * JPEG_YCBCR_FIXED(0.71414) - cb * JPEG_YCBCR_FIXED(0.34414);
    int b = yy + cb * JPEG_YCBCR_FIXED(1.77200);

    reinterpret_cast<uint32_t*>(dst)[0] = dejepeg_pack32(
      static_cast<uint8_t>(0xFF),
      clampToByte(r >> JPEG_YCBCR_PREC),
      clampToByte(g >> JPEG_YCBCR_PREC),
      clampToByte(b >> JPEG_YCBCR_PREC));

    dst += 4;
    pY  += 1;
    pCb += 1;
    pCr += 1;
    i   -= 1;
  }
}
