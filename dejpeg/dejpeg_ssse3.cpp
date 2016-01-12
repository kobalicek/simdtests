// [SimdTests - DeJPEG]
// SIMD optimized JPEG decoding utilities.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSSE3

#include "../simdglobals.h"
#include "./dejpeg.h"

// ============================================================================
// [DeZigZag - SSSE3]
// ============================================================================

#define Z 0x80

template<int A, int B, int C, int D, int E, int F, int G, int H, int I, int J, int K, int L, int M, int N, int O, int P>
SIMD_INLINE __m128i _mm_shuffle_epi8_ssse3(__m128i x) {
  SIMD_ALIGN_VAR(static const uint8_t, mask[16], 16) = { A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P };
  return _mm_shuffle_epi8(x, *reinterpret_cast<const __m128i*>(mask));
}

template<int A, int B, int C, int D, int E, int F, int G, int H>
SIMD_INLINE __m128i _mm_shuffle_epi16_ssse3(__m128i x) {
# define PROC(_x, _y) (_x >= 0x80 ? 0x80 : _x * 2 + _y)
  SIMD_ALIGN_VAR(static const uint8_t, mask[16], 16) = {
    PROC(A, 0), PROC(A, 1), PROC(B, 0), PROC(B, 1),
    PROC(C, 0), PROC(C, 1), PROC(D, 0), PROC(D, 1),
    PROC(E, 0), PROC(E, 1), PROC(F, 0), PROC(F, 1),
    PROC(G, 0), PROC(G, 1), PROC(H, 0), PROC(H, 1)
  };
# undef PROC

  return _mm_shuffle_epi8(x, *reinterpret_cast<const __m128i*>(mask));
}

// [00 | 01 | 05 | 06 | 14 | 15 | 27 | 28]
// [02 | 04 | 07 | 13 | 16 | 26 | 29 | 42]
// [03 | 08 | 12 | 17 | 25 | 30 | 41 | 43]
// [09 | 11 | 18 | 24 | 31 | 40 | 44 | 53]
// [10 | 19 | 23 | 32 | 39 | 45 | 52 | 54]
// [20 | 22 | 33 | 38 | 46 | 51 | 55 | 60]
// [21 | 34 | 37 | 47 | 50 | 56 | 59 | 61]
// [35 | 36 | 48 | 49 | 57 | 58 | 62 | 63]

void dejpeg_dezigzag_ssse3_v1(int16_t* dst, const int16_t* src) {
  // They are not used all at the same time, just to become more readable.
  __m128i x0, x1, x2, x3, x4, x5, x6, x7;
  __m128i y0, y1, y2, y3, y4, y5, y6, y7;
  __m128i t0, t1;

  // y0 <- [0:0 0:1 0:5 0:6 1:6 1:7 3:3 3:4]
  // y1 <- [0:2 0:4 0:7 1:5 2:0 3:2 3:5 5:2]
  // y2 <- [0:3 1:0 1:4 2:1 3:1 3:6 5:1 5:3]
  // y3 <- [1:1 1:3 2:2 3:0 3:7 5:0 5:4 6:5]
  // y4 <- [1:2 2:3 2:7 4:0 4:7 5:5 6:4 6:6]
  // y5 <- [2:4 2:6 4:1 4:6 5:6 6:3 6:7 7:4]
  // y6 <- [2:5 4:2 4:5 5:7 6:2 7:0 7:3 7:5]
  // y7 <- [4:3 4:4 6:0 6:1 7:1 7:2 7:6 7:7]

  x0 = _mm_load_si128((const __m128i *)(src +  0));         // [0:0 0:1 0:2 0:3 0:4 0:5 0:6 0:7]
  x1 = _mm_load_si128((const __m128i *)(src +  8));         // [1:0 1:1 1:2 1:3 1:4 1:5 1:6 1:7]
  x2 = _mm_load_si128((const __m128i *)(src + 16));         // [2:0 2:1 2:2 2:3 2:4 2:5 2:6 2:7]
  x3 = _mm_load_si128((const __m128i *)(src + 24));         // [3:0 3:1 3:2 3:3 3:4 3:5 3:6 3:7]

  y0 = _mm_shuffle_epi16_ssse3<0, 1, 5, 6, Z, Z, Z, Z>(x0); // [0:0 0:1 0:5 0:6 ___ ___ ___ ___]
  y1 = _mm_shuffle_epi16_ssse3<2, 4, 7, Z, Z, Z, Z, Z>(x0); // [0:2 0:4 0:7 ___ ___ ___ ___ ___]
  y2 = _mm_shuffle_epi16_ssse3<3, Z, Z, Z, Z, Z, Z, Z>(x0); // [0:3 ___ ___ ___ ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 6, 7, Z, Z>(x1); // [___ ___ ___ ___ 1:6 1:7 ___ ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, 5, Z, Z, Z, Z>(x1); // [___ ___ ___ 1:5 ___ ___ ___ ___]
  y0 = _mm_or_si128(y0, t0);                                // [0:0 0:1 0:5 0:6 1:6 1:7 ___ ___]
  y1 = _mm_or_si128(y1, t1);                                // [0:2 0:4 0:7 1:5 ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, 3, 4>(x3); // [___ ___ ___ ___ ___ ___ 3:3 3:4]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, 2, 5, Z>(x3); // [___ ___ ___ ___ ___ 3:2 3:5 ___]
  y0 = _mm_or_si128(y0, t0);                                // [0:0 0:1 0:5 0:6 1:6 1:7 3:3 3:4]
  y1 = _mm_or_si128(y1, t1);                                // [0:2 0:4 0:7 1:5 ___ 3:2 3:5 ___]
  _mm_store_si128((__m128i*)(dst +  0), y0);

  x5 = _mm_load_si128((const __m128i *)(src + 40));         // [5:0 5:1 5:2 5:3 5:4 5:5 5:6 5:7]
  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 0, Z, Z, Z>(x2); // [___ ___ ___ ___ 2:0 ___ ___ ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, 1, Z, Z, Z, Z>(x2); // [___ ___ ___ 2:1 ___ ___ ___ ___]
  y1 = _mm_or_si128(y1, t0);                                // [0:2 0:4 0:7 1:5 2:0 3:2 3:5 ___]
  y2 = _mm_or_si128(y2, t1);                                // [0:3 ___ ___ 2:1 ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, Z, 2>(x5); // [___ ___ ___ ___ ___ ___ ___ 5:2]
  t1 = _mm_shuffle_epi16_ssse3<Z, 0, 4, Z, Z, Z, Z, Z>(x1); // [___ 1:0 1:4 ___ ___ ___ ___ ___]
  y1 = _mm_or_si128(y1, t0);                                // [0:2 0:4 0:7 1:5 2:0 3:2 3:5 5:2]
  y2 = _mm_or_si128(y2, t1);                                // [0:3 1:0 1:4 2:1 ___ ___ ___ ___]
  _mm_store_si128((__m128i*)(dst +  8), y1);

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 1, 6, Z, Z>(x3); // [___ ___ ___ ___ 3:1 3:6 ___ ___]
  y3 = _mm_shuffle_epi16_ssse3<1, 3, Z, Z, Z, Z, Z, Z>(x1); // [1:1 1:3 ___ ___ ___ ___ ___ ___]
  y2 = _mm_or_si128(y2, t0);                                // [0:3 1:0 1:4 2:1 3:1 3:6 ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, 1, 3>(x5); // [___ ___ ___ ___ ___ ___ 5:1 5:3]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, 2, Z, Z, Z, Z, Z>(x2); // [___ ___ 2:2 ___ ___ ___ ___ ___]
  y2 = _mm_or_si128(y2, t0);                                // [0:3 1:0 1:4 2:1 3:1 3:6 5:1 5:3]
  y3 = _mm_or_si128(y3, t1);                                // [1:1 1:3 2:2 ___ ___ ___ ___ ___]
  _mm_store_si128((__m128i*)(dst + 16), y2);

  x4 = _mm_load_si128((const __m128i *)(src + 32));         // [4:0 4:1 4:2 4:3 4:4 4:5 4:6 4:7]
  y4 = _mm_shuffle_epi16_ssse3<2, Z, Z, Z, Z, Z, Z, Z>(x1); // [1:2 ___ ___ ___ ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, 0, 7, Z, Z, Z>(x3); // [___ ___ ___ 3:0 3:7 ___ ___ ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, 3, 7, Z, Z, Z, Z, Z>(x2); // [___ 2:3 2:7 ___ ___ ___ ___ ___]
  y3 = _mm_or_si128(y3, t0);                                // [1:1 1:3 2:2 3:0 3:7 ___ ___ ___]
  y4 = _mm_or_si128(y4, t1);                                // [1:2 2:3 2:7 ___ ___ ___ ___ ___]

  x6 = _mm_load_si128((const __m128i *)(src + 48));         // [6:0 6:1 6:2 6:3 6:4 6:5 6:6 6:7]
  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, 0, 4, Z>(x5); // [___ ___ ___ ___ ___ 5:0 5:4 ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, 0, 7, Z, Z, Z>(x4); // [___ ___ ___ 4:0 4:7 ___ ___ ___]
  y3 = _mm_or_si128(y3, t0);                                // [1:1 1:3 2:2 3:0 3:7 5:0 5:4 ___]
  y4 = _mm_or_si128(y4, t1);                                // [1:2 2:3 2:7 4:0 4:7 ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, Z, 5>(x6); // [___ ___ ___ ___ ___ ___ ___ 6:5]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, 5, Z, Z>(x5); // [___ ___ ___ ___ ___ 5:5 ___ ___]
  y3 = _mm_or_si128(y3, t0);                                // [1:1 1:3 2:2 3:0 3:7 5:0 5:4 6:5]
  y4 = _mm_or_si128(y4, t1);                                // [1:2 2:3 2:7 4:0 4:7 5:5 ___ ___]
  _mm_store_si128((__m128i*)(dst + 24), y3);

  y5 = _mm_shuffle_epi16_ssse3<4, 6, Z, Z, Z, Z, Z, Z>(x2); // [2:4 2:6 ___ ___ ___ ___ ___ ___]
  y6 = _mm_shuffle_epi16_ssse3<5, Z, Z, Z, Z, Z, Z, Z>(x2); // [2:5 ___ ___ ___ ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, 4, 6>(x6); // [___ ___ ___ ___ ___ ___ 6:4 6:6]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, 1, 6, Z, Z, Z, Z>(x4); // [___ ___ 4:1 4:6 ___ ___ ___ ___]

  y4 = _mm_or_si128(y4, t0);                                // [1:2 2:3 2:7 4:0 4:7 5:5 6:4 6:6]
  y5 = _mm_or_si128(y5, t1);                                // [2:4 2:6 4:1 4:6 ___ ___ ___ ___]
  _mm_store_si128((__m128i*)(dst + 32), y4);

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 6, Z, Z, Z>(x5); // [___ ___ ___ ___ 5:6 ___ ___ ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, 7, Z, Z, Z, Z>(x5); // [___ ___ ___ 5:7 ___ ___ ___ ___]
  y5 = _mm_or_si128(y5, t0);                                // [2:4 2:6 4:1 4:6 5:6 ___ ___ ___]
  y6 = _mm_or_si128(y6, t1);                                // [2:5 ___ ___ 5:7 ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, 3, 7, Z>(x6); // [___ ___ ___ ___ ___ 6:3 6:7 ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, 2, 5, Z, Z, Z, Z, Z>(x4); // [___ 4:2 4:5 ___ ___ ___ ___ ___]
  y5 = _mm_or_si128(y5, t0);                                // [2:4 2:6 4:1 4:6 5:6 6:3 6:7 ___]
  y6 = _mm_or_si128(y6, t1);                                // [2:5 4:2 4:5 5:7 ___ ___ ___ ___]

  x7 = _mm_load_si128((const __m128i *)(src + 56));         // [7:0 7:1 7:2 7:3 7:4 7:5 7:6 7:7]
  y7 = _mm_shuffle_epi16_ssse3<3, 4, Z, Z, Z, Z, Z, Z>(x4); // [4:3 4:4 ___ ___ ___ ___ ___ ___]

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, Z, Z, 4>(x7); // [___ ___ ___ ___ ___ ___ ___ 7:4]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 2, Z, Z, Z>(x6); // [___ ___ ___ ___ 6:2 ___ ___ ___]
  y5 = _mm_or_si128(y5, t0);                                // [2:4 2:6 4:1 4:6 5:6 6:3 6:7 7:4]
  y6 = _mm_or_si128(y6, t1);                                // [2:5 4:2 4:5 5:7 6:2 ___ ___ ___]
  _mm_store_si128((__m128i*)(dst + 40), y5);

  t0 = _mm_shuffle_epi16_ssse3<Z, Z, 0, 1, Z, Z, Z, Z>(x6); // [___ ___ 6:0 6:1 ___ ___ ___ ___]
  t1 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, Z, 0, 3, 5>(x7); // [___ ___ ___ ___ ___ 7:0 7:3 7:5]
  x7 = _mm_shuffle_epi16_ssse3<Z, Z, Z, Z, 1, 2, 6, 7>(x7); // [___ ___ ___ ___ 7:1 7:2 7:6 7:7]

  y7 = _mm_or_si128(y7, t0);                                // [4:3 4:4 6:0 6:1 ___ ___ ___ ___]
  y6 = _mm_or_si128(y6, t1);                                // [2:5 4:2 4:5 5:7 6:2 7:0 7:3 7:5]
  y7 = _mm_or_si128(y7, x7);                                // [4:3 4:4 6:0 6:1 7:1 7:2 7:6 7:7]

  _mm_store_si128((__m128i*)(dst + 48), y6);
  _mm_store_si128((__m128i*)(dst + 56), y7);
}

// t0 <- [0:00 0:01 0:05 0:06 0:14 0:15 1:11 1:12 0:02 0:04 0:07 0:13 1:00 1:10 1:13 2:10]
// t1 <- [0:03 0:08 0:12 1:01 1:09 1:14 2:09 2:11 0:09 0:11 1:02 1:08 1:15 2:08 2:12 3:05]
// t2 <- [0:10 1:03 1:07 2:00 2:07 2:13 3:04 3:06 1:04 1:06 2:01 2:06 2:14 3:03 3:07 3:12]
// t3 <- [1:05 2:02 2:05 2:15 3:02 3:08 3:11 3:13 2:03 2:04 3:00 3:01 3:09 3:10 3:14 3:15]
#define DEZIGZAG_SHUFFLE(x0, x1, x2, x3, t0, t1, t2, t3) \
  t0 = _mm_shuffle_epi8_ssse3<0 , 1 , 5 , 6 , 14, 15, Z , Z , 2 , 4 , 7 , 13, Z , Z , Z , Z >(x0); \
  t1 = _mm_shuffle_epi8_ssse3<3 , 8 , 12, Z , Z , Z , Z , Z , 9 , 11, Z , Z , Z , Z , Z , Z >(x0); \
  \
  t2 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , Z , Z , 11, 12, Z , Z , Z , Z , 0 , 10, 13, Z >(x1); \
  t3 = _mm_shuffle_epi8_ssse3<Z , Z , Z , 1 , 9 , 14, Z , Z , Z , Z , 2 , 8 , 15, Z , Z , Z >(x1); \
  \
  t0 = _mm_or_si128(t0, t2); \
  t1 = _mm_or_si128(t1, t3); \
  \
  t2 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , 10>(x2); \
  t3 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , Z , Z , 9 , 11, Z , Z , Z , Z , Z , 8 , 12, Z >(x2); \
  \
  t0 = _mm_or_si128(t0, t2); \
  t1 = _mm_or_si128(t1, t3); \
  \
  t2 = _mm_shuffle_epi8_ssse3<10, Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z >(x0); \
  t3 = _mm_shuffle_epi8_ssse3<5 , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z >(x1); \
  \
  x0 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , Z , 5 >(x3); \
  t1 = _mm_or_si128(t1, x0); \
  \
  x0 = _mm_shuffle_epi8_ssse3<Z , 2 , 5 , 15, Z , Z , Z , Z , 3 , 4 , Z , Z , Z , Z , Z , Z >(x2); \
  x1 = _mm_shuffle_epi8_ssse3<Z , 3 , 7 , Z , Z , Z , Z , Z , 4 , 6 , Z , Z , Z , Z , Z , Z >(x1); \
  \
  t3 = _mm_or_si128(t3, x0); \
  t2 = _mm_or_si128(t2, x1); \
  \
  x0 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , 2 , 8 , 11, 13, Z , Z , 0 , 1 , 9 , 10, 14, 15>(x3); \
  x2 = _mm_shuffle_epi8_ssse3<Z , Z , Z , 0 , 7 , 13, Z , Z , Z , Z , 1 , 6 , 14, Z , Z , Z >(x2); \
  x3 = _mm_shuffle_epi8_ssse3<Z , Z , Z , Z , Z , Z , 4 , 6 , Z , Z , Z , Z , Z , 3 , 7 , 12>(x3); \
  \
  t2 = _mm_or_si128(t2, x2); \
  t3 = _mm_or_si128(t3, x0); \
  t2 = _mm_or_si128(t2, x3);

SIMD_CONST_PI(zigzag_lo, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF, 0x00FF00FF);

// This version has less PSHUFB instructions, but more unpacking/packing, which
// makes it slower than v1.
void dejpeg_dezigzag_ssse3_v2(int16_t* dst, const int16_t* src) {
  __m128i x0 = _mm_load_si128((const __m128i *)(src +  0));
  __m128i x4 = _mm_load_si128((const __m128i *)(src +  8));
  __m128i x1 = _mm_load_si128((const __m128i *)(src + 16));
  __m128i x5 = _mm_load_si128((const __m128i *)(src + 24));
  __m128i x2 = _mm_load_si128((const __m128i *)(src + 32));
  __m128i x6 = _mm_load_si128((const __m128i *)(src + 40));
  __m128i x3 = _mm_load_si128((const __m128i *)(src + 48));
  __m128i x7 = _mm_load_si128((const __m128i *)(src + 56));

  __m128i y0 = _mm_packus_epi16(_mm_and_si128(x0, SIMD_GET_PI(zigzag_lo)), _mm_and_si128(x4, SIMD_GET_PI(zigzag_lo)));
  __m128i y1 = _mm_packus_epi16(_mm_and_si128(x1, SIMD_GET_PI(zigzag_lo)), _mm_and_si128(x5, SIMD_GET_PI(zigzag_lo)));
  __m128i y2 = _mm_packus_epi16(_mm_and_si128(x2, SIMD_GET_PI(zigzag_lo)), _mm_and_si128(x6, SIMD_GET_PI(zigzag_lo)));
  __m128i y3 = _mm_packus_epi16(_mm_and_si128(x3, SIMD_GET_PI(zigzag_lo)), _mm_and_si128(x7, SIMD_GET_PI(zigzag_lo)));

  __m128i t0, t1, t2, t3;
  DEZIGZAG_SHUFFLE(y0, y1, y2, y3, t0, t1, t2, t3)

  __m128i y4 = _mm_packus_epi16(_mm_srli_epi16(x0, 8), _mm_srli_epi16(x4, 8));
  __m128i y5 = _mm_packus_epi16(_mm_srli_epi16(x1, 8), _mm_srli_epi16(x5, 8));
  __m128i y6 = _mm_packus_epi16(_mm_srli_epi16(x2, 8), _mm_srli_epi16(x6, 8));
  __m128i y7 = _mm_packus_epi16(_mm_srli_epi16(x3, 8), _mm_srli_epi16(x7, 8));

  __m128i t4, t5, t6, t7;
  DEZIGZAG_SHUFFLE(y4, y5, y6, y7, t4, t5, t6, t7)

  __m128i z0 = _mm_unpacklo_epi8(t0, t4);
  __m128i z4 = _mm_unpackhi_epi8(t0, t4);
  _mm_store_si128((__m128i*)(dst +  0), z0);
  _mm_store_si128((__m128i*)(dst +  8), z4);

  __m128i z1 = _mm_unpacklo_epi8(t1, t5);
  __m128i z5 = _mm_unpackhi_epi8(t1, t5);
  _mm_store_si128((__m128i*)(dst + 16), z1);
  _mm_store_si128((__m128i*)(dst + 24), z5);

  __m128i z2 = _mm_unpacklo_epi8(t2, t6);
  __m128i z6 = _mm_unpackhi_epi8(t2, t6);
  _mm_store_si128((__m128i*)(dst + 32), z2);
  _mm_store_si128((__m128i*)(dst + 40), z6);

  __m128i z3 = _mm_unpacklo_epi8(t3, t7);
  __m128i z7 = _mm_unpackhi_epi8(t3, t7);
  _mm_store_si128((__m128i*)(dst + 48), z3);
  _mm_store_si128((__m128i*)(dst + 56), z7);
}
