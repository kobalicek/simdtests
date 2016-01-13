// [SimdPixel]
// Playground for SIMD pixel manipulation.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSSE3

#include "../simdglobals.h"
#include "./pixops.h"

static inline uint32_t expand16(uint32_t x) { return x | (x << 16); }

void pixops_crossfade_ssse3(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha) {
  if ((alpha & 0x1) || alpha == 0 || alpha == 256)
    return pixops_crossfade_sse2(dst, dstStride, src, srcStride, w, h, alpha);

  uint8_t* pDstRow = static_cast<uint8_t*>(dst);
  const uint8_t* pSrcRow = static_cast<const uint8_t*>(src);

  alpha >>= 1;
  __m128i m = _mm_shuffle_epi32(_mm_cvtsi32_si128(expand16((128 - alpha) | ((alpha) << 8))), _MM_SHUFFLE(0, 0, 0, 0));

  for (uint32_t y = h; y > 0; y--, pDstRow += dstStride, pSrcRow += srcStride) {
    uint32_t* pDst = reinterpret_cast<uint32_t*>(pDstRow);
    const uint32_t* pSrc = reinterpret_cast<const uint32_t*>(pSrcRow);

    uint32_t x = w;
    for (;;) {
      while (x < 4 || !SimdUtils::isAligned(dst, 16)) {
        __m128i d = _mm_cvtsi32_si128(*pDst);
        __m128i s = _mm_cvtsi32_si128(*pSrc);

        d = _mm_unpacklo_epi8(d, s);
        d = _mm_maddubs_epi16(d, m);
        d = _mm_srli_epi16(d, 7);
        d = _mm_packus_epi16(d, d);
        *pDst = _mm_cvtsi128_si32(d);

        pDst++;
        pSrc++;
        x--;
      }

      if (x == 0)
        break;

      while (x >= 16) {
        __m128i d0 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst +  0));
        __m128i s0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc + 0));

        __m128i d2 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 4));
        __m128i s2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc + 4));

        __m128i d4 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst +  8));
        __m128i s4 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc +  8));

        __m128i d6 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 12));
        __m128i s6 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc + 12));

        __m128i d1 = _mm_unpackhi_epi8(d0, s0);
        __m128i d3 = _mm_unpackhi_epi8(d2, s2);

        __m128i d5 = _mm_unpackhi_epi8(d4, s4);
        __m128i d7 = _mm_unpackhi_epi8(d6, s6);

        d0 = _mm_unpacklo_epi8(d0, s0);
        d2 = _mm_unpacklo_epi8(d2, s2);
        d4 = _mm_unpacklo_epi8(d4, s4);
        d6 = _mm_unpacklo_epi8(d6, s6);

        d0 = _mm_maddubs_epi16(d0, m);
        d1 = _mm_maddubs_epi16(d1, m);
        d2 = _mm_maddubs_epi16(d2, m);
        d3 = _mm_maddubs_epi16(d3, m);
        d4 = _mm_maddubs_epi16(d4, m);
        d5 = _mm_maddubs_epi16(d5, m);
        d6 = _mm_maddubs_epi16(d6, m);
        d7 = _mm_maddubs_epi16(d7, m);

        d0 = _mm_srli_epi16(d0, 7);
        d1 = _mm_srli_epi16(d1, 7);
        d2 = _mm_srli_epi16(d2, 7);
        d3 = _mm_srli_epi16(d3, 7);
        d4 = _mm_srli_epi16(d4, 7);
        d5 = _mm_srli_epi16(d5, 7);
        d6 = _mm_srli_epi16(d6, 7);
        d7 = _mm_srli_epi16(d7, 7);

        d0 = _mm_packus_epi16(d0, d1);
        d2 = _mm_packus_epi16(d2, d3);
        d4 = _mm_packus_epi16(d4, d5);
        d6 = _mm_packus_epi16(d6, d7);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 0), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 4), d2);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst +  8), d4);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 12), d6);

        pDst += 16;
        pSrc += 16;
        x -= 16;
      }

      if (x >= 8) {
        __m128i d0 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 0));
        __m128i d2 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 4));
        __m128i s0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc + 0));
        __m128i s2 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc + 4));

        __m128i d1 = _mm_unpackhi_epi8(d0, s0);
        __m128i d3 = _mm_unpackhi_epi8(d2, s2);

        d0 = _mm_unpacklo_epi8(d0, s0);
        d2 = _mm_unpacklo_epi8(d2, s2);

        d0 = _mm_maddubs_epi16(d0, m);
        d1 = _mm_maddubs_epi16(d1, m);
        d2 = _mm_maddubs_epi16(d2, m);
        d3 = _mm_maddubs_epi16(d3, m);

        d0 = _mm_srli_epi16(d0, 7);
        d1 = _mm_srli_epi16(d1, 7);
        d2 = _mm_srli_epi16(d2, 7);
        d3 = _mm_srli_epi16(d3, 7);

        d0 = _mm_packus_epi16(d0, d1);
        d2 = _mm_packus_epi16(d2, d3);

        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 0), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 4), d2);

        pDst += 8;
        pSrc += 8;
        x -= 8;
      }

      if (x >= 4) {
        __m128i d0 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst));
        __m128i s0 = _mm_lddqu_si128(reinterpret_cast<const __m128i*>(pSrc));

        __m128i d1 = _mm_unpackhi_epi8(d0, s0);
        d0 = _mm_unpacklo_epi8(d0, s0);

        d0 = _mm_maddubs_epi16(d0, m);
        d1 = _mm_maddubs_epi16(d1, m);

        d0 = _mm_srli_epi16(d0, 7);
        d1 = _mm_srli_epi16(d1, 7);

        d0 = _mm_packus_epi16(d0, d1);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst), d0);

        pDst += 4;
        pSrc += 4;
        x -= 4;
      }

      if (x == 0)
        break;
    }
  }
}
