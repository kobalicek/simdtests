// [SimdPixel]
// Playground for SIMD pixel manipulation.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSE2

#include "../simdglobals.h"
#include "./pixops.h"

static inline uint32_t expand16(uint32_t x) { return x | (x << 16); }

void pixops_crossfade_sse2(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha) {
  uint8_t* pDstRow = static_cast<uint8_t*>(dst);
  const uint8_t* pSrcRow = static_cast<const uint8_t*>(src);

  __m128i a  = _mm_shuffle_epi32(_mm_cvtsi32_si128(expand16(alpha      )), _MM_SHUFFLE(0, 0, 0, 0));
  __m128i ia = _mm_shuffle_epi32(_mm_cvtsi32_si128(expand16(256 - alpha)), _MM_SHUFFLE(0, 0, 0, 0));

  for (uint32_t y = h; y > 0; y--, pDstRow += dstStride, pSrcRow += srcStride) {
    uint32_t* pDst = reinterpret_cast<uint32_t*>(pDstRow);
    const uint32_t* pSrc = reinterpret_cast<const uint32_t*>(pSrcRow);

    uint32_t x = w;
    for (;;) {
      while (x < 4 || !SimdUtils::isAligned(dst, 16)) {
        __m128i d = _mm_cvtsi32_si128(*pDst);
        __m128i s = _mm_cvtsi32_si128(*pSrc);

        d = _mm_unpacklo_epi8(d, _mm_setzero_si128());
        s = _mm_unpacklo_epi8(s, _mm_setzero_si128());

        d = _mm_mullo_epi16(d, ia);
        s = _mm_mullo_epi16(s, a);

        d = _mm_add_epi16(d, s);
        d = _mm_srli_epi16(d, 8);
        d = _mm_packus_epi16(d, d);
        *pDst = _mm_cvtsi128_si32(d);

        pDst++;
        pSrc++;
        x--;
      }

      if (x == 0)
        break;

      while (x >= 8) {
        __m128i d0 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 0));
        __m128i d2 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst + 4));
        __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrc + 0));
        __m128i s2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrc + 4));

        __m128i d1 = _mm_unpackhi_epi8(d0, _mm_setzero_si128());
        __m128i d3 = _mm_unpackhi_epi8(d2, _mm_setzero_si128());
        __m128i s1 = _mm_unpackhi_epi8(s0, _mm_setzero_si128());
        __m128i s3 = _mm_unpackhi_epi8(s2, _mm_setzero_si128());

        d0 = _mm_unpacklo_epi8(d0, _mm_setzero_si128());
        d2 = _mm_unpacklo_epi8(d2, _mm_setzero_si128());
        s0 = _mm_unpacklo_epi8(s0, _mm_setzero_si128());
        s2 = _mm_unpacklo_epi8(s2, _mm_setzero_si128());

        d0 = _mm_mullo_epi16(d0, ia);
        d1 = _mm_mullo_epi16(d1, ia);
        d2 = _mm_mullo_epi16(d2, ia);
        d3 = _mm_mullo_epi16(d3, ia);

        s0 = _mm_mullo_epi16(s0, a);
        s1 = _mm_mullo_epi16(s1, a);
        s2 = _mm_mullo_epi16(s2, a);
        s3 = _mm_mullo_epi16(s3, a);

        d0 = _mm_add_epi16(d0, s0);
        d1 = _mm_add_epi16(d1, s1);
        d2 = _mm_add_epi16(d2, s2);
        d3 = _mm_add_epi16(d3, s3);

        d0 = _mm_srli_epi16(d0, 8);
        d1 = _mm_srli_epi16(d1, 8);
        d2 = _mm_srli_epi16(d2, 8);
        d3 = _mm_srli_epi16(d3, 8);

        d0 = _mm_packus_epi16(d0, d1);
        d2 = _mm_packus_epi16(d2, d3);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 0), d0);
        _mm_store_si128(reinterpret_cast<__m128i*>(pDst + 4), d2);

        pDst += 8;
        pSrc += 8;
        x -= 8;
      }

      while (x >= 4) {
        __m128i d0 = _mm_load_si128(reinterpret_cast<__m128i*>(pDst));
        __m128i s0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(pSrc));

        __m128i d1 = _mm_unpackhi_epi8(d0, _mm_setzero_si128());
        __m128i s1 = _mm_unpackhi_epi8(s0, _mm_setzero_si128());

        d0 = _mm_unpacklo_epi8(d0, _mm_setzero_si128());
        s0 = _mm_unpacklo_epi8(s0, _mm_setzero_si128());

        d0 = _mm_mullo_epi16(d0, ia);
        d1 = _mm_mullo_epi16(d1, ia);
        s0 = _mm_mullo_epi16(s0, a);
        s1 = _mm_mullo_epi16(s1, a);

        d0 = _mm_add_epi16(d0, s0);
        d1 = _mm_add_epi16(d1, s1);

        d0 = _mm_srli_epi16(d0, 8);
        d1 = _mm_srli_epi16(d1, 8);

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

