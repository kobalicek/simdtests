// [SimdPixel]
// Playground for SIMD pixel manipulation.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./pixops.h"

void pixops_crossfade_ref(void* dst, intptr_t dstStride, const void* src, intptr_t srcStride, uint32_t w, uint32_t h, uint32_t alpha) {
  uint8_t* pDstRow = static_cast<uint8_t*>(dst);
  const uint8_t* pSrcRow = static_cast<const uint8_t*>(src);

  uint32_t ia = 256 - alpha;

  for (uint32_t y = h; y > 0; y--, pDstRow += dstStride, pSrcRow += srcStride) {
    uint32_t* pDst = reinterpret_cast<uint32_t*>(pDstRow);
    const uint32_t* pSrc = reinterpret_cast<const uint32_t*>(pSrcRow);

    for (uint32_t x = w; x > 0; x--, pDst++, pSrc++) {
      uint32_t d = *pDst;
      uint32_t s = *pSrc;

      uint32_t dAG = ((d >> 8) & 0x00FF00FFU) * ia;
      uint32_t dRB = ((d     ) & 0x00FF00FFU) * ia;

      uint32_t sAG = ((s >> 8) & 0x00FF00FFU) * alpha;
      uint32_t sRB = ((s     ) & 0x00FF00FFU) * alpha;

      *pDst = (((dAG + sAG) & 0xFF00FF00U)) + (((dRB + sRB) & 0xFF00FF00U) >> 8);
    }
  }
}
