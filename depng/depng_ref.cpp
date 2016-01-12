// [SimdTests - DePNG]
// SIMD optimized "PNG Reverse Filter" implementation.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./depng.h"

// ============================================================================
// [SimdTests::DePNG - Filter - Ref]
// ============================================================================

void depng_filter_ref(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl) {
  uint32_t y = h;
  uint8_t* u = NULL;

  // Subtract one BYTE that is used to store the `filter` ID.
  bpl--;

  do {
    uint32_t i;
    uint32_t filter = *p++;

    switch (filter) {
      case kPngFilterNone:
        p += bpl;
        break;

      case kPngFilterSub: {
        for (i = bpl - bpp; i != 0; i--, p++)
          p[bpp] = depng_sum(p[bpp], p[0]);

        p += bpp;
        break;
      }

      case kPngFilterUp: {
        for (i = bpl; i != 0; i--, p++, u++)
          p[0] = depng_sum(p[0], u[0]);
        break;
      }

      case kPngFilterAvg: {
        for (i = 0; i < bpp; i++)
          p[i] = depng_sum(p[i], u[i] >> 1);

        u += bpp;
        for (i = bpl - bpp; i != 0; i--, p++, u++)
          p[bpp] = depng_sum(p[bpp], depng_avg(p[0], u[0]));

        p += bpp;
        break;
      }

      case kPngFilterPaeth: {
        for (i = 0; i < bpp; i++)
          p[i] = depng_sum(p[i], u[i]);

        for (i = bpl - bpp; i != 0; i--, p++, u++)
          p[bpp] = depng_sum(p[bpp], depng_paeth_ref(p[0], u[bpp], u[0]));

        p += bpp;
        break;
      }
    }

    u = p - bpl;
  } while (--y != 0);
}

// ============================================================================
// [SimdTests::DePNG - Filter - Opt]
// ============================================================================

// This is a template-specialized implementation that takes an advantage of
// `bpp` being constant, so the C++ compiler has more information for making
// certain optimizations not possible in reference implementation.
template<uint32_t bpp>
static SIMD_INLINE void depng_filter_opt_template(uint8_t* p, uint32_t h, uint32_t bpl) {
  uint32_t y = h;
  uint8_t* u = NULL;

  // Subtract one BYTE that is used to store the `filter` ID.
  bpl--;

  do {
    uint32_t i;
    uint32_t filter = *p++;

    switch (filter) {
      case kPngFilterNone:
        p += bpl;
        break;

      case kPngFilterSub: {
        for (i = bpl - bpp; i != 0; i--, p++)
          p[bpp] = depng_sum(p[bpp], p[0]);

        p += bpp;
        break;
      }

      case kPngFilterUp: {
        for (i = bpl; i != 0; i--, p++, u++)
          p[0] = depng_sum(p[0], u[0]);
        break;
      }

      case kPngFilterAvg: {
        for (i = 0; i < bpp; i++)
          p[i] = depng_sum(p[i], u[i] >> 1);

        u += bpp;
        for (i = bpl - bpp; i != 0; i--, p++, u++)
          p[bpp] = depng_sum(p[bpp], depng_avg(p[0], u[0]));

        p += bpp;
        break;
      }

      case kPngFilterPaeth: {
        for (i = 0; i < bpp; i++)
          p[i] = depng_sum(p[i], u[i]);

        for (i = bpl - bpp; i != 0; i--, p++, u++)
          p[bpp] = depng_sum(p[bpp], depng_paeth_opt(p[0], u[bpp], u[0]));

        p += bpp;
        break;
      }
    }

    u = p - bpl;
  } while (--y != 0);
}

void depng_filter_opt(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl) {
  switch (bpp) {
    case 1: depng_filter_opt_template<1>(p, h, bpl); break;
    case 2: depng_filter_opt_template<2>(p, h, bpl); break;
    case 3: depng_filter_opt_template<3>(p, h, bpl); break;
    case 4: depng_filter_opt_template<4>(p, h, bpl); break;
    case 6: depng_filter_opt_template<6>(p, h, bpl); break;
    case 8: depng_filter_opt_template<8>(p, h, bpl); break;
  }
}
