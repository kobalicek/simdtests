// [SimdPixel]
// Playground for SIMD pixel manipulation.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./pixops.h"

#define BENCH_COUNT 5
#define BENCH_ITER 500

// ============================================================================
// [Utilities]
// ============================================================================

static uint32_t premultiply(uint32_t pixel) {
  uint32_t a = pixel >> 24;
  uint32_t r = (((pixel >> 16) & 0xFF) * a) / 255;
  uint32_t g = (((pixel >>  8) & 0xFF) * a) / 255;
  uint32_t b = (((pixel      ) & 0xFF) * a) / 255;

  return (a << 24) | (r << 16) | (g << 8) | b;
}

static void pixels_fill(uint32_t* dst, int n, uint64_t seed) {
  SimdRandom prnd(seed);
  for (int i = 0; i < n; i++)
    dst[i] = premultiply(prnd.nextUInt32());
}

// ============================================================================
// [SimdTests - PixOps - Check]
// ============================================================================

static void pixops_check(const char* name, PixelOpFunc a, PixelOpFunc b) {
  printf("[CHECK] IMPL=%-20s\n", name);

  enum {
    kW = 1000,
    kH = 1000,
    kCount = kW * kH
  };

  uint32_t* dst = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));
  uint32_t* src = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));

  pixels_fill(dst, kCount, SIMD_UINT64_C(0x2F2E3A4A1A191238));
  pixels_fill(src, kCount, SIMD_UINT64_C(0x3F2E3A4A1A191238));

  uint32_t* aResult = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));
  uint32_t* bResult = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));

  for (uint32_t alpha = 1; alpha < 256; alpha++) {
    ::memcpy(aResult, dst, kCount * sizeof(uint32_t));
    ::memcpy(bResult, dst, kCount * sizeof(uint32_t));

    a(aResult, kW * 4, src, kW * 4, kW, kH, alpha);
    b(bResult, kW * 4, src, kW * 4, kW, kH, alpha);

    for (unsigned int i = 0; i < kCount; i++) {
      uint32_t aPixel = aResult[i];
      uint32_t bPixel = bResult[i];

      if (aPixel != bPixel) {
        printf("ERROR: %0.8X != %0.8X (at %u) (alpha %u)\n", aPixel, bPixel, i, alpha);
      }
    }
  }

  ::free(dst);
  ::free(src);
  ::free(aResult);
  ::free(bResult);
}

// ============================================================================
// [SimdTests - PixOps - Bench]
// ============================================================================

static void pixops_bench(const char* name, PixelOpFunc func) {
  SimdTimer timer;
  uint32_t best = 0xFFFFFFFFU;

  enum {
    kW = 1000,
    kH = 1000,
    kCount = kW * kH
  };

  // Dummy counter to prevent optimizations.
  uint32_t dummy = 0;

  uint32_t* dst = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));
  uint32_t* src = static_cast<uint32_t*>(malloc(kCount * sizeof(uint32_t)));

  for (uint32_t z = 0; z < BENCH_COUNT; z++) {
    uint32_t alpha = 1;

    pixels_fill(dst, kCount, SIMD_UINT64_C(0x0123456789ABCDEF));
    pixels_fill(src, kCount, SIMD_UINT64_C(0xFEDCBA9876543210));

    timer.start();
    for (uint32_t i = 0; i < BENCH_ITER; i++) {
      func(dst, kW * 4, src, kW * 4, kW, kH, alpha);
      dummy += dst[0];

      if (++alpha >= 256)
        alpha = 1;
    }
    timer.stop();

    if (timer.get() < best)
      best = timer.get();
  }

  uint32_t mbps = static_cast<uint32_t>(
    ((static_cast<uint64_t>(kCount * 4) * BENCH_ITER * 1000) / best) / (1024 * 1024));
  printf("[BENCH] IMPL=%-20s [%.2u.%.3u s] (%u MB/s) {dummy=%u}\n", name, best / 1000, best % 1000, mbps, dummy);

  ::free(dst);
  ::free(src);
}

// ============================================================================
// [SimdTests - PixOps - Main]
// ============================================================================

int main(int argc, char* argv[]) {
  pixops_check("crossfade-sse2" , pixops_crossfade_ref, pixops_crossfade_sse2);
  pixops_check("crossfade-ssse3", pixops_crossfade_ref, pixops_crossfade_ssse3);

  pixops_bench("crossfade-ref"  , pixops_crossfade_ref);
  pixops_bench("crossfade-sse2" , pixops_crossfade_sse2);
  pixops_bench("crossfade-ssse3", pixops_crossfade_ssse3);

  return 0;
}
