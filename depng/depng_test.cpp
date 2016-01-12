// [SimdTests - DePNG]
// SIMD optimized "PNG Reverse Filter" implementation.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./depng.h"

// ============================================================================
// [SimdTests::DePNG - Constants]
// ============================================================================

static const char* depng_filter_names[] = {
  "None", "Sub", "Up", "Avg", "Paeth", "Mixed"
};

static const uint32_t depng_bpp_data[] = {
  1, 2, 3, 4, 6, 8
};

static const uint8_t depng_random_data[] = {
  0xD9, 0xFA, 0xA7, 0x20, 0x6B, 0xD3, 0x41, 0xC9, 0x1A, 0x27, 0x2F, 0x64, 0x59,
  0x85, 0x47, 0x1C, 0xFC, 0x3E, 0xA3, 0x5B, 0x3C, 0xD2, 0xB5, 0xB6, 0x80, 0xBB,
  0x84, 0x3C, 0xD4, 0x94, 0x3A, 0x6D, 0xC2, 0x1B, 0x3D, 0x5F, 0x82, 0xD9, 0x1A,
  0x7F, 0xC6, 0x8D, 0x39, 0xDD, 0x07, 0xAD, 0x7A, 0x40, 0x8D, 0x37, 0x56, 0x12,
  0x8B, 0x51, 0xAF, 0x9D, 0x17, 0xBD, 0xD0, 0x61, 0x58, 0xC8, 0x05, 0x44, 0x9B,
  0xCA, 0xD4, 0xD0, 0xD0, 0xB9, 0x83, 0x75, 0x31, 0x4B, 0x09, 0xEC, 0x52, 0xEB,
  0xE5, 0xE8, 0xAA, 0xF6, 0xDD, 0x79, 0x36, 0x61, 0x17, 0xB1, 0x8A, 0x48, 0x00,
  0x1A, 0x9D, 0xDC, 0x51, 0x9F, 0x34, 0x7A, 0x48, 0x56, 0xC9, 0xF3, 0x6A, 0x81,
  0x9B, 0x47, 0x56, 0x64, 0x00, 0x30, 0x60, 0x04, 0x90, 0x4B, 0xC2, 0x48, 0xE3,
  0xED, 0x62, 0xDF, 0x46, 0xEF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF,
  0xFE, 0x94, 0xEE, 0x00, 0xA9, 0x3B, 0x86, 0x9B, 0xD8, 0xEE, 0x3D, 0x9E, 0x32,
  0x00, 0x00, 0x00, 0x00, 0x92, 0x61, 0x9F, 0x3B, 0x22, 0xB0, 0xB9, 0xB3, 0xB0,
  0x01, 0x01, 0x01, 0x01, 0xF4, 0x83, 0xFC, 0x49, 0xA9, 0xD2, 0x89, 0xE0, 0x17,
  0x74, 0x3E, 0xBD, 0x28, 0x74, 0x5E, 0xF8, 0x6D, 0xD2, 0x43, 0xB7, 0x5A, 0xB5,
  0xE6, 0xA4, 0xC7, 0xA4, 0x46, 0xD3, 0x00, 0x1A, 0x26, 0x0C, 0x65, 0x24, 0xAD,
  0xA7, 0xEA, 0xF4, 0xBD, 0xF6, 0x63, 0x2B, 0xEC, 0x1E, 0xDF, 0x0C, 0xBD, 0x50,
  0xEB, 0x71, 0xD9, 0x86, 0x31, 0x62, 0x5E, 0xE7, 0x4D, 0x8B, 0xD1, 0x11, 0x5B,
  0x26, 0x48, 0x9F, 0x8E, 0xE6, 0x7B, 0xE1, 0x0C, 0xF8, 0xCD, 0xF8, 0x90, 0x1E,
  0x4E, 0x24, 0xFE, 0x90, 0xD3, 0xA2, 0x2D, 0xFC, 0x4F, 0x3A, 0x2F, 0x1B, 0xE2,
  0xB8, 0xBF, 0x11, 0x68, 0x80, 0xCB, 0x26, 0xAD, 0x1C, 0x58, 0x4E, 0x57, 0x30,
  0x00, 0x00, 0x00, 0x86, 0x4A, 0x50, 0x36, 0x90, 0x5C, 0x40, 0xA7, 0x38, 0x92,
  0x03, 0xF0, 0x39, 0x82, 0x40, 0xED, 0x39, 0x22, 0x82, 0x90, 0x67, 0xDF, 0x95,
  0x34, 0x15, 0x8A, 0x0F, 0x25, 0x94, 0x56, 0xFD, 0x38, 0x85, 0x9B, 0x06, 0x22
};

// ============================================================================
// [SimdTests::DePNG - Helpers]
// ============================================================================

static uint32_t depng_random_wrap(uint32_t x, uint32_t advance, uint32_t count) {
  x += advance;
  return x < count ? x : x - count;
}

static uint8_t* depng_random_image(uint32_t w, uint32_t h, uint32_t bpp, uint32_t filter, uint32_t seed) {
  uint32_t rCount = sizeof(depng_random_data);

  uint32_t rIndex0 = (seed     ) % rCount;
  uint32_t rIndex1 = (seed * 33) % rCount;

  w *= bpp;
  uint32_t size = (w + 1) * h;

  uint8_t* pImage = static_cast<uint8_t*>(::malloc(size));
  if (pImage == NULL)
    return NULL;

  uint8_t* p = pImage;
  uint32_t f = filter;

  for (uint32_t y = 0; y < h; y++) {
    // NOTE: We always use no filter for the very first for so we don't have to
    // handle special cases in the code. However, when decoding real PNG image,
    // the first row has to be handled as well.
    if (y == 0) {
      *p++ = kPngFilterNone;
    }
    else if (filter < kPngFilterCount) {
      *p++ = static_cast<uint8_t>(filter);
    }
    else {
      if (++f >= kPngFilterCount) f = 0;
      *p++ = static_cast<uint8_t>(f);
    }

    uint32_t x = w;
    for (;;) {
      *p++ = depng_random_data[rIndex0];
      rIndex0 = depng_random_wrap(rIndex0, 1, rCount);
      if (--x == 0) break;

      *p++ = depng_random_data[rIndex1];
      rIndex1 = depng_random_wrap(rIndex1, 2, rCount);
      if (--x == 0) break;
    }
  }

  return pImage;
}

// ============================================================================
// [SimdTests::DePNG - Compare]
// ============================================================================

static bool depng_compare(const char* name, const uint8_t* pA, const uint8_t* pB, uint32_t w, uint32_t h, uint32_t bpp, uint32_t bpl) {
  if (bpl != w * bpp + 1) {
    printf("[ERROR] Invalid BPL=%u given for BPP=%u and Width=%u\n", bpl, bpp, w);
    return false;
  }

  for (uint32_t y = 0; y < h; y++) {
    uint32_t aFilter = pA[0];
    uint32_t bFilter = pB[0];

    if (aFilter != bFilter) {
      printf("[ERROR] IMPL=%-15s  [%ux%u|bpp:%u|bpl=%u at Y=%u X=Filter] Filter %u != %u\n",
        name, w, h, bpp, bpl, y, aFilter, bFilter);
      return false;
    }

    if (aFilter >= kPngFilterCount) {
      printf("[ERROR] IMPL=%-15s  [%ux%u|bpp:%u|bpl=%u at Y=%u X=Filter] Filter %u\n",
        name, w, h, bpp, bpl, y, aFilter);
      return false;
    }

    pA++;
    pB++;

    for (uint32_t x = 0; x < w; x++) {
      for (uint32_t i = 0; i < bpp; i++) {
        uint32_t aVal = pA[i];
        uint32_t bVal = pB[i];

        if (aVal != bVal) {
          printf("[ERROR] IMPL=%-15s  [%ux%u|bpp:%u|bpl=%u at Y=%u|X=%u|Byte=%u] Pixel %u != %u (%s)\n",
            name, w, h, bpp, bpl, y, x, i, aVal, bVal, depng_filter_names[aFilter]);
          return false;
        }
      }

      pA += bpp;
      pB += bpp;
    }
  }

  return true;
}

// ============================================================================
// [SimdTests::DePNG - Check]
// ============================================================================

static bool depng_check(const char* name, DePngFilterFunc ref, DePngFilterFunc opt) {
  printf("[CHECK] IMPL=%-15s\n", name);

  uint32_t seed = 0;
  for (uint32_t filter = 0; filter <= kPngFilterCount; filter++) {
    for (uint32_t h = 1; h < 20; h++) {
      for (uint32_t w = 1; w < 100; w++) {
        for (uint32_t bppIndex = 0; bppIndex < 6; bppIndex++) {
          uint32_t bpp = depng_bpp_data[bppIndex];
          uint32_t bpl = w * bpp + 1;

          uint8_t* pRef = depng_random_image(w, h, bpp, filter, seed);
          uint8_t* pOpt = depng_random_image(w, h, bpp, filter, seed);

          ref(pRef, h, bpp, bpl);
          opt(pOpt, h, bpp, bpl);

          bool ok = depng_compare(name, pRef, pOpt, w, h, bpp, bpl);

          ::free(pRef);
          ::free(pOpt);

          if (!ok)
            return false;

          seed++;
        }
      }
    }
  }

  return true;
}

// ============================================================================
// [SimdTests::DePNG - Bench]
// ============================================================================

static void depng_bench(const char* name, DePngFilterFunc func) {
  SimdTimer timer;

  uint32_t w = 256;
  uint32_t h = 256;
  uint32_t quantity = 1000;
  uint32_t totalTime = 0;

  for (uint32_t filter = 1; filter <= kPngFilterCount; filter++) {
    uint32_t filterTime = 0;

    for (uint32_t bppIndex = 0; bppIndex < 6; bppIndex++) {
      uint32_t bpp = depng_bpp_data[bppIndex];
      uint32_t bpl = w * bpp + 1;

      uint8_t* pImage = depng_random_image(w, h, bpp, filter, 0);

      timer.start();
      for (uint32_t i = 0; i < quantity; i++) {
        func(pImage, h, bpp, bpl);
      }
      timer.stop();

      filterTime += timer.get();
      totalTime += timer.get();

      printf("[BENCH] IMPL=%-15s  [%.2u.%.3u s] [%s:%u]\n",
        name, timer.get() / 1000, timer.get() % 1000, depng_filter_names[filter], bpp);

      ::free(pImage);
    }

    printf("[BENCH] IMPL=%-15s  [%.2u.%.3u s] [%s:ALL]\n",
      name, filterTime / 1000, filterTime % 1000, depng_filter_names[filter]);
  }

  printf("[BENCH] IMPL=%-15s  [%.2u.%.3u s] [Total]\n\n",
    name, totalTime / 1000, totalTime % 1000);
}

// ============================================================================
// [SimdTests::DePNG - Main]
// ============================================================================

int main(int argc, char* argv[]) {
  if (!depng_check("revfilter-opt" , depng_filter_ref, depng_filter_opt )) return 1;
  if (!depng_check("revfilter-sse2", depng_filter_ref, depng_filter_sse2)) return 1;

  depng_bench("revfilter-ref" , depng_filter_ref);
  depng_bench("revfilter-opt" , depng_filter_opt);
  depng_bench("revfilter-sse2", depng_filter_sse2);

  return 0;
}
