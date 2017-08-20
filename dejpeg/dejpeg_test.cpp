// [SimdTests - DeJPEG]
// SIMD optimized JPEG decoding utilities.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./dejpeg.h"

#define BENCH_COUNT 3
#define BENCH_ITER_DEZIGZAG 5000000
#define BENCH_ITER_IDCT 1000000
#define BENCH_YCBCR 1000000

// ============================================================================
// [SimdTests::DeJPEG - Utilities]
// ============================================================================

template<typename T>
static void dejpeg_fill_seq8x8(T* dst) {
  for (int i = 0; i < 64; i++)
    dst[0] = T(i);
}

template<typename T>
static void dejpeg_fill_data8x8(T* dst, int min, int max, unsigned int offset = 0) {
  static const double table[] = {
    0.59083278, 0.71607873, 0.80463743, 0.61216261,
    0.32372465, 0.75365489, 0.34177617, 0.25476189,
    0.43021063, 0.68966455, 0.53471116, 0.80345154,
    0.10321173, 0.10544485, 0.07973516, 0.50994795,
    0.12242540, 0.39569155, 0.34092983, 0.02490654,
    0.08306205, 0.46306336, 0.96708909, 0.70224778,
    0.65308627, 0.15657926, 0.14462084, 0.31709949,
    0.22468948, 0.16985155, 0.37765272, 0.89179553,
    0.98145430, 0.70432421, 0.07281313, 0.77204939,
    0.06260656, 0.73810427, 0.92676377, 0.64641843,
    0.99267340, 0.84705258, 0.51064012, 0.98886629,
    0.30582193, 0.14627043, 0.07068561, 0.37740857,
    0.46265122, 0.61994830, 0.53877411, 0.24022593,
    0.29889454, 0.03764948, 0.79332102, 0.56457740,
    0.50902332, 0.74886765, 0.00633825, 0.37915118,
    0.08329475, 0.68726225, 0.24605709, 0.43404197,
    0.74137674, 0.90377118, 0.04754893, 0.50999720,
    0.21404305, 0.19197957, 0.29286389, 0.35357176,
    0.79264367, 0.27127828, 0.22972804, 0.94551728,
    0.62408120, 0.22804748, 0.77982622, 0.13933479,
    0.29844304, 0.75641065, 0.13080603, 0.50217157,
    0.10244899, 0.26676505, 0.60551821, 0.50029754,
    0.39379406, 0.05539461, 0.77930735, 0.73789579,
    0.27231828, 0.92255640, 0.58626249, 0.48437114,
    0.16742719, 0.70565206, 0.78394555, 0.55203632,
    0.36112592, 0.04635612, 0.62654719, 0.54946801,
    0.82321511, 0.65756787, 0.07394471, 0.39288685,
    0.70626035, 0.85060888, 0.53385926, 0.71634540,
    0.74105780, 0.20400379, 0.22923460, 0.12329607,
    0.00008618, 0.40950272, 0.28721974, 0.00000000,
    0.66980731, 0.66118724, 0.23423347, 0.21200121,
    0.25057092, 0.73761035, 0.17476170, 0.10609926,
    0.92939825, 0.06759644, 0.01943499, 0.24123023,
    0.31366317, 0.35017927, 0.72261192, 0.76317247,
    0.12387586, 0.60138551, 0.90456912, 0.31507560,
    0.84335042, 0.47104808, 0.06190737, 0.27507339,
    0.53257569, 0.17314725, 0.88740422, 0.69179804,
    0.17671502, 0.30833971, 0.31696181, 0.79369190,
    0.24421442, 0.17429168, 0.44177168, 0.93030818,
    0.39179022, 0.13350952, 0.09738642, 0.82683021,
    0.73568170, 0.39357290, 0.26852605, 0.59906637,
    0.12673492, 0.00292828, 0.58202870, 0.74912522,
    0.70339053, 0.22682047, 0.51724632, 0.94343828,
    0.28740611, 0.06879262, 0.30186088, 0.36297955,
    0.54166955, 0.16644372, 0.11399715, 0.16207076,
    0.03804357, 0.90869184, 1.00000000, 0.87044820,
    0.18745118, 0.76402183, 0.10801919, 0.75612246,
    0.85121623, 0.55222927, 0.68102708, 0.85263725,
    0.11734438, 0.87635932, 0.76759812, 0.94980153,
    0.07963626, 0.09154617, 0.44260714, 0.03891884,
    0.66749449, 0.68355304, 0.55360173, 0.39533584,
    0.85677037, 0.59517367, 0.13400087, 0.95705619,
    0.58541782, 0.08735931, 0.19997486, 0.41878701,
    0.61272940, 0.39685917, 0.38463721, 0.95735238,
    0.15668421, 0.15414294, 0.98949966, 0.48749022,
    0.67712971, 0.68025917, 0.13120866, 0.63423287,
    0.23125810, 0.55697325, 0.07152314, 0.62177501,
    0.44509217, 0.02870165, 0.34047027, 0.67045818,
    0.41412399, 0.83180011, 0.23550525, 0.36867960,
    0.14167465, 0.27806542, 0.47724331, 0.60348115,
    0.69265976, 0.78022224, 0.42017292, 0.80791678,
    0.68068532, 0.87043549, 0.90247826, 0.69819724,
    0.72957078, 0.31181102, 0.92038412, 0.53066404,
    0.96118668, 0.25097063, 0.09674919, 0.30427938,
    0.11411108, 0.40091576, 0.42830483, 0.34036910,
    0.85458571, 0.23846644, 0.74743876, 0.34611623,
    0.66110353, 0.76960110, 0.93077682, 0.01057994,
    0.91144785, 0.22983274, 0.04745323, 0.39878106,
    0.89373747, 0.92699427, 0.99117824, 0.08134733,
    0.86771401, 0.76694564, 0.11743421, 0.50348758,
    0.75875547, 0.27996222, 0.76139009, 0.81353597,
    0.01723353, 0.82975543, 0.26078065, 0.96015313,
    0.46883602, 0.20222883, 0.24089862, 0.31141521,
    0.63293214, 0.62085357, 0.09257712, 0.29937766,
    0.80074954, 0.34295318, 0.45165331, 0.25186063,
    0.91329142, 0.58524506, 0.04904435, 0.38372860,
    0.62982636, 0.79220836, 0.64403336, 0.03755513,
    0.23758832, 0.56594344, 0.84145894, 0.42568495,
    0.14162272, 0.46840444, 0.27379008, 0.53465845,
    0.86368423, 0.26103988, 0.64645601, 0.63487090,
    0.65277068, 0.36060306, 0.72011627, 0.64702785,
    0.94045209, 0.40037580, 0.02659500, 0.36216919,
    0.88479002, 0.69896759, 0.48050225, 0.59410044,
    0.49724181, 0.36745825, 0.20439247, 0.27947462,
    0.80516200, 0.12110751, 0.54183499, 0.12617968,
    0.08300157, 0.39046156, 0.96531236, 0.49370228,
    0.64267138, 0.47011973, 0.22462999, 0.11149382,
    0.05905694, 0.72947597, 0.21015221, 0.09778696,
    0.00093556, 0.80898310, 0.82346965, 0.13082356,
    0.79574558, 0.05853443, 0.56377450, 0.72573263,
    0.94560161, 0.13611801, 0.29457805, 0.15119284,
    0.93004186, 0.63487807, 0.00589519, 0.36944532,
    0.83429959, 0.34223025, 0.53044053, 0.67538721,
    0.32885732, 0.21000706, 0.51152149, 0.77033818,
    0.09518500, 0.38533856, 0.52970433, 0.48168568,
    0.73146046, 0.09149309, 0.79657325, 0.17910445,
    0.25525690, 0.18917645, 0.68286309, 0.04178951,
    0.45538362, 0.57240364, 0.55055057, 0.34155747,
    0.34726113, 0.18313940, 0.03737820, 0.80939750,
    0.26161738, 0.08593417, 0.24999455, 0.24853029
  };

  static const unsigned int tableSize = static_cast<int>(sizeof(table) / sizeof(table[0]));

  offset %= tableSize;
  double bias = static_cast<double>(min);
  double scale = static_cast<double>(max - min);

  for (int i = 0; i < 64; i++) {
    dst[i] = static_cast<T>(table[offset] * scale + bias + 0.5);
    if (++offset >= tableSize) offset = 0;
  }
}

template<typename T>
static void dejpeg_compare_data8x8(const T* a, const T* b) {
  for (unsigned int i = 0; i < 64; i++) {
    if (a[i] != b[i]) {
      printf("FAILED [%u] a=%d b=%d\n", i, a[i], b[i]);
    }
  }
}

// ============================================================================
// [SimdTests::DeJPEG - DeZigZag]
// ============================================================================

static void dejpeg_check_dezigzag8x8(const char* name, DeJpegDeZigZag8x8Func a, DeJpegDeZigZag8x8Func b) {
  SIMD_ALIGN_VAR(int16_t, data[64], 16);
  SIMD_ALIGN_VAR(int16_t, out_a[64], 16);
  SIMD_ALIGN_VAR(int16_t, out_b[64], 16);

  printf("[CHECK] IMPL=%-15s\n", name);
  dejpeg_fill_seq8x8(data);

  a(out_a, data);
  b(out_b, data);
  dejpeg_compare_data8x8(out_a, out_b);
}

static void dejpeg_bench_dezigzag8x8(const char* name, DeJpegDeZigZag8x8Func func) {
  SimdTimer timer;
  uint32_t best = 0xFFFFFFFFU;

  SIMD_ALIGN_VAR(int16_t, coeff0[64], 16);
  SIMD_ALIGN_VAR(int16_t, coeff1[64], 16);

  for (uint32_t z = 0; z < BENCH_COUNT; z++) {
    dejpeg_fill_seq8x8(coeff0);

    timer.start();
    for (uint32_t i = 0; i < BENCH_ITER_DEZIGZAG; i++) {
      if ((i & 1) == 0)
        func(coeff1, coeff0);
      else
        func(coeff0, coeff1);
    }
    timer.stop();
    if (timer.get() < best)
      best = timer.get();
  }

  printf("[BENCH] IMPL=%-15s [%.2u.%.3u s]\n", name, best / 1000, best % 1000);
}

// ============================================================================
// [SimdTests::DeJPEG - IDCT]
// ============================================================================

static void dejpeg_check_idct(const char* name, DeJpegIDCTFunc a, DeJpegIDCTFunc b) {
  printf("[CHECK] IMPL=%-15s\n", name);

  SIMD_ALIGN_VAR(int16_t, coeff[64], 16);
  SIMD_ALIGN_VAR(uint16_t, quant[64], 16);

  // Not aligned on purpose.
  uint8_t out_a[64];
  uint8_t out_b[64];

  dejpeg_fill_data8x8(coeff, -256, 255);
  dejpeg_fill_data8x8(quant, 0, 8);

  a(out_a, 8, coeff, quant);
  b(out_b, 8, coeff, quant);

  dejpeg_compare_data8x8(out_a, out_b);
}

static void dejpeg_bench_idct(const char* name, DeJpegIDCTFunc func) {
  SimdTimer timer;
  uint32_t best = 0xFFFFFFFFU;

  // Dummy counter to prevent optimizations.
  uint32_t dummy = 0;

  SIMD_ALIGN_VAR(int16_t, coeff[64], 16);
  SIMD_ALIGN_VAR(uint16_t, quant[64], 16);
  
  uint8_t pixels[64];
  
  dejpeg_fill_data8x8(coeff, -256, 255);
  dejpeg_fill_data8x8(quant, 0, 8);

  for (uint32_t z = 0; z < BENCH_COUNT; z++) {
    timer.start();
    for (uint32_t i = 0; i < BENCH_ITER_IDCT; i++) {
      func(pixels, 8, coeff, quant);
      dummy += pixels[0];
    }
    timer.stop();
    if (timer.get() < best)
      best = timer.get();
  }

  printf("[BENCH] IMPL=%-15s [%.2u.%.3u s] {dummy=%u}\n", name, best / 1000, best % 1000, dummy);
}

// ============================================================================
// [SimdTests::DeJPEG - YCbCrToRGB32]
// ============================================================================

static void dejpeg_check_ycbcr_to_rgb32(const char* name, YCbCrToRgbFunc a, YCbCrToRgbFunc b) {
  printf("[CHECK] IMPL=%-15s\n", name);

  uint8_t y[256];
  uint8_t cb[256];
  uint8_t cr[256];

  uint32_t i, j, k;

  uint32_t aDst[256];
  uint32_t bDst[256];

  for (i = 0; i < 256; i++)
    cr[i] = static_cast<uint8_t>(i);

  for (i = 0; i < 256; i++) {
    ::memset(y, i, 256);
    for (j = 0; j < 256; j++) {
      ::memset(cb, j, 256);

      a(reinterpret_cast<uint8_t*>(aDst), y, cb, cr, 256);
      b(reinterpret_cast<uint8_t*>(bDst), y, cb, cr, 256);

      for (k = 0; k < 256; k++) {
        uint32_t aVal = aDst[k];
        uint32_t bVal = bDst[k];

        if (aVal != bVal) {
          printf("FAILED [y=%d cb=%d cr=%d] a=0x%08X b=0x%08X\n", i, j, k, aVal, bVal);
        }
      }
    }
  }
}

static void dejpeg_bench_ycbcr_to_rgb32(const char* name, YCbCrToRgbFunc func) {
  SimdTimer timer;
  uint32_t best = 0xFFFFFFFFU;

  // Dummy counter to prevent optimizations.
  uint32_t dummy = 0;

  SIMD_ALIGN_VAR(uint8_t, yy[128], 16);
  SIMD_ALIGN_VAR(uint8_t, cb[128], 16);
  SIMD_ALIGN_VAR(uint8_t, cr[128], 16);

  for (uint32_t k = 0; k < 128; k++) {
    yy[k] = k;
    cb[k] = 255 - k;
    cr[k] = 64 + k;
  }

  uint32_t pixels[128];

  for (uint32_t z = 0; z < BENCH_COUNT; z++) {
    timer.start();
    for (uint32_t i = 0; i < BENCH_YCBCR; i++) {
      func(reinterpret_cast<uint8_t*>(pixels), yy, cb, cr, 128);
      dummy += pixels[0];
    }
    timer.stop();
    if (timer.get() < best)
      best = timer.get();
  }

  printf("[BENCH] IMPL=%-15s [%.2u.%.3u s] {dummy=%u}\n", name, best / 1000, best % 1000, dummy);
}

// ============================================================================
// [SimdTests::DeJPEG - Main]
// ============================================================================

int main(int argc, char* argv[]) {
  dejpeg_check_dezigzag8x8("zzag-ssse3-v1", dejpeg_dezigzag_ref, dejpeg_dezigzag_ssse3_v1);
  dejpeg_check_dezigzag8x8("zzag-ssse3-v2", dejpeg_dezigzag_ref, dejpeg_dezigzag_ssse3_v2);
  dejpeg_bench_dezigzag8x8("zzag-ref"  , dejpeg_dezigzag_ref);
  dejpeg_bench_dezigzag8x8("zzag-ssse3-v1", dejpeg_dezigzag_ssse3_v1);
  dejpeg_bench_dezigzag8x8("zzag-ssse3-v2", dejpeg_dezigzag_ssse3_v2);

  printf("\n");

  dejpeg_check_idct("islow-sse2", dejpeg_idct_islow_ref, dejpeg_idct_islow_sse2);
  dejpeg_bench_idct("islow-ref" , dejpeg_idct_islow_ref);
  dejpeg_bench_idct("islow-sse2", dejpeg_idct_islow_sse2);

  printf("\n");

  dejpeg_check_ycbcr_to_rgb32("ycbcr-rgb-sse2", dejpeg_ycbcr_to_rgb32_ref, dejpeg_ycbcr_to_rgb32_sse2);
  dejpeg_bench_ycbcr_to_rgb32("ycbcr-rgb-ref", dejpeg_ycbcr_to_rgb32_ref);
  dejpeg_bench_ycbcr_to_rgb32("ycbcr-rgb-sse2", dejpeg_ycbcr_to_rgb32_sse2);
  return 0;
}
