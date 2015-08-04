// [SimdRgbHsv]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Zlib - See LICENSE.md file in the package.

#include <math.h>
#include <stdio.h>

#include "./simdglobals.h"
#include "./simdrgbhsv.h"

// ============================================================================
// [rgbhsv_fill]
// ============================================================================

static void rgbhsv_fill(float* argb, int length) {
  int i = length;

  for (;;)  {
    float a = 1.0f;
    float r;
    float g;
    float b;

    for (r = 0.0f; r <= 1.0f; r += 0.02f) {
      for (g = 0.0f; g <= 1.0f; g += 0.02f) {
        for (b = 0.0f; b <= 1.0f; b += 0.02f) {
          argb[0] = a;
          argb[1] = r;
          argb[2] = g;
          argb[3] = b;

          argb += 4;
          if (--i == 0) return;
        }
      }
    }
  }
}

// ============================================================================
// [rgbhsv_check]
// ============================================================================

static void rgbhsv_check(
  const char* name,
  ArgbAhsvFunc ahsv_from_argb,
  ArgbAhsvFunc argb_from_ahsv,
  float *argb,
  float* ahsv,
  int length) {

  SIMD_ALIGN_VAR(float, argb_r[4], 16);
  SIMD_ALIGN_VAR(float, ahsv_r[4], 16);

  float ahsv_from_argb_err[4];
  float argb_from_ahsv_err[4];

  float ahsv_from_argb_max[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float argb_from_ahsv_max[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  float display_err = 1e-6f;

  printf("[CHECK] IMPL=%-4s\n", name);

  for (int i = 0; i < length; i++) {
    ::memcpy(argb_r, argb, 4 * sizeof(float));

    ahsv_from_argb(ahsv_r, argb_r, 1);
    argb_from_ahsv(argb_r, ahsv_r, 1);

    ahsv_from_argb_err[0] = SimdUtils::abs(ahsv[0] - ahsv_r[0]);
    ahsv_from_argb_err[1] = SimdUtils::abs(ahsv[1] - ahsv_r[1]);
    ahsv_from_argb_err[2] = SimdUtils::abs(ahsv[2] - ahsv_r[2]);
    ahsv_from_argb_err[3] = SimdUtils::abs(ahsv[3] - ahsv_r[3]);

    ahsv_from_argb_max[0] = SimdUtils::max<float>(ahsv_from_argb_max[0], ahsv_from_argb_err[0]);
    ahsv_from_argb_max[1] = SimdUtils::max<float>(ahsv_from_argb_max[1], ahsv_from_argb_err[1]);
    ahsv_from_argb_max[2] = SimdUtils::max<float>(ahsv_from_argb_max[2], ahsv_from_argb_err[2]);
    ahsv_from_argb_max[3] = SimdUtils::max<float>(ahsv_from_argb_max[3], ahsv_from_argb_err[3]);

    argb_from_ahsv_err[0] = SimdUtils::abs(argb[0] - argb_r[0]);
    argb_from_ahsv_err[1] = SimdUtils::abs(argb[1] - argb_r[1]);
    argb_from_ahsv_err[2] = SimdUtils::abs(argb[2] - argb_r[2]);
    argb_from_ahsv_err[3] = SimdUtils::abs(argb[3] - argb_r[3]);

    argb_from_ahsv_max[0] = SimdUtils::max<float>(argb_from_ahsv_max[0], argb_from_ahsv_err[0]);
    argb_from_ahsv_max[1] = SimdUtils::max<float>(argb_from_ahsv_max[1], argb_from_ahsv_err[1]);
    argb_from_ahsv_max[2] = SimdUtils::max<float>(argb_from_ahsv_max[2], argb_from_ahsv_err[2]);
    argb_from_ahsv_max[3] = SimdUtils::max<float>(argb_from_ahsv_max[3], argb_from_ahsv_err[3]);

    if (ahsv_from_argb_err[0] >= display_err || argb_from_ahsv_err[0] >= display_err ||
        ahsv_from_argb_err[1] >= display_err || argb_from_ahsv_err[1] >= display_err ||
        ahsv_from_argb_err[2] >= display_err || argb_from_ahsv_err[2] >= display_err ||
        ahsv_from_argb_err[3] >= display_err || argb_from_ahsv_err[3] >= display_err) {

      printf("[ERROR] IMPL=%-4s ARGB {%+0.8f %+0.8f %+0.8f %+0.8f} -> AHSV {%+0.8f %+0.8f %+0.8f %+0.8f}\n"
             "                       {%+0.8f %+0.8f %+0.8f %+0.8f}    AHSV {%+0.8f %+0.8f %+0.8f %+0.8f}\n",
        name,
        argb[0], argb[1], argb[2], argb[3],
        ahsv[0], ahsv[1], ahsv[2], ahsv[3],
        argb_r[0], argb_r[1], argb_r[2], argb_r[3],
        ahsv_r[0], ahsv_r[1], ahsv_r[2], ahsv_r[3]);
    }

    argb += 4;
    ahsv += 4;
  }

  if (ahsv_from_argb_max[0] == 0.0f && ahsv_from_argb_max[1] == 0.0f &&
      ahsv_from_argb_max[2] == 0.0f && ahsv_from_argb_max[3] == 0.0f) {
    printf("[CHECK] IMPL=%-4s AHSV<-ARGB Ok\n", name);
  }
  else {
    printf("[CHECK] IMPL=%-4s AHSV<-ARGB MaxErr={%e %e %e %e}\n", name,
      ahsv_from_argb_max[0],
      ahsv_from_argb_max[1],
      ahsv_from_argb_max[2],
      ahsv_from_argb_max[3]);
  }

  if (argb_from_ahsv_max[0] == 0.0f && argb_from_ahsv_max[1] == 0.0f &&
      argb_from_ahsv_max[2] == 0.0f && argb_from_ahsv_max[3] == 0.0f) {
    printf("[CHECK] IMPL=%-4s ARGB<-AHSV Ok\n", name);
  }
  else {
    printf("[CHECK] IMPL=%-4s ARGB<-AHSV MaxErr={%e %e %e %e}\n", name,
      argb_from_ahsv_max[0],
      argb_from_ahsv_max[1],
      argb_from_ahsv_max[2],
      argb_from_ahsv_max[3]);
  };
}

// ============================================================================
// [rgbhsv_bench]
// ============================================================================

void rgbhsv_bench(
  const char* name,
  ArgbAhsvFunc ahsv_from_argb,
  ArgbAhsvFunc argb_from_ahsv,
  float* argb,
  float* ahsv,
  int length) {

  int i;
  int quantity = 1000;

  SimdTimer timer;
  rgbhsv_fill(argb, length);

  timer.start();
  for (i = 0; i < quantity; i++) ahsv_from_argb(ahsv, argb, length);
  timer.stop();
  printf("[BENCH] IMPL=%-4s [%.2u.%.3u s] [AHSV <- ARGB]\n", name, timer.get() / 1000, timer.get() % 1000);

  timer.start();
  for (i = 0; i < quantity; i++) argb_from_ahsv(argb, ahsv, length);
  timer.stop();
  printf("[BENCH] IMPL=%-4s [%.2u.%.3u s] [ARGB <- AHSV]\n", name, timer.get() / 1000, timer.get() % 1000);
}

int main(int argc, char* argv[]) {
  int length = 100000;

  float* argb_data = static_cast<float*>(::malloc(length * 4 * sizeof(float) + 16));
  float* ahsv_data = static_cast<float*>(::malloc(length * 4 * sizeof(float) + 16));

  float* argb = SimdUtils::align(argb_data, 16);
  float* ahsv = SimdUtils::align(ahsv_data, 16);

  rgbhsv_fill(argb, length);
  ahsv_from_argb_ref(ahsv, argb, length);

  rgbhsv_check("sse2", ahsv_from_argb_sse2, argb_from_ahsv_sse2, argb, ahsv, length);

  rgbhsv_bench("ref" , ahsv_from_argb_ref , argb_from_ahsv_ref , argb, ahsv, length);
  rgbhsv_bench("sse2", ahsv_from_argb_sse2, argb_from_ahsv_sse2, argb, ahsv, length);

  ::free(argb_data);
  ::free(ahsv_data);

  return 0;
}
