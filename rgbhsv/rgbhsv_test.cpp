// [SimdTests - RGBHSV]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./rgbhsv.h"

// ============================================================================
// [SimdTests - RGBHSV - Fill]
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
// [SimdTests - RGBHSV - Check]
// ============================================================================

static void rgbhsv_check(
  const char* name,
  ArgbAhsvFunc ahsv_from_argb,
  ArgbAhsvFunc argb_from_ahsv,
  float* argb_src,
  int length) {

  SIMD_ALIGN_VAR(float, tmp[4], 16);

  SIMD_ALIGN_VAR(float, argb_out[4], 16);
  SIMD_ALIGN_VAR(float, argb_ref[4], 16);

  SIMD_ALIGN_VAR(float, ahsv_out[4], 16);
  SIMD_ALIGN_VAR(float, ahsv_ref[4], 16);

  float rgb2hsv_err[4];
  float hsv2rgb_err[4];

  float rgb2hsv_max[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
  float hsv2rgb_max[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

  float display_err = 1e-6f;
  rgbhsv_fill(argb_src, length);

  for (int i = 0; i < length; i++) {
    ::memcpy(tmp, argb_src, 4 * sizeof(float));

    ahsv_from_argb   (ahsv_out, tmp, 1);
    ahsv_from_argb_hq(ahsv_ref, tmp, 1);

    argb_from_ahsv   (argb_out, ahsv_out, 1);
    argb_from_ahsv_hq(argb_ref, ahsv_out, 1);

    rgb2hsv_err[0] = SimdUtils::abs(ahsv_ref[0] - ahsv_out[0]);
    rgb2hsv_err[1] = SimdUtils::abs(ahsv_ref[1] - ahsv_out[1]);
    rgb2hsv_err[2] = SimdUtils::abs(ahsv_ref[2] - ahsv_out[2]);
    rgb2hsv_err[3] = SimdUtils::abs(ahsv_ref[3] - ahsv_out[3]);

    rgb2hsv_max[0] = SimdUtils::max<float>(rgb2hsv_max[0], rgb2hsv_err[0]);
    rgb2hsv_max[1] = SimdUtils::max<float>(rgb2hsv_max[1], rgb2hsv_err[1]);
    rgb2hsv_max[2] = SimdUtils::max<float>(rgb2hsv_max[2], rgb2hsv_err[2]);
    rgb2hsv_max[3] = SimdUtils::max<float>(rgb2hsv_max[3], rgb2hsv_err[3]);

    hsv2rgb_err[0] = SimdUtils::abs(argb_ref[0] - argb_out[0]);
    hsv2rgb_err[1] = SimdUtils::abs(argb_ref[1] - argb_out[1]);
    hsv2rgb_err[2] = SimdUtils::abs(argb_ref[2] - argb_out[2]);
    hsv2rgb_err[3] = SimdUtils::abs(argb_ref[3] - argb_out[3]);

    hsv2rgb_max[0] = SimdUtils::max<float>(hsv2rgb_max[0], hsv2rgb_err[0]);
    hsv2rgb_max[1] = SimdUtils::max<float>(hsv2rgb_max[1], hsv2rgb_err[1]);
    hsv2rgb_max[2] = SimdUtils::max<float>(hsv2rgb_max[2], hsv2rgb_err[2]);
    hsv2rgb_max[3] = SimdUtils::max<float>(hsv2rgb_max[3], hsv2rgb_err[3]);

    if (rgb2hsv_err[0] >= display_err || hsv2rgb_err[0] >= display_err ||
        rgb2hsv_err[1] >= display_err || hsv2rgb_err[1] >= display_err ||
        rgb2hsv_err[2] >= display_err || hsv2rgb_err[2] >= display_err ||
        rgb2hsv_err[3] >= display_err || hsv2rgb_err[3] >= display_err) {

      printf("[ERROR] IMPL=%-4s ARGB{%+0.8f %+0.8f %+0.8f %+0.8f} -> AHSV{%+0.8f %+0.8f %+0.8f %+0.8f} (Ref)\n"
             "                                                       AHSV{%+0.8f %+0.8f %+0.8f %+0.8f} (Out)\n"
             "                  AHSV{%+0.8f %+0.8f %+0.8f %+0.8f} -> ARGB{%+0.8f %+0.8f %+0.8f %+0.8f} (Ref)\n"
             "                                                       ARGB{%+0.8f %+0.8f %+0.8f %+0.8f} (Out)\n",
        name,
        argb_src[0], argb_src[1], argb_src[2], argb_src[3], ahsv_ref[0], ahsv_ref[1], ahsv_ref[2], ahsv_ref[3],
        ahsv_out[0], ahsv_out[1], ahsv_out[2], ahsv_out[3],
        ahsv_out[0], ahsv_out[1], ahsv_out[2], ahsv_out[3], argb_ref[0], argb_ref[1], argb_ref[2], argb_ref[3],
        argb_out[0], argb_out[1], argb_out[2], argb_out[3]);
    }

    argb_src += 4;
  }

  if (rgb2hsv_max[0] == 0.0f && rgb2hsv_max[1] == 0.0f && rgb2hsv_max[2] == 0.0f && rgb2hsv_max[3] == 0.0f) {
    printf("[CHECK] IMPL=%-4s ARGB -> AHSV: OK\n", name);
  }
  else {
    printf("[CHECK] IMPL=%-4s ARGB -> AHSV: MaxErr={%e %e %e %e}\n", name,
      rgb2hsv_max[0],
      rgb2hsv_max[1],
      rgb2hsv_max[2],
      rgb2hsv_max[3]);
  }

  if (hsv2rgb_max[0] == 0.0f && hsv2rgb_max[1] == 0.0f && hsv2rgb_max[2] == 0.0f && hsv2rgb_max[3] == 0.0f) {
    printf("[CHECK] IMPL=%-4s AHSV -> ARGB: OK\n", name);
  }
  else {
    printf("[CHECK] IMPL=%-4s AHSV -> ARGB: MaxErr={%e %e %e %e}\n", name,
      hsv2rgb_max[0],
      hsv2rgb_max[1],
      hsv2rgb_max[2],
      hsv2rgb_max[3]);
  };
}

// ============================================================================
// [SimdTests - RGBHSV - Bench]
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
  printf("[BENCH] IMPL=%-4s ARGB -> AHSV: %.2u.%.3u s\n", name, timer.get() / 1000, timer.get() % 1000);

  timer.start();
  for (i = 0; i < quantity; i++) argb_from_ahsv(argb, ahsv, length);
  timer.stop();
  printf("[BENCH] IMPL=%-4s AHSV -> ARGB: %.2u.%.3u s\n", name, timer.get() / 1000, timer.get() % 1000);
}

// ============================================================================
// [SimdTests - RGBHSV - Main]
// ============================================================================

int main(int argc, char* argv[]) {
  int length = 100000;

  float* argb_data = static_cast<float*>(::malloc(length * 4 * sizeof(float) + 16));
  float* ahsv_data = static_cast<float*>(::malloc(length * 4 * sizeof(float) + 16));

  float* argb = SimdUtils::align(argb_data, 16);
  float* ahsv = SimdUtils::align(ahsv_data, 16);

  rgbhsv_fill(argb, length);
  rgbhsv_check("ref" , ahsv_from_argb_ref , argb_from_ahsv_ref , argb, length);

  rgbhsv_fill(argb, length);
  rgbhsv_check("sse2", ahsv_from_argb_sse2, argb_from_ahsv_sse2, argb, length);

  rgbhsv_bench("ref" , ahsv_from_argb_ref , argb_from_ahsv_ref , argb, ahsv, length);
  rgbhsv_bench("sse2", ahsv_from_argb_sse2, argb_from_ahsv_sse2, argb, ahsv, length);

  ::free(argb_data);
  ::free(ahsv_data);

  return 0;
}
