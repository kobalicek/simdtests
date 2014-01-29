#include <math.h>
#include <stdio.h>

#include "rgbhsv.h"
#include "util.h"

// ============================================================================
// [Test]
// ============================================================================

static void argb_fill(float* argb, int length) {
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

static void validate(float *argb, float* ahsv, int length) {
  int i;

  for (i = 0; i < length; i++) {
    XMM_ALIGNED_VAR(float, local_argb[4]);
    XMM_ALIGNED_VAR(float, local_ahsv[4]);

    memcpy(local_argb, argb, sizeof(float)* 4);
    ahsv_from_argb_sse2(local_ahsv, local_argb, 1);
    argb_from_ahsv_sse2(local_argb, local_ahsv, 1);

    // float should_ahsv[4];
    // ahsv_from_argb_c(should_ahsv, argb, 1);

    if (!(fuzzyEq(ahsv[0], local_ahsv[0]) && fuzzyEq(ahsv[1], local_ahsv[1]) &&
          fuzzyEq(ahsv[2], local_ahsv[2]) && fuzzyEq(ahsv[3], local_ahsv[3]))) {
      printf("ERROR: HSV(%+0.4f %+0.4f %+0.4f) ... HSV(%+0.4f %+0.4f %+0.4f)\n",
        ahsv[1], ahsv[2], ahsv[3],
        local_ahsv[1], local_ahsv[2], local_ahsv[3]);
    }

    if (!(fuzzyEq(argb[0], local_argb[0]) && fuzzyEq(argb[1], local_argb[1]) &&
          fuzzyEq(argb[2], local_argb[2]) && fuzzyEq(argb[3], local_argb[3]))) {
      printf("ERROR: RGB(%+0.4f %+0.4f %+0.4f) ... HSV(%+0.4f %+0.4f %+0.4f)\n"
             "       RGB(%+0.4f %+0.4f %+0.4f) ... ...(%+0.4f %+0.4f %+0.4f)\n\n",
        argb[1], argb[2], argb[3],
        local_ahsv[1], local_ahsv[2], local_ahsv[3],
        local_argb[1], local_argb[2], local_argb[3],
        ahsv[1], ahsv[2], ahsv[3]);
      fflush(stdout);

      memcpy(local_argb, argb, sizeof(float)* 4);
      ahsv_from_argb_sse2(ahsv, argb, 1);
      argb_from_ahsv_sse2(argb, ahsv, 1);
    }

    argb += 4;
    ahsv += 4;
  }
}

int main(int argc, char* argv[])
{
  Timer timer;

  int size = 100000;
  int i;
  float* argb = (float*)malloc(size * 4 * sizeof(float) + 16);
  float* ahsv = (float*)malloc(size * 4 * sizeof(float) + 16);

  argb = (float*)( ((size_t)argb + 15) & -16 );
  ahsv = (float*)( ((size_t)ahsv + 15) & -16 );

  argb_fill(argb, size);
  ahsv_from_argb_c(ahsv, argb, size);
  validate(argb, ahsv, size);

  timer.start();
  for (i = 0; i < 1000; i++) ahsv_from_argb_c(ahsv, argb, size);
  timer.stop();
  printf("AHSV_FROM_ARGB (C)   : %u\n", (unsigned)timer.get());

  timer.start();
  for (i = 0; i < 1000; i++) argb_from_ahsv_c(argb, ahsv, size);
  timer.stop();
  printf("ARGB_FROM_AHSV (C)   : %u\n", (unsigned)timer.get());

  argb_fill(argb, size);

  timer.start();
  for (i = 0; i < 1000; i++) ahsv_from_argb_sse2(ahsv, argb, size);
  timer.stop();
  printf("AHSV_FROM_ARGB (SSE2): %u\n", (unsigned)timer.get());

  timer.start();
  for (i = 0; i < 1000; i++) argb_from_ahsv_sse2(argb, ahsv, size);
  timer.stop();
  printf("ARGB_FROM_AHSV (SSE2): %u\n", (unsigned)timer.get());

  return 0;
}
