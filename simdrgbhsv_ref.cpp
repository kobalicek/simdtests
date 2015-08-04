// [SimdRgbHsv]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Zlib - See LICENSE.md file in the package.

#include "./simdglobals.h"
#include "./simdrgbhsv.h"

// ============================================================================
// [Pure C Implementation]
// ============================================================================

void ahsv_from_argb_ref(float* dst, const float* src, int length) {
  for (int i = length; i; i--, dst += 4, src += 4) {
    float a = src[0];
    float r = src[1];
    float g = src[2];
    float b = src[3];

    float m = SimdUtils::min(r, g, b);
    float v = SimdUtils::max(r, g, b);
    float c = v - m;

    float h = 0.0f;
    float s = 0.0f;
    float x;

    if (c != 0.0f) {
      s = c / v;

      if (v == r) {
        h = g - b;
        x = 1.0f;
      }
      else if (v == g) {
        h = b - r;
        x = 1.0f / 3.0f;
      }
      else {
        h = r - g;
        x = 2.0f / 3.0f;
      }

      h /= 6.0f * c;
      h += x;

      if (h >= 1.0f)
        h -= 1.0f;
    }

    dst[0] = a;
    dst[1] = h;
    dst[2] = s;
    dst[3] = v;
  }
}

void argb_from_ahsv_ref(float* dst, const float* src, int length) {
  for (int i = length; i; i--, dst += 4, src += 4) {
    float a = src[0];
    float h = src[1];
    float s = src[2];
    float v = src[3];

    // The HUE should be at range [0, 1], convert 1.0 to 0.0 if needed.
    if (h >= 1.0f)
      h -= 1.0f;

    dst[0] = a;

    h *= 6.0f;
    int index = static_cast<int>(h);

    float f = h - static_cast<float>(index);
    float p = v * (1.0f - s);
    float q = v * (1.0f - s * f);
    float t = v * (1.0f - s * (1.0f - f));

    switch (index) {
      case 0: dst[1] = v; dst[2] = t; dst[3] = p; break;
      case 1: dst[1] = q; dst[2] = v; dst[3] = p; break;
      case 2: dst[1] = p; dst[2] = v; dst[3] = t; break;
      case 3: dst[1] = p; dst[2] = q; dst[3] = v; break;
      case 4: dst[1] = t; dst[2] = p; dst[3] = v; break;
      case 5: dst[1] = v; dst[2] = p; dst[3] = q; break;
    }
  }
}
