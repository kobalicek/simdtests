// [SimdTests - RGBHSV]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Public Domain <unlicense.org>
#include "../simdglobals.h"
#include "./rgbhsv.h"

// ============================================================================
// [Pure C Implementation]
// ============================================================================

template<typename T>
void SIMD_INLINE ahsv_from_argb_t(float* dst, const float* src, int length) {
  for (int i = length; i; i--, dst += 4, src += 4) {
    T r = static_cast<T>(src[1]);
    T g = static_cast<T>(src[2]);
    T b = static_cast<T>(src[3]);

    T m = SimdUtils::min(r, g, b);
    T v = SimdUtils::max(r, g, b);
    T c = v - m;

    T h = T(0);
    T s = T(0);
    T x;

    if (c != T(0)) {
      s = c / v;

      if (v == r) {
        h = g - b;
        x = T(1);
      }
      else if (v == g) {
        h = b - r;
        x = T(1) / T(3);
      }
      else {
        h = r - g;
        x = T(2) / T(3);
      }

      h /= T(6) * c;
      h += x;

      if (h >= T(1))
        h -= T(1);
    }

    dst[0] = src[0];
    dst[1] = static_cast<float>(h);
    dst[2] = static_cast<float>(s);
    dst[3] = static_cast<float>(v);
  }
}

template<typename T>
void SIMD_INLINE argb_from_ahsv_t(float* dst, const float* src, int length) {
  for (int i = length; i; i--, dst += 4, src += 4) {
    T h = static_cast<T>(src[1]);
    T s = static_cast<T>(src[2]);
    T v = static_cast<T>(src[3]);

    // The HUE should be at range [0, 1], convert 1.0 to 0.0 if needed.
    dst[0] = src[0];
    if (h >= T(1))
      h -= T(1);

    h *= T(6);
    int index = static_cast<int>(h);

    T f = h - static_cast<T>(index);
    T p = v * (T(1) - s);
    T q = v * (T(1) - s * f);
    T t = v * (T(1) - s * (T(1) - f));

    switch (index) {
      case 0: dst[1] = float(v); dst[2] = float(t); dst[3] = float(p); break;
      case 1: dst[1] = float(q); dst[2] = float(v); dst[3] = float(p); break;
      case 2: dst[1] = float(p); dst[2] = float(v); dst[3] = float(t); break;
      case 3: dst[1] = float(p); dst[2] = float(q); dst[3] = float(v); break;
      case 4: dst[1] = float(t); dst[2] = float(p); dst[3] = float(v); break;
      case 5: dst[1] = float(v); dst[2] = float(p); dst[3] = float(q); break;
    }
  }
}

void ahsv_from_argb_ref(float* dst, const float* src, int length) {
  ahsv_from_argb_t<float>(dst, src, length);
}

void argb_from_ahsv_ref(float* dst, const float* src, int length) {
  argb_from_ahsv_t<float>(dst, src, length);
}

void ahsv_from_argb_hq(float* dst, const float* src, int length) {
  ahsv_from_argb_t<double>(dst, src, length);
}

void argb_from_ahsv_hq(float* dst, const float* src, int length) {
  argb_from_ahsv_t<double>(dst, src, length);
}
