// [SimdGlobals]
// Useful definitions for making SIMD implementation benchmarks.
//
// [License]
// Public Domain <unlicense.org>
#ifndef _SIMDGLOBALS_H
#define _SIMDGLOBALS_H

#include <math.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(_WIN32)
# include <windows.h>
#else
# include <sys/time.h>
#endif

// ============================================================================
// [Instruction Sets]
// ============================================================================

#if defined(USE_SSE)
# include <xmmintrin.h>
#endif // USE_SSE

#if defined(USE_SSE2)
# include <emmintrin.h>
#endif // USE_SSE2

#if defined(USE_SSSE3)
# include <tmmintrin.h>
#endif // USE_SSE3

// ============================================================================
// [Port]
// ============================================================================

#if defined(_MSC_VER)
# define SIMD_UINT64_C(x) x##ui64
#else
# define SIMD_UINT64_C(x) x##ull
#endif

#if defined(_MSC_VER)
# define SIMD_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
# define SIMD_INLINE inline __attribute__((__always_inline__))
#else
# define SIMD_INLINE inline
#endif

#if defined(_MSC_VER)
#define SIMD_ALIGN_VAR(type, name, alignment) \
  __declspec(align(alignment)) type name
#else
#define SIMD_ALIGN_VAR(type, name, alignment) \
  type __attribute__((__aligned__(alignment))) name
#endif // _MSC_VER

#define SIMD_CONST_PS(name, val0, val1, val2, val3) \
  SIMD_ALIGN_VAR(static const float, _xmm_const_##name[4], 16) = { \
    static_cast<float>(val3), \
    static_cast<float>(val2), \
    static_cast<float>(val1), \
    static_cast<float>(val0)  \
  }

#define SIMD_CONST_PI(name, val0, val1, val2, val3) \
  SIMD_ALIGN_VAR(static const int, _xmm_const_##name[4], 16) = { \
    static_cast<int>(val3), \
    static_cast<int>(val2), \
    static_cast<int>(val1), \
    static_cast<int>(val0)  \
  }

#define SIMD_GET_SS(name) (*(const  float  *)_xmm_const_##name)
#define SIMD_GET_PS(name) (*(const __m128  *)_xmm_const_##name)
#define SIMD_GET_SI(name) (*(const int     *)_xmm_const_##name)
#define SIMD_GET_PI(name) (*(const __m128i *)_xmm_const_##name)

// Shuffle floats in `src` by using SSE2 `pshufd` instead of `shufps`, if possible.
#define SIMD_SHUFFLE_PS(src, imm) \
  _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(src), imm))

#if defined(abs)
# undef abs
#endif

#if defined(min)
# undef min
#endif

#if defined(max)
# undef max
#endif

// ============================================================================
// [SimdUtils]
// ============================================================================

namespace SimdUtils {
  template<typename T>
  static SIMD_INLINE T abs(T a) { return a >= 0 ? a : -a; }

  template<typename T>
  static SIMD_INLINE T max(T a, T b) { return a > b ? a : b; }
  template<typename T>
  static SIMD_INLINE T max(T a, T b, T c) { return max<T>(max<T>(a, b), c); }

  template<typename T>
  static SIMD_INLINE T min(T a, T b) { return a < b ? a : b; }
  template<typename T>
  static SIMD_INLINE T min(T a, T b, T c) { return min<T>(min<T>(a, b), c); }

  template<typename T>
  static SIMD_INLINE bool fuzzyEq(T a, T b, T epsilon = T(1e-8)) { return abs<T>(a - b) < epsilon; }

  template<typename T>
  static SIMD_INLINE T align(T p, uint32_t alignment) {
    uint32_t mask = alignment - 1;
    return (T)( ((uintptr_t)p + mask) & ~static_cast<uintptr_t>(mask) );
  }

  template<typename T>
  static SIMD_INLINE bool isAligned(T p, uint32_t alignment) {
    uint32_t mask = alignment - 1;
    return ((uintptr_t)p & static_cast<uintptr_t>(mask)) == 0;
  }

  template<typename T>
  static SIMD_INLINE uint32_t alignDiff(T p, uint32_t alignment) {
    uint32_t mask = alignment - 1;
    return (alignment - static_cast<uint32_t>((uintptr_t)p & mask)) & mask;
  }
};

// ============================================================================
// [SimdTimer]
// ============================================================================

struct SimdTimer {
  SIMD_INLINE SimdTimer() : _cnt(0) {}

  SIMD_INLINE uint32_t get() const { return _cnt; }
  SIMD_INLINE void start() { _cnt = now(); }
  SIMD_INLINE void stop() { _cnt = now() - _cnt; }

  static SIMD_INLINE uint32_t now() {
#if defined(_WIN32)
    return GetTickCount();
#else
    timeval ts;
    gettimeofday(&ts,0);
    return (uint32_t)(ts.tv_sec * 1000 + (ts.tv_usec / 1000));
#endif
  }

  uint32_t _cnt;
};

// ============================================================================
// [SimdRandom]
// ============================================================================

//! Simple PRNG.
struct SimdRandom {
  // --------------------------------------------------------------------------
  // [Construction / Destruction]
  // --------------------------------------------------------------------------

  SIMD_INLINE SimdRandom(uint64_t seed = 0) { reset(seed); }

  // The constants used are the constants suggested as `23/18/5`.
  enum {
    kStep1 = 23,
    kStep2 = 18,
    kStep3 = 5
  };

  //! Reset the PRNG and initialize it to the given `seed`.
  void reset(uint64_t seed = 0) {
    // The number is arbitrary, it means nothing.
    static const uint64_t kZeroSeed = SIMD_UINT64_C(0x1F0A2BE71D163FA0);

    // Generate the state data by using splitmix64.
    for (uint32_t i = 0; i < 2; i++) {
      uint64_t x = (seed += SIMD_UINT64_C(0x9E3779B97F4A7C15));
      x = (x ^ (x >> 30)) * SIMD_UINT64_C(0xBF58476D1CE4E5B9);
      x = (x ^ (x >> 27)) * SIMD_UINT64_C(0x94D049BB133111EB);
      x = (x ^ (x >> 31));
      _state[i] = x != 0 ? x : kZeroSeed;
    }
  }

  SIMD_INLINE uint64_t nextUInt64() {
    uint64_t x = _state[0];
    uint64_t y = _state[1];

    x ^= x << kStep1;
    y ^= y >> kStep3;

    x ^= x >> kStep2;
    x ^= y;

    _state[0] = y;
    _state[1] = x;

    return x + y;
  }

  SIMD_INLINE uint32_t nextUInt32() {
    return static_cast<uint32_t>(nextUInt64() >> 32);
  }

  // --------------------------------------------------------------------------
  // [Members]
  // --------------------------------------------------------------------------

  uint64_t _state[2];
};

#endif // _SIMDGLOBALS_H
