#ifndef _UTIL_H
#define _UTIL_H

#include <math.h>
#include <stdlib.h>

// SSE.
#include <xmmintrin.h>

// SSE2.
#if defined(USE_SSE2)
#include <emmintrin.h>
#endif // USE_SSE2

// SSSE3.
#if defined(USE_SSSE3)
#include <tmmintrin.h>
#endif // USE_SSE3

#if defined(_MSC_VER)
# define INLINE __forceinline
#else
# define INLINE inline
#endif

#if defined(_MSC_VER)
#include <windows.h>
#else
#include <sys/time.h>
#endif 

// ============================================================================
// [Inlines]
// ============================================================================

static INLINE float maxf(float a, float b) { return a > b ? a : b; }
static INLINE float minf(float a, float b) { return a < b ? a : b; }

static INLINE double maxf(double a, double b) { return a > b ? a : b; }
static INLINE double minf(double a, double b) { return a < b ? a : b; }

static inline bool fuzzyEq(float a, float b) { return (fabs(a - b) < 1e-6f); }

// ============================================================================
// [SSE / SSE2]
// ============================================================================

#if defined(_MSC_VER)
#define XMM_ALIGNED_VAR(_Type_, _Name_) \
  __declspec(align(16)) _Type_ _Name_
#else
#define XMM_ALIGNED_VAR(_Type_, _Name_) \
  _Type_ __attribute__((aligned(16))) _Name_
#endif // _MSC_VER

#define XMM_CONSTANT_PS(name, val0, val1, val2, val3) \
  XMM_ALIGNED_VAR(static const float, _xmm_const_##name[4]) = { val3, val2, val1, val0 }

#define XMM_CONSTANT_PI(name, val0, val1, val2, val3) \
  XMM_ALIGNED_VAR(static const int, _xmm_const_##name[4]) = { val3, val2, val1, val0 }

#define XMM_GET_SS(name) (*(const  float  *)_xmm_const_##name)
#define XMM_GET_PS(name) (*(const __m128  *)_xmm_const_##name)
#define XMM_GET_SI(name) (*(const int     *)_xmm_const_##name)
#define XMM_GET_PI(name) (*(const __m128i *)_xmm_const_##name)

// Shuffle Float4 by using SSE2 instruction.
#define XMM_SHUFFLE_PS(src, imm) \
  _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(src), imm))

static __m128 INLINE _mm_load_ps_my(const void* mem) { return _mm_load_ps((const float*)mem); }
static void INLINE _mm_store_ps_my(void* mem, __m128 reg) { _mm_store_ps((float*)mem, reg); }

// ============================================================================
// [Timer]
// ============================================================================

struct Timer {
  INLINE Timer() : cnt(0) {}
  INLINE unsigned int get() const { return cnt; }
  INLINE void start() { cnt = sGetTime(); }
  INLINE void stop() { cnt = sGetTime() - cnt; }

  static INLINE unsigned int sGetTime() {
#if defined(_MSC_VER)
    return GetTickCount();
#else
    timeval ts;
    gettimeofday(&ts,0);
    return (unsigned int)(ts.tv_sec * 1000 + (ts.tv_usec / 1000));
#endif
  }

  unsigned int cnt;
};

#endif // _UTIL_H
