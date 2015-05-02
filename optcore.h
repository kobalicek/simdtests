#ifndef _OPTCORE_H
#define _OPTCORE_H

// ============================================================================
// [Dependencies]
// ============================================================================

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#include <windows.h>
#else
#include <sys/time.h>
#endif 

// ============================================================================
// [Instructions]
// ============================================================================

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

// ============================================================================
// [Portability]
// ============================================================================

#if defined(_MSC_VER)
# define OPT_INLINE __forceinline
#else
# define OPT_INLINE inline
#endif

// ============================================================================
// [Helpers]
// ============================================================================

static OPT_INLINE float maxf(float a, float b) { return a > b ? a : b; }
static OPT_INLINE float minf(float a, float b) { return a < b ? a : b; }

static OPT_INLINE double maxf(double a, double b) { return a > b ? a : b; }
static OPT_INLINE double minf(double a, double b) { return a < b ? a : b; }

static OPT_INLINE bool fuzzyEq(float a, float b) { return (fabs(a - b) < 1e-6f); }

// ============================================================================
// [OptUtils]
// ============================================================================

#if defined(_MSC_VER)
#define XMM_ALIGN_VAR(_Type_, _Name_) \
  __declspec(align(16)) _Type_ _Name_
#else
#define XMM_ALIGN_VAR(_Type_, _Name_) \
  _Type_ __attribute__((aligned(16))) _Name_
#endif // _MSC_VER

#define XMM_CONST_PS(name, val0, val1, val2, val3) \
  XMM_ALIGN_VAR(static const float, _xmm_const_##name[4]) = { val3, val2, val1, val0 }

#define XMM_CONST_PI(name, val0, val1, val2, val3) \
  XMM_ALIGN_VAR(static const int, _xmm_const_##name[4]) = { val3, val2, val1, val0 }

#define XMM_GET_SS(name) (*(const  float  *)_xmm_const_##name)
#define XMM_GET_PS(name) (*(const __m128  *)_xmm_const_##name)
#define XMM_GET_SI(name) (*(const int     *)_xmm_const_##name)
#define XMM_GET_PI(name) (*(const __m128i *)_xmm_const_##name)

// Shuffle Float4 by using SSE2 instruction.
#define XMM_SHUFFLE_PS(src, imm) \
  _mm_castsi128_ps(_mm_shuffle_epi32(_mm_castps_si128(src), imm))

static __m128 OPT_INLINE _mm_load_ps_my(const void* mem) { return _mm_load_ps((const float*)mem); }
static void OPT_INLINE _mm_store_ps_my(void* mem, __m128 reg) { _mm_store_ps((float*)mem, reg); }

// ============================================================================
// [OptTimer]
// ============================================================================

struct OptTimer {
  OPT_INLINE OptTimer() : cnt(0) {}
  OPT_INLINE unsigned int get() const { return cnt; }
  OPT_INLINE void start() { cnt = sGetTime(); }
  OPT_INLINE void stop() { cnt = sGetTime() - cnt; }

  static OPT_INLINE unsigned int sGetTime() {
#if defined(_WIN32)
    return GetTickCount();
#else
    timeval ts;
    gettimeofday(&ts,0);
    return (unsigned int)(ts.tv_sec * 1000 + (ts.tv_usec / 1000));
#endif
  }

  unsigned int cnt;
};

#endif // _OPTCORE_H
