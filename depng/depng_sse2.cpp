// [SimdTests - DePNG]
// SIMD optimized "PNG Reverse Filter" implementation.
//
// [License]
// Public Domain <unlicense.org>
#define USE_SSE2

#include "../simdglobals.h"
#include "./depng.h"

// ============================================================================
// [SimdTests::DePNG - Filter - SSE2]
// ============================================================================

#define DEPNG_SSE2_SLL_ADDB_1X(P0, T0, Shift) \
  do { \
    T0 = _mm_slli_si128(P0, Shift); \
    P0 = _mm_add_epi8(P0, T0); \
  } while (0)

#define DEPNG_SSE2_SLL_ADDB_2X(P0, T0, P1, T1, Shift) \
  do { \
    T0 = _mm_slli_si128(P0, Shift); \
    T1 = _mm_slli_si128(P1, Shift); \
    P0 = _mm_add_epi8(P0, T0); \
    P1 = _mm_add_epi8(P1, T1); \
  } while (0)

#define DEPNG_SSE2_PAETH(Dst, A, B, C) \
  do { \
    __m128i MinAB = _mm_min_epi16(A, B); \
    __m128i MaxAB = _mm_max_epi16(A, B); \
    __m128i DivAB = _mm_mulhi_epu16(_mm_sub_epi16(MaxAB, MinAB), rcp3); \
    \
    MinAB = _mm_sub_epi16(MinAB, C); \
    MaxAB = _mm_sub_epi16(MaxAB, C); \
    \
    Dst = _mm_add_epi16(C  , _mm_andnot_si128(_mm_srai_epi16(_mm_add_epi16(DivAB, MinAB), 15), MaxAB)); \
    Dst = _mm_add_epi16(Dst, _mm_andnot_si128(_mm_srai_epi16(_mm_sub_epi16(DivAB, MaxAB), 15), MinAB)); \
  } while (0)

template<uint32_t bpp>
static SIMD_INLINE void depng_filter_sse2_template(uint8_t* p, uint32_t h, uint32_t bpl) {
  uint32_t y = h;
  uint8_t* u = NULL;

  // Subtract one BYTE that is used to store the `filter` ID.
  bpl--;

  do {
    uint32_t i;
    uint32_t filter = *p++;

    switch (filter) {
      // ----------------------------------------------------------------------
      // [None]
      // ----------------------------------------------------------------------

      case kPngFilterNone:
        p += bpl;
        break;

      // ----------------------------------------------------------------------
      // [Sub]
      // ----------------------------------------------------------------------

      // This is one of the easiest filters to parallelize. Although it looks
      // like the data dependency is too high, it's simply additions, which are
      // really easy to parallelize. The following formula:
      //
      //     Y1' = BYTE(Y1 + Y0')
      //     Y2' = BYTE(Y2 + Y1')
      //     Y3' = BYTE(Y3 + Y2')
      //     Y4' = BYTE(Y4 + Y3')
      //
      // Expanded to (with byte casts removed, as they are implicit in our case):
      //
      //     Y1' = Y1 + Y0'
      //     Y2' = Y2 + Y1 + Y0'
      //     Y3' = Y3 + Y2 + Y1 + Y0'
      //     Y4' = Y4 + Y3 + Y2 + Y1 + Y0'
      //
      // Can be implemented like this by taking advantage of SIMD:
      //
      //     +-----------+-----------+-----------+-----------+----->
      //     |    Y1     |    Y2     |    Y3     |    Y4     | ...
      //     +-----------+-----------+-----------+-----------+----->
      //                   Shift by 1 and PADDB
      //     +-----------+-----------+-----------+-----------+
      //     |           |    Y1     |    Y2     |    Y3     | ----+
      //     +-----------+-----------+-----------+-----------+     |
      //                                                           |
      //     +-----------+-----------+-----------+-----------+     |
      //     |    Y1     |   Y1+Y2   |   Y2+Y3   |   Y3+Y4   | <---+
      //     +-----------+-----------+-----------+-----------+
      //                   Shift by 2 and PADDB
      //     +-----------+-----------+-----------+-----------+
      //     |           |           |    Y1     |   Y1+Y2   | ----+
      //     +-----------+-----------+-----------+-----------+     |
      //                                                           |
      //     +-----------+-----------+-----------+-----------+     |
      //     |    Y1     |   Y1+Y2   | Y1+Y2+Y3  |Y1+Y2+Y3+Y4| <---+
      //     +-----------+-----------+-----------+-----------+
      //
      // The size of the register doesn't matter here. The Y0' dependency has
      // been omitted to make the flow cleaner, however, it can be added to Y1
      // before processing or it can be shifted to the first cell so the first
      // addition would be performed against [Y0', Y1, Y2, Y3].

      case kPngFilterSub: {
        i = bpl - bpp;

        if (i >= 32) {
          // Align to 16-BYTE boundary.
          uint32_t j = SimdUtils::alignDiff(p + bpp, 16);
          for (i -= j; j != 0; j--, p++)
            p[bpp] = depng_sum(p[bpp], p[0]);

          if (bpp == 1) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t2;

            // Process 64 BYTEs at a time.
            p0 = _mm_cvtsi32_si128(p[0]);
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 1));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 17));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 33));
              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 49));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 1);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 2);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 1), p0);

              p0 = _mm_srli_si128(p0, 15);
              t2 = _mm_srli_si128(p2, 15);
              p1 = _mm_add_epi8(p1, p0);
              p3 = _mm_add_epi8(p3, t2);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 1);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 2);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 17), p1);

              p1 = _mm_unpackhi_epi8(p1, p1);
              p1 = _mm_unpackhi_epi16(p1, p1);
              p1 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 3, 3, 3));

              p2 = _mm_add_epi8(p2, p1);
              p3 = _mm_add_epi8(p3, p1);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 33), p2);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 49), p3);
              p0 = _mm_srli_si128(p3, 15);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 1));

              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 1);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 2);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 4);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 8);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 1), p0);
              p0 = _mm_srli_si128(p0, 15);

              p += 16;
              i -= 16;
            }
          }
          else if (bpp == 2) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t2;

            // Process 64 BYTEs at a time.
            p0 = _mm_cvtsi32_si128(reinterpret_cast<uint16_t*>(p)[0]);
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 2));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 18));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 34));
              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 50));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 2);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 2), p0);

              p0 = _mm_srli_si128(p0, 14);
              t2 = _mm_srli_si128(p2, 14);
              p1 = _mm_add_epi8(p1, p0);
              p3 = _mm_add_epi8(p3, t2);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 2);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 18), p1);

              p1 = _mm_unpackhi_epi16(p1, p1);
              p1 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 3, 3, 3));

              p2 = _mm_add_epi8(p2, p1);
              p3 = _mm_add_epi8(p3, p1);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 34), p2);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 50), p3);
              p0 = _mm_srli_si128(p3, 14);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 2));
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 2);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 4);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 8);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 2), p0);
              p0 = _mm_srli_si128(p0, 14);

              p += 16;
              i -= 16;
            }
          }
          else if (bpp == 3) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t2;
            __m128i ext3b = _mm_set1_epi32(0x01000001);

            // Process 64 BYTEs at a time.
            p0 = _mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(p)[0] & 0x00FFFFFFU);
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 3));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 19));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 35));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 3);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 6);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t2, 12);

              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 51));
              t0 = _mm_srli_si128(p0, 13);
              t2 = _mm_srli_si128(p2, 13);

              p1 = _mm_add_epi8(p1, t0);
              p3 = _mm_add_epi8(p3, t2);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 3);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 6);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t2, 12);
              _mm_store_si128(reinterpret_cast<__m128i*>(p +  3), p0);

              p0 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 3, 3, 3));
              p0 = _mm_srli_epi32(p0, 8);
              p0 = _mm_mul_epu32(p0, ext3b);

              p0 = _mm_shufflelo_epi16(p0, _MM_SHUFFLE(0, 2, 1, 0));
              p0 = _mm_shufflehi_epi16(p0, _MM_SHUFFLE(1, 0, 2, 1));

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 19), p1);
              p2 = _mm_add_epi8(p2, p0);
              p0 = _mm_shuffle_epi32(p0, _MM_SHUFFLE(1, 3, 2, 1));

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 35), p2);
              p0 = _mm_add_epi8(p0, p3);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 51), p0);
              p0 = _mm_srli_si128(p0, 13);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 3));

              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 3);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 6);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 12);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 3), p0);
              p0 = _mm_srli_si128(p0, 13);

              p += 16;
              i -= 16;
            }
          }
          else if (bpp == 4) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t1, t2;

            // Process 64 BYTEs at a time.
            p0 = _mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(p)[0]);
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 4));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 20));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 36));
              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 52));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t1, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t1, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 4), p0);

              p0 = _mm_srli_si128(p0, 12);
              t2 = _mm_srli_si128(p2, 12);

              p1 = _mm_add_epi8(p1, p0);
              p3 = _mm_add_epi8(p3, t2);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t1, 4);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t1, 8);

              p0 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 3, 3, 3));
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 20), p1);

              p2 = _mm_add_epi8(p2, p0);
              p0 = _mm_add_epi8(p0, p3);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 36), p2);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 52), p0);
              p0 = _mm_srli_si128(p0, 12);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 4));

              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 4);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 4), p0);
              p0 = _mm_srli_si128(p0, 12);

              p += 16;
              i -= 16;
            }
          }
          else if (bpp == 6) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t1;

            p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p));
            p0 = _mm_slli_epi64(p0, 16);
            p0 = _mm_srli_epi64(p0, 16);

            // Process 64 BYTEs at a time.
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 6));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 22));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 38));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t1, 6);
              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t1, 12);

              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 54));
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 6), p0);

              p0 = _mm_srli_si128(p0, 10);
              t1 = _mm_srli_si128(p2, 10);

              p1 = _mm_add_epi8(p1, p0);
              p3 = _mm_add_epi8(p3, t1);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t1, 6);
              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t1, 12);
              p0 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 2, 3, 2));

              p0 = _mm_shufflelo_epi16(p0, _MM_SHUFFLE(1, 3, 2, 1));
              p0 = _mm_shufflehi_epi16(p0, _MM_SHUFFLE(2, 1, 3, 2));

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 22), p1);
              p2 = _mm_add_epi8(p2, p0);
              p0 = _mm_shuffle_epi32(p0, _MM_SHUFFLE(1, 3, 2, 1));

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 38), p2);
              p0 = _mm_add_epi8(p0, p3);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 54), p0);
              p0 = _mm_srli_si128(p0, 10);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 6));

              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 6);
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 12);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 6), p0);
              p0 = _mm_srli_si128(p0, 10);

              p += 16;
              i -= 16;
            }
          }
          else if (bpp == 8) {
            __m128i p0, p1, p2, p3;
            __m128i t0, t1, t2;

            // Process 64 BYTEs at a time.
            p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p));
            while (i >= 64) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 8));
              p1 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 24));
              p2 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 40));
              p3 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 56));

              DEPNG_SSE2_SLL_ADDB_2X(p0, t0, p2, t1, 8);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 8), p0);

              p0 = _mm_srli_si128(p0, 8);
              t2 = _mm_shuffle_epi32(p2, _MM_SHUFFLE(3, 2, 3, 2));
              p1 = _mm_add_epi8(p1, p0);

              DEPNG_SSE2_SLL_ADDB_2X(p1, t0, p3, t1, 8);
              p0 = _mm_shuffle_epi32(p1, _MM_SHUFFLE(3, 2, 3, 2));
              p3 = _mm_add_epi8(p3, t2);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 24), p1);

              p2 = _mm_add_epi8(p2, p0);
              p0 = _mm_add_epi8(p0, p3);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 40), p2);
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 56), p0);
              p0 = _mm_srli_si128(p0, 8);

              p += 64;
              i -= 64;
            }

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              p0 = _mm_add_epi8(p0, *reinterpret_cast<__m128i*>(p + 8));
              DEPNG_SSE2_SLL_ADDB_1X(p0, t0, 8);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 8), p0);
              p0 = _mm_srli_si128(p0, 8);

              p += 16;
              i -= 16;
            }
          }
        }

        for (; i != 0; i--, p++)
          p[bpp] = depng_sum(p[bpp], p[0]);

        p += bpp;
        break;
      }

      // ----------------------------------------------------------------------
      // [Up]
      // ----------------------------------------------------------------------

      // This is actually the easiest filter that doesn't require any kind of
      // specialization for a particular BPP. Even C++ compiler like GCC is
      // able to parallelize a naive implementation. However, MSC compiler does
      // not parallelize the naive implementation so the SSE2 implementation
      // provided greatly boosted the performance on Windows.
      //
      //     +-----------+-----------+-----------+-----------+
      //     |    Y1     |    Y2     |    Y3     |    Y4     |
      //     +-----------+-----------+-----------+-----------+
      //                           PADDB
      //     +-----------+-----------+-----------+-----------+
      //     |    U1     |    U2     |    U3     |    U4     | ----+
      //     +-----------+-----------+-----------+-----------+     |
      //                                                           |
      //     +-----------+-----------+-----------+-----------+     |
      //     |   Y1+U1   |   Y2+U2   |   Y3+U3   |   Y4+U4   | <---+
      //     +-----------+-----------+-----------+-----------+

      case kPngFilterUp: {
        i = bpl;

        if (i >= 24) {
          // Align to 16-BYTE boundary.
          uint32_t j = SimdUtils::alignDiff(p, 16);
          for (i -= j; j != 0; j--, p++, u++)
            p[0] = depng_sum(p[0], u[0]);

          // Process 64 BYTEs at a time.
          while (i >= 64) {
            __m128i u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u));
            __m128i u1 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 16));

            __m128i p0 = _mm_load_si128 (reinterpret_cast<__m128i*>(p));
            __m128i p1 = _mm_load_si128 (reinterpret_cast<__m128i*>(p + 16));

            __m128i u2 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 32));
            __m128i u3 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 48));

            p0 = _mm_add_epi8(p0, u0);
            p1 = _mm_add_epi8(p1, u1);

            __m128i p2 = _mm_load_si128 (reinterpret_cast<__m128i*>(p + 32));
            __m128i p3 = _mm_load_si128 (reinterpret_cast<__m128i*>(p + 48));

            p2 = _mm_add_epi8(p2, u2);
            p3 = _mm_add_epi8(p3, u3);

            _mm_store_si128(reinterpret_cast<__m128i*>(p     ), p0);
            _mm_store_si128(reinterpret_cast<__m128i*>(p + 16), p1);
            _mm_store_si128(reinterpret_cast<__m128i*>(p + 32), p2);
            _mm_store_si128(reinterpret_cast<__m128i*>(p + 48), p3);

            p += 64;
            u += 64;
            i -= 64;
          }

          // Process 8 BYTEs at a time.
          while (i >= 8) {
            __m128i u0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(u));
            __m128i p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p));

            p0 = _mm_add_epi8(p0, u0);
            _mm_storel_epi64(reinterpret_cast<__m128i*>(p), p0);

            p += 8;
            u += 8;
            i -= 8;
          }
        }

        for (; i != 0; i--, p++, u++)
          p[0] = depng_sum(p[0], u[0]);
        break;
      }

      // ----------------------------------------------------------------------
      // [Avg]
      // ----------------------------------------------------------------------

      // This filter is extremely difficult for low BPP values as there is
      // a huge sequential data dependency, I didn't succeed to solve it.
      // 1-3 BPP implementations are pretty bad and I would like to hear about
      // a way to improve those. The implementation for 4 BPP and more is
      // pretty good, as these is less data dependency between individual bytes.
      //
      // Sequental Approach:
      //
      //     Y1' = byte((2*Y1 + U1 + Y0') >> 1)
      //     Y2' = byte((2*Y2 + U2 + Y1') >> 1)
      //     Y3' = byte((2*Y3 + U3 + Y2') >> 1)
      //     Y4' = byte((2*Y4 + U4 + Y3') >> 1)
      //     Y5' = ...
      //
      // Expanded, `U1 + Y0'` replaced with `U1`:
      //
      //     Y1' = byte((2*Y1 + U1) >> 1)
      //     Y2' = byte((2*Y2 + U2 +
      //           byte((2*Y1 + U1) >> 1)) >> 1)
      //     Y3' = byte((2*Y3 + U3 +
      //           byte((2*Y2 + U2 +
      //           byte((2*Y1 + U1) >> 1)) >> 1)) >> 1)
      //     Y4' = byte((2*Y4 + U4 +
      //           byte((2*Y3 + U3 +
      //           byte((2*Y2 + U2 +
      //           byte((2*Y1 + U1) >> 1)) >> 1)) >> 1)) >> 1)
      //     Y5' = ...

      case kPngFilterAvg: {
        for (i = 0; i < bpp; i++)
          p[i] = depng_sum(p[i], u[i] >> 1);

        i = bpl - bpp;
        u += bpp;

        if (i >= 32) {
          // Align to 16-BYTE boundary.
          uint32_t j = SimdUtils::alignDiff(p + bpp, 16);
          __m128i zero = _mm_setzero_si128();

          for (i -= j; j != 0; j--, p++, u++)
            p[bpp] = depng_sum(p[bpp], depng_avg(p[0], u[0]));

          if (bpp == 1) {
            // This is one of the most difficult AVG filters. 1-BPP has a huge
            // sequential dependency, which is nearly impossible to parallelize.
            // The code below is the best I could have written, it's a mixture
            // of C++ and SIMD. Maybe using a pure C would be even better than
            // this code, but, I tried to take advantage of 8 BYTE fetches at
            // least. Unrolling the loop any further doesn't lead to an
            // improvement.
            //
            // I know that the code looks terrible, but it's a bit faster than
            // a pure specialized C++ implementation I used to have before.
            uint32_t t0 = p[0];
            uint32_t t1;

            // Process 8 BYTEs at a time.
            while (i >= 8) {
              __m128i p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p + 1));
              __m128i u0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(u));

              p0 = _mm_unpacklo_epi8(p0, zero);
              u0 = _mm_unpacklo_epi8(u0, zero);

              p0 = _mm_slli_epi16(p0, 1);
              p0 = _mm_add_epi16(p0, u0);

              t1 = _mm_cvtsi128_si32(p0);
              p0 = _mm_srli_si128(p0, 4);
              t0 = ((t0 + t1) >> 1) & 0xFF; t1 >>= 16;
              p[1] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF;
              t1 = _mm_cvtsi128_si32(p0);
              p0 = _mm_srli_si128(p0, 4);
              p[2] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF; t1 >>= 16;
              p[3] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF;
              t1 = _mm_cvtsi128_si32(p0);
              p0 = _mm_srli_si128(p0, 4);
              p[4] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF; t1 >>= 16;
              p[5] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF;
              t1 = _mm_cvtsi128_si32(p0);
              p[6] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF; t1 >>= 16;
              p[7] = static_cast<uint8_t>(t0);

              t0 = ((t0 + t1) >> 1) & 0xFF;
              p[8] = static_cast<uint8_t>(t0);

              p += 8;
              u += 8;
              i -= 8;
            }
          }
          // TODO: Not complete / Not working.
          /*
          else if (bpp == 2) {
            // Process 16 BYTEs at a time.
            // __m128i shf = _mm_setr_epi32(0x80000000, 0, 0, 0);
            __m128i msk = _mm_set1_epi16(0x01FE);
            __m128i t0 = _mm_unpacklo_epi8(
              _mm_cvtsi32_si128(reinterpret_cast<uint16_t*>(p)[0]), zero);

            __m128i scale = _mm_setr_epi32(0x00000000, 0x00000000, 0x00000000, 0x00000000);

            // Sequental Approach:
            //   y1' = byte{(256y1 + 128x1 + 128y0') / 256}
            //   y2' = byte{(256y2 + 128x2 + 128y1') / 256}
            //   y3' = byte{(256y3 + 128x3 + 128y2') / 256}
            //   y4' = byte{(256y4 + 128x4 + 128y3') / 256}
            //   y5' = ...
            while (i >= 8) {
              __m128i p0, p2, p3;

              p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p + 2));
              p2 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(u));

              p0 = _mm_unpacklo_epi8(p0, zero);
              p2 = _mm_unpacklo_epi8(p2, zero);

              p0 = _mm_slli_epi16(p0, 1);
              p0 = _mm_add_epi16(p0, p2);
              p0 = _mm_add_epi16(p0, t0);
              p0 = _mm_and_si128(p0, msk);

              // P0/P2
              p2 = _mm_shuffle_epi32(p0, _MM_SHUFFLE(1, 2, 3, 0));
              p0 = _mm_slli_si128(p0, 12);
              p2 = _mm_slli_epi16(p2, 1);

              //p0 = _mm_srli_si128(p0, 8); // [Z Z 2 0]

              //p2 = _mm_srli_epi64(p2, 32); // [Z 3 Z 1]
              //p0 = _mm_srli_epi64(p0, 32); // [Z 2 Z 0]

              p3 = _mm_shuffle_epi32(p0, _MM_SHUFFLE(1, 1, 1, 0));
              p3 = _mm_srli_epi32(p3, 1);
              p3 = _mm_and_si128(p3, msk);

              p += 8;
              u += 8;
              i -= 8;
            }
          }
          else if (bpp == 3) {
          }
          */
          else if (bpp == 4) {
            __m128i m00FF = _mm_set1_epi16(0x00FF);
            __m128i m01FF = _mm_set1_epi16(0x01FF);

            __m128i t1 = _mm_unpacklo_epi8(
              _mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(p)[0]), zero);

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              __m128i p0, p1;
              __m128i u0, u1;

              p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 4));
              u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u));

              p1 = p0;                          // HI | Move Ln
              p0 = _mm_unpacklo_epi8(p0, zero); // LO | Unpack Ln

              u1 = u0;                          // HI | Move Up
              p0 = _mm_slli_epi16(p0, 1);       // LO | << 1

              u0 = _mm_unpacklo_epi8(u0, zero); // LO | Unpack Up
              p0 = _mm_add_epi16(p0, t1);       // LO | Add Last

              p1 = _mm_unpackhi_epi8(p1, zero); // HI | Unpack Ln
              p0 = _mm_add_epi16(p0, u0);       // LO | Add Up
              p0 = _mm_and_si128(p0, m01FF);    // LO | & 0x01FE

              u1 = _mm_unpackhi_epi8(u1, zero); // HI | Unpack Up
              t1 = _mm_slli_si128(p0, 8);       // LO | Get Last
              p0 = _mm_slli_epi16(p0, 1);       // LO | << 1

              p1 = _mm_slli_epi16(p1, 1);       // HI | << 1
              p0 = _mm_add_epi16(p0, t1);       // LO | Add Last
              p0 = _mm_srli_epi16(p0, 2);       // LO | >> 2

              p1 = _mm_add_epi16(p1, u1);       // HI | Add Up
              p0 = _mm_and_si128(p0, m00FF);    // LO | & 0x00FF
              t1 = _mm_srli_si128(p0, 8);       // LO | Get Last

              p1 = _mm_add_epi16(p1, t1);       // HI | Add Last
              p1 = _mm_and_si128(p1, m01FF);    // HI | & 0x01FE

              t1 = _mm_slli_si128(p1, 8);       // HI | Get Last
              p1 = _mm_slli_epi16(p1, 1);       // HI | << 1

              t1 = _mm_add_epi16(t1, p1);       // HI | Add Last
              t1 = _mm_srli_epi16(t1, 2);       // HI | >> 2
              t1 = _mm_and_si128(t1, m00FF);    // HI | & 0x00FF

              p0 = _mm_packus_epi16(p0, t1);
              t1 = _mm_srli_si128(t1, 8);       // HI | Get Last
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 4), p0);

              p += 16;
              u += 16;
              i -= 16;
            }
          }
          else if (bpp == 6) {
            __m128i t1 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p));

            // Process 16 BYTEs at a time.
            while (i >= 16) {
              __m128i p0, p1, p2;
              __m128i u0, u1, u2;

              u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u));
              t1 = _mm_unpacklo_epi8(t1, zero);
              p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 6));

              p1 = _mm_srli_si128(p0, 6);       // P1 | Extract
              u1 = _mm_srli_si128(u0, 6);       // P1 | Extract

              p2 = _mm_srli_si128(p0, 12);      // P2 | Extract
              u2 = _mm_srli_si128(u0, 12);      // P2 | Extract

              p0 = _mm_unpacklo_epi8(p0, zero); // P0 | Unpack
              u0 = _mm_unpacklo_epi8(u0, zero); // P0 | Unpack

              p1 = _mm_unpacklo_epi8(p1, zero); // P1 | Unpack
              u1 = _mm_unpacklo_epi8(u1, zero); // P1 | Unpack

              p2 = _mm_unpacklo_epi8(p2, zero); // P2 | Unpack
              u2 = _mm_unpacklo_epi8(u2, zero); // P2 | Unpack

              u0 = _mm_add_epi16(u0, t1);       // P0 | Add Last
              u0 = _mm_srli_epi16(u0, 1);       // P0 | >> 1
              p0 = _mm_add_epi8(p0, u0);        // P0 | Add (Up+Last)/2

              u1 = _mm_add_epi16(u1, p0);       // P1 | Add P0
              u1 = _mm_srli_epi16(u1, 1);       // P1 | >> 1
              p1 = _mm_add_epi8(p1, u1);        // P1 | Add (Up+Last)/2

              u2 = _mm_add_epi16(u2, p1);       // P2 | Add P1
              u2 = _mm_srli_epi16(u2, 1);       // P2 | >> 1
              p2 = _mm_add_epi8(p2, u2);        // P2 | Add (Up+Last)/2

              p0 = _mm_slli_si128(p0, 4);
              p0 = _mm_packus_epi16(p0, p1);
              p0 = _mm_slli_si128(p0, 2);
              p0 = _mm_srli_si128(p0, 4);

              p2 = _mm_packus_epi16(p2, p2);
              p2 = _mm_slli_si128(p2, 12);
              p0 = _mm_or_si128(p0, p2);

              _mm_store_si128(reinterpret_cast<__m128i*>(p + 6), p0);
              t1 = _mm_srli_si128(p0, 10);

              p += 16;
              u += 16;
              i -= 16;
            }
          }
          else if (bpp == 8) {
            // Process 16 BYTEs at a time.
            __m128i t1 = _mm_unpacklo_epi8(
              _mm_loadl_epi64(reinterpret_cast<__m128i*>(p)), zero);

            while (i >= 16) {
              __m128i p0, p1;
              __m128i u0, u1;

              u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u));
              p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 8));

              u1 = u0;                          // HI | Move Up
              p1 = p0;                          // HI | Move Ln
              u0 = _mm_unpacklo_epi8(u0, zero); // LO | Unpack Up
              p0 = _mm_unpacklo_epi8(p0, zero); // LO | Unpack Ln

              u0 = _mm_add_epi16(u0, t1);       // LO | Add Last
              p1 = _mm_unpackhi_epi8(p1, zero); // HI | Unpack Ln
              u0 = _mm_srli_epi16(u0, 1);       // LO | >> 1
              u1 = _mm_unpackhi_epi8(u1, zero); // HI | Unpack Up

              p0 = _mm_add_epi8(p0, u0);        // LO | Add (Up+Last)/2
              u1 = _mm_add_epi16(u1, p0);       // HI | Add LO
              u1 = _mm_srli_epi16(u1, 1);       // HI | >> 1
              p1 = _mm_add_epi8(p1, u1);        // HI | Add (Up+LO)/2

              p0 = _mm_packus_epi16(p0, p1);
              t1 = p1;                          // HI | Get Last
              _mm_store_si128(reinterpret_cast<__m128i*>(p + 8), p0);

              p += 16;
              u += 16;
              i -= 16;
            }
          }
        }

        for (; i != 0; i--, p++, u++)
          p[bpp] = depng_sum(p[bpp], depng_avg(p[0], u[0]));

        p += bpp;
        break;
      }

      // ----------------------------------------------------------------------
      // [Paeth]
      // ----------------------------------------------------------------------

      case kPngFilterPaeth: {
        if (bpp == 1) {
          // There is not much to optimize for 1BPP. The only thing this code
          // does is to keep `p0` and `u0` values from the current iteration
          // to the next one (they become `pz` and `uz`).
          uint32_t pz = 0;
          uint32_t uz = 0;
          uint32_t u0;

          for (i = 0; i < bpl; i++) {
            u0 = u[i];
            pz = (static_cast<uint32_t>(p[i]) + depng_paeth_opt(pz, u0, uz)) & 0xFF;

            p[i] = static_cast<uint8_t>(pz);
            uz = u0;
          }

          p += bpl;
        }
        else {
          for (i = 0; i < bpp; i++)
            p[i] = depng_sum(p[i], u[i]);

          i = bpl - bpp;

          if (i >= 32) {
            // Align to 16-BYTE boundary.
            uint32_t j = SimdUtils::alignDiff(p + bpp, 16);

            __m128i zero = _mm_setzero_si128();
            __m128i rcp3 = _mm_set1_epi16(0xAB << 7);

            for (i -= j; j != 0; j--, p++, u++)
              p[bpp] = depng_sum(p[bpp], depng_paeth_opt(p[0], u[bpp], u[0]));

            // TODO: Not complete.
            /*
            if (bpp == 2) {
            }
            */
            if (bpp == 3) {
              __m128i pz = _mm_unpacklo_epi8(_mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(p)[0] & 0x00FFFFFFU), zero);
              __m128i uz = _mm_unpacklo_epi8(_mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(u)[0] & 0x00FFFFFFU), zero);
              __m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0x0000FFFF, 0x00000000, 0x00000000);

              // Process 8 BYTEs at a time.
              while (i >= 8) {
                __m128i p0, p1;
                __m128i u0, u1;

                u0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(u + 3));
                p0 = _mm_loadl_epi64(reinterpret_cast<__m128i*>(p + 3));

                u0 = _mm_unpacklo_epi8(u0, zero);
                p0 = _mm_unpacklo_epi8(p0, zero);
                u1 = _mm_srli_si128(u0, 6);

                DEPNG_SSE2_PAETH(uz, pz, u0, uz);
                uz = _mm_and_si128(uz, mask);
                p0 = _mm_add_epi8(p0, uz);

                DEPNG_SSE2_PAETH(uz, p0, u1, u0);
                uz = _mm_and_si128(uz, mask);
                uz = _mm_slli_si128(uz, 6);
                p0 = _mm_add_epi8(p0, uz);

                p1 = _mm_srli_si128(p0, 6);
                u0 = _mm_srli_si128(u1, 6);

                DEPNG_SSE2_PAETH(u0, p1, u0, u1);
                u0 = _mm_slli_si128(u0, 12);

                p0 = _mm_add_epi8(p0, u0);
                pz = _mm_srli_si128(p0, 10);
                uz = _mm_srli_si128(u1, 4);

                p0 = _mm_packus_epi16(p0, p0);
                _mm_storel_epi64(reinterpret_cast<__m128i*>(p + 3), p0);

                p += 8;
                u += 8;
                i -= 8;
              }
            }
            else if (bpp == 4) {
              __m128i pz = _mm_unpacklo_epi8(_mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(p)[0]), zero);
              __m128i uz = _mm_unpacklo_epi8(_mm_cvtsi32_si128(reinterpret_cast<uint32_t*>(u)[0]), zero);
              __m128i mask = _mm_setr_epi32(0xFFFFFFFF, 0xFFFFFFFF, 0, 0);

              // Process 16 BYTEs at a time.
              while (i >= 16) {
                __m128i p0, p1;
                __m128i u0, u1;

                p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 4));
                u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 4));

                p1 = _mm_unpackhi_epi8(p0, zero);
                p0 = _mm_unpacklo_epi8(p0, zero);
                u1 = _mm_unpackhi_epi8(u0, zero);
                u0 = _mm_unpacklo_epi8(u0, zero);

                DEPNG_SSE2_PAETH(uz, pz, u0, uz);
                uz = _mm_and_si128(uz, mask);
                p0 = _mm_add_epi8(p0, uz);
                uz = _mm_shuffle_epi32(u0, _MM_SHUFFLE(1, 0, 3, 2));

                DEPNG_SSE2_PAETH(u0, p0, uz, u0);
                u0 = _mm_slli_si128(u0, 8);
                p0 = _mm_add_epi8(p0, u0);
                pz = _mm_srli_si128(p0, 8);

                DEPNG_SSE2_PAETH(uz, pz, u1, uz);
                uz = _mm_and_si128(uz, mask);
                p1 = _mm_add_epi8(p1, uz);
                uz = _mm_shuffle_epi32(u1, _MM_SHUFFLE(1, 0, 3, 2));

                DEPNG_SSE2_PAETH(u1, p1, uz, u1);
                u1 = _mm_slli_si128(u1, 8);
                p1 = _mm_add_epi8(p1, u1);
                pz = _mm_srli_si128(p1, 8);

                p0 = _mm_packus_epi16(p0, p1);
                _mm_store_si128(reinterpret_cast<__m128i*>(p + 4), p0);

                p += 16;
                u += 16;
                i -= 16;
              }
            }
            else if (bpp == 6) {
              __m128i pz = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<__m128i*>(p)), zero);
              __m128i uz = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<__m128i*>(u)), zero);
              __m128i mask = _mm_setr_epi32(0, 0xFFFFFFFF, 0, 0);

              // Process 16 BYTEs at a time.
              while (i >= 16) {
                __m128i p0, p1, p2;
                __m128i u0, u1, u2;

                p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 6));
                u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 6));

                p1 = _mm_srli_si128(p0, 6);
                p0 = _mm_unpacklo_epi8(p0, zero);
                u1 = _mm_srli_si128(u0, 6);
                u0 = _mm_unpacklo_epi8(u0, zero);

                DEPNG_SSE2_PAETH(uz, pz, u0, uz);
                p0 = _mm_add_epi8(p0, uz);
                p2 = _mm_srli_si128(p1, 6);
                u2 = _mm_srli_si128(u1, 6);
                p1 = _mm_unpacklo_epi8(p1, zero);
                u1 = _mm_unpacklo_epi8(u1, zero);

                DEPNG_SSE2_PAETH(u0, p0, u1, u0);
                p1 = _mm_add_epi8(p1, u0);
                p2 = _mm_unpacklo_epi8(p2, zero);
                u2 = _mm_unpacklo_epi8(u2, zero);

                DEPNG_SSE2_PAETH(u0, p1, u2, u1);
                p2 = _mm_add_epi8(p2, u0);

                p0 = _mm_slli_si128(p0, 4);
                p0 = _mm_packus_epi16(p0, p1);
                p0 = _mm_slli_si128(p0, 2);
                p0 = _mm_srli_si128(p0, 4);

                p2 = _mm_shuffle_epi32(p2, _MM_SHUFFLE(1, 0, 1, 0));
                u2 = _mm_shuffle_epi32(u2, _MM_SHUFFLE(1, 0, 1, 0));

                pz = _mm_shuffle_epi32(_mm_unpackhi_epi32(p1, p2), _MM_SHUFFLE(3, 3, 1, 0));
                uz = _mm_shuffle_epi32(_mm_unpackhi_epi32(u1, u2), _MM_SHUFFLE(3, 3, 1, 0));

                p2 = _mm_packus_epi16(p2, p2);
                p2 = _mm_slli_si128(p2, 12);

                p0 = _mm_or_si128(p0, p2);
                _mm_store_si128(reinterpret_cast<__m128i*>(p + 6), p0);

                p += 16;
                u += 16;
                i -= 16;
              }
            }
            else if (bpp == 8) {
              __m128i pz = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<__m128i*>(p)), zero);
              __m128i uz = _mm_unpacklo_epi8(_mm_loadl_epi64(reinterpret_cast<__m128i*>(u)), zero);

              // Process 16 BYTEs at a time.
              while (i >= 16) {
                __m128i p0, p1;
                __m128i u0, u1;

                p0 = _mm_load_si128(reinterpret_cast<__m128i*>(p + 8));
                u0 = _mm_loadu_si128(reinterpret_cast<__m128i*>(u + 8));

                p1 = _mm_unpackhi_epi8(p0, zero);
                p0 = _mm_unpacklo_epi8(p0, zero);
                u1 = _mm_unpackhi_epi8(u0, zero);
                u0 = _mm_unpacklo_epi8(u0, zero);

                DEPNG_SSE2_PAETH(uz, pz, u0, uz);
                p0 = _mm_add_epi8(p0, uz);

                DEPNG_SSE2_PAETH(pz, p0, u1, u0);
                pz = _mm_add_epi8(pz, p1);
                uz = u1;

                p0 = _mm_packus_epi16(p0, pz);
                _mm_store_si128(reinterpret_cast<__m128i*>(p + 8), p0);

                p += 16;
                u += 16;
                i -= 16;
              }
            }
          }

          for (; i != 0; i--, p++, u++)
            p[bpp] = depng_sum(p[bpp], depng_paeth_opt(p[0], u[bpp], u[0]));

          p += bpp;
        }
        break;
      }
    }

    u = p - bpl;
  } while (--y != 0);
}

void depng_filter_sse2(uint8_t* p, uint32_t h, uint32_t bpp, uint32_t bpl) {
  switch (bpp) {
    case 1: depng_filter_sse2_template<1>(p, h, bpl); break;
    case 2: depng_filter_sse2_template<2>(p, h, bpl); break;
    case 3: depng_filter_sse2_template<3>(p, h, bpl); break;
    case 4: depng_filter_sse2_template<4>(p, h, bpl); break;
    case 6: depng_filter_sse2_template<6>(p, h, bpl); break;
    case 8: depng_filter_sse2_template<8>(p, h, bpl); break;
  }
}
