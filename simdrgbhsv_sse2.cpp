// [SimdRgbHsv]
// SIMD optimized RGB/HSV conversion.
//
// [License]
// Zlib - See LICENSE.md file in the package.

#define USE_SSE2

#include "./simdglobals.h"
#include "./simdrgbhsv.h"

// ============================================================================
// [SSE/SSE2 Implementation]
// ============================================================================

// Shortcuts:
// 'p?' - Positive number (p6 == +6.0f, p0 == +0.0f)
// 'r?' - Reciprocal of number (r6 == 1/6.0f)
// 'n?' - Negative number (n1 == -1.0f, n0 == -0.0f)
// 'ep' - Epsilon, a very small number near zero (1e-8f)

// 'sn' - Sign bit (0x80000000)
// 'ab' - Everything but sign (0x7FFFFFFF)
// 'nm' - Number (0xFFFFFFFF)

SIMD_CONST_PI(sn         , 0x80000000, 0x80000000, 0x80000000, 0x80000000);
SIMD_CONST_PI(abs        , 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF, 0x7FFFFFFF);
SIMD_CONST_PI(full       , 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF);
SIMD_CONST_PI(sn_sn_p0_p0, 0x80000000, 0x80000000, 0x00000000, 0x00000000);

SIMD_CONST_PS(p0         , 0.0f , 0.0f , 0.0f , 0.0f);
SIMD_CONST_PS(p1         , 1.0f , 1.0f , 1.0f , 1.0f);
SIMD_CONST_PS(eps        , 1e-8f, 1e-8f, 1e-8f, 1e-8f);
SIMD_CONST_PS(p0_p0_p0_p6, 0.0f , 0.0f , 0.0f , 6.0f);
SIMD_CONST_PS(p1_p1_m2_p0, 1.0f , 1.0f ,-2.0f , 0.0f);
SIMD_CONST_PS(m1_m1_m1_p1,-1.0f ,-1.0f ,-1.0f , 1.0f);
SIMD_CONST_PS(m6_m6_p6_p0,-6.0f ,-6.0f , 6.0f , 0.0f);
SIMD_CONST_PS(m6_m6_m6_m6,-6.0f ,-6.0f ,-6.0f ,-6.0f);

SIMD_CONST_PS(p4o6_p2o6_p3o6_p0  , 4.0f / 6.0f, 2.0f / 6.0f, 3.0f / 6.0f, 0.0f);
SIMD_CONST_PS(m4o6_m4o6_m4o6_m4o6,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f,-4.0f / 6.0f);

static __m128 SIMD_INLINE _mm_load_ps_my(const void* mem) { return _mm_load_ps((const float*)mem); }
static void SIMD_INLINE _mm_store_ps_my(void* mem, __m128 reg) { _mm_store_ps((float*)mem, reg); }

void ahsv_from_argb_sse2(float* dst, const float* src, int length) {
  int i = length;

  while ((i -= 4) >= 0) {
    __m128 xG, xB, xA, xR;
    __m128 xH, xS, xV, xC;
    __m128 xX, xY, xZ;

    // Load 4 ARGB Pixels.
    xC = _mm_load_ps(src +  0);                            // xC <- [B0|G0|R0|A0]
    xS = _mm_load_ps(src +  4);                            // xS <- [B1|G1|R1|A1]
    xR = _mm_load_ps(src +  8);                            // xR <- [B2|G2|R2|A2]
    xV = _mm_load_ps(src + 12);                            // xV <- [B3|G3|R3|A3]

    // Transpose.
    //
    // What we get: xA == [A3 A2 A1 A0] - Alpha channel.
    //              xR == [R3 R2 R1 R0] - Red   channel.
    //              xG == [G3 G2 G1 G0] - Green channel.
    //              xB == [B3 B2 B1 B0] - Blue  channel.
    //
    // What we use: xC - Temporary.
    //              xS - Temporary.
    //              xV - Temporary.
    xA = _mm_unpackhi_ps(xC, xS);                          // xA <- [B1|B0|G1|G0]
    xB = _mm_unpackhi_ps(xR, xV);                          // xB <- [B3|B2|G3|G2]
    xC = _mm_unpacklo_ps(xC, xS);                          // xC <- [R1|R0|A1|A0]
    xR = _mm_unpacklo_ps(xR, xV);                          // xR <- [R3|R2|A3|A2]

    xG = _mm_movelh_ps(xA, xB);                            // xG <- [G3|G2|G1|G0]
    xB = _mm_movehl_ps(xB, xA);                            // xB <- [B3|B2|B1|B0]
    xA = _mm_movelh_ps(xC, xR);                            // xA <- [A3|A2|A1|A0]
    xR = _mm_movehl_ps(xR, xC);                            // xR <- [R3|R2|R1|R0]

    // Calculate Value, Chroma, and Saturation.
    //
    // What we get: xC == [C3 C2 C1 C0 ] - Chroma.
    //              xV == [V3 V2 V1 V0 ] - Value == Max(R, G, B).
    //              xS == [S3 S2 S1 S0 ] - Saturation, possibly incorrect due to division
    //                                     by zero, corrected at the end of the algorithm.
    //
    // What we use: xR
    //              xG
    //              xB
    xS = _mm_max_ps(xG, xB);                               // xS <- [max(G, B)]
    xC = _mm_min_ps(xG, xB);                               // xC <- [min(G, B)]

    xS = _mm_max_ps(xS, xR);                               // xS <- [max(G, B, R)]
    xC = _mm_min_ps(xC, xR);                               // xC <- [min(G, B, R)]

    xV = xS;                                               // xV <- [V    ]
    xS = _mm_sub_ps(xS, xC);                               // xS <- [V - m]
    xS = _mm_div_ps(xS, xV);                               // xS <- [S    ]

    xC = _mm_sub_ps(xC, xV);                               // xC <- [V + m]
 
    // Calculate Hue.
    //
    // What we get: xG - Hue
    //              xC - Chroma * 6.
    //
    // What we use: xR - Destroyed during calculation.
    //              xG - Destroyed during calculation.
    //              xB - Destroyed during calculation.
    //              xC - Chroma.
    //              xX - Mask.
    //              xY - Mask.
    //              xZ - Mask.
    xZ = _mm_cmpeq_ps(xV, xG);                             // xZ <- [V==G]
    xX = _mm_cmpneq_ps(xV, xR);                            // xX <- [V!=R]

    xY = _mm_and_ps(xZ, xX);                               // xY <- [V!=R && V==G]
    xZ = _mm_andnot_ps(xZ, xX);                            // xZ <- [V!=R && V!=G]

    xY = _mm_xor_ps(xY, SIMD_GET_PS(full));                // xY <- [V==R || V!=G]
    xZ = _mm_xor_ps(xZ, SIMD_GET_PS(full));                // xZ <- [V==R || V==G]
    
    xR = _mm_and_ps(xR, xX);                               // xR <- [X!=0 ? R : 0]
    xB = _mm_and_ps(xB, xZ);                               // xB <- [Z!=0 ? B : 0]
    xG = _mm_and_ps(xG, xY);                               // xG <- [Y!=0 ? G : 0]

    xZ = _mm_andnot_ps(xZ, SIMD_GET_PS(sn));               // xZ <- [sign(!Z)]
    xY = _mm_andnot_ps(xY, SIMD_GET_PS(sn));               // xY <- [sign(!Y)]

    xG = _mm_xor_ps(xG, xZ);                               // xG <- [Y!=0 ? (Z==0 ? G : -G) : 0]
    xR = _mm_xor_ps(xR, xY);                               // xR <- [X!=0 ? (Y==0 ? R : -R) : 0]

    // G is now accumulator.
    xG = _mm_add_ps(xG, xR);                               // xG <- [Rx + Gx]
    xB = _mm_xor_ps(xB, xY);                               // xB <- [Z!=0 ? (Y==0 ? B : -B) : 0]

    xC = _mm_mul_ps(xC, SIMD_GET_PS(m6_m6_m6_m6));         // xC <- [C*6     ]
    xG = _mm_sub_ps(xG, xB);                               // xG <- [Rx+Gx+Bx]

    xH = _mm_and_ps(xX, SIMD_GET_PS(m4o6_m4o6_m4o6_m4o6)); // xH <- [V==R ?0 :-4/6]
    xG = _mm_div_ps(xG, xC);                               // xG <- [(Rx+Gx+Bx)/6C]

    // Correct the achromatic case - H/S may be infinite (or near) due to division by zero.
    xH = _mm_xor_ps(xH, xZ);                               // xH <- [V==R ? 0 : V==G ? -4/6 : 4/6]
    xC = _mm_cmple_ps(SIMD_GET_PS(eps), xC);
    xH = _mm_add_ps(xH, SIMD_GET_PS(p1));                  // xH <- [V==R ? 1 : V==G ?  2/6 :10/6]

    xG = _mm_add_ps(xG, xH);

    // Normalize H to a fraction. If it's greater than or equal to 1 then 1 is subtracted 
    // to get the Hue at [0..1) domain.
    xH = _mm_cmple_ps(SIMD_GET_PS(p1), xG);

    xH = _mm_and_ps(xH, SIMD_GET_PS(p1));
    xS = _mm_and_ps(xS, xC);
    xG = _mm_and_ps(xG, xC);
    xG = _mm_sub_ps(xG, xH);

    // Transpose.
    xC = _mm_unpacklo_ps(xS, xV);                          // xC <- [V1|S1|V0|S0]
    xS = _mm_unpackhi_ps(xS, xV);                          // xS <- [V3|S3|V2|S2]

    xB = _mm_unpacklo_ps(xA, xG);                          // xB <- [H1|A1|H0|A0]
    xA = _mm_unpackhi_ps(xA, xG);                          // xA <- [H3|A3|H2|A2]

    xG = _mm_movelh_ps(xB, xC);                            // xG <- [V0|S0|H0|A0] 
    xR = _mm_movelh_ps(xA, xS);                            // xR <- [V2|S2|H2|A2] 

    xB = _mm_shuffle_ps(xB, xC, _MM_SHUFFLE(3, 2, 3, 2));  // xB <- [V1|S1|H1|A1]
    xA = _mm_shuffle_ps(xA, xS, _MM_SHUFFLE(3, 2, 3, 2));  // xA <- [V3|S3|H3|A3]

    // Store 4 AHSV Pixels.
    _mm_store_ps(dst +  0, xG);
    _mm_store_ps(dst +  4, xB);
    _mm_store_ps(dst +  8, xR);
    _mm_store_ps(dst + 12, xA);

    dst += 16;
    src += 16;
  }

  // Process the remaining pixels using C.
  if ((i += 4) > 0)
    ahsv_from_argb_ref(dst, src, i);
}

void argb_from_ahsv_sse2(float* dst, const float* src, int length) {
  int i = length;

  while ((i -= 4) >= 0) {
    __m128 h0, h1, h2, h3;
    __m128 x0, x1, x2, x3;
    __m128 a0, a1;

    // Load 4 AHSV Pixels.
    h0 = _mm_load_ps_my(src + 0);                          // h0 <- [V           |S           |H           |A          ]
    h1 = _mm_load_ps_my(src + 4);                          // h1 <- [V           |S           |H           |A          ]
    h2 = _mm_load_ps_my(src + 8);                          // h2 <- [V           |S           |H           |A          ]
    h3 = _mm_load_ps_my(src + 12);                         // h3 <- [V           |S           |H           |A          ]

    // Prepare HUE for RGB components (per pixel).
    x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]
    x1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(1, 1, 1, 3));     // x1 <- [H           |H           |H           |V          ]
    x2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(1, 1, 1, 3));     // x2 <- [H           |H           |H           |V          ]
    x3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(1, 1, 1, 3));     // x3 <- [H           |H           |H           |V          ]

    // Calculate intervals from HUE.
    x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
    x1 = _mm_sub_ps(x1, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x1 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
    x2 = _mm_sub_ps(x2, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x2 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
    x3 = _mm_sub_ps(x3, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x3 <- [H-4/6       |H-2/6       |H-3/6       |V          ]

    x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
    x1 = _mm_and_ps(x1, SIMD_GET_PS(abs));                 // x1 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
    x2 = _mm_and_ps(x2, SIMD_GET_PS(abs));                 // x2 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
    x3 = _mm_and_ps(x3, SIMD_GET_PS(abs));                 // x3 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]

    x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
    x1 = _mm_mul_ps(x1, SIMD_GET_PS(m6_m6_p6_p0));         // x1 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
    x2 = _mm_mul_ps(x2, SIMD_GET_PS(m6_m6_p6_p0));         // x2 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
    x3 = _mm_mul_ps(x3, SIMD_GET_PS(m6_m6_p6_p0));         // x3 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]

    x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
    x1 = _mm_add_ps(x1, SIMD_GET_PS(p1_p1_m2_p0));         // x1 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
    x2 = _mm_add_ps(x2, SIMD_GET_PS(p1_p1_m2_p0));         // x2 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]
    x3 = _mm_add_ps(x3, SIMD_GET_PS(p1_p1_m2_p0));         // x3 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

    // Bound intervals.
    x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
    x1 = _mm_max_ps(x1, SIMD_GET_PS(m1_m1_m1_p1));
    x2 = _mm_max_ps(x2, SIMD_GET_PS(m1_m1_m1_p1));
    x3 = _mm_max_ps(x3, SIMD_GET_PS(m1_m1_m1_p1));

    x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
    x1 = _mm_min_ps(x1, SIMD_GET_PS(p0));                  // x1 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
    x2 = _mm_min_ps(x2, SIMD_GET_PS(p0));                  // x2 <- [(R-1)       |(G-1)       |(B-1)       |0          ]
    x3 = _mm_min_ps(x3, SIMD_GET_PS(p0));                  // x3 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

    // Prepare S/V vectors.
    a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
    a1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]
    h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]
    h1 = SIMD_SHUFFLE_PS(h1, _MM_SHUFFLE(3, 3, 3, 0));     // h1 <- [V           |V           |V           |A          ]

    // Multiply with 'S*V' and add 'V'.
    x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
    x1 = _mm_mul_ps(x1, a1);                               // x1 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
    a0 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]
    a1 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(2, 2, 2, 2));     // a1 <- [S           |S           |S           |S          ]

    x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
    x1 = _mm_mul_ps(x1, h1);                               // x1 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
    h2 = SIMD_SHUFFLE_PS(h2, _MM_SHUFFLE(3, 3, 3, 0));     // h2 <- [V           |V           |V           |A          ]
    h3 = SIMD_SHUFFLE_PS(h3, _MM_SHUFFLE(3, 3, 3, 0));     // h3 <- [V           |V           |V           |A          ]

    x2 = _mm_mul_ps(x2, a0);                               // x2 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
    x3 = _mm_mul_ps(x3, a1);                               // x3 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
    x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

    x2 = _mm_mul_ps(x2, h2);                               // x2 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
    x3 = _mm_mul_ps(x3, h3);                               // x3 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
    x1 = _mm_add_ps(x1, h1);                               // x1 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

    x2 = _mm_add_ps(x2, h2);                               // x2 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]
    x3 = _mm_add_ps(x3, h3);                               // x3 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

    // Store 4 ARGB Pixels.
    _mm_store_ps_my(dst + 0, x0);
    _mm_store_ps_my(dst + 4, x1);
    _mm_store_ps_my(dst + 8, x2);
    _mm_store_ps_my(dst + 12, x3);

    dst += 16;
    src += 16;
  }
  i += 4;

  while (i) {
    __m128 h0;
    __m128 x0;
    __m128 a0;

    // Load 1 AHSV Pixel.
    h0 = _mm_load_ps_my(src);                              // h0 <- [V           |S           |H           |A          ]

    // Prepare HUE for RGB components (per pixel).
    x0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(1, 1, 1, 3));     // x0 <- [H           |H           |H           |V          ]

    // Calculate intervals from HUE.
    x0 = _mm_sub_ps(x0, SIMD_GET_PS(p4o6_p2o6_p3o6_p0));   // x0 <- [H-4/6       |H-2/6       |H-3/6       |V          ]
    a0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(2, 2, 2, 2));     // a0 <- [S           |S           |S           |S          ]

    x0 = _mm_and_ps(x0, SIMD_GET_PS(abs));                 // x0 <- [Abs(H-4/6)  |Abs(H-2/6)  |Abs(H-3/6)  |0          ]
    h0 = SIMD_SHUFFLE_PS(h0, _MM_SHUFFLE(3, 3, 3, 0));     // h0 <- [V           |V           |V           |A          ]

    x0 = _mm_mul_ps(x0, SIMD_GET_PS(m6_m6_p6_p0));         // x0 <- [-Abs(H*6-4) |-Abs(H*6-2) |Abs(H*6-3)  |0          ]
    x0 = _mm_add_ps(x0, SIMD_GET_PS(p1_p1_m2_p0));         // x0 <- [1-Abs(H*6-4)|1-Abs(H*6-2)|Abs(H*6-3)-2|0          ]

    // Bound intervals.
    x0 = _mm_max_ps(x0, SIMD_GET_PS(m1_m1_m1_p1));
    x0 = _mm_min_ps(x0, SIMD_GET_PS(p0));                  // x0 <- [(R-1)       |(G-1)       |(B-1)       |0          ]

    // Multiply with 'S*V' and add 'V'.
    x0 = _mm_mul_ps(x0, a0);                               // x0 <- [(R-1)*S     |(G-1)*S     |(B-1)*S     |0          ]
    x0 = _mm_mul_ps(x0, h0);                               // x0 <- [(R-1)*S*V   |(G-1)*S*V   |(B-1)*S*V   |0          ]
    x0 = _mm_add_ps(x0, h0);                               // x0 <- [(R-1)*S*V+V |(G-1)*S*V+V |(B-1)*S*V+V |A          ]

    // Store 1 ARGB Pixel.
    _mm_store_ps_my(dst, x0);
    i--;
  }
}
