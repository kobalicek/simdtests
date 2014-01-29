#ifndef _RGBHSV_H
#define _RGBHSV_H

// Pure C++ Implementation.
void ahsv_from_argb_c(float* dst, const float* src, int length);
void argb_from_ahsv_c(float* dst, const float* src, int length);

// SSE2 Enhanced Implementation.
void ahsv_from_argb_sse2(float* dst, const float* src, int length);
void argb_from_ahsv_sse2(float* dst, const float* src, int length);

#endif // _RGBHSV_H
