OptRgbHsv - Optimized RGB/HSV Conversion by using SIMD
======================================================

Official Repository: https://github.com/kobalicek/rgbhsv

Support the Project: [![Donate](https://www.paypalobjects.com/en_US/i/btn/btn_donate_LG.gif)](
  https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QDRM6SRNG7378&lc=EN;&item_name=rgbhsv&currency_code=EUR)

Introduction
------------

OptRgbHsv is an implementation of RGB/HSV conversion by using SSE and SSE2 instruction sets. The implementation is able to convert 4 pixels at a time. It targets only SSE and SSE2 instruction sets and operates at scanline level; however, the instruction set is not that important as the logic can be applied to any other SIMD instruction set.

The code provided in fact implements ARGB<->AHSV conversion, but it's referred as RGB/HSV for the sake of simplicity. It expects normalized input of ARGB and AHSV data ranging from 0 to 1. The code targets CPU only; there are many solutions taking advantage of GPU shaders.

See [http://en.wikipedia.org/wiki/HSL_and_HSV](HSL and HSV) article on wikipedia that describes HSV colorspace in detail.

RGB to HSV
----------

The RGB to HSV conversion is based on the formula shown below where RGB and HSV are input and output variables, respectively:

    m = min(R, G, B);
    v = max(R, G, B);
    c = v - m;

    H = (c == 0) ? (0.0) :
        (v == R) ? (1.0 + (G - B) / 6*c) :
        (v == G) ? (2/6 + (B - R) / 6*c) :
                   (4/6 + (R - G) / 6*c) ;
    S = (c == 0) ? (0.0) : (c / v);
    V = v;

When translated into C++ code it is full of conditional expressions that are usually translated to machine code as branches. It's extremely difficult to write a good SIMD implementation that can process more pixels at a time, because all branches for all pixels have to be executed independently and the bit manipulation used to select the correct result is actually more expensive than branching.

TODO: More docs.

HSV to RGB
----------

TODO: More docs.
