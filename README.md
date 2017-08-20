SimdTests
=========

This is a playground used to test SIMD optimized functions related mostly to 2D computer graphics. It contains validation of SIMD optimized functions and benchmarks that show relative performance gain. Some experiments contain SSE3 and SSSE3 version that can be also used to compare with SSE2 baseline implementation.

  * [Official Repository (kobalicek/simdtests)](https://github.com/kobalicek/simdtests)
  * [Public Domain](https://unlicense.org/)

The repository contains the following concepts:

  * dejpeg - SIMD optimized JPEG decoding functions (dezigzag, idct, ycbcr)
  * depng - SIMD optimized PNG reverse filter implementation (revfilter)
  * pixops - SIMD optimized low-level pixel operations (crossfade)
  * rgbhsv - SIMD optimized RGB<->HSV conversion
  * trigo - SIMD optimized trigonometric functions

Support
-------

Please consider a donation if this repository saved you some time when implementing your own variation:

  * [Donate by PayPal](https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=QDRM6SRNG7378&lc=EN;&item_name=simdtests&currency_code=EUR)

Authors & Maintainers
---------------------

  * Petr Kobalicek <kobalicek.petr@gmail.com>
