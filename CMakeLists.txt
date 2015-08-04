CMake_Minimum_Required(VERSION 2.8.12)
Project(SimdRgbHsv C CXX)

Add_Executable(SimdRgbHsv
  simdrgbhsv.h
  simdrgbhsv_ref.cpp
  simdrgbhsv_sse2.cpp
  simdglobals.h
  simdtest.cpp)
