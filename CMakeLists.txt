CMake_Minimum_Required(VERSION 2.8.4)
Project(RgbHsv C CXX)

Add_Executable(RgbHsvTest
  src/main.cpp
  src/rgbhsv.cpp
  src/rgbhsv.h
  src/util.h)
