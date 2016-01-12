cmake_minimum_required(VERSION 3.1)
project(SimdTests C CXX)

set(SIMD_CFLAGS_SSE2)
set(SIMD_CFLAGS_SSE3)
set(SIMD_CFLAGS_SSSE3)

if("${CMAKE_CXX_COMPILER_ID}" MATCHES "^(GNU|Clang)$")
  set(SIMD_CFLAGS_SSE2 -msse2)
  set(SIMD_CFLAGS_SSE3 -msse3)
  set(SIMD_CFLAGS_SSSE3 -mssse3)
endif()

macro(simd_add_test _target _files)
  foreach(_file ${_files})
    set(_cflags "")

    if(${_file} MATCHES "_sse2\\.")
      set(_cflags ${SIMD_CFLAGS_SSE2})
    endif()

    if(${_file} MATCHES "_sse3\\.")
      set(_cflags ${SIMD_CFLAGS_SSE3})
    endif()

    if(${_file} MATCHES "_ssse3\\.")
      set(_cflags ${SIMD_CFLAGS_SSSE3})
    endif()

    if("${_cflags}")
      set_source_files_properties(${_file} PROPERTIES COMPILE_FLAGS ${_cflags})
    endif()
  endforeach()

  add_executable(${_target} ${_files})
endmacro()

set(SIMD_COMMON_SRC
  simdglobals)

set(SIMD_DEJPEG_SRC
  dejpeg/dejpeg.h
  dejpeg/dejpeg_ref.cpp
  dejpeg/dejpeg_sse2.cpp
  dejpeg/dejpeg_ssse3.cpp
  dejpeg/dejpeg_test.cpp)

set(SIMD_DEPNG_SRC
  depng/depng.h
  depng/depng_ref.cpp
  depng/depng_sse2.cpp
  depng/depng_test.cpp)

set(SIMD_RGBHSV_SRC
  rgbhsv/rgbhsv.h
  rgbhsv/rgbhsv_ref.cpp
  rgbhsv/rgbhsv_sse2.cpp
  rgbhsv/rgbhsv_test.cpp)

set(SIMD_PIXOPS_SRC
  pixops/pixops.h
  pixops/pixops_ref.cpp
  pixops/pixops_sse2.cpp
  pixops/pixops_ssse3.cpp
  pixops/pixops_test.cpp)

simd_add_test(test_dejpeg "${SIMD_COMMON_SRC};${SIMD_DEJPEG_SRC}")
simd_add_test(test_depng  "${SIMD_COMMON_SRC};${SIMD_DEPNG_SRC}")
simd_add_test(test_rgbhsv "${SIMD_COMMON_SRC};${SIMD_RGBHSV_SRC}")
simd_add_test(test_pixops "${SIMD_COMMON_SRC};${SIMD_PIXOPS_SRC}")
