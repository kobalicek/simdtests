//
// Lol Engine
//
// Copyright: (c) 2010-2011 Sam Hocevar <sam@hocevar.net>
//   This program is free software; you can redistribute it and/or
//   modify it under the terms of the Do What The Fuck You Want To
//   Public License, Version 2, as published by Sam Hocevar. See
//   http://sam.zoy.org/projects/COPYING.WTFPL for more details.
//

//
// The Real class
// --------------
//

#ifndef __LOL_MATH_REAL_H__
#define __LOL_MATH_REAL_H__

#include <stdint.h>

/* Avoid issues with NaCl headers */
#undef log2

namespace lol
{

class real;

real sin(real const &x);
real cos(real const &x);
real tan(real const &x);
real asin(real const &x);
real acos(real const &x);
real atan(real const &x);
real atan2(real const &y, real const &x);

/* Hyperbolic functions */
real sinh(real const &x);
real cosh(real const &x);
real tanh(real const &x);

/* Exponential and logarithmic functions */
real exp(real const &x);
real exp2(real const &x);
real log(real const &x);
real log2(real const &x);
real log10(real const &x);
real frexp(real const &x, int *exp);
real ldexp(real const &x, int exp);
real modf(real const &x, real *iptr);

/* Power functions */
real re(real const &x);
real sqrt(real const &x);
real cbrt(real const &x);
real pow(real const &x, real const &y);

/* Rounding, absolute value, remainder etc. */
real ceil(real const &x);
real copysign(real const &x, real const &y);
real floor(real const &x);
real fabs(real const &x);
real round(real const &x);
real fmod(real const &x, real const &y);


class real
{
public:
    real();
    real(real const &x);
    real const &operator =(real const &x);

    real(float f);
    real(double f);
    real(int i);
    real(unsigned int i);

    real(char const *str);

    operator float() const;
    operator double() const;
    operator int() const;
    operator unsigned int() const;

    real operator +() const;
    real operator -() const;
    real operator +(real const &x) const;
    real operator -(real const &x) const;
    real operator *(real const &x) const;
    real operator /(real const &x) const;
    real const &operator +=(real const &x);
    real const &operator -=(real const &x);
    real const &operator *=(real const &x);
    real const &operator /=(real const &x);

    bool operator ==(real const &x) const;
    bool operator !=(real const &x) const;
    bool operator <(real const &x) const;
    bool operator >(real const &x) const;
    bool operator <=(real const &x) const;
    bool operator >=(real const &x) const;

    bool operator !() const;
    operator bool() const;

    void hexprint() const;
    void print(int ndigits = 150) const;

    /* Additional operators using base C++ types */
#define __LOL_REAL_OP_HELPER_GENERIC(op, type) \
    inline real operator op(type x) const { return *this op (real)x; } \
    inline real const &operator op##=(type x) { return *this = (*this op x); }
#define __LOL_REAL_OP_HELPER_FASTMULDIV(op, type) \
    inline real operator op(type x) const \
    { \
        real tmp = *this; return tmp op##= x; \
    } \
    inline real const &operator op##=(type x) \
    { \
        /* If multiplying or dividing by a power of two, take a shortcut */ \
        if ((m_signexp << 1) && x && !(x & (x - 1))) \
        { \
            while (x >>= 1) \
                m_signexp += 1 op 2 - 1; /* 1 if op is *, -1 if op is / */ \
        } \
        else \
            *this = *this op (real)x; \
        return *this; \
    }
#define __LOL_REAL_OP_HELPER_INT(type) \
    __LOL_REAL_OP_HELPER_GENERIC(+, type) \
    __LOL_REAL_OP_HELPER_GENERIC(-, type) \
    __LOL_REAL_OP_HELPER_FASTMULDIV(*, type) \
    __LOL_REAL_OP_HELPER_FASTMULDIV(/, type)
#define __LOL_REAL_OP_HELPER_FLOAT(type) \
    __LOL_REAL_OP_HELPER_GENERIC(+, type) \
    __LOL_REAL_OP_HELPER_GENERIC(-, type) \
    __LOL_REAL_OP_HELPER_GENERIC(*, type) \
    __LOL_REAL_OP_HELPER_GENERIC(/, type)

    __LOL_REAL_OP_HELPER_INT(int)
    __LOL_REAL_OP_HELPER_INT(unsigned int)
    __LOL_REAL_OP_HELPER_FLOAT(float)
    __LOL_REAL_OP_HELPER_FLOAT(double)

    /* Constants */
    static real const R_0;
    static real const R_1;
    static real const R_2;
    static real const R_3;
    static real const R_10;

    static real const R_E;
    static real const R_LOG2E;
    static real const R_LOG10E;
    static real const R_LN2;
    static real const R_LN10;
    static real const R_PI;
    static real const R_PI_2;
    static real const R_PI_3;
    static real const R_PI_4;
    static real const R_1_PI;
    static real const R_2_PI;
    static real const R_2_SQRTPI;
    static real const R_SQRT2;
    static real const R_SQRT3;
    static real const R_SQRT1_2;

    /* XXX: changing this requires tuning real::fres (the number of
     * Newton-Raphson iterations) and real::print (the number of printed
     * digits) */
    enum {
      BIGITS = 16,
      BIGIT_BITS = 32
    };

    uint32_t m_mantissa[BIGITS];
    uint32_t m_signexp;
};

} /* namespace lol */

#endif // __LOL_MATH_REAL_H__

