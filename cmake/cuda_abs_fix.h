#pragma once
#include <cmath>
#include <cstdlib>

// When __CORRECT_ISO_CPP_MATH_H_PROTO is defined, libstdc++ sometimes skips defining
// the float and double overloads for std::abs, causing bessel_function.tcc to fail.
// We explicitly define them here before CUDA pulls in the rest of the standard library.
namespace std {
    inline float abs(float __x) { return __builtin_fabsf(__x); }
    inline double abs(double __x) { return __builtin_fabs(__x); }
    inline long double abs(long double __x) { return __builtin_fabsl(__x); }
}
