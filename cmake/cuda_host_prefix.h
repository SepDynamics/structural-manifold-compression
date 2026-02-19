#pragma once

// Ensures std::min/std::max and other math overloads are visible to the CUDA host compiler
// before device_functions.hpp pulls them in. This prevents conflicts introduced by glibc 2.38+.
#include <algorithm>
#include <cmath>
#include <limits>
#include <type_traits>
