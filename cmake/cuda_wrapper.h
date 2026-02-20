#pragma once

// Block bits/mathcalls.h from using noexcept for these exact functions
#define __mathcalls_h_noexcept_override

#include <math.h>

#undef __MATHDECL
#define __MATHDECL(type, function,suffix, args) \
  extern type function args

#include <cuda_runtime.h>
