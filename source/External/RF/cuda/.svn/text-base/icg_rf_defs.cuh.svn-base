#ifndef ICG_RF_DEFS_CUH_
#define ICG_RF_DEFS_CUH_

#include "cudatemplates/devicememorylinear.hpp"
#include "cudatemplates/devicememorypitched.hpp"
#include "cudatemplates/copy.hpp"

#define ICG_RF_THROW_ERROR(message) \
{ \
  fprintf(stderr,"\n\nError: " #message "\n"); \
  fprintf(stderr,"  file:       %s\n",__FILE__); \
  fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
  fprintf(stderr,"  line:       %d\n\n",__LINE__); \
  return false; \
}

#define ICG_RF_CHECK_CUDA_ERROR() \
  if (cudaError_t err = cudaGetLastError()) \
{ \
  fprintf(stderr,"\n\nCUDAError: %s\n",cudaGetErrorString(err)); \
  fprintf(stderr,"  file:       %s\n",__FILE__); \
  fprintf(stderr,"  function:   %s\n",__FUNCTION__); \
  fprintf(stderr,"  line:       %d\n\n",__LINE__); \
  return false; \
  }

//-----------------------------------------------------------------------------
//! Round a / b to nearest higher integer value.
//! @param[in] a Numerator
//! @param[in] b Denominator
//! @return a / b rounded up
inline unsigned int divUp(unsigned int a, unsigned int b) {
  return (a % b != 0) ? (a / b + 1) : (a / b);
}

#endif //ICG_RF_DEFS_CUH_
