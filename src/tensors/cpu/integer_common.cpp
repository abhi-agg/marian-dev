#include "integer_common.h"

namespace marian {
namespace cpu {
namespace integer {

size_t computeAlignment(void* address) {
  auto ptr = reinterpret_cast<std::uintptr_t>(address);
    if ((ptr % 512) == 0) {
      return 512;
    } else if ((ptr % 256) == 0) {
      return 256;
    } else if ((ptr % 128) == 0) {
      return 128;
    } else if ((ptr % 64) == 0) {
      return 64;
    } else if ((ptr % 32) == 0) {
      return 32;
    } else if ((ptr % 16) == 0) {
      return 16;
    } else if ((ptr % 8) == 0) {
      return 8;
    } else if ((ptr % 4) == 0) {
      return 4;
    } else if ((ptr % 2) == 0) {
      return 2;
    } else {
      return 1;
    }
}

// This operates on floats after processing so doesn't care about int8_t vs int16_t.
void AddBias(marian::Tensor C, const marian::Tensor Bias) {
  float* y = C->data();
  const float* x = C->data();
  const float* bias = Bias->data();

  const int m = C->shape().elements() / C->shape()[-1];
  const int n = C->shape()[-1];

  for(int j = 0; j < m; ++j) {
    int i = 0;
#ifdef __AVX512F__
    int n16 = n & ~15;
    for(; i < n16; i += 16) {
      __m512 ai = _mm512_loadu_ps(x + j * n + i);
      __m512 bi = _mm512_loadu_ps(bias + i);
      __m512 yi = _mm512_add_ps(ai, bi);
      _mm512_storeu_ps(y + j * n + i, yi);
    }
#else
    int n4 = (n / 4) * 4;
    for(; i < n4; i += 4) {
      __m128 ai = _mm_loadu_ps(x + j * n + i);
      __m128 bi = _mm_loadu_ps(bias + i);
      __m128 yi = _mm_add_ps(ai, bi);
      _mm_storeu_ps(y + j * n + i, yi);
    }
#endif
    for(; i < n; i++) {
      y[j * n + i] = x[j * n + i] + bias[i];
    }
  }
}

} //integer
} //cpu
} //marian