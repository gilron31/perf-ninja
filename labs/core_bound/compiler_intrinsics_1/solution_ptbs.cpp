
#include "solution.h"
#include <immintrin.h>
#include <memory>

void imageSmoothing(const InputVector &input, uint8_t radius,
                    OutputVector &output) {
  int pos = 0;
  int currentSum = 0;
  int size = static_cast<int>(input.size());

  // 1. left border - time spend in this loop can be ignored, no need to
  // optimize it
  for (int i = 0; i < std::min<int>(size, radius); ++i) {
    currentSum += input[i];
  }

  int limit = std::min(radius + 1, size - radius);
  for (pos = 0; pos < limit; ++pos) {
    currentSum += input[pos + radius];
    output[pos] = currentSum;
  }

  // 2. main loop.
  limit = size - radius;

  constexpr size_t step = 32;
  for (; pos < limit; pos += step) {

    __m256i minuses8 = _mm256_loadu_epi8(&input[pos - radius - 1]);
    __m256i pluses8 = _mm256_loadu_epi8(&input[pos + radius]);
    __m512i outputs = _mm512_set1_epi16(currentSum);
    __m512i minuses16 = _mm512_cvtepi8_epi16(minuses8);
    __m512i pluses16 = _mm512_cvtepi8_epi16(pluses8);
    outputs = _mm512_add_epi16(outputs, pluses16);
    outputs = _mm512_sub_epi16(outputs, minuses16);

    _mm512_storeu_epi16(&output, outputs);

    currentSum = output[pos + step - 1];
  }

  for (; pos < limit; ++pos) {
    currentSum -= input[pos - radius - 1];
    currentSum += input[pos + radius];
    output[pos] = currentSum;
  }

  // 3. special case, executed only if size <= 2*radius + 1
  limit = std::min(radius + 1, size);
  for (; pos < limit; pos++) {
    output[pos] = currentSum;
  }

  // 4. right border - time spend in this loop can be ignored, no need to
  // optimize it
  for (; pos < size; ++pos) {
    currentSum -= input[pos - radius - 1];
    output[pos] = currentSum;
  }
}
