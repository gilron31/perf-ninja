
#include "solution.h"
#include <array>
#include <immintrin.h>

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

  constexpr size_t STEP = 8;

  uint16_t scratch_pluses[STEP] = {};
  uint16_t scratch_minuses[STEP] = {};
  for (; pos <= limit - STEP; pos += STEP) {

    __m128i outputs = _mm_set1_epi16(currentSum);
    for (size_t i = 0; i < STEP; i++) {
      scratch_pluses[i] = input[pos - radius - 1 + i];
      scratch_minuses[i] = input[pos + radius + i];
    }
    __m128i minuses16 = _mm_loadu_si16(&scratch_minuses);
    __m128i pluses16 = _mm_loadu_si16(&scratch_pluses);
    outputs = _mm_add_epi16(outputs, pluses16);
    outputs = _mm_sub_epi16(outputs, minuses16);

    _mm_storeu_si16(&output[pos], outputs);

    currentSum = output[pos + STEP - 1];
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
