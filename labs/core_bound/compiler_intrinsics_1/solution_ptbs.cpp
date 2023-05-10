
#include "solution.h"
#include <array>
#include <immintrin.h>
#include <iostream>

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
  uint16_t scratch_diffs[STEP] = {};
  for (; pos <= limit - STEP; pos += STEP) {

    for (size_t i = 0; i < STEP; i++) {
      scratch_minuses[i] = input[pos - radius - 1 + i];
      scratch_pluses[i] = input[pos + radius + i];
    }
    __m128i minuses16 = _mm_set_epi8(                 //
        0, scratch_minuses[7], 0, scratch_minuses[6], //
        0, scratch_minuses[5], 0, scratch_minuses[4], //
        0, scratch_minuses[3], 0, scratch_minuses[2], //
        0, scratch_minuses[1], 0, scratch_minuses[0]);
    __m128i pluses16 = _mm_set_epi8(                //
        0, scratch_pluses[7], 0, scratch_pluses[6], //
        0, scratch_pluses[5], 0, scratch_pluses[4], //
        0, scratch_pluses[3], 0, scratch_pluses[2], //
        0, scratch_pluses[1], 0, scratch_pluses[0]);
    __m128i diffs = _mm_sub_epi16(pluses16, minuses16);

    _mm_storeu_si128((__m128i *)&scratch_diffs, diffs);

    for (size_t i = 0; i < STEP; i++) {
      // if (scratch_diffs[i] != scratch_pluses[i] - scratch_minuses[i]) {
      //   std::cout << "ERROR" << std::endl;
      // }
      // std::cout << scratch_diffs[i] << ", " << scratch_pluses[i] << ", "
      //           << scratch_minuses[i] << std::endl;
      currentSum += scratch_diffs[i];
      output[pos + i] = currentSum;
    }
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
