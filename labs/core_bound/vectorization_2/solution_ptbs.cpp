#include "solution.hpp"

uint16_t checksum(const Blob &blob) {
  uint32_t acc = 0;
  for (auto value : blob) {
    acc += value;
  }
  acc = uint16_t(acc) + (acc >> 16);
  acc = uint16_t(acc) + (acc >> 16);
  return acc;
}
