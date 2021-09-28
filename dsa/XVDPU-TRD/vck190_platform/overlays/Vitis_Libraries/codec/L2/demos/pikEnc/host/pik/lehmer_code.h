// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_LEHMER_CODE_H_
#define PIK_LEHMER_CODE_H_

// Library to compute the Lehmer code of a permutation and to reconstruct the
// permutation from its Lehmer code. For more details on Lehmer codes, see
// http://en.wikipedia.org/wiki/Lehmer_code

#include <cstring>
#include <memory>
#include <vector>

namespace pik {

// Computes the Lehmer code of the permutation sigma[0..len) and puts the
// result into code[0..len).
void ComputeLehmerCode(const int* sigma, int len, int* code);

// Decodes the Lehmer code in code[0..len) and puts the resulting permutation
// into sigma[0..len).
void DecodeLehmerCode(const int* code, int len, int* sigma);

// This class is an optimized Lehmer-like coder that takes the remaining
// number of possible values into account to reduce the bit usage.
class PermutationCoder {
 public:
  explicit PermutationCoder(int num_bits)
      : nbits_(num_bits), num_values_(1 << nbits_), values_(num_values_) {
    for (int i = 0; i < num_values_; ++i) values_[i] = i;
  }
  PermutationCoder(int num_bits, const unsigned char values[])
      : nbits_(num_bits), num_values_(1 << nbits_), values_(num_values_) {
    for (int i = 0; i < num_values_; ++i) values_[i] = values[i];
  }
  // number of bits needed to represent the next code.
  int num_bits() const { return nbits_; }

  // Removes (and return) the value coded by 'code'. Returns -1 in
  // case of error (invalid slot).
  int Remove(int code) {
    if (code >= num_values_ || code < 0) {
      return -1;
    }
    const int value = values_[code];
    DoRemove(code);
    return value;
  }

  // Removes 'value' from the list and assign a code + number-of-bits
  // for it. Returns false if value is not codable.
  bool RemoveValue(int value, int* code, int* nbits) {
    for (int i = 0; i < num_values_; ++i) {
      if (values_[i] == value) {
        *code = i;
        *nbits = nbits_;
        DoRemove(i);
        return true;
      }
    }
    return false;  // invalid/non-existing value was passed.
  }

 private:
  void DoRemove(int pos) {
    --num_values_;
    if (pos < num_values_) {
      memmove(&values_[pos], &values_[pos + 1],
              (num_values_ - pos) * sizeof(values_[0]));
    }
    if (((1 << nbits_) >> 1) >= num_values_) {
      --nbits_;
    }
  }

  int nbits_;
  int num_values_;
  std::vector<unsigned char> values_;
};

}  // namespace pik

#endif  // PIK_LEHMER_CODE_H_
