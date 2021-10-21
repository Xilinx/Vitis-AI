// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

// TODO(veluca): Lehmer coding takes up to 5% of the decoding time on very
// small images (32x32 pixels), and up to 1.5% for moderate-sized images.
// However, since computing and reversing Lehmer coding can be seen as a
// variation of the Fisher-Yates shuffle, we can reduce the time taken by this
// step significantly by changing the implementation to a linear time one.

#include "pik/lehmer_code.h"

#include <vector>

namespace pik {

int FindIndexAndRemove(int val, int* s, int len) {
  int idx = 0;
  for (int i = 0; i < len; ++i) {
    if (s[i] == val) {
      s[i] = -1;
      break;
    } else if (s[i] != -1) {
      ++idx;
    }
  }
  return idx;
}

void ComputeLehmerCode(const int* sigma, const int len, int* code) {
  std::vector<int> stdorder(len);
  for (int i = 0; i < len; ++i) {
    stdorder[i] = i;
  }
  for (int i = 0; i < len; ++i) {
    code[i] = FindIndexAndRemove(sigma[i], &stdorder[0], len);
  }
}

// Result is guaranteed to be one of s[0] .. s[len - 1]
int FindValueAndRemove(int idx, int* s, int len) {
  int pos = 0;
  int val = 0;
  for (int i = 0; i < len; ++i) {
    if (s[i] == -1) continue;
    if (pos == idx) {
      val = s[i];
      s[i] = -1;
      break;
    }
    ++pos;
  }
  return val;
}

// sigma[0] .. sigma[len - 1] are guaranteed to be in range 0 .. (len - 1)
void DecodeLehmerCode(const int* code, int len, int* sigma) {
  std::vector<int> stdorder(len);
  for (int i = 0; i < len; ++i) {
    stdorder[i] = i;
  }
  for (int i = 0; i < len; ++i) {
    sigma[i] = FindValueAndRemove(code[i], &stdorder[0], len);
  }
}

}  // namespace pik
