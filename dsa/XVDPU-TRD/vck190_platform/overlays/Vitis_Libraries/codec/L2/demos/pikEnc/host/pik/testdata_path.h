// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_TESTDATA_PATH_H_
#define PIK_TESTDATA_PATH_H_

#include <string>


namespace pik {

static inline std::string GetTestDataPath(const std::string& filename) {
  return std::string(TEST_DATA_PATH "/") + filename;
}

}  // namespace pik

#endif  // PIK_TESTDATA_PATH_H_
