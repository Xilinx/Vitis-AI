// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CONTEXT_MAP_ENCODE_H_
#define PIK_CONTEXT_MAP_ENCODE_H_

#include <stdint.h>
#include <cstddef>
#include <vector>

namespace pik {

// Encodes the given context map to the bit stream. The number of different
// histogram ids is given by num_histograms.
void EncodeContextMap(const std::vector<uint8_t>& context_map,
                      size_t num_histograms, size_t* storage_ix,
                      uint8_t* storage);

}  // namespace pik

#endif  // PIK_CONTEXT_MAP_ENCODE_H_
