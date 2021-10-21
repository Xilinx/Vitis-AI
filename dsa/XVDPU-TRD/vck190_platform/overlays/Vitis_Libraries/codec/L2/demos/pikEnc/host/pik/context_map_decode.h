// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_CONTEXT_MAP_DECODE_H_
#define PIK_CONTEXT_MAP_DECODE_H_

#include <stddef.h>
#include <stdint.h>
#include <vector>

#include "pik/bit_reader.h"

namespace pik {

// Reads the context map from the bit stream. On calling this function,
// context_map->size() must be the number of possible context ids.
// Sets *num_htrees to the number of different histogram ids in
// *context_map.
bool DecodeContextMap(std::vector<uint8_t>* context_map, size_t* num_htrees,
                      BitReader* input);

}  // namespace pik

#endif  // PIK_CONTEXT_MAP_DECODE_H_
