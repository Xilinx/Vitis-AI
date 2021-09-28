// Copyright 2019 Google LLC
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "pik/padded_bytes.h"

namespace pik {

void PaddedBytes::IncreaseCapacityTo(size_t capacity) {
  PIK_ASSERT(capacity > capacity_);

  // write_bits.h writes up to 7 bytes past the end.
  CacheAlignedUniquePtr new_data = AllocateArray(capacity + 8);
  if (new_data == nullptr) {
    // Allocation failed, discard all data to ensure this is noticed.
    size_ = capacity_ = 0;
    return;
  }

  if (data_ == nullptr) {
    // First allocation: ensure first byte is initialized (won't be copied).
    new_data[0] = 0;
  } else {
    // Subsequent resize: copy existing data to new location.
    memcpy(new_data.get(), data_.get(), size_);
    // Ensure that the first new byte is initialized, to allow write_bits to
    // safely append to the newly-resized PaddedBytes.
    new_data[size_] = 0;
  }

  capacity_ = capacity;
  std::swap(new_data, data_);
}

}  // namespace pik
