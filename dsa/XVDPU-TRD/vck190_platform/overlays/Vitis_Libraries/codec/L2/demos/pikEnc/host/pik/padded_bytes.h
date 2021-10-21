// Copyright 2017 Google Inc. All Rights Reserved.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PIK_PADDED_BYTES_H_
#define PIK_PADDED_BYTES_H_

// std::vector replacement with padding to reduce bounds checks in WriteBits

#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <memory>

#include "pik/cache_aligned.h"
#include "pik/status.h"

namespace pik {

// Provides a subset of the std::vector interface with some differences:
// - allows WriteBits to write 64 bits at a time without bounds checking;
// - ONLY zero-initializes the first byte (required by WriteBits);
// - ensures cache-line alignment.
class PaddedBytes {
 public:
  // Required for output params.
  PaddedBytes() : size_(0), capacity_(0) {}

  explicit PaddedBytes(size_t size) : size_(size), capacity_(0) {
    if (size != 0) IncreaseCapacityTo(size);
  }

  PaddedBytes(const PaddedBytes& other) : size_(other.size_), capacity_(0) {
    if (size_ != 0) IncreaseCapacityTo(size_);
    if (data() != nullptr) memcpy(data(), other.data(), size_);
  }
  PaddedBytes& operator=(const PaddedBytes& other) {
    // Self-assignment is safe.
    resize(other.size());
    if (data() != nullptr) memmove(data(), other.data(), size_);
    return *this;
  }

  // default is not OK - need to set other.size_ to 0!
  PaddedBytes(PaddedBytes&& other)
      : size_(other.size_),
        capacity_(other.capacity_),
        data_(std::move(other.data_)) {
    other.size_ = other.capacity_ = 0;
  }
  PaddedBytes& operator=(PaddedBytes&& other) {
    size_ = other.size_;
    capacity_ = other.capacity_;
    data_ = std::move(other.data_);

    if (&other != this) {
      other.size_ = other.capacity_ = 0;
    }
    return *this;
  }

  void swap(PaddedBytes& other) {
    std::swap(size_, other.size_);
    std::swap(capacity_, other.capacity_);
    std::swap(data_, other.data_);
  }

  void reserve(size_t capacity) {
    if (capacity > capacity_) IncreaseCapacityTo(capacity);
  }
  // NOTE: unlike vector, this does not initialize the new data!
  // However, we guarantee that write_bits can safely append after
  // the resize, as we zero-initialize the first new byte of data.
  void resize(size_t size) {
    if (size > capacity_) IncreaseCapacityTo(size);
    size_ = (data() == nullptr) ? 0 : size;
  }
  // Amortized constant complexity due to exponential growth.
  void push_back(uint8_t x) {
    if (size_ == capacity_) {
      IncreaseCapacityTo(std::max<size_t>(3 * capacity_ / 2, 64));
      if (data() == nullptr) return;
    }

    data_[size_++] = x;
  }

  size_t size() const { return size_; }
  size_t capacity() const { return capacity_; }

  uint8_t* data() { return data_.get(); }
  const uint8_t* data() const { return data_.get(); }

  // std::vector operations implemented in terms of the public interface above.

  void clear() { resize(0); }
  bool empty() const { return size() == 0; }

  void assign(std::initializer_list<uint8_t> il) {
    resize(il.size());
    memcpy(data(), il.begin(), il.size());
  }

  uint8_t* begin() { return data(); }
  const uint8_t* begin() const { return data(); }
  uint8_t* end() { return begin() + size(); }
  const uint8_t* end() const { return begin() + size(); }

  uint8_t& operator[](const size_t i) {
    PIK_ASSERT(i < size());
    return data()[i];
  }
  const uint8_t& operator[](const size_t i) const {
    PIK_ASSERT(i < size());
    return data()[i];
  }

  uint8_t& back() {
    PIK_ASSERT(size() != 0);
    return data()[size() - 1];
  }
  const uint8_t& back() const {
    PIK_ASSERT(size() != 0);
    return data()[size() - 1];
  }

  template <typename T>
  void append(const T& other) {
    size_t old_size = size();
    resize(size() + other.size());
    memcpy(data() + old_size, other.data(), other.size());
  }

 private:
  // Copies existing data to newly allocated "data_". If allocation fails,
  // data() == nullptr and size_ = capacity_ = 0.
  void IncreaseCapacityTo(size_t capacity);

  size_t size_;
  size_t capacity_;
  CacheAlignedUniquePtr data_;
};

template <typename T>
static inline void Append(const T& s, PaddedBytes* out,
                          size_t* PIK_RESTRICT byte_pos) {
  memcpy(out->data() + *byte_pos, s.data(), s.size());
  *byte_pos += s.size();
  PIK_CHECK(*byte_pos <= out->size());
}

}  // namespace pik

#endif  // PIK_PADDED_BYTES_H_
