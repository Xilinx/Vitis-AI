/*
 * Copyright 2022-2023 Advanced Micro Devices Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 */

#pragma once
#include <memory>
#include <vector>
#include <string>
namespace vitis {
namespace ai {
class VariableBitIterator {
 public:
  // std::unique_ptr<VariableBitIterator> create(char* data, size_t bit_width,
  //                            size_t byte_offset, size_t bit_offset);

  VariableBitIterator(unsigned char* data, size_t bit_width, size_t byte_offset,
                      size_t bit_offset);

 public:
  VariableBitIterator() = delete;
  ~VariableBitIterator() = default;
  VariableBitIterator(const VariableBitIterator& other) = default;
  VariableBitIterator& operator=(const VariableBitIterator& rhs) = default;
  bool operator!=(const VariableBitIterator& rhs) { return !(*this == rhs); }
  bool operator==(const VariableBitIterator& rhs) {
    return true && data_ == rhs.data_           //
           && byte_offset_ == rhs.byte_offset_  //
           && bit_offset_ == rhs.bit_offset_    //
           && bit_width_ == rhs.bit_width_;
  }
  VariableBitIterator operator++() {
    *this = this->next(1u);
    return *this;
  }

  VariableBitIterator operator+(size_t offset_in_elt) {
    // *this = this->next(offset_in_elt);
    // return *this;
    return this->next(offset_in_elt);
  }

  size_t operator*() { return get(); }

 public:
 public:
  size_t mask(size_t num_of_bits) const;
  std::pair<size_t, size_t> read(size_t byte_offset, size_t bit_offset,
                                 size_t num_of_bits);
  size_t write(size_t byte_offset, size_t bit_offset, size_t num_of_bits,
               size_t val);
  size_t get();
  void set(size_t val);
  VariableBitIterator next(size_t n_of_element = 1u);
  std::string to_string() const;

 private:
  unsigned char* data_;
  size_t bit_width_;
  size_t byte_offset_;
  size_t bit_offset_;
};

class VariableBitView {
 public:
  VariableBitView(unsigned char* data, size_t bit_width, const size_t elements)
      : data_{data}, bit_width_{bit_width}, elements_{elements} {}
  VariableBitIterator begin() {
    return VariableBitIterator(&data_[0], bit_width_, 0u, 0u);
  }
  VariableBitIterator end() { return begin().next(elements_); }

 public:
  unsigned char* data_;
  size_t bit_width_;
  size_t elements_;
};

}  // namespace ai
}  // namespace vitis
