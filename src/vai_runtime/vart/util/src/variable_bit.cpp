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
 */
#include "vitis/ai/variable_bit.hpp"

#include <glog/logging.h>

#include <memory>
#include <sstream>
#include <algorithm>

namespace vitis {
namespace ai {

VariableBitIterator::VariableBitIterator(unsigned char* data, size_t bit_width,
                                         size_t byte_offset, size_t bit_offset)
    : data_{data},
      bit_width_{bit_width},
      byte_offset_{byte_offset},
      bit_offset_{bit_offset} {}

size_t VariableBitIterator::mask(size_t num_of_bits) const {
  return ((1ul << num_of_bits) - 1u);
}

/**
 * @return <value, num of bits for read success>
 * sample:
 *  data_={..., [a7 a6 a5 a4 a3 a2 a1 a0], [b7 b6 b5 b4 b3 b2 b1 b0],
 *                                ^ bit_offset = 2;
 *              ^ byte_offset = n;
 *  read(n,2,2) => <[a3 a2], 2>
 *
 *  data_={..., [a7 a6 a5 a4 a3 a2 a1 a0], [b7 b6 b5 b4 b3 b2 b1 b0],
 *                    ^ bit_offset = 6;
 *              ^ byte_offset = n;
 *  read(n,6,10) => <[a7 a6], 2>
 *
 */
std::pair<size_t, size_t> VariableBitIterator::read(size_t byte_offset,
                                                    size_t bit_offset,
                                                    size_t num_of_bits) {
  if (num_of_bits == 0) {
    return std::make_pair((size_t)(0u), (size_t)(0u));
  }
  auto left = (8u - bit_offset);
  auto read_size = std::min(left, num_of_bits);
  auto this_mask = mask(read_size) << bit_offset;
  auto value = ((data_[byte_offset]) & (this_mask)) >> bit_offset;
  LOG_IF(INFO, false) << "read() return: byte_offset " << byte_offset << " "
                      << "bit_offset " << bit_offset << " "
                      << "num_of_bits " << num_of_bits << " "
                      << "value " << value << " "
                      << "read_size " << read_size;
  return std::pair<size_t, size_t>(value, read_size);
}
/**
 *
 * @return write success bit num
 * sample 1:
 * data_ ={...,[x7 x6 x5 x4 x3 x2 x1 x0], ...}
 *             ^byte_offset = n
 *                         ^bit_offset = 4
 * num_of_bit=3,  val=[- - - - - a2 a1 a0]
 * write(n,4,3,val) => data_= {...,[x7 a2 a1 a0 x3 x2 x1 x0], ...}  => return:3
 *
 * sample 2:
 * data_ ={...,[x7 x6 x5 x4 x3 x2 x1 x0], ...}
 *             ^byte_offset = n
 *                               ^bit_offset = 2
 * num_of_bit=10,  val=[a7 a6 a5 a4 a3 a2 a1 a0],[- - - - - - b1 b0]
 * write(n,2,10,val) => data_= {...,[a5 a4 a3 a2 a1 a0 x1 x0], ...}  => return:6
 *
 */
size_t VariableBitIterator::write(size_t byte_offset, size_t bit_offset,
                                  size_t num_of_bits, size_t val) {
  if (num_of_bits == 0) {
    return (size_t)0u;
  }
  auto left = (8u - bit_offset);
  auto write_size = std::min(left, num_of_bits);
  auto this_mask = mask(write_size) << bit_offset;
  // sample 1 :
  // data_[n]: [a7 a6 a5 a4 a3 a2 a1 a0] => [a7 0 0 0 a3 a2 a1 a0]
  auto origin_data = data_[byte_offset] & (~this_mask);
  // sample 1:
  // val[a7 a6 a5 a4 a3 a2 a1 a0] => [0 a2 a1 a0 0 0 0 0]
  auto concat_data = (val << bit_offset) & this_mask;
  // put value to high position
  // sample 1:
  // data_[n] = {...,[x7 a2 a1 a0 x3 x2 x1 x0], ...}
  data_[byte_offset] = origin_data | concat_data;
  LOG_IF(INFO, false) << "write() : byte_offset " << byte_offset << " "
                      << "bit_offset " << bit_offset << " "
                      << "num_of_bits " << num_of_bits << " "
                      << "val " << val << " "
                      << "data_[byte_offset] " << (size_t)data_[byte_offset]
                      << " "
                      << "write_size " << write_size;
  return write_size;
}
/**
 * sample :
 * data_={...[a7 a6 a5 a4 a3 a2 a1 a0],[b7 b6 b5 b4 b3 b2 b1 b0],[c7 c6 c5 c4 c3
 * c2 c1 c0], ...}
 *           ^byte_offset_=n
 *                             ^bit_offset_2
 * bit_width_= 5
 * for(i=0;i<4;i++){
 *     cout << get() << endl;
 * }
 * output:
 * [a6,a5,a4,a3,a2]
 * [b3,b2,b1,b0,a7]
 * [c0,b7,b6,b5,b4]
 * [c5,c4,c3,c2,c1]
 *
 */
size_t VariableBitIterator::get() {
  LOG_IF(INFO, false) << "byte_offset_ " << byte_offset_ << " "
                      << "bit_offset_ " << bit_offset_ << " "
                      << "data_[byte_offset_] " << (size_t)data_[byte_offset_]
                      << " ";
  size_t ret = 0u;
  size_t value = 0u;
  size_t read_size = 0u;
  auto num_of_bits = 0u;
  auto byte_offset = byte_offset_;
  auto bit_offset = bit_offset_;
  auto bits_left = bit_width_;
  for (std::tie(value, read_size) = read(byte_offset, bit_offset, bits_left);
       read_size > 0u;
       std::tie(value, read_size) = read(byte_offset, bit_offset, bits_left)) {
    // for body
    // ret = (ret << read_size) + value;
    // when need get data from multi bytes, concat rule: fetch from low, put the
    // data taken later to the high position
    ret = ret + (value << num_of_bits);
    // iterator next
    num_of_bits += read_size;
    byte_offset = byte_offset + (bit_offset + read_size) / 8u;
    bit_offset = (bit_offset + read_size) % 8u;
    bits_left = bits_left - read_size;
  }
  return ret;
}
/**
 * sample 1:
 *data_={...[a7 a6 a5 a4 a3 a2 a1 a0],[b7 b6 b5 b4 b3 b2 b1 b0],[c7 c6 c5 c4 c3
 * c2 c1 c0], ...}
 *          ^ byte_offset_=n
 *                ^bit_offset=6
 * bit_width_ = 12
 * val={[x7 x6 x5 x4 x3 x2 x1 x0],[- - - - y3 y2 y1 y0]}
 * set(val) = {...[x1 x0 a5 a4 a3 a2 a1 a0],[y1 y0 x7 x6 x5 x4 x3 x2],[c7 c6 c5
 *c4 c3 c2 y3 y2], ...}
 *
 * sample 2:
 * data_={}
 * sample: byte_offset=0, bit_offset=0,
 * bit_width_=4
 * for(i=0;i<16;i++){
 *    set(i);
 * }
 * => data_={0x10,0x32,0x54,0x76,0x98,0xba,0xdc,0xfe}
 */
void VariableBitIterator::set(size_t val) {
  auto write_size = 0u;
  auto byte_offset = byte_offset_;
  auto bit_offset = bit_offset_;
  auto bits_left = bit_width_;
  auto value = val;
  for (write_size = write(byte_offset, bit_offset, bits_left, value);  //
       write_size > 0u;                                                //
       write_size = write(byte_offset, bit_offset, bits_left, value)) {
    byte_offset = byte_offset + (bit_offset + write_size) / 8u;
    bit_offset = (bit_offset + write_size) % 8u;
    bits_left = bits_left - write_size;
    value = value >> write_size;
  }
}
VariableBitIterator VariableBitIterator::next(size_t n_of_element) {
  auto bit_offset = bit_offset_ + bit_width_ * n_of_element;
  auto byte_offset = byte_offset_ + bit_offset / 8u;
  bit_offset = bit_offset % 8u;
  return VariableBitIterator(data_, bit_width_, byte_offset, bit_offset);
}

std::string VariableBitIterator::to_string() const {
  std::ostringstream str;
  str << "it[";
  str << "bitwidth=" << bit_width_ << ",";
  str << "byte_offset=" << byte_offset_ << ",";
  str << "bit_offset=" << bit_offset_;
  str << "]";
  return str.str();
}
}  // namespace ai
}  // namespace vitis
