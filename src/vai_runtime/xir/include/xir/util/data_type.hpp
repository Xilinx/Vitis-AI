/*
 * Copyright 2019 Xilinx Inc.
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

#pragma once
#include <climits>
#include <cstdint>
#include <string>

#include "xir/XirExport.hpp"

namespace xir {

struct XIR_DLLESPEC DataType {
  enum Type { INT, UINT, XINT, XUINT, FLOAT, BFLOAT, UNKNOWN };

  DataType();
  DataType(const std::string& type);
  DataType(const Type& type, const std::int32_t& bit_width);

  const bool valid() const;
  const std::string to_string() const;

  XIR_DLLESPEC friend bool operator==(const DataType& lhs, const DataType& rhs);
  XIR_DLLESPEC friend bool operator!=(const DataType& lhs, const DataType& rhs);

  Type type;
  std::int32_t bit_width;
};

// helper function
template <typename T>
const std::int32_t& get_bit_width() {
  static constexpr std::int32_t bit_width = sizeof(T) * CHAR_BIT;
  return bit_width;
}

template <typename T>
DataType create_data_type() {
  return DataType{"UNKNOWN0"};
};

template <>
XIR_DLLESPEC DataType create_data_type<float>();
template <>
XIR_DLLESPEC DataType create_data_type<double>();
template <>
XIR_DLLESPEC DataType create_data_type<char>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int16_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int32_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::int64_t>();
template <>
XIR_DLLESPEC DataType create_data_type<unsigned char>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint16_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint32_t>();
template <>
XIR_DLLESPEC DataType create_data_type<std::uint64_t>();

}  // namespace xir
