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

#include "xir/util/data_type.hpp"
#include <UniLog/UniLog.hpp>
#include <regex>
#include <string>

namespace xir {

DataType::Type str_to_type(const std::string& type) {
  if ("INT" == type || "int" == type) {
    return DataType::INT;
  } else if ("UINT" == type || "uint" == type) {
    return DataType::UINT;
  } else if ("XINT" == type || "xint" == type) {
    return DataType::XINT;
  } else if ("XUINT" == type || "xuint" == type) {
    return DataType::XUINT;
  } else if ("FLOAT" == type || "float" == type) {
    return DataType::FLOAT;
  } else if ("BFLOAT" == type || "bfloat" == type) {
    return DataType::BFLOAT;
  } else {
    UNI_LOG_WARNING << "The type \"" << type
                    << "\" is set to xir::DataType::UNKNOWN.";
    return DataType::UNKNOWN;
  }
}

std::string type_to_str(const DataType::Type& type) {
  switch (type) {
    case DataType::INT:
      return "INT";
    case DataType::UINT:
      return "UINT";
    case DataType::XINT:
      return "XINT";
    case DataType::XUINT:
      return "XUINT";
    case DataType::FLOAT:
      return "FLOAT";
    case DataType::BFLOAT:
      return "BFLOAT";
    default:
      return "UNKNOWN";
  }
}

DataType::DataType() : type(DataType::UNKNOWN), bit_width(0) {}

DataType::DataType(const std::string& type) {
  static const std::regex data_type_pattern{"([A-Za-z]+)([0-9]+)"};
  std::smatch data_type_matchs;
  UNI_LOG_CHECK(std::regex_match(type, data_type_matchs, data_type_pattern),
                XIR_INVALID_DATA_TYPE)
      << "\"" << type << "\" is an invalid data type, please check it again.";
  this->type = str_to_type(data_type_matchs[1].str());
  this->bit_width = std::stoi(data_type_matchs[2].str());
}

DataType::DataType(const Type& type, const std::int32_t& bit_width)
    : type(type), bit_width(bit_width) {}

const std::string DataType::to_string() const {
  return (type_to_str(this->type) + std::to_string(this->bit_width));
}

const bool DataType::valid() const {
  return ((this->type != DataType::UNKNOWN) && (this->bit_width > 0));
}

bool operator==(const DataType& lhs, const DataType& rhs) {
  return lhs.type == rhs.type && lhs.bit_width == rhs.bit_width;
}

bool operator!=(const DataType& lhs, const DataType& rhs) {
  return lhs.type != rhs.type || lhs.bit_width != rhs.bit_width;
}

template <>
DataType create_data_type<float>() {
  return DataType{DataType::FLOAT, get_bit_width<float>()};
};

template <>
DataType create_data_type<double>() {
  return DataType{DataType::FLOAT, get_bit_width<double>()};
};

template <>
DataType create_data_type<char>() {
  return DataType{DataType::INT, get_bit_width<char>()};
};

template <>
DataType create_data_type<std::int16_t>() {
  return DataType{DataType::INT, get_bit_width<std::int16_t>()};
};

template <>
DataType create_data_type<std::int32_t>() {
  return DataType{DataType::INT, get_bit_width<std::int32_t>()};
};

template <>
DataType create_data_type<std::int64_t>() {
  return DataType{DataType::INT, get_bit_width<std::int64_t>()};
};

template <>
DataType create_data_type<unsigned char>() {
  return DataType{DataType::UINT, get_bit_width<unsigned char>()};
};

template <>
DataType create_data_type<std::uint16_t>() {
  return DataType{DataType::UINT, get_bit_width<std::uint16_t>()};
};

template <>
DataType create_data_type<std::uint32_t>() {
  return DataType{DataType::UINT, get_bit_width<std::uint32_t>()};
};

template <>
DataType create_data_type<std::uint64_t>() {
  return DataType{DataType::UINT, get_bit_width<std::uint64_t>()};
};

}  // namespace xir
