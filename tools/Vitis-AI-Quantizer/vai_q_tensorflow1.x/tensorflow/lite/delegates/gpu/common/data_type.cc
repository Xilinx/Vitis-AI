/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/gpu/common/data_type.h"

#include <stddef.h>
#include <string>

namespace tflite {
namespace gpu {

size_t SizeOf(DataType data_type) {
  switch (data_type) {
    case DataType::UINT8:
    case DataType::INT8:
      return 1;
    case DataType::FLOAT16:
    case DataType::INT16:
    case DataType::UINT16:
      return 2;
    case DataType::FLOAT32:
    case DataType::INT32:
    case DataType::UINT32:
      return 4;
    case DataType::FLOAT64:
    case DataType::INT64:
    case DataType::UINT64:
      return 8;
    case DataType::UNKNOWN:
      return 0;
  }
  return 0;
}

std::string ToString(DataType data_type) {
  switch (data_type) {
    case DataType::FLOAT16:
      return "float16";
    case DataType::FLOAT32:
      return "float32";
    case DataType::FLOAT64:
      return "float64";
    case DataType::INT16:
      return "int16";
    case DataType::INT32:
      return "int32";
    case DataType::INT64:
      return "int64";
    case DataType::INT8:
      return "int8";
    case DataType::UINT16:
      return "uint16";
    case DataType::UINT32:
      return "uint32";
    case DataType::UINT64:
      return "uint64";
    case DataType::UINT8:
      return "uint8";
    case DataType::UNKNOWN:
      return "unknown";
  }
  return "undefined";
}

}  // namespace gpu
}  // namespace tflite
