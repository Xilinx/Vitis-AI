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
 */

#include <glog/logging.h>
#include "./convert_tensor.hpp"

vitis::ai::Tensor::DataType convert_data_type(xir::DataType data_type) {
  auto ret = vitis::ai::Tensor::DataType::UNKNOWN;
  switch (data_type.type) {
    case xir::DataType::XINT:
    case xir::DataType::INT:
      switch (data_type.bit_width) {
        case 4:
          LOG(FATAL) << "does not support 4bits";
          break;
        case 8:
          ret = vitis::ai::Tensor::DataType::INT8;
          break;
        case 16:
          ret = vitis::ai::Tensor::DataType::INT16;
          break;
        case 32:
          ret = vitis::ai::Tensor::DataType::INT32;
          break;
        case 64:
          ret = vitis::ai::Tensor::DataType::UINT64;
          break;
        default:
          LOG(FATAL) << "unknown bitwidth " << data_type.bit_width;
      }
      break;
    case xir::DataType::XUINT:
    case xir::DataType::UINT:
      switch (data_type.bit_width) {
        case 4:
          LOG(FATAL) << "does not support 4bits";
          break;
        case 8:
          ret = vitis::ai::Tensor::DataType::UINT8;
          break;
        case 16:
          ret = vitis::ai::Tensor::DataType::UINT16;
          break;
        case 32:
          ret = vitis::ai::Tensor::DataType::UINT32;
          break;
        case 64:
          ret = vitis::ai::Tensor::DataType::UINT64;
          break;
        default:
          LOG(FATAL) << "unknown bitwidth " << data_type.bit_width;
      }
      break;
    case xir::DataType ::FLOAT:
      ret = vitis::ai::Tensor::DataType::FLOAT;
      break;
    case xir::DataType ::UNKNOWN:
      ret = vitis::ai::Tensor::DataType::UNKNOWN;
      break;
    default:
      LOG(FATAL) << "cannot convert data type";
  };
  return ret;
}

xir::DataType convert_data_type(vitis::ai::Tensor::DataType data_type) {
  auto ret = xir::DataType{xir::DataType::UNKNOWN, 8};
  switch (data_type) {
    case vitis::ai::Tensor::DataType::INT8:
      ret = xir::DataType{xir::DataType::XINT, 8};
      break;
    case vitis::ai::Tensor::DataType::UINT8:
      // DIRTY HACK, vitis::ai::Tensor does not support XINT/XUINT
      ret = xir::DataType{xir::DataType::XUINT, 8};
      break;
    case vitis::ai::Tensor::DataType::INT16:
      ret = xir::DataType{xir::DataType::INT, 16};
      break;
    case vitis::ai::Tensor::DataType::UINT16:
      ret = xir::DataType{xir::DataType::UINT, 16};
      break;
    case vitis::ai::Tensor::DataType::INT32:
      ret = xir::DataType{xir::DataType::INT, 32};
      break;
    case vitis::ai::Tensor::DataType::UINT32:
      ret = xir::DataType{xir::DataType::UINT, 32};
      break;
    case vitis::ai::Tensor::DataType::INT64:
      ret = xir::DataType{xir::DataType::INT, 64};
      break;
    case vitis::ai::Tensor::DataType::UINT64:
      ret = xir::DataType{xir::DataType::UINT, 64};
      break;
    case vitis::ai::Tensor::DataType::FLOAT:
      ret = xir::DataType{xir::DataType::FLOAT, 32};
      break;
    case vitis::ai::Tensor::DataType::DOUBLE:
      LOG(FATAL) << "xir does not support DOUBLE";
      break;
    case vitis::ai::Tensor::DataType::UNKNOWN:
      ret = xir::DataType{xir::DataType::UNKNOWN, 8};
      break;
    default:
      LOG(FATAL) << "unknown type";
  };
  return ret;
}

using std::unique_ptr;
unique_ptr<vitis::ai::Tensor> convert_tensor(const xir::Tensor* xir_tensor) {
  return std::unique_ptr<vitis::ai::Tensor>(
      new vitis::ai::Tensor(xir_tensor->get_name(), xir_tensor->get_shape(),
                            convert_data_type(xir_tensor->get_data_type())));
}

unique_ptr<xir::Tensor> convert_tensor(
    const vitis::ai::Tensor* vitis_ai_tensor) {
  return xir::Tensor::create(
      vitis_ai_tensor->get_name(), vitis_ai_tensor->get_dims(),
      convert_data_type(vitis_ai_tensor->get_data_type()));
}

std::vector<unique_ptr<vitis::ai::Tensor>> convert_tensors(
    const std::vector<const xir::Tensor*>& xir_tensors) {
  auto ret = std::vector<unique_ptr<vitis::ai::Tensor>>(xir_tensors.size());
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = convert_tensor(xir_tensors[i]);
  }
  return ret;
}
