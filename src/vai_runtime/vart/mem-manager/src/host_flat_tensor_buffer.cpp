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
 * Modifications Copyright (C) 2022 Advanced Micro Devices, Inc. All Rights
 * Reserved.
 */

#include "vart/mm/host_flat_tensor_buffer.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <xir/util/tool_function.hpp>

#include "vart/util_4bit.hpp"

namespace vart {
namespace mm {

std::vector<int32_t> get_strides(const xir::Tensor* tensor, bool ignore_def) {
  auto shape = tensor->get_shape();
  auto data_type = tensor->get_data_type();
  auto strides = std::vector<int32_t>(shape.size());
  if (tensor->has_attr("strides") && !ignore_def) {
    strides = tensor->get_attr<std::vector<int32_t>>("strides");
    UNI_LOG_CHECK(strides.size() == shape.size(),
                  VART_TENSOR_BUFFER_CONSTRUCTION_FAIL)
        << "shape(size=" << shape.size()
        << ") and strides(size=" << strides.size()
        << ") have to be in same size";
    auto step = data_type.bit_width;
    for (int idx = shape.size() - 1; idx >= 0; idx--) {
      UNI_LOG_CHECK(strides[idx] >= step, VART_TENSOR_BUFFER_CONSTRUCTION_FAIL)
          << "Invalid strides! (strides=" << xir::to_string(strides)
          << ", shape=" << xir::to_string(shape)
          << ", bit_width=" << data_type.bit_width << ")";
      step *= shape[idx];
    }
  } else if (tensor->has_attr("stride") && !ignore_def) {
    auto stride = tensor->get_attr<std::vector<int32_t>>("stride");
    UNI_LOG_CHECK(stride.size() == shape.size(),
                  VART_TENSOR_BUFFER_CONSTRUCTION_FAIL)
        << "shape(size=" << shape.size()
        << ") and stride(size=" << stride.size() << ") have to be in same size";
    auto step = 1;
    for (int idx = shape.size() - 1; idx >= 0; idx--) {
      UNI_LOG_CHECK(stride[idx] >= step, VART_TENSOR_BUFFER_CONSTRUCTION_FAIL)
          << "Invalid stride! (stride=" << xir::to_string(stride)
          << ", shape=" << xir::to_string(shape) << ")";
      strides[idx] = stride[idx] * data_type.bit_width;
      step *= shape[idx];
    }
  } else {
    auto step = data_type.bit_width;
    for (int idx = shape.size() - 1; idx >= 0; idx--) {
      strides[idx] = step;
      step *= shape[idx];
    }
  }
  return strides;
}

static int32_t get_last_continued_dim(int32_t bit_width,
                                      std::vector<int32_t> shape,
                                      std::vector<int32_t> strides) {
  auto last_continued_dim = shape.size();
  auto step = 1;
  for (int idx = shape.size() - 1; idx >= 0; idx--) {
    if (step * bit_width != strides[idx]) break;
    step *= shape[idx];
    last_continued_dim--;
  }
  return last_continued_dim;
}

HostFlatTensorBuffer::HostFlatTensorBuffer(const xir::Tensor* tensor)
    : TensorBuffer(tensor),
      data_type(tensor_->get_data_type()),
      shape(tensor_->get_shape()),
      strides(get_strides(tensor_)),
      last_continued_dim(
          get_last_continued_dim(data_type.bit_width, shape, strides)) {
  data_ = new char[static_cast<uint32_t>(
      std::ceil(shape.front() * strides.front() / 8.f))];
}

HostFlatTensorBuffer::HostFlatTensorBuffer(const xir::Tensor* tensor,
                                           std::vector<int32_t> strides)
    : TensorBuffer(tensor),
      data_type(tensor_->get_data_type()),
      shape(tensor_->get_shape()),
      strides(strides),
      last_continued_dim(
          get_last_continued_dim(data_type.bit_width, shape, strides)) {
  data_ = new char[static_cast<uint32_t>(
      std::ceil(shape.front() * strides.front() / 8.f))];
}

HostFlatTensorBuffer::~HostFlatTensorBuffer() { delete[] data_; }

static size_t size_of_element_in_bytes(size_t num_of_element,
                                       size_t bit_width) {
  auto ceil = [](size_t a, size_t b) {
    return a / b + ((a % b == 0) ? 0u : 1u);
  };
  return ceil(num_of_element * bit_width, 8u);
}

std::pair<uint64_t, size_t> HostFlatTensorBuffer::data(
    const std::vector<int> idx) {
  auto valid_size = 1U;
  for (int k = shape.size() - 1; k >= static_cast<int>(last_continued_dim);
       k--) {
    valid_size *= shape[k];
  }
  if (idx.size() == 0U) {
    return {reinterpret_cast<uint64_t>(data_), size_of_element_in_bytes(valid_size, get_tensor()->get_data_type().bit_width)};
  }
  UNI_LOG_CHECK(idx.size() == shape.size(), VART_TENSOR_BUFFER_INVALID_INDEX)
      << "shape=" << xir::to_string(shape) << ", index=" << xir::to_string(idx);
  auto offset = 0U;
  for (auto k = 0U; k < idx.size(); k++) {
    UNI_LOG_CHECK(idx[k] >= 0 && idx[k] < static_cast<int32_t>(shape[k]),
                  VART_TENSOR_BUFFER_INVALID_INDEX)
        << "shape=" << xir::to_string(shape)
        << ", index=" << xir::to_string(idx);
    offset += idx[k] * strides[k];
  }
  UNI_LOG_CHECK(offset % 8 == 0, VART_TENSOR_BUFFER_INVALID_INDEX)
      << "offset on unaligned addr! "
      << "idx=" << xir::to_string(idx)
      << ", strides=" << xir::to_string(strides);
  auto element_offset = 0U;
  auto step = 1U;
  for (int k = shape.size() - 1; k >= static_cast<int>(last_continued_dim);
       k--) {
    element_offset += idx[k] * step;
    step *= shape[k];
  }
  return {reinterpret_cast<uint64_t>(data_ + offset / 8),
          size_of_element_in_bytes(valid_size - element_offset,
                                   get_tensor()->get_data_type().bit_width)};
}

static int32_t max_common_divisor(int32_t a, int32_t b) {
  int32_t c = b;
  while (a % b != 0) {
    c = a % b;
    a = b;
    b = c;
  }
  return c;
}

static void clear_buffer(HostFlatTensorBuffer* buffer) {
  auto ptr = reinterpret_cast<uint8_t*>(buffer->data({}).first);
  for (auto idx = 0;
       idx < std::ceil(buffer->shape.front() * buffer->strides.front() / 8.f);
       idx++) {
    ptr[idx] = 0U;
  }
}

static inline void set_data_4bit(uint8_t* ptr, const std::vector<int32_t>& idx,
                                 const std::vector<int32_t>& strides,
                                 uint8_t value) {
  auto offset = 0U;
  for (auto k = 0U; k < idx.size(); k++) {
    offset += idx[k] * strides[k];
  }
  ptr[offset / 8] = (offset % 8 == 0)
                        ? ((ptr[offset / 8] & 0xf0) | value)
                        : ((ptr[offset / 8] & 0x0f) | (value << 4));
}

static inline uint8_t get_data_4bit(uint8_t* ptr,
                                    const std::vector<int32_t>& idx,
                                    const std::vector<int32_t>& strides) {
  auto offset = 0U;
  for (auto k = 0U; k < idx.size(); k++) {
    offset += idx[k] * strides[k];
  }
  return (offset % 8 == 0) ? (ptr[offset / 8] & 0x0f) : (ptr[offset / 8] >> 4);
}

static uint32_t get_file_size(const std::string& file_name) {
  std::ifstream infile(file_name, std::ios::binary | std::ios::ate);
  UNI_LOG_CHECK(infile.is_open(), VART_FAILED_FILE_OPERATION)
      << "Cannot open file " << file_name;
  return infile.tellg();
}

static void init_from_file_4bit(HostFlatTensorBuffer* buffer,
                                const std::string& file_name) {
  clear_buffer(buffer);
  auto ptr = reinterpret_cast<uint8_t*>(buffer->data({}).first);
  // TODO: check file size
  std::ifstream infile(file_name, std::ios_base::in | std::ios_base::binary);
  UNI_LOG_CHECK(infile.is_open(), VART_FAILED_FILE_OPERATION)
      << "Cannot open " << file_name;
  auto num = buffer->get_tensor()->get_element_num();
  auto idx = std::vector<int32_t>(buffer->shape.size(), 0U);
  uint8_t word;
  // TODO: support odd num
  for (auto idx_byte = 0U; idx_byte < std::ceil(num / 2.f); idx_byte++) {
    infile.read(reinterpret_cast<char*>(&word), 1U);
    set_data_4bit(ptr, idx, buffer->strides, (word & 0x0f));
    bump_idx(idx, buffer->shape);
    set_data_4bit(ptr, idx, buffer->strides, (word >> 4));
    bump_idx(idx, buffer->shape);
  }
}

static void init_from_file_common(HostFlatTensorBuffer* buffer,
                                  std::string file_name) {
  clear_buffer(buffer);
  auto ptr = reinterpret_cast<uint8_t*>(buffer->data({}).first);

  auto num = buffer->get_tensor()->get_element_num();
  auto bytes = buffer->data_type.bit_width / 8;

  auto file_size = get_file_size(file_name);
  UNI_LOG_CHECK(file_size == static_cast<uint32_t>(num * bytes),
                VART_UNEXPECTED_FILE_SIZE)
      << "Failed to initialize tensor buffer "
      << buffer->get_tensor()->get_name() << " from file " << file_name
      << ". file size: " << file_size << " expected size: " << num * bytes;

  std::ifstream infile(file_name, std::ios_base::in | std::ios_base::binary);
  UNI_LOG_CHECK(infile.is_open(), VART_FAILED_FILE_OPERATION)
      << "Cannot open " << file_name;
  auto idx = std::vector<int32_t>(buffer->shape.size(), 0);
  uint8_t word;
  for (auto idx_num = 0; idx_num < num; idx_num++) {
    auto offset = 0U;
    for (auto k = 0U; k < idx.size(); k++) {
      offset += idx[k] * buffer->strides[k] / 8;
    }
    for (auto idx_bytes = 0; idx_bytes < bytes; idx_bytes++) {
      infile.read(reinterpret_cast<char*>(&word), 1U);
      ptr[offset + idx_bytes] = word;
    }
    bump_idx(idx, buffer->shape);
  }
}

static int32_t get_unit(HostFlatTensorBuffer* buffer) {
  int32_t unit = buffer->data_type.bit_width;
  auto strides = buffer->strides;
  for (auto iter = strides.begin(); iter != strides.end();
       iter = std::next(iter)) {
    unit = max_common_divisor(unit, *iter);
  }
  return unit;
}

void init_from_file(HostFlatTensorBuffer* buffer, std::string file_name) {
  auto unit = get_unit(buffer);
  if (unit == 4)
    init_from_file_4bit(buffer, file_name);
  else if (unit % 8 == 0)
    init_from_file_common(buffer, file_name);
  else
    UNI_LOG_FATAL(VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
        << "init_from_file, strides=" << xir::to_string(buffer->strides);
}

static void dump_to_file_4bit(HostFlatTensorBuffer* buffer,
                              std::string file_name) {
  auto ptr = reinterpret_cast<uint8_t*>(buffer->data({}).first);
  // TODO: check file size
  std::ofstream outfile(file_name, std::ios_base::out | std::ios_base::binary |
                                       std::ios_base::trunc);
  UNI_LOG_CHECK(outfile.is_open(), VART_FAILED_FILE_OPERATION)
      << "Cannot open " << file_name;
  auto num = buffer->get_tensor()->get_element_num();
  auto idx = std::vector<int32_t>(buffer->shape.size(), 0U);
  uint8_t word;
  // TODO: support odd num
  for (auto idx_byte = 0U; idx_byte < std::ceil(num / 2.f); idx_byte++) {
    word = 0U;
    word = word | get_data_4bit(ptr, idx, buffer->strides);
    bump_idx(idx, buffer->shape);
    word = word | (get_data_4bit(ptr, idx, buffer->strides) << 4);
    bump_idx(idx, buffer->shape);
    outfile.write(reinterpret_cast<char*>(&word), 1U);
  }
}

static void dump_to_file_common(HostFlatTensorBuffer* buffer,
                                std::string file_name) {
  auto ptr = reinterpret_cast<uint8_t*>(buffer->data({}).first);
  // TODO: check file size
  std::ofstream outfile(file_name, std::ios_base::out | std::ios_base::binary |
                                       std::ios_base::trunc);
  UNI_LOG_CHECK(outfile.is_open(), VART_FAILED_FILE_OPERATION)
      << "Cannot open " << file_name;
  auto num = buffer->get_tensor()->get_element_num();
  auto bytes = buffer->data_type.bit_width / 8;
  auto idx = std::vector<int32_t>(buffer->shape.size(), 0);
  for (auto idx_num = 0; idx_num < num; idx_num++) {
    auto offset = 0U;
    for (auto k = 0U; k < idx.size(); k++) {
      offset += idx[k] * buffer->strides[k] / 8;
    }
    for (auto idx_bytes = 0; idx_bytes < bytes; idx_bytes++) {
      outfile.write(reinterpret_cast<char*>(ptr + offset + idx_bytes), 1U);
    }
    bump_idx(idx, buffer->shape);
  }
}

void dump_to_file(HostFlatTensorBuffer* buffer, std::string file_name) {
  auto unit = get_unit(buffer);
  if (unit == 4)
    dump_to_file_4bit(buffer, file_name);
  else if (unit % 8 == 0)
    dump_to_file_common(buffer, file_name);
  else
    UNI_LOG_FATAL(VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
        << "dump_to_file, strides=" << xir::to_string(buffer->strides);
}

static void tensorbuffer_copy_common(HostFlatTensorBuffer* buffer_src,
                                     HostFlatTensorBuffer* buffer_dest) {
  auto ptr_src = reinterpret_cast<uint8_t*>(buffer_src->data({}).first);
  auto ptr_dest = reinterpret_cast<uint8_t*>(buffer_dest->data({}).first);
  auto num = buffer_src->get_tensor()->get_element_num();
  auto bytes = buffer_src->data_type.bit_width / 8;
  auto idx_src = std::vector<int32_t>(buffer_src->shape.size(), 0);
  auto idx_dest = std::vector<int32_t>(buffer_dest->shape.size(), 0);
  for (auto idx_num = 0; idx_num < num; idx_num++) {
    auto offset_src = 0U;
    for (auto k = 0U; k < idx_src.size(); k++) {
      offset_src += idx_src[k] * buffer_src->strides[k] / 8;
    }
    auto offset_dest = 0U;
    for (auto k = 0U; k < idx_dest.size(); k++) {
      offset_dest += idx_dest[k] * buffer_dest->strides[k] / 8;
    }
    for (auto idx_bytes = 0; idx_bytes < bytes; idx_bytes++) {
      *(ptr_dest + offset_dest + idx_bytes) =
          *(ptr_src + offset_src + idx_bytes);
    }
    bump_idx(idx_src, buffer_src->shape);
    bump_idx(idx_dest, buffer_dest->shape);
  }
}

void tensorbuffer_copy(HostFlatTensorBuffer* buffer_src,
                       HostFlatTensorBuffer* buffer_dest) {
  UNI_LOG_CHECK(buffer_src->data_type == buffer_dest->data_type,
                VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "tensorbuffer_copy, data types mismatch, "
      << buffer_src->data_type.to_string() << " to "
      << buffer_dest->data_type.to_string();
  UNI_LOG_CHECK(buffer_src->get_tensor()->get_element_num() ==
                    buffer_dest->get_tensor()->get_element_num(),
                VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "tensorbuffer_copy, element numbers mismatch, "
      << xir::to_string(buffer_src->shape) << " to "
      << xir::to_string(buffer_dest->shape);

  auto unit = max_common_divisor(get_unit(buffer_src), get_unit(buffer_dest));
  if (unit % 8 == 0)
    tensorbuffer_copy_common(buffer_src, buffer_dest);
  else
    UNI_LOG_FATAL(VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
        << "tensorbuffer_copy, unsupported strides, "
        << xir::to_string(buffer_src->strides) << " to "
        << xir::to_string(buffer_dest->strides);
}

template <typename T>
static T DPURound(double data, T data_min, T data_max) {
  T rlt = 0;
  if (data > data_max) {
    rlt = data_max;
  } else if (data < data_min) {
    rlt = data_min;
  } else if (data < 0 && (data - std::floor(data)) == 0.5) {
    rlt = static_cast<T>(std::ceil(data));
  } else {
    rlt = static_cast<T>(std::round(data));
  }
  return rlt;
}

static int32_t float_2_xint(float data, int32_t fix_point, int32_t bit_width) {
  auto step = std::pow(2.f, fix_point);
  auto lower_bound = -1 * std::pow(2, bit_width - 1);
  auto upper_bound = std::pow(2, bit_width - 1) - 1;
  data *= step;
  return DPURound<int32_t>(data, lower_bound, upper_bound);
}

static uint32_t float_2_xuint(float data, int32_t fix_point,
                              int32_t bit_width) {
  auto step = std::pow(2.f, fix_point);
  auto lower_bound = 0U;
  auto upper_bound = std::pow(2, bit_width) - 1;
  data *= step;
  return DPURound<uint32_t>(data, lower_bound, upper_bound);
}

static void copy_to_fix_buffer_4bit(TensorBuffer* buffer_in,
                                    TensorBuffer* buffer_out,
                                    int32_t fix_point) {
  auto num = buffer_in->get_tensor()->get_element_num();
  auto idx =
      std::vector<int32_t>(buffer_out->get_tensor()->get_shape().size(), 0U);
  auto ptr = reinterpret_cast<uint8_t*>(buffer_out->data({}).first);
  uint8_t value;
  for (auto idx_num = 0; idx_num < num; idx_num++) {
    auto float_value = *reinterpret_cast<float*>(buffer_in->data(idx).first);
    if (buffer_out->get_tensor()->get_data_type().type == xir::DataType::XINT) {
      auto tmp =
          float_2_xint(float_value, fix_point,
                       buffer_out->get_tensor()->get_data_type().bit_width);
      value = *reinterpret_cast<uint8_t*>(&tmp);
    } else {
      auto tmp =
          float_2_xuint(float_value, fix_point,
                        buffer_out->get_tensor()->get_data_type().bit_width);
      value = *reinterpret_cast<uint8_t*>(&tmp);
    }
    set_data_4bit(ptr, idx, get_strides(buffer_out->get_tensor()),
                  value & 0x0f);
    bump_idx(idx, buffer_out->get_tensor()->get_shape());
  }
}

static void copy_to_fix_buffer_common(TensorBuffer* buffer_in,
                                      TensorBuffer* buffer_out,
                                      int32_t fix_point) {
  auto num = buffer_in->get_tensor()->get_element_num();
  auto idx =
      std::vector<int32_t>(buffer_out->get_tensor()->get_shape().size(), 0U);
  char* data_src;
  char* data_dst;
  int32_t signed_value;
  uint32_t unsigned_value;
  for (auto idx_num = 0; idx_num < num; idx_num++) {
    auto float_value = *reinterpret_cast<float*>(buffer_in->data(idx).first);
    if (buffer_out->get_tensor()->get_data_type().type == xir::DataType::XINT) {
      signed_value =
          float_2_xint(float_value, fix_point,
                       buffer_out->get_tensor()->get_data_type().bit_width);
      data_src = reinterpret_cast<char*>(&signed_value);
    } else {
      unsigned_value =
          float_2_xuint(float_value, fix_point,
                        buffer_out->get_tensor()->get_data_type().bit_width);
      data_src = reinterpret_cast<char*>(&unsigned_value);
    }
    data_dst = reinterpret_cast<char*>(buffer_out->data(idx).first);
    for (auto idx_byte = 0;
         idx_byte < buffer_out->get_tensor()->get_data_type().bit_width / 8;
         idx_byte++) {
      data_dst[idx_byte] = data_src[idx_byte];
    }
    bump_idx(idx, buffer_out->get_tensor()->get_shape());
  }
}

std::pair<std::unique_ptr<HostFlatTensorBuffer>, std::unique_ptr<xir::Tensor>>
transform_to_fix_buffer(TensorBuffer* buffer, int32_t fix_point,
                        int32_t bit_width, bool if_signed,
                        std::string round_mode) {
  UNI_LOG_CHECK(
      buffer->get_tensor()->get_data_type().type == xir::DataType::FLOAT &&
          buffer->get_tensor()->get_data_type().bit_width == 32,
      VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint only supports FLOAT32, but input is "
      << buffer->get_tensor()->get_data_type().to_string();
  UNI_LOG_CHECK(round_mode == "DPU_ROUND", VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint only supports DPU_ROUND mode";
  auto tensor = buffer->get_tensor();
  auto new_tensor = xir::Tensor::create(
      tensor->get_name() + "_fix_manually", tensor->get_shape(),
      {if_signed ? xir::DataType::XINT : xir::DataType::XUINT, bit_width});
  auto new_buffer = std::make_unique<HostFlatTensorBuffer>(
      new_tensor.get(), get_strides(new_tensor.get(), true));
  if (bit_width == 4)
    copy_to_fix_buffer_4bit(buffer, new_buffer.get(), fix_point);
  else if (bit_width % 8 == 0 && bit_width <= 32)
    copy_to_fix_buffer_common(buffer, new_buffer.get(), fix_point);
  else
    UNI_LOG_FATAL(VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
        << "transform_float_2_xint doesn't support " << bit_width << "bit data";
  return {std::move(new_buffer), std::move(new_tensor)};
}

void transform_to_fix_buffer(TensorBuffer* buffer_src,
                             TensorBuffer* buffer_dest,
                             std::string round_mode) {
  auto type_src = buffer_src->get_tensor()->get_data_type();
  UNI_LOG_CHECK(
      type_src.type == xir::DataType::FLOAT && type_src.bit_width == 32,
      VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint only supports FLOAT32 input, but it is "
      << type_src.to_string();
  auto type_dest = buffer_dest->get_tensor()->get_data_type();
  UNI_LOG_CHECK(type_dest.type == xir::DataType::XINT ||
                    type_dest.type == xir::DataType::XUINT,
                VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint only supports XINT/XUINT output, but it is "
      << type_dest.to_string();

  UNI_LOG_CHECK(round_mode == "DPU_ROUND", VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint only supports DPU_ROUND mode";

  UNI_LOG_CHECK(buffer_src->get_tensor()->get_element_num() ==
                    buffer_dest->get_tensor()->get_element_num(),
                VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
      << "transform_float_2_xint, element numbers mismatch, "
      << xir::to_string(buffer_src->get_tensor()->get_shape()) << " to "
      << xir::to_string(buffer_dest->get_tensor()->get_shape());

  auto fix_point = buffer_dest->get_tensor()->get_attr<int32_t>("fix_point");
  if (type_dest.bit_width == 4)
    copy_to_fix_buffer_4bit(buffer_src, buffer_dest, fix_point);
  else if (type_dest.bit_width % 8 == 0 && type_dest.bit_width <= 32)
    copy_to_fix_buffer_common(buffer_src, buffer_dest, fix_point);
  else
    UNI_LOG_FATAL(VART_TENSOR_BUFFER_UNSUPPORT_FORMAT)
        << "transform_float_2_xint doesn't support " << type_dest.bit_width
        << "bit output";
}

}  // namespace mm
}  // namespace vart
