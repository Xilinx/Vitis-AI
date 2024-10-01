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
#include "vart/tensor_buffer.hpp"

#include <glog/logging.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <sstream>

#include "./runner_helper.hpp"
#include "vart/tensor_buffer_unowned_device.hpp"
#include "vitis/ai/env_config.hpp"
#include "xir/tensor/tensor.hpp"

DEF_ENV_PARAM(DEBUG_RUNNER, "0");

namespace vart {

TensorBuffer::TensorBuffer(const xir::Tensor* tensor) : tensor_{tensor} {}

const xir::Tensor* TensorBuffer::get_tensor() const { return tensor_; }

std::string TensorBuffer::to_string(TensorBuffer::location_t value) {
  std::string ret;
  switch (value) {
    case TensorBuffer::location_t::HOST_VIRT:
      ret = "HOST_VIRT";
      break;
    case TensorBuffer::location_t::HOST_PHY:
      ret = "HOST_PHY";
      break;
    default:
      ret =
          std::string("DEVICE_") +
          std::to_string((int)value - (int)TensorBuffer::location_t::DEVICE_0);
  }
  return ret;
}

std::string TensorBuffer::to_string() const {
  std::ostringstream out;
  out << "TensorBuffer{"
      << "@" << (void*)this;
  auto xir_tensor = get_tensor();
  out << ",tensor=" << xir_tensor->to_string();
  out << ",location=" << TensorBuffer::to_string(get_location());
  out << ",data=[";
  auto dims = xir_tensor->get_shape();
  int batch = dims[0];
  for (auto i = 0; i < batch; ++i) {
    auto idx = std::vector<std::int32_t>(dims.size());
    idx[0] = i;
    uint64_t data;
    size_t size;
    std::tie(data, size) = const_cast<TensorBuffer*>(this)->data(idx);
    if (i != 0) {
      out << ",";
    }
    out << "(";
    out << std::hex << "Virt=0x" << data << ", " << std::dec << size;
    if (get_location() == location_t::HOST_PHY ||
        (int)get_location() >= (int)location_t::DEVICE_0) {
      uint64_t phy_data;
      size_t phy_size;
      std::tie(phy_data, phy_size) =
          const_cast<TensorBuffer*>(this)->data_phy(idx);
      out << std::hex << ",Phy=0x" << phy_data << ", " << std::dec << phy_size;
    }
    out << ")";
  }
  out << "]";
  out << "}";
  return out.str();
}
void TensorBuffer::copy_from_host(size_t batch_idx, const void* buf,
                                  size_t size, size_t offset) {
  auto idx = vart::get_index_zeros(get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  idx[0] = (int)batch_idx;
  std::tie(data, tensor_size) = this->data(idx);
  auto copy_size = size;
  CHECK_LT(offset, tensor_size);
  if (tensor_size < size + offset) {
    copy_size = tensor_size - offset;
  }
  // DIRTY HACK: support copy partial data to destination, WARNING:
  // data will be lost.
  // CHECK_GE(tensor_size, size + offset)
  //     << "size=" << size << ";offset=" << offset;
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "copy_from_host:"
      << "data " << (void*)data << " "  //
      << "data+offset " << reinterpret_cast<void*>(data + offset) << "buf "
      << (void*)buf << " "            //
      << "offset " << offset << " "   //
      << "size " << copy_size << " "  //
      ;
  memcpy(reinterpret_cast<void*>(data + offset), buf, copy_size);
}

void TensorBuffer::copy_to_host(size_t batch_idx, void* buf, size_t size,
                                size_t offset) {
  auto idx = vart::get_index_zeros(get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  idx[0] = (int)batch_idx;
  std::tie(data, tensor_size) = this->data(idx);
  auto copy_size = size;
  CHECK_LT(offset, tensor_size);
  if (tensor_size < size + offset) {
    copy_size = tensor_size - offset;
  }
  // DIRTY HACK: support copy partial data to destination, WARNING:
  // data will be lost.
  // CHECK_GE(tensor_size, size + offset)
  //     << "size=" << size << ";offset=" << offset;
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "copy_to_host:"
      << "data " << (void*)data << " "  //
      << "data+offset " << reinterpret_cast<void*>(data + offset) << "buf "
      << (void*)buf << " "            //
      << "offset " << offset << " "   //
      << "size " << copy_size << " "  //
      ;
  memcpy(buf, reinterpret_cast<void*>(data + offset), copy_size);
}

static void copy_tensor_buffer_real_from_host_to_host(
    vart::TensorBuffer* tb_from, vart::TensorBuffer* tb_to, size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_from->data(idx);
    tb_to->copy_from_host(batch, reinterpret_cast<const void*>(data),
                          tensor_size, 0u);
  }
}

static void copy_tensor_buffer_real_from_host_to_phy(
    vart::TensorBuffer* tb_from, vart::TensorBuffer* tb_to, size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  auto single_batch_size = tb_from->get_tensor()->get_data_size() /
                           tb_from->get_tensor()->get_shape()[0];
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_from->data(idx);
    CHECK_LE(single_batch_size, tensor_size);
    tb_to->copy_from_host(batch, reinterpret_cast<void*>(data),
                          single_batch_size, 0u);
  }
}

static void copy_tensor_buffer_real_from_phy_to_host(
    vart::TensorBuffer* tb_from, vart::TensorBuffer* tb_to, size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  auto single_batch_size = tb_from->get_tensor()->get_data_size() /
                           tb_from->get_tensor()->get_shape()[0];
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_to->data(idx);
    CHECK_LE(single_batch_size, tensor_size);
    tb_from->copy_to_host(batch, reinterpret_cast<void*>(data),
                          single_batch_size, 0u);
  }
}

static void copy_tensor_buffer_real_from_host_to_device(
    vart::TensorBuffer* tb_from, vart::TensorBuffer* tb_to, size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  auto single_batch_size = tb_from->get_tensor()->get_data_size() /
                           tb_from->get_tensor()->get_shape()[0];
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_from->data(idx);
    CHECK_LE(single_batch_size, tensor_size);
    tb_to->copy_from_host(batch, reinterpret_cast<const void*>(data),
                          single_batch_size, 0u);
  }
}

static void copy_tensor_buffer_real_from_device_to_host(
    vart::TensorBuffer* tb_from, vart::TensorBuffer* tb_to, size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  auto single_batch_size = tb_from->get_tensor()->get_data_size() /
                           tb_from->get_tensor()->get_shape()[0];
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_to->data(idx);
    CHECK_LE(single_batch_size, tensor_size);
    tb_from->copy_to_host(batch, reinterpret_cast<void*>(data),
                          single_batch_size, 0u);
  }
}

static void copy_tensor_buffer_real_from_phy_to_phy(vart::TensorBuffer* tb_from,
                                                    vart::TensorBuffer* tb_to,
                                                    size_t batch_size) {
  auto idx = vart::get_index_zeros(tb_from->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  auto single_batch_size = tb_from->get_tensor()->get_data_size() /
                           tb_from->get_tensor()->get_shape()[0];
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tb_to->data(idx);
    CHECK_LE(single_batch_size, tensor_size);
    tb_from->copy_to_host(batch, reinterpret_cast<void*>(data),
                          single_batch_size, 0u);
    tb_to->sync_for_write(0, single_batch_size);
  }
  //  tb_to->sync_for_write(0, tb_to->get_tensor()->get_data_size());
}

static void copy_tensor_buffer_real(vart::TensorBuffer* tb_from,
                                    vart::TensorBuffer* tb_to,
                                    size_t batch_size) {
  // no checking
  if (tb_from->get_location() == vart::TensorBuffer::location_t::HOST_VIRT &&
      tb_to->get_location() == vart::TensorBuffer::location_t::HOST_VIRT) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "copy tensor buffer virt to virt";
    copy_tensor_buffer_real_from_host_to_host(tb_from, tb_to, batch_size);
  } else if (tb_from->get_location() ==
                 vart::TensorBuffer::location_t::HOST_VIRT &&
             tb_to->get_location() ==
                 vart::TensorBuffer::location_t::HOST_PHY) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "copy tensor buffer virt to phy";

    copy_tensor_buffer_real_from_host_to_phy(tb_from, tb_to, batch_size);
  } else if (tb_from->get_location() ==
                 vart::TensorBuffer::location_t::HOST_PHY &&
             tb_to->get_location() ==
                 vart::TensorBuffer::location_t::HOST_VIRT) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "copy tensor buffer phy to virt";

    copy_tensor_buffer_real_from_phy_to_host(tb_from, tb_to, batch_size);
  } else if (tb_from->get_location() ==
                 vart::TensorBuffer::location_t::HOST_VIRT &&
             tb_to->get_location() >=
                 vart::TensorBuffer::location_t::DEVICE_0) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
        << "copy tensor buffer virt to device";

    copy_tensor_buffer_real_from_host_to_device(tb_from, tb_to, batch_size);
  } else if (tb_from->get_location() >=
                 vart::TensorBuffer::location_t::DEVICE_0 &&
             tb_to->get_location() ==
                 vart::TensorBuffer::location_t::HOST_VIRT) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
        << "copy tensor buffer device to virt";
    copy_tensor_buffer_real_from_device_to_host(tb_from, tb_to, batch_size);
  } else if (tb_from->get_location() ==
                 vart::TensorBuffer::location_t::HOST_PHY &&
             tb_to->get_location() ==
                 vart::TensorBuffer::location_t::HOST_PHY) {
    copy_tensor_buffer_real_from_phy_to_phy(tb_from, tb_to, batch_size);
  } else {
    LOG(FATAL) << "TODO: from device to phy / phy to device / device to device";
  }
}

void tensor_buffer_datatype_transform(vart::TensorBuffer* tb_from,
                                      vart::TensorBuffer* tb_to, float scale) {
  auto tensor_from = tb_from->get_tensor();
  auto tensor_to = tb_to->get_tensor();
  auto from_batch_size = tensor_from->get_shape()[0];
  auto to_batch_size = tensor_to->get_shape()[0];
  size_t batch_size = std::min(from_batch_size, to_batch_size);
  std::int32_t from_dim_num = tensor_from->get_shape().size();
  auto to_dim_num = tensor_to->get_shape().size();
  CHECK_EQ(from_dim_num, to_dim_num);
  for (auto i = 1; i < from_dim_num; ++i) {
    CHECK_EQ(tensor_from->get_shape().at(i), tensor_to->get_shape().at(i))
        << "dim size is not same at dim " << i;
  }
  auto dim = std::vector<int32_t>(from_dim_num);
  auto view_from = std::pair<uint64_t, size_t>(0u, 0u);
  auto view_to = std::pair<uint64_t, size_t>(0u, 0u);
  auto from_data_type = tensor_from->get_data_type().type;
  auto to_data_type = tensor_to->get_data_type().type;
  size_t size_from = tensor_from->get_element_num() / from_batch_size;
  size_t size_to = tensor_to->get_element_num() / to_batch_size;
  CHECK_EQ(size_from, size_to) << "element numbers is not same";
  for (auto batch = 0u; batch < batch_size; ++batch) {
    dim[0] = (int)batch;
    view_from = tb_from->data(dim);
    view_to = tb_to->data(dim);
    for (auto i = 0u; i < size_from; ++i) {
      if (from_data_type == xir::DataType::FLOAT &&
          to_data_type == xir::DataType::XINT) {
        auto from_value = ((float*)view_from.first)[i];
        auto to_value = (int8_t)(from_value * scale);
        ((int8_t*)view_to.first)[i] = to_value;
      } else if (from_data_type == xir::DataType::XINT &&
                 to_data_type == xir::DataType::FLOAT) {
        auto from_value = ((int8_t*)view_from.first)[i];
        auto to_value = ((float)from_value) * scale;
        ((float*)view_to.first)[i] = to_value;
      } else {
        LOG(FATAL) << "unsupported data type conversion: from "
                   << (int)from_data_type << " to " << (int)to_data_type;
      }
    }
  }
}

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();

  return tensor->template get_attr<int>("fix_point");
}

static void dump_tensor_buffer(vart::TensorBuffer* tensor,
                               unsigned int batch_size,
                               const std::string& pre_file_name) {
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  auto idx = vart::get_index_zeros(tensor->get_tensor());
  uint64_t data = 0u;
  size_t tensor_size = 0;
  for (auto batch = 0u; batch < batch_size; ++batch) {
    idx[0] = (int)batch;
    std::tie(data, tensor_size) = tensor->data(idx);

    auto filename = pre_file_name + std::to_string(batch) + std::string(".bin");
    CHECK(std::ofstream(filename, mode).write((char*)data, tensor_size).good())
        << " faild to write to " << filename;
  }
}
static void copy_tensor_buffer_float_to_int(vart::TensorBuffer* tb_from,
                                            vart::TensorBuffer* tb_to,
                                            int batch_size) {
  CHECK(tb_from->get_location() <= vart::TensorBuffer::location_t::HOST_PHY)
      << " host can't access From Tensorbuffer.";
  auto tensor_from = tb_from->get_tensor();
  auto tensor_to = tb_to->get_tensor();
  int fixpos = get_fix_point(tensor_to);
  auto scale = std::exp2f(1.0f * (float)fixpos);
  auto new_tensor =
      xir::Tensor::create(tensor_from->get_name(), tensor_from->get_shape(),
                          {xir::DataType::XINT, 8});
  auto tb_from_fix = alloc_cpu_flat_tensor_buffer(new_tensor.get());
  tensor_buffer_datatype_transform(tb_from, tb_from_fix.get(), scale);
  copy_tensor_buffer_real(tb_from_fix.get(), tb_to, batch_size);
}

static void copy_tensor_buffer_int_to_float(vart::TensorBuffer* tb_from,
                                            vart::TensorBuffer* tb_to,
                                            int batch_size) {
  CHECK(tb_to->get_location() <= vart::TensorBuffer::location_t::HOST_PHY)
      << "host can't access To Tensorbuffer.";
  auto tensor_from = tb_from->get_tensor();
  auto tensor_to = tb_to->get_tensor();
  int fixpos = get_fix_point(tensor_from);
  auto scale = std::exp2f(-1.0f * (float)fixpos);
  auto new_tensor = xir::Tensor::create(
      tensor_to->get_name(), tensor_to->get_shape(), {xir::DataType::XINT, 8});
  auto tb_to_fix = alloc_cpu_flat_tensor_buffer(new_tensor.get());
  copy_tensor_buffer_real(tb_from, tb_to_fix.get(), batch_size);
  tensor_buffer_datatype_transform(tb_to_fix.get(), tb_to, scale);
}

// copy tensor
void TensorBuffer::copy_tensor_buffer(vart::TensorBuffer* tb_from,
                                      vart::TensorBuffer* tb_to) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "copy_tensor_buffer: "
                                        << " from:" << tb_from->to_string()  //
                                        << " to:" << tb_to->to_string();
  auto tensor_from = tb_from->get_tensor();
  auto tensor_to = tb_to->get_tensor();
  CHECK_EQ(tensor_from->get_name(), tensor_to->get_name());
  auto from_batch_size = tensor_from->get_shape()[0];
  auto to_batch_size = tensor_to->get_shape()[0];
  auto from_single_batch_size =
      tensor_from->get_element_num() / from_batch_size;
  auto to_single_batch_size = tensor_to->get_element_num() / to_batch_size;
  CHECK_EQ(from_single_batch_size, to_single_batch_size);
  auto batch_size = std::min(from_batch_size, to_batch_size);
  std::int32_t from_dim_num = tensor_from->get_shape().size();
  auto to_dim_num = tensor_to->get_shape().size();
  CHECK_EQ(from_dim_num, to_dim_num);
  for (auto i = 1; i < from_dim_num; ++i) {
    CHECK_EQ(tensor_from->get_shape().at(i),
             tensor_to->get_shape().at(i))
        << "dim size is not same at dim " << i  //
        << " from:" << tb_from->to_string()     //
        << " to:" << tb_to->to_string();
  }
  auto from_data_type = tensor_from->get_data_type().type;
  auto to_data_type = tensor_to->get_data_type().type;
  if (from_data_type == to_data_type) {  // XINT->XINT & FLOAT-> FLOAT
    copy_tensor_buffer_real(tb_from, tb_to, batch_size);
  } else if (from_data_type == xir::DataType::FLOAT &&
             to_data_type == xir::DataType::XINT) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "cope tensor buffer float to xint";
    if (ENV_PARAM(DEBUG_RUNNER) >= 2) {
      dump_tensor_buffer(tb_from, batch_size,
                         std::string("float_to_int_tb_from_"));
    }
    copy_tensor_buffer_float_to_int(tb_from, tb_to, batch_size);
    if (ENV_PARAM(DEBUG_RUNNER) >= 2) {
      dump_tensor_buffer(tb_to, batch_size, std::string("float_to_int_tb_to_"));
    }

  } else if (from_data_type == xir::DataType::XINT &&
             to_data_type == xir::DataType::FLOAT) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "cope tensor buffer xint to float";
    if (ENV_PARAM(DEBUG_RUNNER) >= 2) {
      dump_tensor_buffer(tb_from, batch_size,
                         std::string("int_to_float_tb_from_"));
    }
    copy_tensor_buffer_int_to_float(tb_from, tb_to, batch_size);
    if (ENV_PARAM(DEBUG_RUNNER) >= 2) {
      dump_tensor_buffer(tb_to, batch_size, std::string("int_to_float_tb_to_"));
    }

  } else {
    LOG(FATAL) << "unsupported data type conversion: from "
               << (int)from_data_type << " to " << (int)to_data_type;
  }
  return;
}

std::unique_ptr<TensorBuffer> TensorBuffer::create_unowned_device_tensor_buffer(
    const xir::Tensor* tensor, uint64_t batch_addr[], size_t addr_arrsize) {
  CHECK(batch_addr != nullptr) << "batch_addr is null, "
                               << "tensor: " << tensor->to_string();
  CHECK_LE(addr_arrsize, tensor->get_shape()[0])
      << "addr_arrsize should not exceed batchsize " << tensor->get_shape()[0]
      << " tensor: " << tensor->to_string();
  CHECK(tensor->has_attr("ddr_addr"))
      << "tensor should have ddr_addr attribute, "
      << "tensor: " << tensor->to_string();
  CHECK_EQ(tensor->template get_attr<int>("ddr_addr"), 0)
      << "tensor to create unowned device tensor buffer should have ddr_addr "
         "attribute 0"
      << "tensor: " << tensor->to_string();

  return std::make_unique<TensorBufferUnownedDevice>(tensor, batch_addr,
                                                     addr_arrsize);
}

}  // namespace vart
