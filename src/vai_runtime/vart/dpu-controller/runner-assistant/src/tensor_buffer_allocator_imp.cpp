/*
 *  Copyright 2022-2023 Advanced Micro Devices, Inc.
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

#include "./tensor_buffer_allocator_imp.hpp"

#include <algorithm>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <xir/graph/subgraph.hpp>
#include <xir/tensor/tensor.hpp>
#include <UniLog/UniLog.hpp>

#include "./tensor_buffer_imp_host.hpp"
#include "./tensor_buffer_imp_host_phy.hpp"
#include "./tensor_buffer_imp_view.hpp"
#include "vart/zero_copy_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"

DEF_ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR, "0")
namespace {
using reg_type_t = vart::reg_type_t;

TensorBufferAllocatorImp::TensorBufferAllocatorImp(const xir::Attrs* attrs)
    : attrs_{attrs} {}

static size_t get_device_id(const xir::Attrs* attrs) {
  size_t ret = std::numeric_limits<size_t>::max();
  if (attrs->has_attr("__device_id__")) {
    ret = attrs->get_attr<size_t>("__device_id__");
  }
  return ret;
}

static size_t get_device_core_id(const xir::Attrs* attrs) {
  size_t ret = std::numeric_limits<size_t>::max();
  if (attrs->has_attr("__device_core_id__")) {
    ret = attrs->get_attr<size_t>("__device_core_id__");
  }
  return ret;
}

static std::string get_cu_name(const xir::Attrs* attrs) {
  std::string ret;
  if (attrs->has_attr("__cu_name__")) {
    ret = attrs->get_attr<std::string>("__cu_name__");
  }
  return ret;
}

static size_t get_reg_id(const xir::Tensor* tensor) {
  auto reg_id = std::numeric_limits<size_t>::max();
  if (tensor->has_attr("reg_id")) {
    reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  };
  return reg_id;
}

static vart::TensorBuffer::location_t get_location(
    const xir::Subgraph* subgraph, const xir::Attrs* attrs) {
  auto ret = vart::TensorBuffer::location_t::HOST_VIRT;
  auto attr_name = subgraph->get_name() + ":__tensor_buffer_location__";
  if (attrs->has_attr(attr_name)) {
    ret = (vart::TensorBuffer::location_t)attrs->template get_attr<int>(
        attr_name);
  };
  return ret;
}

static size_t get_offset(const xir::Tensor* tensor) {
  size_t ret = 0u;
  if (tensor->has_attr("ddr_addr")) {
    ret = (size_t)tensor->template get_attr<int>("ddr_addr");
  };
  return ret;
}

static size_t get_batch(const xir::Tensor* tensor) {
  return tensor->get_shape()[0];
}

static size_t get_batch(const xir::Attrs* tensor) {
  auto ret = std::numeric_limits<size_t>::max();
  if (tensor->has_attr("__batch__")) {
    ret = (size_t)tensor->template get_attr<size_t>("__batch__");
  } else {
    LOG(FATAL) << "__batch__ is not set";
  }
  return ret;
}

static size_t get_size(const xir::Tensor* tensor) {
  return (size_t)tensor->get_data_size() / tensor->get_shape()[0];
}

struct reg_info_t {
  vart::reg_basic_info_t basic_info_;
  vart::TensorBuffer::location_t location;
  size_t batch;
  size_t device_id;
  size_t device_core_id;
  std::string cu_name;
  std::shared_ptr<vart::TensorBuffer> reg_tensor_buffer;
  std::shared_ptr<std::vector<char>> content;
  const xir::Subgraph* subgraph;
};
static std::string to_string(const reg_info_t& reg_info) {
  std::ostringstream str;
  str << "reg_info_t{";
  str << "id=" << reg_info.basic_info_.reg_id << ";";
  str << "type=" << vart::to_string(reg_info.basic_info_.type) << ";";
  str << "location=" << vart::TensorBuffer::to_string(reg_info.location) << ";";
  str << "size=" << reg_info.basic_info_.size << ";";
  str << "batch=" << reg_info.batch << ";";
  str << "device_id=" << reg_info.device_id << ";";
  str << "device_core_id=" << reg_info.device_core_id << ";";
  str << "cu_name=" << reg_info.cu_name << ";";
  str << "backstore="
      << (reg_info.reg_tensor_buffer == nullptr
              ? std::string("null")
              : reg_info.reg_tensor_buffer->to_string());
  str << "}";
  return str.str();
}

static std::string get_allocator_key(const xir::Subgraph* subgraph,
                                     const xir::Attrs* attrs, size_t reg_id) {
  std::string suffix = "";
  // if(attrs->has_attr ("__subgraph_suffix__")) TODO: for U50, we
  // need to add a suffix, so that there are parameters for each
  // device core.
  //
  std::ostringstream str;
  str << "reg" << reg_id << "_sg_";
  str << (void*)subgraph;
  return str.str();
}
static std::vector<std::unique_ptr<reg_info_t>>
extract_reg_info_from_subgraph_and_attrs(const xir::Subgraph* subgraph_,
                                         const xir::Attrs* attrs) {
  auto device_id = get_device_id(attrs);
  auto device_core_id = get_device_core_id(attrs);
  auto cu_name = get_cu_name(attrs);
  auto batch = get_batch(attrs);
  auto location = get_location(subgraph_, attrs);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR))
      << "device_id " << device_id << " "            //
      << "device_core_id " << device_core_id << " "  //
      << "cu_name " << cu_name << " "                //
      ;
  auto ret = std::vector<std::unique_ptr<reg_info_t>>();
  auto reg_id_to_parameter_value = std::map<std::string, std::vector<char>>();
  if (subgraph_->has_attr("reg_id_to_parameter_value")) {
    reg_id_to_parameter_value =
        subgraph_->get_attr<std::map<std::string, std::vector<char>>>(
            "reg_id_to_parameter_value");
  }
  auto basic_infos = vart::extract_reg_info_from_subgraph(subgraph_);
  ret.resize(basic_infos.size());
  for (auto i = 0u; i < basic_infos.size(); ++i) {
    if (basic_infos[i].type != reg_type_t::INVALID) {
      ret[i] = std::unique_ptr<reg_info_t>(new reg_info_t{});
      ret[i]->basic_info_ = basic_infos[i];
    }
  }

  for (auto i = 0u; i < ret.size(); ++i) {
    if (ret[i] == nullptr) {
      continue;
    }
    auto reg_id = ret[i]->basic_info_.reg_id;
    auto reg_type = ret[i]->basic_info_.type;
    auto reg_size = ret[i]->basic_info_.size;
    auto reg_id_int = reg_id;
    auto allocator_key =
        get_allocator_key(subgraph_, attrs, (size_t)reg_id_int);
    auto& reg_info = *ret[reg_id_int].get();
    {  // begin initialize the reg_info_t
      auto ok = true;
      reg_info.location = location;
      ok = ok && reg_size > 0;
      ok = ok && (reg_info.basic_info_.reg_id < MAX_REG_ID_SIZE &&
                  reg_info.basic_info_.reg_id >= 0);
      reg_info.batch = reg_type == vart::reg_type_t::CONST ? 1u : batch;
      reg_info.device_id = device_id;
      reg_info.device_core_id = device_core_id;
      reg_info.cu_name = cu_name;
      reg_info.subgraph = subgraph_;
      if (reg_type == vart::reg_type_t::CONST) {
        static std::mutex mtx;
        std::lock_guard<std::mutex> lock(mtx);
        auto it_value = reg_id_to_parameter_value.find(std::string("REG_") +
                                                       std::to_string(reg_id));
        CHECK(it_value != reg_id_to_parameter_value.end());
        reg_info.content =
            vitis::ai::WeakStore<std::string, std::vector<char>>::create(
                allocator_key);
        if (reg_info.content->empty()) {
          *reg_info.content = std::move(it_value->second);
        }
      }
      CHECK(ok);
    }  // end initialization
  }
  return ret;
}

static std::vector<std::unique_ptr<reg_info_t>> collect_reg_info(
    const xir::Subgraph* subgraph,
    const std::vector<const xir::Tensor*>& tensors, const xir::Attrs* attrs) {
  auto reg_infos = extract_reg_info_from_subgraph_and_attrs(subgraph, attrs);
  if (reg_infos.empty()) {
    return {};
  }
  for (auto i = 0u; i < tensors.size(); ++i) {
    auto tensor = tensors[i];
    auto location_from_tensor = tensor->template get_attr<int>("location");
    if (location_from_tensor == 0) {
      continue;
    }
    auto reg_id = get_reg_id(tensor);
    UNI_LOG_CHECK(reg_id < reg_infos.size(), VART_TENSOR_INFO_ERROR)
      << " tensor:" << tensor->to_string();
    auto offset = get_offset(tensor);
    auto location = get_location(subgraph, attrs);
    auto batch = get_batch(tensor);
    auto size = get_size(tensor);
    auto new_size = offset + size;
    UNI_LOG_CHECK(reg_infos[reg_id] != nullptr, VART_XRT_NULL_PTR)
        << "cannot find reg_info: reg_id=" << reg_id;
    UNI_LOG_CHECK(reg_infos[reg_id]->basic_info_.reg_id == reg_id, VART_TENSOR_INFO_ERROR)
        << "reg id conflict: tensor = " << tensor->to_string();
    UNI_LOG_CHECK((int)reg_infos[reg_id]->location == (int)location, VART_TENSOR_INFO_ERROR)
        << "location conflict: tensor = " << tensor->to_string()
        << " reg=" << to_string(*reg_infos[reg_id]);
    if (reg_infos[reg_id]->basic_info_.type != reg_type_t::CONST) {
      UNI_LOG_CHECK(reg_infos[reg_id]->batch == batch, VART_TENSOR_INFO_ERROR)
          << "batch conflict: tensor = " << tensor->to_string()
          << " reg=" << to_string(*reg_infos[reg_id]);
    }
    bool log_warn = ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR) &&
                    new_size > reg_infos[reg_id]->basic_info_.size;
    LOG_IF(WARNING, log_warn)
        << "ddr execeed? new_size=" << new_size << " reg_infos[" << reg_id
        << "]=" << to_string(*reg_infos[reg_id])
        << " tensor=" << tensor->to_string();
  }
  if (ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR)) {
    int c = 0;
    for (auto& info : reg_infos) {
      LOG(INFO) << "info[" << c++ << "]="
                << (info == nullptr ? std::string("null") : to_string(*info));
    }
  }
  return reg_infos;
}

static std::string get_reg_tensor_buffer_key(const reg_info_t& reg_info) {
  static uint64_t counter = 0u;
  std::ostringstream str;
  str << "reg_" << reg_info.basic_info_.reg_id;
  switch (reg_info.basic_info_.type) {
    case reg_type_t::CONST: {
      str << "_sg_" << (void*)reg_info.subgraph;
      // TODO: for U50, it is also device_core_specific
      auto device_id = reg_info.device_id;
      str << "_device_" << device_id;
      break;
    }
    case reg_type_t::DATA_GLOBAL: {
      str << "_sg_" << (void*)reg_info.subgraph;
      auto device_core_id = reg_info.device_core_id;
      str << "_device_core_id_" << device_core_id;
      break;
    }
    case reg_type_t::DATA_LOCAL:
    case reg_type_t::DATA_LOCAL_INPUT:
    case reg_type_t::DATA_LOCAL_OUTPUT: {
      // DATA_LOCAL is not shared, so a unique key should be returned.
      str << "_" << counter++;
      break;
    }
    default: {
      CHECK(false) << "not a valid type" << to_string(reg_info);
    }
  }
  return str.str();
}

static std::shared_ptr<vart::TensorBuffer> create_tensor_buffer_for_reg(
    reg_info_t& reg_info) {
  // this function is thread-safe, protected by a mutex.
  auto ret = std::shared_ptr<vart::TensorBuffer>();
  auto location = reg_info.location;
  // key is important, it determines which scope the tensor buffer is shared.
  auto key = get_reg_tensor_buffer_key(reg_info);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR)) << "key=" << key;
  switch (location) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      // it is not need to create a shared back store for a reg. For
      // HostVirt, every tensor buffer point to a individual unique
      // buffer.
      break;
    default:
      auto tensor = xir::Tensor::create(
          std::string("__reg__") + std::to_string(reg_info.basic_info_.reg_id) +
              "__",
          std::vector<std::int32_t>{(int)reg_info.batch,
                                    (int)reg_info.basic_info_.size},
          xir::DataType{xir::DataType::XINT, 8});
      tensor->set_attr<int>("reg_id", reg_info.basic_info_.reg_id);
      tensor->set_attr<int>("ddr_addr", 0);
      tensor->set_attr<int>("location", 1);
      ret = vitis::ai::
          WeakStore<std::string, vart::dpu::TensorBufferExtImpHostPhy>::create(
              key,  // key is important
              tensor.get(), location, reg_info.device_id, reg_info.cu_name,
              reg_info.content);
      break;
  }
  return ret;
}

static void create_tensor_buffer_for_reg(
    std::vector<std::unique_ptr<reg_info_t>>& reg_infos) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);

  for (auto& reg_info : reg_infos) {
    if (reg_info == nullptr) {
      continue;
    }
    reg_info->reg_tensor_buffer = create_tensor_buffer_for_reg(*reg_info);
    LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR))
        << "allocate backstore tensor buffer: " << to_string(*reg_info);
  }

  return;
}

static std::unique_ptr<vart::TensorBuffer> create_host_tensor_buffer(
    const xir::Tensor* tensor) {
  auto value = new vart::dpu::TensorBufferExtImpHost(tensor);
  return std::unique_ptr<vart::TensorBuffer>(value);
}

static std::unique_ptr<vart::TensorBuffer> create_tensor_buffer_view(
    const xir::Tensor* tensor, std::shared_ptr<vart::TensorBuffer> backstore) {
  auto offset = get_offset(tensor);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TENSOR_BUFFER_ALLOCATOR) >= 2)
      << "allocate tensor buffer: " << tensor->to_string() << ";"
      << " offset = 0x" << std::hex << offset << std::dec << ";"
      << "backstore=" << (void*)backstore.get();
  return std::unique_ptr<vart::TensorBuffer>(
      new vart::dpu::TensorBufferExtImpView(tensor, offset, backstore));
}

static std::unique_ptr<vart::TensorBuffer> create_tensor_buffer(
    const xir::Subgraph* subgraph, const xir::Attrs* attrs,
    const xir::Tensor* tensor, reg_info_t* reg_info) {
  auto ret = std::unique_ptr<vart::TensorBuffer>();
  auto location = get_location(subgraph, attrs);
  if (reg_info != nullptr) {
    location = reg_info->location;
  }
  switch (location) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      ret = create_host_tensor_buffer(tensor);
      break;
    default:
      if (reg_info != nullptr && reg_info->reg_tensor_buffer != nullptr) {
        ret = create_tensor_buffer_view(tensor, reg_info->reg_tensor_buffer);
      } else {
        auto device_id = get_device_id(attrs);
        auto cu_name = get_cu_name(attrs);
        auto content = std::shared_ptr<std::vector<char>>();
        ret = std::make_unique<vart::dpu::TensorBufferExtImpHostPhy>(
            tensor, location, device_id, cu_name, content);
      }
      break;
  }
  UNI_LOG_CHECK(ret != nullptr, VART_XRT_NULL_PTR) << "not implemented?";
  return ret;
}

static std::vector<std::unique_ptr<vart::TensorBuffer>> create_tensor_buffers(
    const xir::Subgraph* subgraph, const xir::Attrs* attrs,
    const std::vector<const xir::Tensor*>& tensors,
    const std::vector<std::unique_ptr<reg_info_t>>& reg_infos) {
  auto ret = std::vector<std::unique_ptr<vart::TensorBuffer>>();
  ret.reserve(tensors.size());
  for (auto i = 0u; i < tensors.size(); ++i) {
    auto reg_id = get_reg_id(tensors[i]);
    // reg_id out of range?
    reg_info_t* reg_info = nullptr;
    if (reg_id < reg_infos.size()) {
      reg_info = reg_infos[reg_id].get();
    }
    ret.emplace_back(
        create_tensor_buffer(subgraph, attrs, tensors[i], reg_info));
  }
  // create a fake tensor buffer view for get base address of the
  // whole tensor buffer for reg. It is bad that these tensor buffers
  // are dangling for HostVirt, but if it is HostVirt, these tensor
  // buffers are not in use.
  for (auto& reg_info : reg_infos) {
    if (reg_info == nullptr) {
      continue;
    }
    if (reg_info->reg_tensor_buffer
        /* TODO: check the buffer's localtion is not HostVirt.
           &&
           reg_info.sed */) {
      ret.emplace_back(create_tensor_buffer_view(
          xir::Tensor::clone(reg_info->reg_tensor_buffer->get_tensor()).get(),
          reg_info->reg_tensor_buffer));
    }
  }
  return ret;
}
static bool is_included(const std::string& tensor_name,
                        const std::vector<const xir::Tensor*>& tensors) {
  bool ret = false;
  for (auto& t : tensors) {
    if (t->get_name() == tensor_name) return true;
  }
  return ret;
}
std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
          std::vector<std::unique_ptr<vart::TensorBuffer>>>
TensorBufferAllocatorImp::allocate(
    const xir::Subgraph* subgraph,
    const std::vector<const xir::Tensor*>& input_tensors,
    const std::vector<const xir::Tensor*>& output_tensors) {
  auto tensors = std::vector<const xir::Tensor*>();
  tensors.insert(tensors.end(), input_tensors.begin(), input_tensors.end());
  tensors.insert(tensors.end(), output_tensors.begin(), output_tensors.end());

  auto reg_infos = collect_reg_info(subgraph, tensors, attrs_);
  create_tensor_buffer_for_reg(reg_infos);
  auto all_tensor_buffers =
      create_tensor_buffers(subgraph, attrs_, tensors, reg_infos);
  UNI_LOG_CHECK(all_tensor_buffers.size() >=
           (input_tensors.size() + output_tensors.size()),
           VART_TENSOR_INFO_ERROR)
      << "allocate error: allocator return tensor_buffers.size must >= "
         "input_tensors.size() + "
         "output_tensors.size()";
  auto input_tensor_buffers =
      std::vector<std::unique_ptr<vart::TensorBuffer>>();
  auto output_tensor_buffers =
      std::vector<std::unique_ptr<vart::TensorBuffer>>();
  for (auto& tensor_buffer : all_tensor_buffers) {
    auto name = tensor_buffer->get_tensor()->get_name();
    if (is_included(name, output_tensors)) {
      output_tensor_buffers.push_back(std::move(tensor_buffer));
    } else {
      input_tensor_buffers.push_back(std::move(tensor_buffer));
    }
  }
  return std::make_pair(std::move(input_tensor_buffers),
                        std::move(output_tensor_buffers));
}

TensorBufferAllocatorImp::~TensorBufferAllocatorImp() {}

}  // namespace

extern "C" vart::assistant::TensorBufferAllocator*
create_tensor_buffer_allocator(const xir::Attrs* attrs) {
  return new TensorBufferAllocatorImp(attrs);
}
