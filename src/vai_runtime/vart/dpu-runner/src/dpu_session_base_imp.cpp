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
#include "./dpu_session_base_imp.hpp"

#include <glog/logging.h>

#include <cmath>
#include <fstream>
#include <numeric>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/weak.hpp>
#include <UniLog/UniLog.hpp>

#include "vart/runner.hpp"
#include "xir/util/tool_function.hpp"

DEF_ENV_PARAM(DEBUG_DPU_RUNNER, "0");
#if IS_EDGE
DEF_ENV_PARAM(XLNX_TENSOR_BUFFER_LOCATION, "1" /* HOST_PHY */);
#else
DEF_ENV_PARAM(XLNX_TENSOR_BUFFER_LOCATION, "2" /* DEVICE */);
#endif
DEF_ENV_PARAM_2(XLNX_DPU_DEVICE_CORES, "", std::vector<size_t>);
namespace vart {
namespace dpu {
// size_t DpuSessionBaseImp::session_count = 0;
namespace REG_ID_TO_SEGMENT_TYPE {
enum { CODE = 0, CONST = 1, DATA = 2 };
}
static std::vector<std::string> get_input_tensor_names(
    const xir::Subgraph* subgraph);

static std::vector<std::string> get_all_tensor_names(
    const xir::Subgraph* subgraph) {
  auto ops = subgraph->get_ops();
  auto inputs = get_input_tensor_names(subgraph);
  auto ret = std::vector<std::string>();
  ret.reserve(ops.size() + inputs.size());
  std::transform(ops.begin(), ops.end(), std::back_inserter(ret),
                 [](auto op) { return op->get_output_tensor()->get_name(); });
  ret.insert(ret.end(), inputs.begin(), inputs.end());
  return ret;
}

// my_ to avoid name confliction with the member function.
size_t DpuSessionBaseImp::my_get_device_core_id(size_t cu_size,
                                                xir::Attrs* attrs) {
  UNI_LOG_CHECK(cu_size > 0u, VART_DEVICE_BUSY)
      << "cannot create a dpu controller, no device is available";
  auto core_list = ENV_PARAM(XLNX_DPU_DEVICE_CORES);

  if (core_list.empty()) {
    core_list.resize(cu_size);
    std::iota(core_list.begin(), core_list.end(), 0);
  }
  static auto session_count = 0u;
  UNI_LOG_CHECK(core_list.size() > 0u, VART_DEVICE_BUSY)
      << "cannot create a dpu session, no core id is available";

  auto device_core_id = core_list[(session_count) % core_list.size()];
  if (attrs) {
    auto device_id = 0u;
    if (!attrs->has_attr("__device_core_id__")) {
      attrs->set_attr<size_t>("__device_core_id__", device_core_id);
      session_count = session_count + 1u;
    }
    device_core_id = attrs->get_attr<size_t>("__device_core_id__");

    device_id = dpu_controller_->get_device_id(device_core_id);
    if (attrs->has_attr("__device_id__")) {
      UNI_LOG_CHECK(device_id == attrs->get_attr<size_t>("__device_id__"), VART_DEVICE_MISMATCH)
          << "The __device_id__ attr must match with cu get from "
             "device_core_id";
    } else {
      attrs->set_attr<size_t>("__device_id__", device_id);
    }

    if (!attrs_->has_attr("__batch__")) {
      attrs_->set_attr<size_t>(
          "__batch__", get_dpu_controller()->get_batch_size(device_core_id));
    }
  } else {
    session_count = session_count + 1u;
    UNI_LOG_CHECK(device_core_id < cu_size, VART_DEVICE_MISMATCH)
        << "Invaild device_core_id, device_core_id must < cu_size ( " << cu_size
        << " )";
  }
  return device_core_id;
}

DpuSessionBaseImp::DpuSessionBaseImp(xir::Attrs* attrs)
    : default_attrs_{xir::Attrs::create()},
      attrs_{attrs == nullptr ? default_attrs_.get() : attrs},
      kernel_{},  // kernel_ is initialized after contruction. because
                  // to construct the kernel, we need dpu_controller
                  // which is not initialized yet.
      dpu_controller_{xir::DpuController::get_instance()},
      device_core_id_(
          my_get_device_core_id(dpu_controller_->get_num_of_dpus(), attrs_)) {}

void DpuSessionBaseImp::initialize() {
  my_input_tensors_ = init_input_tensors(kernel_->get_subgraph());
  my_output_tensors_ = init_output_tensors(kernel_->get_subgraph());
  my_all_tensors_ =
      init_tensors(kernel_->get_subgraph(),
                   get_all_tensor_names(kernel_->get_subgraph()), false);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER))
      << "session is created."
      << "subgraph: " << kernel_->get_subgraph()->get_name();
  for (const auto& tensor : my_input_tensors_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER)) << "input tensor:" << tensor;
  }
  for (const auto& tensor : my_output_tensors_) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER)) << "output tensor:" << tensor;
  }
}

static std::vector<std::string> get_input_tensor_names(
    const xir::Subgraph* subgraph) {
  auto ret = std::vector<std::string>();
  for (auto tensor : subgraph->get_sorted_input_tensors()) {
    ret.emplace_back(tensor->get_name());
  }
  return ret;
}

std::vector<my_tensor_t> DpuSessionBaseImp::init_input_tensors(
    const xir::Subgraph* subgraph) {
  return init_tensors(subgraph, get_input_tensor_names(subgraph), true);
}

std::vector<my_tensor_t> DpuSessionBaseImp::init_output_tensors(
    const xir::Subgraph* subgraph) {
  auto ret = std::vector<std::string>();
  for (auto tensor : subgraph->get_sorted_output_tensors()) {
    ret.emplace_back(tensor->get_name());
  }
  return init_tensors(subgraph, ret, true);
}

template <typename T>
static inline std::ostream& operator<<(std::ostream& out,
                                       const std::vector<T>& v) {
  int c = 0;
  out << "[";
  for (const auto x : v) {
    if (c++ != 0) {
      out << ",";
    }
    out << x;
  }
  out << "]";
  return out;
}

struct op_output_tensor_ddr {
  size_t reg_id;
  size_t ddr_addr;
  size_t location;
};

static bool belong_to_subgraph(const xir::Op* op,
                               const xir::Subgraph* subgraph) {
  auto ops = subgraph->get_ops();
  return ops.find(op) != ops.end();
}

static op_output_tensor_ddr get_op_output_tensor_ddr(
    const xir::Op* op, const xir::Subgraph* subgraph) {
  auto tensor = op->get_output_tensor();
  if (op->get_type() == "download" && belong_to_subgraph(op, subgraph)) {
    auto input_ops = op->get_input_ops("input");
    UNI_LOG_CHECK(input_ops.size() == 1u, VART_OUT_OF_RANGE)
        << "There must be only one pre_op for download op";
    tensor = input_ops[0]->get_output_tensor();
  } else if (!tensor->has_attr("reg_id") ||
             (!belong_to_subgraph(op, subgraph))) {
    auto fanout_ops = op->get_fanout_ops();
    auto subgraph_ops1 = subgraph->get_ops();
    auto subgraph_ops =
        std::vector<const xir::Op*>(subgraph_ops1.begin(), subgraph_ops1.end());
    auto ops = std::vector<const xir::Op*>();
    std::sort(fanout_ops.begin(), fanout_ops.end());
    std::sort(subgraph_ops.begin(), subgraph_ops.end());
    std::set_intersection(fanout_ops.begin(), fanout_ops.end(),
                          subgraph_ops.begin(), subgraph_ops.end(),
                          std::back_inserter(ops));
    UNI_LOG_CHECK(ops.size() == 1u, VART_XMODEL_ERROR)
        << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
    auto upload_op = ops.front();
    UNI_LOG_CHECK(upload_op->get_type() == "upload", VART_XMODEL_ERROR)
        << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
    /*    auto up_next_op = upload_op->get_fanout_ops();
        CHECK_EQ(up_next_op.size(), 1u)
            << "illegal xmodel. op:" << op->get_name() << "  has no ddr info";
        auto next_op = up_next_op.front();
    */
    tensor = upload_op->get_output_tensor();
  }
  UNI_LOG_CHECK(tensor->has_attr("reg_id"), VART_TENSOR_INFO_ERROR)
    << "op_name " << op->get_name();
  UNI_LOG_CHECK(tensor->has_attr("ddr_addr"), VART_TENSOR_INFO_ERROR)
    << "op_name " << op->get_name();
  UNI_LOG_CHECK(tensor->has_attr("location"), VART_TENSOR_INFO_ERROR)
    << "op_name " << op->get_name();
  auto reg_id = (size_t)tensor->template get_attr<int>("reg_id");
  auto ddr_addr = (size_t)tensor->template get_attr<int>("ddr_addr");
  auto location = (size_t)tensor->template get_attr<int>("location");

  return op_output_tensor_ddr{reg_id, ddr_addr, location};
}

static void set_ddr_info(xir::Tensor* tensor, op_output_tensor_ddr& info) {
  tensor->template set_attr<int>("reg_id", (int)info.reg_id);
  tensor->template set_attr<int>("ddr_addr", (int)info.ddr_addr);
  tensor->template set_attr<int>("location", (int)info.location);
}

std::vector<int> get_stride(const xir::Tensor* vitis_tensor) {
  auto ret = std::vector<int>{};
  if (vitis_tensor->has_attr("stride")) {
    ret = vitis_tensor->get_attr<std::vector<int>>("stride");
  }
  return ret;
}

static bool is_normal_stride(const xir::Tensor* vitis_tensor) {
  auto ret = true;
  auto shape = vitis_tensor->get_shape();
  auto stride = 1;
  auto strides = get_stride(vitis_tensor);
  auto c = shape.size() - 1;
  for (auto s = strides.rbegin(); s != strides.rend(); ++s) {
    ret = ret && stride == (*s);
    stride = stride * shape[c];
    c = c - 1;
  }
  return ret;
}

std::vector<my_tensor_t> DpuSessionBaseImp::init_tensors(
    const xir::Subgraph* subgraph, const std::vector<std::string>& tensor_names,
    bool check_stride) {
  auto graph = subgraph->get_graph();
  auto ret = std::vector<my_tensor_t>{};
  ret.reserve(tensor_names.size());

  std::transform(
      tensor_names.begin(), tensor_names.end(), std::back_inserter(ret),
      [&graph, subgraph, this, check_stride](const auto& tensor_name) {
        auto xir_tensor = graph->get_tensor(tensor_name);
        UNI_LOG_CHECK(xir_tensor != nullptr, VART_NULL_PTR)
          << "cannot find tensor: " << tensor_name;
        auto op = xir_tensor->get_producer();
        UNI_LOG_CHECK(op != nullptr, VART_NULL_PTR)
            << "cannot find tensor's producer: " << tensor_name;
        auto tensor_ddr = get_op_output_tensor_ddr(op, subgraph);
        auto dims = xir_tensor->get_shape();
        // dirty HACK, batch size is decided by VART, not the model yet.
        // ugly code here: please be careful.
        auto size = (size_t)xir_tensor->get_data_size() / dims[0];
        dims[0] = this->get_num_of_engines();
        LOG_IF(INFO, ENV_PARAM(DEBUG_DPU_RUNNER) >= 2)
            << "tensor " << tensor_name << " ddr_addr " << tensor_ddr.ddr_addr
            << " size " << xir_tensor->get_data_size() << " , " << size
            << "; dims=" << xir_tensor->get_shape() << " , " << dims;
        auto attrs = xir_tensor->get_attrs();
        auto vitis_tensor = xir::Tensor::create(xir_tensor->get_name(), dims,
                                                xir_tensor->get_data_type());
        vitis_tensor->set_attrs(std::move(attrs));
        set_ddr_info(vitis_tensor.get(), tensor_ddr);
        if (check_stride) {
          CHECK(is_normal_stride(vitis_tensor.get()))
              << "stride=" << xir::to_string(get_stride(vitis_tensor.get()))
              << ";tensor=" << vitis_tensor->to_string();
        }
        auto ret = my_tensor_t{xir_tensor,               //
                               std::move(vitis_tensor),  //
                               tensor_ddr.reg_id,        //
                               tensor_ddr.ddr_addr,      //
                               size,                     //
                               tensor_ddr.location};
        return ret;
      });
  return ret;
}

static void print_tensor_attr_keys(const xir::Tensor* tensor) {
  if (ENV_PARAM(DEBUG_DPU_RUNNER) >= 3) {
    auto attrs = tensor->get_attrs();
    LOG(INFO) << "tensor " << tensor->get_name()
              << "debug_info :" << attrs->debug_info();
    auto keys = attrs->get_keys();
    for (auto& key : keys) {
      LOG(INFO) << "dpu_session_base_imp: attr key : " << key;
    }
  }
}
static std::vector<const xir::Tensor*> get_tensor_pointer(
    const std::vector<my_tensor_t>& tensors) {
  for (auto& t : tensors) {
    print_tensor_attr_keys(t.get_tensor());
  }
  auto ret = std::vector<const xir::Tensor*>();
  ret.reserve(tensors.size());
  std::transform(tensors.begin(), tensors.end(), std::back_inserter(ret),
                 [](auto& tensor) { return tensor.get_tensor(); });
  return ret;
}

std::vector<const xir::Tensor*> DpuSessionBaseImp::get_input_tensors() const {
  return get_tensor_pointer(my_input_tensors_);
};
std::vector<const xir::Tensor*> DpuSessionBaseImp::get_output_tensors() const {
  return get_tensor_pointer(my_output_tensors_);
}

size_t DpuSessionBaseImp::get_num_of_engines() const {
  /*  auto core_id =
get_dpu_controller()->get_core_id(get_device_core_id()); auto
batch_from_hbm_txt = vart::dpu::get_engine_hbm(core_id).size(); */
  auto batch_from_dpu_controller = const_cast<DpuSessionBaseImp*>(this)
                                       ->get_dpu_controller()
                                       ->get_batch_size(get_device_core_id());
  /*  CHECK_EQ(batch_from_hbm_txt, batch_from_dpu_controller)
      << ", logic error, please check hbm_address_assignment.txt or dpu IP";
   */
  return batch_from_dpu_controller;
}
}  // namespace dpu
}  // namespace vart
std::ostream& operator<<(std::ostream& out, const my_tensor_t& my_tensor) {
  out << "mytensor{";
  out << my_tensor.get_name() << ":(";
  int fixpos = my_tensor.get_tensor()->template get_attr<int>("fix_point");
  auto dims = my_tensor.vitis_tensor_->get_shape();
  for (auto i = 0u; i < dims.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << dims[i];
  }
  out << "), fixpos=" << fixpos;
  out << "}";
  return out;
}
