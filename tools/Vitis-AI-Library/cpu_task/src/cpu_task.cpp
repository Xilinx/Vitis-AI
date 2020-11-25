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

#include "cpu_task.hpp"

#include <glog/logging.h>

#include <thread>

#include "batch_tensor_buffer_view.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/op/op_def.hpp"
#include "xir/tensor/tensor.hpp"
DEF_ENV_PARAM(DEBUG_CPU_TASK, "0")
// defined in op_imp.cpp
extern std::unique_ptr<vart::OpImp> create_op_imp(const xir::Op* op);

namespace {

static std::vector<std::unique_ptr<xir::Tensor>> copy_set_to_vector_tensor(
    const std::set<const xir::Tensor*>& tensors) {
  auto ret = std::vector<std::unique_ptr<xir::Tensor>>();
  ret.reserve(tensors.size());
  for (auto b : tensors) {
    // clone it, it is pontentially possible to change the batch size.
    // see get_input_tensors and get_output_tensors
    auto dims = b->get_shape();
    // not change the dimetions here, because we cannot make sure that
    // 1st dimention is always the batch size for all tensors.
    auto x = xir::Tensor::create(b->get_name(), dims, b->get_data_type());
    ret.emplace_back(std::move(x));
  }
  return ret;
}

static std::vector<const xir::Tensor*> get_all_tensors(
    const std::vector<const xir::Op*>& ops) {
  return vitis::ai::vec_map(
      ops, [](const xir::Op* op) { return op->get_output_tensor(); });
}

static std::vector<std::unique_ptr<vart::OpImp>> create_op_imp_vec(
    const std::vector<const xir::Op*>& ops) {
  return vitis::ai::vec_map(ops, [](const xir::Op* op) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK) >= 2)
        << "create op_imp " << op->get_name() << " :: " << op->get_type();
    return create_op_imp(op);
  });
}

CpuTask::CpuTask(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : attrs_{attrs}, inputs_{}, outputs_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
      << "@" << (void*)this << " cpu task is created for subgraph "
      << subgraph->get_name();
  // Q: why we need to copy input and output tensors?
  // A: converting std::set to std::vector
  inputs_ = copy_set_to_vector_tensor(subgraph->get_input_tensors());
  outputs_ = copy_set_to_vector_tensor(subgraph->get_output_tensors());
  ops_ = subgraph->topological_sort();
  op_imp_ = create_op_imp_vec(ops_);
  tensors_ = get_all_tensors(ops_);
  auto subgraph_inputs = subgraph->get_input_tensors();
  tensors_.insert(tensors_.begin(), subgraph_inputs.begin(),
                  subgraph_inputs.end());
  tensor_buffers_ = vart::alloc_cpu_flat_tensor_buffers(tensors_);
  tensor_name_2_index_ = build_tensor_name_2_index();
  tensor_buffer_views_ = build_tensor_buffer_views();
  my_op_args_ = build_my_op_args();
}

CpuTask::~CpuTask() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
      << "@" << (void*)this << " cpu task is destroyed";
}

std::unordered_map<std::string, size_t> CpuTask::build_tensor_name_2_index() {
  auto ret = std::unordered_map<std::string, size_t>();
  auto index = 0u;
  for (auto& tensor_buffer : tensor_buffers_) {
    ret[tensor_buffer->get_tensor()->get_name()] = index++;
  }
  return ret;
}

std::vector<std::unique_ptr<vart::BatchTensorBufferView>>
CpuTask::build_tensor_buffer_views() {
  return vitis::ai::vec_map(
      tensor_buffers_,
      [](const std::unique_ptr<vart::TensorBuffer>& tb)
          -> std::unique_ptr<vart::BatchTensorBufferView> {
        return std::make_unique<vart::BatchTensorBufferView>(tb.get());
      });
}

std::vector<MyOpArgs> CpuTask::build_my_op_args() {
  return vitis::ai::vec_map(ops_, [this](const xir::Op* op) -> MyOpArgs {
    auto inputs = std::vector<vart::OpImpArg>();
    auto op_def = op->get_opdef();
    inputs = vitis::ai::vec_map(
        op_def->input_args(),
        [op, this](const xir::OpArgDef& op_arg_def) -> vart::OpImpArg {
          auto name = op_arg_def.name;
          auto input_ops = op->get_input_ops(name);
          return vart::OpImpArg{
              name, vitis::ai::vec_map(
                        input_ops,
                        [this](const xir::Op* input_op) -> vart::TensorBuffer* {
                          return find_tensor_buffer_view(
                              input_op->get_output_tensor()->get_name());
                        })};
        });
    vart::TensorBuffer* output =
        find_tensor_buffer_view(op->get_output_tensor()->get_name());
    return MyOpArgs{inputs, output};
  });
}

vart::TensorBuffer* CpuTask::find_tensor_buffer(const std::string& name) {
  vart::TensorBuffer* ret =
      tensor_buffers_[tensor_name_2_index_.at(name)].get();
  CHECK(ret != nullptr) << "cannot find tensor buffer. name = " << name;
  return ret;
}

vart::BatchTensorBufferView* CpuTask::find_tensor_buffer_view(
    const std::string& name) {
  vart::BatchTensorBufferView* ret =
      tensor_buffer_views_[tensor_name_2_index_.at(name)].get();
  CHECK(ret != nullptr) << "cannot find tensor buffer. name = " << name;
  return ret;
}

size_t CpuTask::get_batch_size(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) const {
  int batch_size = -1;
  auto update_batch_size =
      [&batch_size](const std::vector<vart::TensorBuffer*>& tbs) {
        for (auto b : tbs) {
          if (batch_size == -1) {
            batch_size = (int)b->get_tensor()->get_shape()[0];
          } else {
            CHECK_EQ(batch_size, b->get_tensor()->get_shape()[0])
                << "all tensor must have same batch size: " << b->to_string();
          }
        }
      };
  update_batch_size(input);
  update_batch_size(output);
  CHECK_NE(batch_size, -1);
  return (size_t)batch_size;
}

size_t CpuTask::get_batch_step(const std::vector<vart::TensorBuffer*>& input,
                               const std::vector<vart::TensorBuffer*>& output) {
  int batch_size = -1;
  auto update_batch_size =
      [&batch_size, this](const std::vector<vart::TensorBuffer*>& tbs) mutable {
        for (auto b : tbs) {
          if (batch_size == -1) {
            batch_size = find_tensor_buffer(b->get_tensor()->get_name())
                             ->get_tensor()
                             ->get_shape()[0];
          } else {
            CHECK_EQ(batch_size, find_tensor_buffer(b->get_tensor()->get_name())
                                     ->get_tensor()
                                     ->get_shape()[0])
                << "all tensor must have same batch size: " << b->to_string();
          }
        }
      };
  update_batch_size(input);
  update_batch_size(output);
  CHECK_NE(batch_size, -1);
  return (size_t)batch_size;
}

static bool host_accessible(vart::TensorBuffer::location_t loc) {
  return loc == vart::TensorBuffer::location_t::HOST_PHY ||
         loc == vart::TensorBuffer::location_t::HOST_VIRT;
}

void CpuTask::update_tensor_buffer_view(
    size_t batch_index, const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto update =
      [this, batch_index](const std::vector<vart::TensorBuffer*>& tbs) mutable {
        for (auto& tb : tbs) {
          auto accessible = host_accessible(tb->get_location());
          auto view = find_tensor_buffer_view(tb->get_tensor()->get_name());
          if (accessible) {
            view->update_batch_index(tb, batch_index);
          } else {
            auto internal = find_tensor_buffer(tb->get_tensor()->get_name());
            auto view_device =
                std::make_unique<vart::BatchTensorBufferView>(internal);
            view_device->update_batch_index(tb, batch_index);
            view->update_batch_index(internal, 0);
            vart::TensorBuffer::copy_tensor_buffer(view_device.get(), internal);
          }
          LOG(INFO) << "updating " << tb->get_tensor()->get_name();
        }
      };
  update(input);
  update(output);
}

std::pair<uint32_t, int> CpuTask::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto batch_size = get_batch_size(input, output);
  auto batch_step = get_batch_step(input, output);
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
      << "@" << (void*)this << " start to run: "
      << " inputs= " << to_string(input) << " "
      << " outputs= " << to_string(output) << " "  //
      << "batch_size " << batch_size << " "        //
      << "batch_step " << batch_step << " "        //
      ;
  auto size = op_imp_.size();
  CHECK_EQ(size, my_op_args_.size()) << "must be equal";
  for (auto batch_index = 0u; batch_index < batch_size;
       batch_index = batch_index + batch_step) {
    update_tensor_buffer_view(batch_index, input, output);
    for (auto i = 0u; i < size; ++i) {
      auto& inputs = my_op_args_[i].inputs;
      auto& output = my_op_args_[i].output;
      auto error_code = op_imp_[i]->calculate(inputs, output);
      CHECK_EQ(error_code, 0);
    }
  }
  return std::make_pair(0u, 0);
}

int CpuTask::wait(int jobid, int timeout) { return 0; }

static std::vector<const xir::Tensor*> copy(
    std::vector<std::unique_ptr<xir::Tensor>>& from) {
  auto ret = std::vector<const xir::Tensor*>();
  ret.reserve(from.size());
  for (auto& b : from) {
    ret.push_back(const_cast<const xir::Tensor*>(b.get()));
  }
  return ret;
}

std::vector<const xir::Tensor*> CpuTask::get_input_tensors() {
  return copy(inputs_);
}

std::vector<const xir::Tensor*> CpuTask::get_output_tensors() {
  return copy(outputs_);
}

}  // namespace

extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return new CpuTask(subgraph, attrs);
}
