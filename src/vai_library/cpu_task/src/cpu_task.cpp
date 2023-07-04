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

#include "cpu_task.hpp"

#include <glog/logging.h>

#include <thread>
#include <vart/trace/trace.hpp>
#include <vitis/ai/profiling.hpp>

#include <UniLog/UniLog.hpp>
#include "tensor_buffer_proxy.hpp"
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/op/op_def.hpp"
#include "xir/tensor/tensor.hpp"
DEF_ENV_PARAM(DEBUG_CPU_TASK, "0")
DEF_ENV_PARAM(XLNX_ENABLE_DUMP, "0")
// defined in op_imp.cpp
extern std::unique_ptr<vart::OpImp> create_op_imp(const xir::Op* op,
                                                  xir::Attrs* attrs);

namespace {

static std::vector<std::unique_ptr<xir::Tensor>> copy_set_to_vector_tensor(
    const std::vector<const xir::Tensor*>& tensors) {
  auto ret = std::vector<std::unique_ptr<xir::Tensor>>();
  ret.reserve(tensors.size());
  for (auto& b : tensors) {
    ret.emplace_back(xir::Tensor::clone(b));
  }
  return ret;
}

static std::vector<const xir::Tensor*> get_all_tensors(
    const std::vector<const xir::Op*>& ops) {
  return vitis::ai::vec_map(
      ops, [](const xir::Op* op) { return op->get_output_tensor(); });
}

static std::vector<std::unique_ptr<vart::OpImp>> create_op_imp_vec(
    const std::vector<const xir::Op*>& ops, xir::Attrs* attrs) {
  return vitis::ai::vec_map(ops, [attrs](const xir::Op* op) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK) >= 2)
        << "create op_imp " << op->get_name() << " :: " << op->get_type();
    return create_op_imp(op, attrs);
  });
}

CpuTask::CpuTask(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : default_attrs_{xir::Attrs::create()},
      attrs_{attrs == nullptr ? default_attrs_.get() : attrs},
      subgraph_{subgraph},
      inputs_{},
      outputs_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
      << "@" << (void*)this << " cpu task is created for subgraph "
      << subgraph->get_name();
  vitis::ai::trace::add_subgraph(subgraph_);
  // Q: why we need to copy input and output tensors?
  // A: converting std::set to std::vector
  inputs_ = copy_set_to_vector_tensor(subgraph->get_sorted_input_tensors());
  outputs_ = copy_set_to_vector_tensor(subgraph->get_sorted_output_tensors());
  ops_ = subgraph->topological_sort();
  op_imp_ = create_op_imp_vec(ops_, attrs_);
  tensors_ = get_all_tensors(ops_);
  auto subgraph_inputs = subgraph->get_sorted_input_tensors();
  tensors_.insert(tensors_.begin(), subgraph_inputs.begin(),
                  subgraph_inputs.end());
  tensor_buffers_ = vart::alloc_cpu_flat_tensor_buffers(tensors_);
  proxy_tensor_buffers_ = vitis::ai::vec_map(
      tensor_buffers_, [](const std::unique_ptr<vart::TensorBuffer>& tb) {
        return std::make_unique<vart::TensorBufferProxy>(
            const_cast<vart::TensorBuffer*>(tb.get()), tb->get_tensor());
      });
  tensor_name_2_index_ = build_tensor_name_2_index();
  my_op_args_ = build_my_op_args();
  if (!attrs_->has_attr("__batch__")) {
    // dirty hack, allocator needs this field and I forgot why?
    attrs_->set_attr<size_t>("__batch__", (size_t)tensors_[0]->get_shape()[0]);
  }
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
                          return find_proxy_tensor_buffer(
                              input_op->get_output_tensor()->get_name());
                        })};
        });
    vart::TensorBuffer* output =
        find_proxy_tensor_buffer(op->get_output_tensor()->get_name());
    return MyOpArgs{inputs, output};
  });
}

vart::TensorBuffer* CpuTask::find_tensor_buffer(const std::string& name) {
  vart::TensorBuffer* ret =
      tensor_buffers_[tensor_name_2_index_.at(name)].get();
  // CHECK(ret != nullptr) << "cannot find tensor buffer. name = " << name;
  UNI_LOG_CHECK(ret != nullptr, VAILIB_CPU_RUNNER_TENSOR_BUFFER_NOT_FIND)
      << "name = " << name;
  return ret;
}

vart::TensorBufferProxy* CpuTask::find_proxy_tensor_buffer(
    const std::string& name) {
  vart::TensorBufferProxy* ret =
      proxy_tensor_buffers_[tensor_name_2_index_.at(name)].get();
  // CHECK(ret != nullptr) << "cannot find tensor buffer. name = " << name;
  UNI_LOG_CHECK(ret != nullptr, VAILIB_CPU_RUNNER_TENSOR_BUFFER_NOT_FIND)
      << "name = " << name;
  return ret;
}

static bool host_accessible(vart::TensorBuffer::location_t loc) {
  return loc == vart::TensorBuffer::location_t::HOST_PHY ||
         loc == vart::TensorBuffer::location_t::HOST_VIRT;
}

std::pair<uint32_t, int> CpuTask::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  auto size = op_imp_.size();
  // CHECK_EQ(size, my_op_args_.size()) << "must be equal";
  UNI_LOG_CHECK(size == my_op_args_.size(),
                VAILIB_CPU_RUNNER_TENSOR_BUFFER_NOT_FIND)
      << "must be equal";
  auto subgraph_name = subgraph_->get_name();
  auto subgraph_depth = subgraph_->get_depth();
  vitis::ai::trace::add_trace("cpu-task", subgraph_name, subgraph_depth,
                              vitis::ai::trace::func_start);

  __TIC__(CPU_UPDATE_INPUT)
  update_proxy(input);
  __TOC__(CPU_UPDATE_INPUT)
  __TIC__(CPU_UPDATE_OUTPUT)
  update_proxy(output);
  __TOC__(CPU_UPDATE_OUTPUT)
  __TIC__(CPU_SYNC_FOR_READ)
  maybe_sync_for_read(input);
  __TOC__(CPU_SYNC_FOR_READ)
  for (auto i = 0u; i < size; ++i) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
        << "op: " << ops_[i]->get_type()                         //
        << "\n\tname: " << ops_[i]->get_name()                   //
        << "\n\tinputs: " << to_string(my_op_args_[i].inputs)    //
        << "\n\toutput: " << my_op_args_[i].output->to_string()  //
        ;
    auto& inputs = my_op_args_[i].inputs;
    auto& output1 = my_op_args_[i].output;
    LOG_IF(INFO, ENV_PARAM(DEEPHI_PROFILING))
        << "op_name : " << ops_[i]->get_name();
    __TIC__(CPU_OP_EXEC)
    auto error_code = op_imp_[i]->calculate(inputs, output1);
    __TOC__(CPU_OP_EXEC)
    if (ENV_PARAM(XLNX_ENABLE_DUMP)) {
      auto dir = std::string("dump") + "/" +
                 vitis::ai::to_valid_file_name(subgraph_->get_name()) + "/";
      int batch_base = 0;
      if (attrs_->has_attr("__batch_base__")) {
        batch_base = attrs_->get_attr<int>("__batch_base__");
      }
      vart::dump_tensor_buffer(dir, output1, batch_base);
    }
    CHECK_EQ(error_code, 0);
  }
  __TIC__(CPU_SYNC_FOR_WRITE)
  maybe_sync_for_write(output);
  __TOC__(CPU_SYNC_FOR_WRITE)
  // TODO: do we need to restore the proxy?
  vitis::ai::trace::add_trace("cpu-task", subgraph_name, subgraph_depth,
                              vitis::ai::trace::func_end);
  return std::make_pair(0u, 0);
}

void CpuTask::maybe_sync_for_read(const std::vector<vart::TensorBuffer*>& b) {
  for (auto& x : b) {
    maybe_sync_for_read(x);
  }
}

void CpuTask::maybe_sync_for_write(const std::vector<vart::TensorBuffer*>& b) {
  for (auto& x : b) {
    maybe_sync_for_write(x);
  }
}

void CpuTask::maybe_sync_for_read(vart::TensorBuffer* b) {
  switch (b->get_location()) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      // do nothing
      break;
    case vart::TensorBuffer::location_t::HOST_PHY:
      // TODO: check continous
      b->sync_for_read(0, b->get_tensor()->get_data_size());
      break;
    default:
      // update_proxy already copy the tensor buffer
      // do nothing LOG(FATAL) << "Not supported!";
      break;
  }
}

void CpuTask::maybe_sync_for_write(vart::TensorBuffer* b) {
  switch (b->get_location()) {
    case vart::TensorBuffer::location_t::HOST_VIRT:
      // do nothing
      break;
    case vart::TensorBuffer::location_t::HOST_PHY:
      // TODO: check continous
      b->sync_for_write(0, b->get_tensor()->get_data_size());
      break;
    default:
      // update_proxy already copy the tensor buffer
      // LOG(FATAL) << "Not supported!";
      break;
  }
}

void CpuTask::update_proxy(const std::vector<vart::TensorBuffer*>& inputs) {
  for (auto& tb : inputs) {
    auto accessible = host_accessible(tb->get_location());
    auto tensor_buffer = find_tensor_buffer(tb->get_tensor()->get_name());
    auto proxy_tensor_buffer =
        find_proxy_tensor_buffer(tb->get_tensor()->get_name());
    if (accessible) {
      proxy_tensor_buffer->update_backend(tb);
    } else {
      // no need to restore the proxy, because it is updated anyway.
      proxy_tensor_buffer->update_backend(tensor_buffer);
      vart::TensorBuffer::copy_tensor_buffer(tb, tensor_buffer);
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_CPU_TASK))
        << "updating " << tb->get_tensor()->get_name()
        << " proxy_tensor_buffer=" << proxy_tensor_buffer->to_string()
        << " backend=" << tb->to_string();
  }
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

extern "C" vart::Runner* create_runner(const xir::Subgraph* subgraph,
                                       const std::string& mode) {
  return new CpuTask(subgraph, nullptr);
}
