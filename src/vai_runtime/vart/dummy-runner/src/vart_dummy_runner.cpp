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

#include <thread>

#include "../src/runner_helper.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_ext.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/plugin.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/tensor/tensor.hpp"
DEF_ENV_PARAM(DUMMY_RUNNER_BATCH_SIZE, "1");
DEF_ENV_PARAM(DUMMY_RUNNER_PROCESS_TIME, "2");
DEF_ENV_PARAM(DEBUG_DUMMY_RUNNER, "0")
namespace {
class DummyRunner : public vart::RunnerExt {
 public:
  explicit DummyRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs);
  DummyRunner(const DummyRunner& other) = delete;

  virtual ~DummyRunner();

 public:
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<vart::TensorBuffer*>& input,
      const std::vector<vart::TensorBuffer*>& output) override;
  virtual int wait(int jobid, int timeout) override;
  virtual std::vector<const xir::Tensor*> get_input_tensors() override;
  virtual std::vector<const xir::Tensor*> get_output_tensors() override;
  virtual std::vector<vart::TensorBuffer*> get_inputs() override;
  virtual std::vector<vart::TensorBuffer*> get_outputs() override;

 private:
  void thread_main();

 private:
  std::vector<std::unique_ptr<xir::Tensor>> inputs_;
  std::vector<std::unique_ptr<xir::Tensor>> outputs_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> input_tensor_buffers_;
  std::vector<std::unique_ptr<vart::TensorBuffer>> output_tensor_buffers_;
};

DummyRunner::DummyRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    : inputs_{}, outputs_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMMY_RUNNER))
      << "@" << (void*)this << " dummy runner is created for subgraph "
      << subgraph->get_name();
  auto input_set = subgraph->get_sorted_input_tensors();
  inputs_.reserve(input_set.size());
  for (auto b : input_set) {
    auto dims = b->get_shape();
    dims[0] = ENV_PARAM(DUMMY_RUNNER_BATCH_SIZE);
    auto x = xir::Tensor::create(b->get_name(), dims, b->get_data_type());
    x->set_attrs(b->get_attrs());
    for (auto op : b->get_producer()->get_fanout_ops()) {
      if (subgraph->has_op(op)) {
        if (op->get_type() == "upload") {
          if (op->get_output_tensor()->has_attr("stride")) {
            x->set_attr("stride",
                        op->get_output_tensor()->get_attr<std::vector<int32_t>>(
                            "stride"));
            break;
          }
        }
      }
    }
    inputs_.emplace_back(std::move(x));
  }
  auto output_set = subgraph->get_sorted_output_tensors();
  outputs_.reserve(output_set.size());
  for (auto b : output_set) {
    auto dims = b->get_shape();
    dims[0] = ENV_PARAM(DUMMY_RUNNER_BATCH_SIZE);
    auto x = xir::Tensor::create(b->get_name(), dims, b->get_data_type());
    x->set_attrs(b->get_attrs());
    outputs_.emplace_back(std::move(x));
  }
  input_tensor_buffers_ = vart::alloc_cpu_flat_tensor_buffers(
      vitis::ai::vector_unique_ptr_get_const(inputs_));
  output_tensor_buffers_ = vart::alloc_cpu_flat_tensor_buffers(
      vitis::ai::vector_unique_ptr_get_const(outputs_));
}

DummyRunner::~DummyRunner() {}

std::pair<uint32_t, int> DummyRunner::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DUMMY_RUNNER))
      << "@" << (void*)this << " start to run: "
      << " inputs= " << to_string(input) << " "    //
      << " outputs= " << to_string(output) << " "  //
      << "processing time =" << ENV_PARAM(DUMMY_RUNNER_PROCESS_TIME) << " ms";
  std::this_thread::sleep_for(
      std::chrono::milliseconds(ENV_PARAM(DUMMY_RUNNER_PROCESS_TIME)));
  return std::make_pair(0u, 0);
}

int DummyRunner::wait(int jobid, int timeout) { return 0; }

static std::vector<const xir::Tensor*> copy(
    std::vector<std::unique_ptr<xir::Tensor>>& from) {
  auto ret = std::vector<const xir::Tensor*>();
  ret.reserve(from.size());
  for (auto& b : from) {
    ret.push_back(const_cast<const xir::Tensor*>(b.get()));
  }
  return ret;
}
std::vector<const xir::Tensor*> DummyRunner::get_input_tensors() {
  return copy(inputs_);
}
std::vector<const xir::Tensor*> DummyRunner::get_output_tensors() {
  return copy(outputs_);
}
std::vector<vart::TensorBuffer*> DummyRunner::get_inputs() {
  return vitis::ai::vector_unique_ptr_get(input_tensor_buffers_);
}

std::vector<vart::TensorBuffer*> DummyRunner::get_outputs() {
  return vitis::ai::vector_unique_ptr_get(output_tensor_buffers_);
}

}  // namespace

static vart::Runner* create_dummy_runner_with_attrs(
    const xir::Subgraph* subgraph, xir::Attrs* attrs) {
  return new DummyRunner(subgraph, attrs);
}

#if VART_DUMMY_RUNNER_USE_DLL == 1
extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return create_dummy_runner_with_attrs(subgraph, attrs);
}
extern "C" {
void* vart_dummy_runner_hook = nullptr;
}
#else
namespace {
vitis::ai::StaticPluginRegister __register(
    "vart-dummy-runner", "create_runner_with_attrs",
    (void*)&create_dummy_runner_with_attrs);
}  // namespace
extern "C" {
void* vart_dummy_runner_hook = &__register;
}
#endif
