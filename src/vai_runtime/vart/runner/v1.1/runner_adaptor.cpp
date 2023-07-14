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
#include "./runner_adaptor.hpp"

#include "./tensor_buffer_adaptor.hpp"
#include "convert_tensor.hpp"
#include "vart/runner.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/tensor.hpp"

DEF_ENV_PARAM(DEBUG_RUNNER, "0");

namespace vitis {
namespace ai {
using std::unique_ptr;
static std::vector<unique_ptr<vart::TensorBuffer>> convert_tensor_buffer(
    std::vector<TensorBuffer*> tensor_buffers) {
  auto ret = std::vector<unique_ptr<vart::TensorBuffer>>(tensor_buffers.size());
  for (auto i = 0u; i < ret.size(); ++i) {
    ret[i] = std::unique_ptr<vart::TensorBuffer>(
        new TensorBufferAdaptor(tensor_buffers[i]));
  }
  return ret;
}
RunnerAdaptor::RunnerAdaptor(std::shared_ptr<GraphHolder> graph,
                             std::shared_ptr<xir::Attrs> attrs,
                             const xir::Subgraph* subgraph)
    : v1_2_runner_{vart::Runner::create_runner_with_attrs(subgraph,
                                                          attrs.get())},
      input_tensors_{convert_tensors(v1_2_runner_->get_input_tensors())},
      output_tensors_{convert_tensors(v1_2_runner_->get_output_tensors())},
      graph_{graph},
      attrs_{attrs} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "RunnerAdaptor @" << (void*)this
      << " created. graph=" << graph_->graph_->get_name();
}
RunnerAdaptor::~RunnerAdaptor() {
  input_tensors_.clear();
  output_tensors_.clear();
  v1_2_runner_.reset();
  auto name = graph_->graph_->get_name();
  graph_.reset();
  CHECK(input_args_.empty()) << "resource leak;";
  CHECK(output_args_.empty()) << "resource leak;";
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
      << "RunnerAdaptor @" << (void*)this << " destroyed. graph=" << name << ";"
      << "input_args_.size()=" << input_args_.size() << ";"
      << "output_args_.size()=" << output_args_.size() << ";";
}

std::pair<uint32_t, int> RunnerAdaptor::execute_async(
    const std::vector<TensorBuffer*>& input,
    const std::vector<TensorBuffer*>& output) {
  auto input_adaptor = convert_tensor_buffer(input);
  auto output_adaptor = convert_tensor_buffer(output);
  auto ret = v1_2_runner_->execute_async(vector_unique_ptr_get(input_adaptor),
                                         vector_unique_ptr_get(output_adaptor));
  auto it = input_args_.find(ret.first);
  CHECK(it == input_args_.end()) << "resource leak";
  input_args_.emplace(std::piecewise_construct,
                      std::forward_as_tuple(ret.first),
                      std::forward_as_tuple(std::move(input_adaptor)));
  it = output_args_.find(ret.first);
  CHECK(it == output_args_.end()) << "resource leak";
  output_args_.emplace(std::piecewise_construct,
                       std::forward_as_tuple(ret.first),
                       std::forward_as_tuple(std::move(output_adaptor)));
  return ret;
}

int RunnerAdaptor::wait(int jobid, int timeout) {
  auto status = v1_2_runner_->wait(jobid, timeout);
  auto key = (uint32_t)jobid;
  auto it = input_args_.find(key);
  CHECK(it != input_args_.end()) << "resource leak";
  input_args_.erase(it);
  it = output_args_.find(key);
  CHECK(it != output_args_.end()) << "resource leak";
  output_args_.erase(it);
  return status;
}

DpuRunner::TensorFormat RunnerAdaptor::get_tensor_format() {
  auto ret = DpuRunner::TensorFormat::NCHW;
  auto format = v1_2_runner_->get_tensor_format();
  switch (format) {
    case vart::Runner::TensorFormat::NCHW:
      ret = DpuRunner::TensorFormat::NCHW;
      break;
    case vart::Runner::TensorFormat::NHWC:
      ret = DpuRunner::TensorFormat::NHWC;
      break;
  };
  return ret;
}

std::vector<Tensor*> RunnerAdaptor::get_input_tensors() {
  return vector_unique_ptr_get(input_tensors_);
}

std::vector<Tensor*> RunnerAdaptor::get_output_tensors() {
  return vector_unique_ptr_get(output_tensors_);
}

}  // namespace ai
}  // namespace vitis
