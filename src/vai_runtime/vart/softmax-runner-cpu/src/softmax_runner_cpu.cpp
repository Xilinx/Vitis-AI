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

#include "vart/softmax_runner_cpu.hpp"

#include <cmath>
#include <cstdint>
#include <queue>

#include "../src/runner_helper.hpp"
#include "vart/assistant/tensor_buffer_allocator.hpp"
#include "vart/runner_ext.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/env_config.hpp"
#include "xir/graph/subgraph.hpp"
#include "xir/sfm_controller.hpp"
#include "xir/tensor/tensor.hpp"

DEF_ENV_PARAM(DEBUG_SOFTMAX_RUNNER, "0")
DEF_ENV_PARAM(DEBUG_TEST, "0");

void CPUCalcSoftmax(const int8_t* data, size_t size, float* result,
                    float scale) {
  assert(data && result);

  float max = (float)(data[0] * scale);

  std::vector<float> input(size);
  input[0] = max;
  for (size_t i = 1; i < size; i++) {
    input[i] = data[i] * scale;
    if (input[i] > max) max = input[i];
  }

  double sum = 0.0f;
  for (size_t i = 0; i < size; i++) {
    result[i] = exp(input[i] - max);
    sum += result[i];
  }

  for (size_t i = 0; i < size; i++) {
    result[i] /= sum;
  }
}

namespace vart {

static std::vector<std::int32_t> reshape_tensor_to_three_dim(
    std::vector<std::int32_t> in) {
  CHECK_GE(in.size(), 2) << "input dimension is less than 2";
  std::int32_t mid = 1;
  for (unsigned int i = 1; i < in.size() - 1; i++) {
    mid *= in[i];
  }
  return std::vector<std::int32_t>{in.front(), mid, in.back()};
}

SoftmaxRunnerCPU::SoftmaxRunnerCPU(const xir::Subgraph* subgraph,
                                   xir::Attrs* attrs)
    : input_{}, output_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "@" << (void*)this << " softmax runner is created for subgraph "
      << subgraph->get_name();

  auto input_set = subgraph->get_sorted_input_tensors();
  CHECK_EQ(input_set.size(), 1u);
  auto output_set = subgraph->get_sorted_output_tensors();
  CHECK_EQ(output_set.size(), 1u);

  size_t batch =
      attrs->has_attr("__batch__") ? attrs->get_attr<size_t>("__batch__") : 1;

  std::set<xir::Tensor*> inputTensors;
  std::set<xir::Tensor*> outputTensors;

  for (auto tensor : input_set) {
    std::vector<std::int32_t> shape = tensor->get_shape();
    shape[0] = batch;
    inputTensors.emplace(
        xir::Tensor::create(tensor->get_name(), shape, tensor->get_data_type())
            .release());
  }
  for (auto tensor : output_set) {
    std::vector<std::int32_t> shape = tensor->get_shape();
    shape[0] = batch;
    outputTensors.emplace(
        xir::Tensor::create(tensor->get_name(), shape, tensor->get_data_type())
            .release());
  }

  attrs->set_attr<int>(subgraph->get_name() + ":__tensor_buffer_location__", 1);

  auto allocator = vart::assistant::TensorBufferAllocator::create(attrs);
  auto tensor_buffers = allocator->allocate(
      subgraph, std::vector<const xir::Tensor*>{*inputTensors.begin()},
      std::vector<const xir::Tensor*>{*outputTensors.begin()});
  input_ = std::move((tensor_buffers.first)[0]);
  output_ = std::move((tensor_buffers.second)[0]);
}

SoftmaxRunnerCPU::~SoftmaxRunnerCPU() {}

std::pair<uint32_t, int> SoftmaxRunnerCPU::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  CHECK_EQ(input.size(), 1u) << "only support single input";
  CHECK_EQ(output.size(), 1u) << "only support single output";

  auto input_buffer = input[0];
  auto output_buffer = output[0];

  auto input_shape =
      reshape_tensor_to_three_dim(input_buffer->get_tensor()->get_shape());
  auto output_shape =
      reshape_tensor_to_three_dim(output_buffer->get_tensor()->get_shape());

  auto input_batch_size = input_shape[0];
  auto output_batch_size = output_shape[0];

  // Few Checks
  CHECK_EQ(input_batch_size, output_batch_size);
  CHECK_EQ(input_shape.size(), output_shape.size());
  CHECK_EQ(input_shape.size(), 3u);
  for (auto i = 0u; i < input_shape.size(); ++i) {
    CHECK_EQ(input_shape[i], output_shape[i]);
  }

  auto cls = (uint32_t)input_shape[2];
  auto softmaxOut = std::make_unique<float[]>(cls);

  std::vector<int> idx = vart::get_index_zeros(input_buffer->get_tensor());
  for (auto b = 0; b < input_batch_size; ++b) {
    idx[0] = b;
    int8_t* data = (int8_t*)input_buffer->data(idx).first;

    int fixpos =
        input_buffer->get_tensor()->template get_attr<int>("fix_point");
    float scale = std::exp2f(-1.0f * (float)fixpos);

    CPUCalcSoftmax(data, cls, softmaxOut.get(), scale);
    float* out_data = (float*)output[0]->data(idx).first;
    for (uint32_t i = 0; i < cls; ++i) out_data[i] = softmaxOut[i];
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "@" << (void*)this << " start to run: "
      << " inputs= " << to_string(input) << " "  //
      << " outputs= " << to_string(output);
  return std::make_pair(0u, 0);
}

int SoftmaxRunnerCPU::wait(int jobid, int timeout) { return 0; }

std::vector<const xir::Tensor*> SoftmaxRunnerCPU::get_input_tensors() {
  return {input_->get_tensor()};
}

std::vector<const xir::Tensor*> SoftmaxRunnerCPU::get_output_tensors() {
  return {output_->get_tensor()};
}

std::vector<vart::TensorBuffer*> SoftmaxRunnerCPU::get_inputs() {
  return {input_.get()};
}

std::vector<vart::TensorBuffer*> SoftmaxRunnerCPU::get_outputs() {
  return {output_.get()};
}

}  // namespace vart

extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return new vart::SoftmaxRunnerCPU(subgraph, attrs);
}
