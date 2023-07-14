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

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>
#include <thread>
#include <utility>
#include <vart/runner.hpp>
#include <vector>
#include <xir/graph/graph.hpp>

#include "../src/runner_helper.hpp"
#include "vart/assistant/tensor_buffer_allocator.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/softmax_runner_cpu.hpp"
#include "vart/tensor_buffer.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"
#include "vitis/ai/thread_pool.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/tensor/tensor.hpp"

DEF_ENV_PARAM(DEBUG_TEST, "0");
DEF_ENV_PARAM(TEST_ZERO_COPY, "0");

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      float* output) {
  float max = input[0] * scale;
  std::vector<float> scaledIn(cls);
  scaledIn.push_back(max);
  for (unsigned int i = 1; i < cls; ++i) {
    scaledIn[1] = input[i] * scale;
    if (max < scaledIn[i]) max = scaledIn[i];
  }
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(scaledIn[i] - max);
    sum += output[i];
  }
  for (unsigned int i = 0; i < cls; ++i) output[i] /= sum;
}

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      unsigned int group, float* output) {
  for (unsigned int i = 0; i < group; ++i) {
    softmax_c(input, scale, cls, output);
    input += cls;
    output += cls;
  }
}

static void compare(int cls, int group, signed char* input, float* output1,
                    float* output2) {
  for (auto g = 0; g < group; ++g) {
    for (auto i = 0; i < cls; ++i) {
      auto idx = g * cls + i;
      auto diff = output1[idx] - output2[idx];
      if (ENV_PARAM(DEBUG_TEST) || (diff != 0.0 && std::abs(diff) > 0.001)) {
        std::cout << " i   = " << i << " g   = " << g << " idx = " << idx << " "
                  << (int)input[idx] << " : " << output1[idx] << " "
                  << output2[idx] << " " << std::abs(diff) << std::endl;
      }
    }
  }
}

std::vector<std::int32_t> reshape_tensor_to_three_dim(
    std::vector<std::int32_t> in) {
  CHECK_GE(in.size(), 2) << "input dimension is less than 2";
  std::vector<std::int32_t> ret;
  ret.reserve(3);
  ret.push_back(in.front());
  std::int32_t mid = 1;
  for (unsigned int i = 1; i < in.size() - 1; i++) {
    mid *= in[i];
  }
  ret.push_back(mid);
  ret.push_back(in.back());
  return ret;
}

static int get_fix_pos(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point")) << "tensor = " << tensor->to_string();
  int fixpos = tensor->template get_attr<int>("fix_point");
  return fixpos;
}

int main(int argc, char* argv[]) {
  // prepare subgraph
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "SM-CPU") {
      s = c;
      break;
    }
  }
  LOG(INFO) << "sugraph: " << s->get_name();

  // prepare attrs
  auto attrs = xir::Attrs::create();
  attrs->set_attr<size_t>("__device_id__", 0u);
  attrs->set_attr<size_t>("__batch__", 1);
  attrs->set_attr<int>(s->get_name() + ":__tensor_buffer_location__", 1);

  // prepare tensors
  CHECK_EQ(s->get_sorted_input_tensors().size(), 1u);
  CHECK_EQ(s->get_sorted_output_tensors().size(), 1u);

  auto input_tensor = *(s->get_sorted_input_tensors().begin());
  auto output_tensor = *(s->get_sorted_output_tensors().begin());

  LOG(INFO) << "input_tensor info:  " << input_tensor->to_string();
  LOG(INFO) << "output_tensor info:  " << output_tensor->to_string();

  auto input_shape = reshape_tensor_to_three_dim(input_tensor->get_shape());

  auto input_tensor_size = input_shape[0] * input_shape[1] * input_shape[2];
  LOG(INFO) << "input tensor size: " << input_shape[0] << " " << input_shape[1]
            << " " << input_shape[2];

  // use subgraph, attrs and tensors to apply tensor_buffer
  std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
            std::vector<std::unique_ptr<vart::TensorBuffer>>>
      tensor_buffers;

  auto allocator = vart::assistant::TensorBufferAllocator::create(attrs.get());

  LOG(INFO) << "allocate tensor buffers at VIRT";
  tensor_buffers.first.emplace_back(
      vart::alloc_cpu_flat_tensor_buffer(input_tensor));
  tensor_buffers.second.emplace_back(
      vart::alloc_cpu_flat_tensor_buffer(output_tensor));

  auto input_tensor_buffer = tensor_buffers.first[0].get();
  auto output_tensor_buffer = tensor_buffers.second[0].get();

  // set random input value
  uint64_t input_addr = 0u;
  size_t input_size = 0u;
  uint64_t output_addr = 0u;
  size_t output_size = 0u;

  auto input_dim_idx = vart::get_index_zeros(input_tensor_buffer->get_tensor());
  std::tie(input_addr, input_size) = input_tensor_buffer->data(input_dim_idx);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "input_addr: " << (void*)input_addr << "; input_size: " << input_size;

  auto output_dim_idx =
      vart::get_index_zeros(output_tensor_buffer->get_tensor());
  std::tie(output_addr, output_size) =
      output_tensor_buffer->data(output_dim_idx);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "output_addr: " << (void*)output_addr
                                      << "; output_size: " << output_size;

  int8_t* inputPtr = reinterpret_cast<int8_t*>(input_addr);
  for (auto i = 0; i < input_tensor_size; ++i) {
    inputPtr[i] = i % 10 + 1;
  }

  // run softmax
  auto runner = std::make_unique<vart::SoftmaxRunnerCPU>(s, attrs.get());
  __TIC__(sfmx);

  auto v = runner->execute_async({input_tensor_buffer}, {output_tensor_buffer});
  auto status = runner->wait((int)v.first, -1);
  CHECK_EQ(status, 0) << "failed to run sm-cpu";

  __TOC__(sfmx);

  // compare result with cpu result
  auto group = input_shape[1];
  auto cls = input_shape[2];

  auto fixpos = get_fix_pos(input_tensor);
  float scale = std::exp2f(-1.0f * (float)fixpos);

  std::vector<float> output_c(cls * group);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << " fixpos=" << fixpos << " cls=" << cls << " group=" << group
      << " scale=" << scale;

  softmax_c(inputPtr, scale, cls, group, &output_c[0]);

  compare(cls, group, inputPtr, reinterpret_cast<float*>(output_addr),
          &output_c[0]);

  return 0;
}
