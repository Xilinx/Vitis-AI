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
#include "../src/softmax_runner.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/tensor_buffer.hpp"
#include "vart/tensor_buffer_allocator.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/profiling.hpp"
#include "vitis/ai/thread_pool.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/tensor/tensor.hpp"

DEF_ENV_PARAM(NUM_OF_GROUP, "1")
DEF_ENV_PARAM(NUM_OF_CLASS, "10")
DEF_ENV_PARAM(SFM_FIX_POS, "4")
DEF_ENV_PARAM(DEBUG_TEST, "0");
DEF_ENV_PARAM(TEST_ZERO_COPY, "0");

static void softmax_c(const int8_t* input, float scale, unsigned int cls,
                      float* output) {
  float sum = 0.f;
  for (unsigned int i = 0; i < cls; ++i) {
    output[i] = exp(input[i] * scale);
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
        cout << " i=" << i               //
             << " g = " << g             //
             << " idx = " << idx << " "  //
             << (int)input[idx] << ": " << output1[idx] << " " << output2[idx]
             << " " << diff << endl;
      }
    }
  }
}

int main(int argc, char* argv[]) {
  // prepare subgraph
  auto graph = xir::Graph::deserialize(argv[1]);
  auto root = graph->get_root_subgraph();
  xir::Subgraph* s = nullptr;
  for (auto c : root->get_children()) {
    if (c->get_attr<std::string>("device") == "CPU") {
      s = c;
      break;
    }
  }
  LOG(INFO) << "sugraph: " << s->get_name();

  for (auto t : s->get_input_tensors()) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
        << "subgraph input tensor info: " << t->to_string();
  }
  for (auto t : s->get_output_tensors()) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
        << "subgraph output tensor info: " << t->to_string();
  }

  // prepare attrs
  auto attrs = xir::Attrs::create();
  attrs->set_attr<size_t>("__device_id__", 0u);
  attrs->set_attr<size_t>("__batch__", 1);
  attrs->set_attr<int>("__tensor_buffer_location__", 1);
  attrs->set_attr<std::string>("__cu_name__", "DPU");
  LOG(INFO) << "attrs: " << (void*)attrs.get();

  // prepare tensors by hand simulate subgraph get_tensors function
  auto input_tensor = xir::Tensor::create(
      "input", {1, ENV_PARAM(NUM_OF_GROUP), ENV_PARAM(NUM_OF_CLASS)},
      xir::DataType{"XINT8"});
  auto output_tensor = xir::Tensor::create(
      "output", {1, ENV_PARAM(NUM_OF_GROUP), ENV_PARAM(NUM_OF_CLASS)},
      xir::DataType{"FLOAT32"});
  LOG(INFO) << "input_tensor info(create by hand):  "
            << input_tensor->to_string();
  LOG(INFO) << "output_tensor info(create by hand):  "
            << output_tensor->to_string();
  auto input_shape = input_tensor->get_shape();
  auto input_tensor_size = input_shape[0] * input_shape[1] * input_shape[2];
  LOG(INFO) << "input tensor size: " << input_shape[0] << " " << input_shape[1]
            << " " << input_shape[2];

  // use subgraph, attrs and tensors to apply tensor_buffer
  // std::vector<std::unique_ptr<vart::TensorBuffer>> tensor_buffers;
  std::pair<std::vector<std::unique_ptr<vart::TensorBuffer>>,
            std::vector<std::unique_ptr<vart::TensorBuffer>>>
      tensor_buffers;
  auto allocator = vart::TensorBufferAllocator::create(attrs.get());
  if (ENV_PARAM(TEST_ZERO_COPY) == 1) {
    LOG(INFO) << "allocate tensor buffers at PHY";
    tensor_buffers = allocator->allocate(
        s, std::vector<const xir::Tensor*>{input_tensor.get()},
        std::vector<const xir::Tensor*>{output_tensor.get()});
  } else {
    LOG(INFO) << "allocate tensor buffers at VIRT";
    tensor_buffers.first.emplace_back(
        // std::make_unique<vart::mm::HostFlatTensorBuffer>(input_tensor.get()));
        // //
        vart::alloc_cpu_flat_tensor_buffer(input_tensor.get()));
    tensor_buffers.second.emplace_back(
        // std::make_unique<vart::mm::HostFlatTensorBuffer>(output_tensor.get()));
        vart::alloc_cpu_flat_tensor_buffer(output_tensor.get()));
  }
  auto input_tensor_buffer = tensor_buffers.first[0].get();
  auto output_tensor_buffer = tensor_buffers.second[0].get();

  uint64_t input_addr = 0u;
  size_t input_size = 0u;
  uint64_t output_addr = 0u;
  size_t output_size = 0u;
  std::tie(input_addr, input_size) = input_tensor_buffer->data({0, 0, 0});
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << "input_addr: " << (void*)input_addr << "; input_size: " << input_size;
  std::tie(output_addr, output_size) = output_tensor_buffer->data({0, 0, 0});
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST)) << "output_addr: " << (void*)output_addr
                                      << "; output_size: " << output_size;

  // set random input value
  std::default_random_engine random(time(NULL));
  std::uniform_int_distribution<int8_t> random_int8(0, 60);
  for (auto i = 0; i < input_tensor_size; ++i) {
    *((unsigned char*)input_addr + i) = random_int8(random);
  }

  auto runner = std::make_unique<vart::SoftmaxRunner>(s, attrs.get());
  __TIC__(sfmx);
  runner->execute_async({input_tensor_buffer}, {output_tensor_buffer});
  __TOC__(sfmx);

  // compare result
  auto cls = ENV_PARAM(NUM_OF_CLASS);
  auto group = ENV_PARAM(NUM_OF_GROUP);
  auto fixpos = ENV_PARAM(SFM_FIX_POS);
  float scale = std::exp2f(-1.0f * (float)fixpos);
  vector<float> output_c(cls * group);
  LOG_IF(INFO, ENV_PARAM(DEBUG_TEST))
      << " fixpos=" << fixpos << " cls=" << cls << " group=" << group
      << " scale=" << scale;
  softmax_c((int8_t*)(input_addr), scale, cls, group, &output_c[0]);
  compare(cls, group, (int8_t*)(input_addr), (float*)output_addr, &output_c[0]);

  return 0;
}
