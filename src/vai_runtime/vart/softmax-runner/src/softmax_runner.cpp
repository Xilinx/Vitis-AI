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

#include "vart/softmax_runner.hpp"

#include <cmath>
#include <cstdint>

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

namespace vart {

SoftmaxRunner::SoftmaxRunner(const xir::Subgraph* subgraph, xir::Attrs* attrs)
    :  //
       // TODO: one controller per runner.
      controller_{xir::SfmController::get_instance()},
      input_{},
      output_{} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "@" << (void*)this << " softmax runner is created for subgraph "
      << subgraph->get_name();
  auto input_set = subgraph->get_sorted_input_tensors();
  CHECK_EQ(input_set.size(), 1u);
  auto output_set = subgraph->get_sorted_output_tensors();
  CHECK_EQ(output_set.size(), 1u);

  attrs->set_attr<size_t>("__device_id__", 0u);
  attrs->set_attr<size_t>("__batch__", 1);
  attrs->set_attr<int>(subgraph->get_name() + ":__tensor_buffer_location__", 1);

  auto allocator = vart::assistant::TensorBufferAllocator::create(attrs);
  auto tensor_buffers = allocator->allocate(
      subgraph, std::vector<const xir::Tensor*>{*input_set.begin()},
      std::vector<const xir::Tensor*>{*output_set.begin()});
  input_ = std::move((tensor_buffers.first)[0]);
  output_ = std::move((tensor_buffers.second)[0]);
}

SoftmaxRunner::~SoftmaxRunner() {}

std::pair<uint32_t, int> SoftmaxRunner::execute_async(
    const std::vector<vart::TensorBuffer*>& input,
    const std::vector<vart::TensorBuffer*>& output) {
  CHECK_EQ(input.size(), 1u) << "only support single input";
  CHECK_EQ(output.size(), 1u) << "only support single output";
  auto input_phy = prepare_input(input[0]);
  auto output_phy = prepare_output(output[0]);
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "@" << (void*)this << " start to run: "
      << " inputs= " << to_string(input) << " "  //
      << " outputs= " << to_string(output);
  start_controller(input_phy, output_phy);
  finalize_output(output_phy, output[0]);
  return std::make_pair(0u, 0);
}

static void debug_tensorbuffer(vart::TensorBuffer* tb) {
  uint64_t tb_addr = 0u;
  size_t tb_size = 0u;
  auto dim_idx = vart::get_index_zeros(tb->get_tensor());
  std::tie(tb_addr, tb_size) = tb->data(dim_idx);
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "addr: " << (void*)tb_addr << ", "
      << "size: " << tb_size;

  auto tb_tensor = tb->get_tensor();
  for (auto i = 0; i < tb_tensor->get_element_num(); ++i) {
    LOG(INFO) << "index: " << i << ", "
              << "value: " << (int)(*((unsigned char*)tb_addr + i));
  }
}

static void debug_tensorbuffer_float(vart::TensorBuffer* tb) {
  uint64_t tb_addr = 0u;
  size_t tb_size = 0u;
  auto dim_idx = vart::get_index_zeros(tb->get_tensor());
  std::tie(tb_addr, tb_size) = tb->data(dim_idx);
  LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
      << "addr: " << (void*)tb_addr << ", "
      << "size: " << tb_size;

  auto tb_tensor = tb->get_tensor();
  for (auto i = 0; i < tb_tensor->get_element_num(); ++i) {
    LOG(INFO) << "index: " << i << ", "
              << "value: " << *((float*)tb_addr + i);
  }
}

vart::TensorBuffer* SoftmaxRunner::prepare_input(vart::TensorBuffer* user) {
  auto ret = input_.get();
  if (user->get_location() == TensorBuffer::location_t::HOST_VIRT) {
    if (ENV_PARAM(DEBUG_SOFTMAX_RUNNER) >= 2) {
      LOG(INFO) << "user input virt tensor buffer info:";
      debug_tensorbuffer(user);
    }
    vart::TensorBuffer::copy_tensor_buffer(user /* from user */,
                                           ret /* to internal*/);
    if (ENV_PARAM(DEBUG_SOFTMAX_RUNNER) >= 2) {
      LOG(INFO) << "internal input  phy tensor buffer info:";
      debug_tensorbuffer(ret);
    }
  } else {
    ret = user;
  }
  return ret;
}

vart::TensorBuffer* SoftmaxRunner::prepare_output(vart::TensorBuffer* user) {
  auto ret = output_.get();
  if (user->get_location() == TensorBuffer::location_t::HOST_VIRT) {
    // no copy
  } else {
    ret = user;
  }
  return ret;
}

void SoftmaxRunner::finalize_output(vart::TensorBuffer* internal,
                                    vart::TensorBuffer* output) {
  if (output == internal) {
    // zero copy, do nothing
  } else {
    if (ENV_PARAM(DEBUG_SOFTMAX_RUNNER) >= 2) {
      LOG(INFO) << "internal output phy tensor buffer info:";
      debug_tensorbuffer_float(internal);
    }
    vart::TensorBuffer::copy_tensor_buffer(output_.get() /*from internal */,
                                           output /*to user*/);
    if (ENV_PARAM(DEBUG_SOFTMAX_RUNNER) >= 2) {
      LOG(INFO) << "user output virt tensor buffer info:";
      debug_tensorbuffer_float(output);
    }
  }
}

static int get_fix_pos(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point")) << "tensor = " << tensor->to_string();
  int fixpos = tensor->template get_attr<int>("fix_point");
  return fixpos;
}
static std::vector<std::int32_t> reshape_tensor_to_three_dim(
    std::vector<std::int32_t> in) {
  CHECK_GE(in.size(), 2) << "input dimension is less than 2";
  std::int32_t mid = 1;
  for (unsigned int i = 1; i < in.size() - 1; i++) {
    mid *= in[i];
  }
  return std::vector<std::int32_t>{in.front(), mid, in.back()};
}

void SoftmaxRunner::start_controller(vart::TensorBuffer* input,
                                     vart::TensorBuffer* output) {
  CHECK(input->get_location() != TensorBuffer::location_t::HOST_VIRT)
      << "input=" << input->to_string();
  CHECK(output->get_location() != TensorBuffer::location_t::HOST_VIRT)
      << "output=" << output->to_string();
  // a
  auto input_shape =
      reshape_tensor_to_three_dim(input->get_tensor()->get_shape());
  auto output_shape =
      reshape_tensor_to_three_dim(output->get_tensor()->get_shape());
  auto input_batch_size = input_shape[0];
  auto output_batch_size = output_shape[0];
  CHECK(output->get_tensor()->get_data_type().type == xir::DataType::FLOAT)
      << "output=" << output->to_string();
  CHECK_EQ(input_batch_size, output_batch_size);  // TODO CHECK DIMS same.
  CHECK_EQ(input_shape.size(), output_shape.size());
  CHECK_EQ(input_shape.size(), 3u);  // {batch, group, cls}
  for (auto i = 0u; i < input_shape.size(); ++i) {
    CHECK_EQ(input_shape[i], output_shape[i]);
  }
  auto group = (uint32_t)input_shape[1];
  auto cls = (uint32_t)input_shape[2];
  auto batch_size = input_batch_size;
  auto offset = (uint32_t)0u;  // TODO: read from ENV_PARAM();
  auto fixpos = get_fix_pos(input->get_tensor());
  for (auto batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    uint64_t input_addr = 0u;
    size_t input_size = 0u;
    uint64_t output_addr = 0u;
    size_t output_size = 0u;
    auto input_dim_idx = vart::get_index_zeros(input->get_tensor());
    auto output_dim_idx = vart::get_index_zeros(output->get_tensor());
    input_dim_idx[0] = batch_idx;
    output_dim_idx[0] = batch_idx;
    std::tie(input_addr, input_size) = input->data_phy(input_dim_idx);
    std::tie(output_addr, output_size) = output->data_phy(output_dim_idx);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOFTMAX_RUNNER))
        << "batch_size: " << batch_size << "; group: " << group
        << "; cls: " << cls << "; offset: " << offset << "; fixpos: " << fixpos;
    controller_->run_xrt_cu(device_core_id_, input_addr, cls, group, fixpos,
                            output_addr, offset);
  }

  return;
}

int SoftmaxRunner::wait(int jobid, int timeout) { return 0; }

std::vector<const xir::Tensor*> SoftmaxRunner::get_input_tensors() {
  return {input_->get_tensor()};
}

std::vector<const xir::Tensor*> SoftmaxRunner::get_output_tensors() {
  return {output_->get_tensor()};
}

std::vector<vart::TensorBuffer*> SoftmaxRunner::get_inputs() {
  return {input_.get()};
}

std::vector<vart::TensorBuffer*> SoftmaxRunner::get_outputs() {
  return {output_.get()};
}

}  // namespace vart
extern "C" vart::Runner* create_runner_with_attrs(const xir::Subgraph* subgraph,
                                                  xir::Attrs* attrs) {
  return new vart::SoftmaxRunner(subgraph, attrs);
}
