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

#include "vart/vart.h"

#include <glog/logging.h>

#include <vart/runner_ext.hpp>
#include <vart/tensor_buffer.hpp>

extern "C" vart_runner_t vart_create_runner(xir_subgraph_t subgraph,
                                            const char* mode) {
  return static_cast<vart_runner_t>(
      vart::Runner::create_runner(static_cast<xir::Subgraph*>(subgraph),
                                  std::string(mode))
          .release());
}

extern "C" void vart_destroy_runner(vart_runner_t runner) {
  delete static_cast<vart::Runner*>(runner);
}

extern "C" vart_job_id_and_status_t vart_runner_execute_async(
    vart_runner_t runner, vart_tensor_buffer_t inputs[], size_t num_of_inputs,
    vart_tensor_buffer_t outputs[], size_t num_of_outputs) {
  vart_job_id_and_status_t ret;
  std::tie(ret.job_id, ret.status) =
      static_cast<vart::Runner*>(runner)->execute_async(
          std::vector<vart::TensorBuffer*>(
              reinterpret_cast<vart::TensorBuffer**>(inputs),
              reinterpret_cast<vart::TensorBuffer**>(inputs) + num_of_inputs),
          std::vector<vart::TensorBuffer*>(
              reinterpret_cast<vart::TensorBuffer**>(outputs),
              reinterpret_cast<vart::TensorBuffer**>(outputs) + num_of_outputs))

      ;
  return ret;
}

extern "C" int vart_runner_wait(vart_runner_t runner, int jobid, int timeout) {
  return static_cast<vart::Runner*>(runner)->wait(jobid, timeout);
}

extern "C" size_t vart_runner_get_num_of_input_tensors(vart_runner_t runner) {
  return static_cast<vart::Runner*>(runner)->get_input_tensors().size();
}
extern "C" void vart_runner_get_input_tensors(vart_runner_t runner,
                                              xir_tensor_t inputs[]) {
  using const_xir_tensor_pointer = const xir::Tensor*;
  auto tensors = static_cast<vart::Runner*>(runner)->get_input_tensors();
  std::copy(tensors.begin(), tensors.end(),
            (const_xir_tensor_pointer*)(inputs));
}

extern "C" size_t vart_runner_get_num_of_output_tensors(vart_runner_t runner) {
  return static_cast<vart::Runner*>(runner)->get_output_tensors().size();
}

extern "C" void vart_runner_get_output_tensors(vart_runner_t runner,
                                               xir_tensor_t outputs[]) {
  using const_xir_tensor_pointer = const xir::Tensor*;
  auto tensors = static_cast<vart::Runner*>(runner)->get_output_tensors();
  std::copy(tensors.begin(), tensors.end(),
            (const_xir_tensor_pointer*)(outputs));
}

extern "C" xir_tensor_t vart_tensor_buffer_get_tensor(vart_tensor_buffer_t tb) {
  return (xir_tensor_t) static_cast<vart::TensorBuffer*>(tb)->get_tensor();
}

extern "C" vart_tensor_buffer_address_t vart_tensor_buffer_data(
    vart_tensor_buffer_t tb, int32_t idx[], size_t num_of_idx) {
  vart_tensor_buffer_address_t ret;
  std::tie(ret.addr, ret.size) = static_cast<vart::TensorBuffer*>(tb)->data(
      std::vector<int32_t>(idx, idx + num_of_idx));
  return ret;
}

extern "C" void vart_destroy_tensor_buffer(vart_tensor_buffer_t tb) {
  delete static_cast<vart::TensorBuffer*>(tb);
}

extern "C" void vart_runner_get_inputs(vart_runner_t runner,
                                       vart_tensor_buffer_t inputs[]) {
  auto self1 = static_cast<vart::Runner*>(runner);
  auto self2 = dynamic_cast<vart::RunnerExt*>(self1);
  CHECK(self2) << "runtime error: it is not an instance of vart::RunnerExt";
  auto ibs = self2->get_inputs();
  std::copy(ibs.begin(), ibs.end(), (void**)inputs);
  return;
}
/** @brief return the allocated output tensor buffers.
 *
 * potentially more efficient
 * */

extern "C" void vart_runner_get_outputs(vart_runner_t runner,
                                        vart_tensor_buffer_t outputs[]) {
  auto self1 = static_cast<vart::Runner*>(runner);
  auto self2 = dynamic_cast<vart::RunnerExt*>(self1);
  CHECK(self2) << "runtime error: it is not an instance of vart::RunnerExt";
  auto obs = self2->get_outputs();
  std::copy(obs.begin(), obs.end(), (void**)outputs);
  return;
}

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: undecided-unix */
/* End: */
