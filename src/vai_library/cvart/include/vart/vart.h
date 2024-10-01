
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

#pragma once
#include <xir/cxir.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef void* vart_runner_t;
typedef void* vart_tensor_buffer_t;

/**
 * @brief Factory fucntion to create an instance of CPU/SIM/DPU runner by
 * subgraph.
 * @param subgraph  XIR Subgraph
 * @param mode 3 mode supported: 'ref' - CPU runner, 'sim' - Simulation
 * runner,  'run' - DPU runner.
 * @return An instance of CPU/SIM/DPU runner.
 */

vart_runner_t vart_create_runner(xir_subgraph_t subgraph, const char* mode);
/**
 * @brief destroy a runner
 * */
void vart_destroy_runner(vart_runner_t runner);
/**
 * @brief execute_async
 *
 * @param input A vector of TensorBuffer create by all input tensors of
 * runner.
 *
 * @param output A vector of TensorBuffe create by all output tensors of
 * runner.
 *
 * @return pair<jodid, status> status 0 for exit successfully, others for
 * customized warnings or errors
 *
 */
typedef struct {
  uint32_t job_id;
  int status;
} vart_job_id_and_status_t;

vart_job_id_and_status_t vart_runner_execute_async(
    vart_runner_t runner, vart_tensor_buffer_t inputs[], size_t num_of_inputs,
    vart_tensor_buffer_t outputs[], size_t num_of_outputs);

/**
 * @brief wait
 *
 * @details modes: 1. Blocking wait for specific ID. 2. Non-blocking wait for
 * specific ID. 3. Blocking wait for any ID. 4. Non-blocking wait for any ID
 *
 * @param jobid job id, neg for any id, others for specific job id
 *
 * @param timeout timeout, neg for block for ever, 0 for non-block, pos for
 * block with a limitation(ms).
 *
 * @return status 0 for exit successfully, others for customized warnings or
 * errors
 *
 */
int vart_runner_wait(vart_runner_t runner, int jobid, int timeout);
/**
 *@brief Get all input tensors of runner.
 *@return All input tensors. A vector of raw pointer to the input tensor.
 */
size_t vart_runner_get_num_of_input_tensors(vart_runner_t runner);
void vart_runner_get_input_tensors(vart_runner_t runner, xir_tensor_t inputs[]);
/**
 *@brief Get all output tensors of runner.
 *@return All output tensors. A vector of raw pointer to the output tensor.
 */
size_t vart_runner_get_num_of_output_tensors(vart_runner_t runner);
void vart_runner_get_output_tensors(vart_runner_t runner,
                                    xir_tensor_t outputs[]);

xir_tensor_t vart_tensor_buffer_get_tensor(vart_tensor_buffer_t tb);

typedef struct {
  uint64_t addr;
  size_t size;
} vart_tensor_buffer_address_t;
vart_tensor_buffer_address_t vart_tensor_buffer_data(vart_tensor_buffer_t tb,
                                                     int32_t idx[],
                                                     size_t num_of_idx);

/// extended API, not stable yet.  defined in libvart-dpu-runner.so
/** @brief return the allocated input tensor buffers.
 *
 * potentially more efficient
 * */
void vart_runner_get_inputs(vart_runner_t runner,
                            vart_tensor_buffer_t inputs[]);
/** @brief return the allocated output tensor buffers.
 *
 * potentially more efficient
 * */
void vart_runner_get_outputs(vart_runner_t runner,
                             vart_tensor_buffer_t outputs[]);

void vart_destroy_tensor_buffer(vart_tensor_buffer_t tb);

#ifdef __cplusplus
}
#endif

/* Local Variables: */
/* mode:c */
/* c-basic-offset: 2 */
/* coding: undecided-unix */
/* End: */
