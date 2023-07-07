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
#pragma once

#include <map>
#include <string>
#include <vector>
#include <xir/graph/graph.hpp>

#define MAX_REG_ID_SIZE  256

namespace vart {
// avoid dependency on XRT
struct xrt_bo_t {
  void* xrt_handle;
#ifdef _WIN32
  void* xrt_bo_handle;
#else
  unsigned int xrt_bo_handle;
#endif
};
enum class reg_type_t : int {
  INVALID,     //
  CONST,       // for v1 reg_id_to_context_type == "CONST"
  DATA_LOCAL,  // for v1 reg_id_to_context_type == "DATA" and the
               // corresponding tensors are mix input, output and
               // intermediate internal tensors
  // to support zero copy, only single such reg type is supported,
  // otherwise, zero copy is disabled, i.e. get_input_buffer_size
  // return -1;
  DATA_GLOBAL,  // for v2 reg_id_to_context_type_v2 == "WORKSPACE"
  // to support zero copy, only single such reg type is supported,
  // otherwise, zero copy is disabled, i.e. get_input_buffer_size
  // return -1;
  DATA_LOCAL_INPUT,   // for v2 reg_id_to_context_type_v2 == "INTERACE"
                      // and the corresponding tensors are inputs only.
  DATA_LOCAL_OUTPUT,  // for v2 reg_id_to_context_type_v2 == "INTERACE"
                      // and the corresponding tensors are outputs only.
};

struct reg_basic_info_t {
  size_t reg_id;
  reg_type_t type;
  size_t size;
};

static inline std::string to_string(reg_type_t t) {
  const char* name[] = {"INVALID",          "CONST",
                        "DATA_LOCAL",       "DATA_GLOBAL",
                        "DATA_LOCAL_INPUT", "DATA_LOCAL_OUTPUT"};
  return std::string(name[(int)t]);
}

/**
 * @brief the total size of the buffer including all inputs and paddings
 *
 * For DPU model with N inputs, all the inputs should be
 * organized as one buffer with padding between them
 *
 * | padding_0	input_0	padding_1	input_1 ... input_N-1  padding_N |
 *
 * For each input, there is an offset which takes the start
 * address of padding_0 as the base. We will provide helper
 * functions for users (just an example, the final signature may be
 * changed):
 *
 * @return The total size of the buffer including all inputs and paddings.
 * return -1 if zero copy is not supported.
 *
 */
int get_input_buffer_size(const xir::Subgraph* subgraph);

/**
 * @brief return the vector of input offset for zero copy
 */

std::vector<size_t> get_input_offset(const xir::Subgraph* subgraph);

/**
 * @brief the total size of the buffer including all outputs and paddings
 *
 * For DPU model with N outputs, all the outputs should be
 * organized as one buffer with padding between them
 *
 * | padding_0  output_0 padding_1   output_1 ... output_N-1  padding_N |
 *
 * For each output, there is an offset which takes the start
 * address of padding_0 as the base. We will provide helper
 * functions for users (just an example, the final signature may be
 * changed):
 *
 * @return The total size of the buffer including all outputs and paddings.
 * return -1 if zero copy is not supported.
 *
 */
int get_output_buffer_size(const xir::Subgraph* subgraph);

/**
 * @brief return the vector of output offset for zero copy
 */

std::vector<size_t> get_output_offset(const xir::Subgraph* subgraph);

std::vector<vart::reg_basic_info_t> extract_reg_info_from_subgraph(
    const xir::Subgraph* subgraph);
}  // namespace vart
