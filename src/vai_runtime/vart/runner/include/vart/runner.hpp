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

#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vart/tensor_buffer.hpp"

namespace xir {
class Subgraph;
class Attrs;
}  // namespace xir

namespace vart {

template <typename InputType, typename OutputType = InputType>
class BaseRunner {
 public:
  virtual ~BaseRunner() = default;
  /**
   * @brief execute_async
   *
   * @param in inputs with a customized type
   *
   * @param out outputs with a customized type
   *
   * @return pair<jodid, status> status 0 for exit successfully, others for
   * customized warnings or errors
   *
   */
  virtual std::pair<std::uint32_t, int> execute_async(InputType input,
                                                      OutputType output) = 0;

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
  virtual int wait(int jobid, int timeout = -1) = 0;
};

class Runner : public BaseRunner<const std::vector<TensorBuffer*>&> {
 public:
  /**
   * @brief Factory fucntion to create an instance of CPU/SIM/DPU runner by
   * subgraph.
   * @param subgraph  XIR Subgraph
   * @param mode 3 mode supported: 'ref' - CPU runner, 'sim' - Simulation
   * runner,  'run' - DPU runner.
   * @return An instance of CPU/SIM/DPU runner.
   */
  static std::unique_ptr<Runner> create_runner(
      const xir::Subgraph* subgraph, const std::string& mode = std::string(""));

  /**
   * @brief Factory fucntion to create an instance of CPU/SIM/DPU runner by
   * subgraph, and attrs
   *
   * @param subgraph  XIR Subgraph
   *
   * @param attrs XIR attrs object, this object is shared among all
   * runners on the same graph.
   *
   * @param attrs["mode"], 3 mode supported: 'ref' - CPU runner, 'sim' -
   * Simulation runner,  'run' - DPU runner.
   *
   * @return An instance of CPU/SIM/DPU runner.
   */
  static std::unique_ptr<Runner> create_runner_with_attrs(
      const xir::Subgraph* subgraph, xir::Attrs* attrs);
  //# Overload method with model directory for DPUV1
  /** @brief create dpu runner by model_directory
   *
   * because one DPU model may contains more than one DPU kernels(or DPU
   * subgraph), this function returns a vector of a dpu runner.
   *
   * @param model_directory the directory name which contrains meta.json
   * @return a vector of dpu runner
   **/
  static std::vector<std::unique_ptr<Runner>> create_runner(
      const std::string& model_directory);

 public:
  enum class TensorFormat { NHWC = 0, NCHW };

 public:
  virtual ~Runner() = default;
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
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<TensorBuffer*>& input,
      const std::vector<TensorBuffer*>& output) = 0;

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
  virtual int wait(int jobid, int timeout) = 0;

  /**
   *@brief Get tensor format of runner.
   *@return TensorFormat : NHWC / HCHW
   */
  virtual TensorFormat get_tensor_format();
  /**
   *@brief Get all input tensors of runner.
   *@return All input tensors. A vector of raw pointer to the input tensor.
   */
  virtual std::vector<const xir::Tensor*> get_input_tensors() = 0;
  /**
   *@brief Get all output tensors of runner.
   *@return All output tensors. A vector of raw pointer to the output tensor.
   */
  virtual std::vector<const xir::Tensor*> get_output_tensors() = 0;
};
struct Meta {
  // target name
  std::string target;
  // the shared library which contains a real factory method.
  std::string lib;
  // directory name of meta.json
  std::string dirname;
};

struct DpuMeta : public Meta {
  // the file name of a DPU model.
  std::string filename;
  // one DPU model may contains more than one DPU kernels.
  std::vector<std::string> kernels;
  // model configurations
  std::string config_file;
};
}  // namespace vart
