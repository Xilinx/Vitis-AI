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
   * @param input inputs with a customized type
   *
   * @param output outputs with a customized type
   *
   * @return pair<jobid, status> status 0 for exit successfully, others for
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

/**
 * @class Runner 
 * @brief Class of the Runner, provides API to use the runner.  
 *   The runner instance has a number of member functions to control 
 *   the execution and get the input and output tensors of the runner. 
 
   Sample code:

   @code
    // This example assumes that you have a DPU subgraph called dpu_subgraph.
    // The way to create a DPU runner to run dpu_subgraph is shown below.
    
    // create runner
    auto runner = vart::Runner::create_runner(dpu_subgraph, ”run”);
    // get input tensors
    auto input_tensors = runner->get_input_tensors();
    // get input tensor buffers
    auto input_tensor_buffers = std::vector<vart::TensorBuffer*>();
        for (auto input : input_tensors) {
            auto t = vart::alloc_cpu_flat_tensor_buffer(input);
            input_tensor_buffers.emplace_back(t.get());
    }
    // get output tensors
    auto output_tensors = runner->get_output_tensors();
    // get output tensor buffers
    auto output_tensor_buffers = std::vector< vart::TensorBuffer*>();
    for (auto output : output _tensors) {
        auto t = vart::alloc_cpu_flat_tensor_buffer(output);
                output_tensor_buffers.emplace_back(t.get());
    }
    // sync input tensor buffers
    for (auto& input : input_tensor_buffers) {
        input->sync_for_write(0, input->get_tensor()->get_data_size() /
                input->get_tensor()->get_shape()[0]);
    }
    // run runner
    auto v = runner->execute_async(input_tensor_buffers, output_tensor_buffers);
    auto status = runner->wait((int)v.first, 1000000000);
    // sync output tensor buffers
    for (auto& output : output_tensor_buffers) {
        output->sync_for_read(0, output->get_tensor()->get_data_size() /
        output->get_tensor()->get_shape()[0]);
    }
   @endcode
 * */
class Runner : public BaseRunner<const std::vector<TensorBuffer*>&> {
 public:
  /**
   * @brief Factory function to create an instance of DPU runner by
   * subgraph.
   * @param subgraph  XIR Subgraph
   * @param mode 1 mode supported: 'run' - DPU runner.
   * @return An instance of DPU runner.

   Sample code:

     @code
     // This API can be used like:
     auto runner = vart::Runner::create_runner(subgraph, "run");
     @endcode
   */
  static std::unique_ptr<Runner> create_runner(
      // inner use: 3 mode supported: 'ref' - CPU runner, 'sim' - Simulation * runner,  'run' - DPU runner.
      const xir::Subgraph* subgraph, const std::string& mode = std::string(""));

  /**
   * @brief Factory function to create an instance of DPU runner by
   * subgraph, and attrs
   *
   * @param subgraph  XIR Subgraph
   *
   * @param attrs XIR attrs object, this object is shared among all
   * runners on the same graph.
   *
   * @param attrs["mode"], 1 mode supported: 'run' - DPU runner.
   *
   * @return An instance of DPU runner.
   */
   // inner use: attrs["mode"], 3 mode supported: 'ref' - CPU runner, 'sim' -
   //  Simulation runner,  'run' - DPU runner.
  static std::unique_ptr<Runner> create_runner_with_attrs(
      const xir::Subgraph* subgraph, xir::Attrs* attrs);

  //# Overload method with model directory for DPUV1
  // brief create dpu runner by model_directory
  //
  // because one DPU model may contains more than one DPU kernels(or DPU
  // subgraph), this function returns a vector of a dpu runner.
  //
  // param model_directory the directory name which contrains meta.json
  // return a vector of dpu runner
  static std::vector<std::unique_ptr<Runner>> create_runner(
      const std::string& model_directory);

 public:
  enum class TensorFormat { NHWC = 0, NCHW };

 public:
  virtual ~Runner() = default;
  /**
   * @brief Executes the runner. This is a blocking function.
   *
   * @param input A vector of TensorBuffer create by all input tensors of
   * runner.
   *
   * @param output A vector of TensorBuffer create by all output tensors of
   * runner.
   *
   * @return pair<jobid, status> status 0 for exit successfully, others for
   * customized warnings or errors
   *
   */
  virtual std::pair<uint32_t, int> execute_async(
      const std::vector<TensorBuffer*>& input,
      const std::vector<TensorBuffer*>& output) = 0;

  /**
   * @brief Waits for the end of DPU processing.
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
   *@brief Get the tensor format of runner.
   *@return TensorFormat : NHWC / HCHW

   Sample code:

     @code
       auto format = runner->get_tensor_format();
       switch (format) {
           case vart::Runner::TensorFormat::NCHW:
               // do something
               break;
           case vart::Runner::TensorFormat::NHWC:
               // do something
               break;
       }
     @endcode
   */
  virtual TensorFormat get_tensor_format();
  /**
   *@brief Get all input tensors of runner.
   *@return All input tensors. A vector of raw pointer to the input tensor.
   
   Sample code:

     @code
      inputTensors = runner->get_input_tensors();
      for (auto input : inputTensor) {
          input->get_name();
          input->get_shape();
          input->get_element_num();
      }
     @endcode
   */
  virtual std::vector<const xir::Tensor*> get_input_tensors() = 0;
  /**
   *@brief Get all output tensors of runner.
   *@return All output tensors. A vector of raw pointer to the output tensor.
   
   Sample code:
     @code
       outputTensors = runner->get_output_tensors();
         for (auto output : outputTensor) {
             output->get_name();
             output->get_shape();
             output->get_element_num();
       }
     @endcode
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
