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

#include <string>
#include <vector>

namespace vart {
namespace xrnn {

/// Abstract class that represents RNN Model Parser
class RNNModelParser {
 public:
  virtual ~RNNModelParser() = default;

  /// Return the size of a input frame in bytes
  /// @param : aligned : Returns aligned size if true, else return original size
  virtual int get_model_input_seq_dim(bool aligned = true) const = 0;

  /// Return the size of a output frame in bytes
  /// @param : aligned : Returns aligned size if true, else return original size
  virtual int get_model_output_seq_dim(bool aligned = true) const = 0;

  /// Return number of layers in the model
  virtual int get_num_layers() const = 0;

  /// Return the supported batch size
  virtual int get_batch_size() const = 0;

  /// Return the supported target device
  virtual const std::string& get_target_device() const = 0;

  /// Returns the compiled/reformatted weights of the model
  virtual const std::vector<char>& get_weights() const = 0;

  /// Returns DDR reg config for a specific layer and register
  /// Output is a vector whose values depends on type of register
  /// If register not found, empty vector is returned
  virtual std::vector<uint32_t> get_ddr_reg_config(
      int layer_idx, const std::string& reg_name) const = 0;

  /// Get Model end insructions
  virtual std::vector<char> get_end_instructions() const = 0;
  /// Get first instructions of a particular layer
  virtual std::vector<char> get_first_instructions(int layer_idx) const = 0;
  /// Get loop instructions of a particular layer
  virtual std::vector<char> get_loop_instructions(int layer_idx) const = 0;
};

}  // namespace xrnn
}  // namespace vart
