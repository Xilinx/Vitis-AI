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
#include <iostream>
#include <map>
#include <vector>

#include "rnn_model_parser.hpp"

struct LAYER_T {
  int direction;
  int size;
};

enum CONFIG_NAME { LOAD0 = 0, LOAD1 = 1, SAVE0 = 2, SAVE1 = 3 };

// Register names for external memory config
const std::array<std::string, 4> ext_mem_reg = {
    "load_src_reg_0", "load_src_reg_1", "save_dst_reg_0", "save_dst_reg_1"};

// Register names for internal memory config
const std::array<std::string, 2> int_mem_reg = {"load_dst_reg_0",
                                                "load_dst_reg_1"};
enum INT_CONFIG_NAME { LOAD_DST0 = 0, LOAD_DST1 = 1 };

// Offsets for reg w.r.t layer header (except Layer-0)
constexpr uint32_t HEAD_LEN_OFFSET = 0;
constexpr uint32_t FINSTR_ADDR_OFFSET = 4;
constexpr uint32_t FINSTR_LEN_OFFSET = 5;
constexpr uint32_t LINSTR_ADDR_OFFSET = 6;
constexpr uint32_t LINSTR_LEN_OFFSET = 7;

// Offsets for "load_dst_reg0/1" w.r.t batch_header start
constexpr uint32_t LOAD_SRC0_OFFSET = 0;
constexpr uint32_t LOAD_SRC1_OFFSET = 1;
constexpr uint32_t LOAD_DST0_OFFSET = 4;
constexpr uint32_t LOAD_DST1_OFFSET = 5;
constexpr uint32_t SAVE_DST0_OFFSET = 8;
constexpr uint32_t SAVE_DST1_OFFSET = 9;

class ModelConfig {
 public:
  explicit ModelConfig(const std::string& model_directory);
  explicit ModelConfig(const vart::xrnn::RNNModelParser* model_parser);
  ~ModelConfig() = default;
  /// Return number of layers in the network
  int get_layer_num() const;
  /// Return byte-offset to instructions of a layer
  int get_layer_instr_len(int layer_num, int batch) const;
  /// Get the number of 128-bit first-instructions in a layer
  int get_first_instr_count(int layer_num, int batch) const;
  /// Get the number of 128-bit loop-instructions in a layer
  int get_loop_instr_count(int layer_num, int batch) const;
  /// Get the number of 128-bit end-instructions in a model
  int get_end_instr_count(int batch) const;
  /// Get instruction count vector;
  const std::vector<int>& get_instr_count_vector(int batch) const;
  ///
  int get_reg_size(int layer_num, CONFIG_NAME config) const;
  ///
  int get_reg_dir(int layer_num, CONFIG_NAME config) const;

  std::pair<char*, size_t> get_instructions();
  bool dump_instructions(const std::string& filename) const;

 private:
  const vart::xrnn::RNNModelParser* model_parser_ = nullptr;
  // number of layers
  int layers_;
  // #layers x ext_mem_reg, all size in int16
  std::vector<std::vector<LAYER_T>> layer_config_;
  // dict [batch#n -> [#instruns]]
  // #instructions are in terms 128-bit #rows
  std::map<std::string, std::vector<int>> batch_lines_;
  std::vector<char> instrs_;

  std::vector<char> compose_instructions() const;
  std::vector<char> compose_instructions_u50() const;
  std::vector<char> compose_instructions_u25() const;
  void fill_ddr_config_u50(std::vector<char>& instructions) const;
  void fill_ddr_config_u25(std::vector<char>& instructions) const;
};
