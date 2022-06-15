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

#include "rnn_xmodel_parser.hpp"

#include <cassert>

namespace vart {
namespace xrnn {

RnnXmodelParser::RnnXmodelParser(const xir::Subgraph* subgraph)
    : subgraph_(subgraph) {
  init();
}

void RnnXmodelParser::init() {
  target_ = subgraph_->get_attr<std::string>("target");
  weights_ = subgraph_->get_attr<std::vector<char>>("weights");
  subgraphs_ = subgraph_->children_topological_sort();
}

int RnnXmodelParser::get_model_input_seq_dim(bool aligned) const {
  const std::vector<const xir::Subgraph*> children =
      subgraph_->children_topological_sort();
  if (aligned) {
    return children[0]
        ->get_attr<std::map<std::string, std::uint32_t>>("load_src_reg_0")
        .at("aligned_size");
  }

  return children[0]
      ->get_attr<std::map<std::string, std::uint32_t>>("load_src_reg_0")
      .at("original_size");
}

int RnnXmodelParser::get_model_output_seq_dim(bool aligned) const {
  const std::vector<const xir::Subgraph*> children =
      subgraph_->children_topological_sort();
  if (aligned) {
    return children[0]
        ->get_attr<std::map<std::string, std::uint32_t>>("save_dst_reg_1")
        .at("aligned_size");
  }

  return children[0]
      ->get_attr<std::map<std::string, std::uint32_t>>("save_dst_reg_1")
      .at("original_size");
}

int RnnXmodelParser::get_num_layers() const {
  return subgraph_->get_attr<int32_t>("cell_size");
}

int RnnXmodelParser::get_batch_size() const {
  return subgraph_->get_attr<int32_t>("batch");
}

const std::string& RnnXmodelParser::get_target_device() const {
  return target_;
}

const std::vector<char>& RnnXmodelParser::get_weights() const {
  return weights_;
}

std::vector<uint32_t> RnnXmodelParser::get_ddr_reg_config(
    int i, const std::string& reg_name) const {
  CHECK_LT(i, subgraphs_.size());
  std::vector<uint32_t> config;
  if (!subgraphs_[i]->has_attr(reg_name)) {
    return config;
  }

  if (reg_name.substr(0, 8) == "load_dst") {
    config = subgraphs_[i]->get_attr<std::vector<uint32_t>>(reg_name);
  } else {
    auto dict =
        subgraphs_[i]->get_attr<std::map<std::string, uint32_t>>(reg_name);
    config.reserve(3);
    config.push_back(dict.at("dir"));
    config.push_back(dict.at("aligned_size"));
    config.push_back(dict.at("original_size"));
  }
  return config;
}

std::vector<char> RnnXmodelParser::get_end_instructions() const {
  return subgraph_->get_attr<std::vector<char>>("end_bin_instr");
}

std::vector<char> RnnXmodelParser::get_first_instructions(int i) const {
  return subgraphs_[i]->get_attr<std::vector<char>>("init_bin_instr");
}

std::vector<char> RnnXmodelParser::get_loop_instructions(int i) const {
  return subgraphs_[i]->get_attr<std::vector<char>>("loop_bin_instr");
}

}  // namespace xrnn
}  // namespace vart
