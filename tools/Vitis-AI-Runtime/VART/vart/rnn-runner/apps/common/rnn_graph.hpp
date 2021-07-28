/*
 * Copyright 2021 Xilinx Inc.
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
#include <memory>
#include <string>
#include <xir/graph/graph.hpp>

std::unique_ptr<xir::Graph> create_rnn_graph(
    const std::string model_dir, const std::string model_name,
    const int num_sequences, const int input_seq_dim, const int output_seq_dim,
    const std::string device_name, const int device_core_id) {
  std::unique_ptr<xir::Graph> fakegraph = xir::Graph::create("lstm");

  auto rs = fakegraph->get_root_subgraph();

  std::map<std::string, std::string> subg_attr = {
      {"run", "libvart-rnn-runner.so"}};
  rs->set_attr<std::map<std::string, std::string>>("runner", subg_attr);
  rs->set_attr<std::string>("device_name", device_name);
  rs->set_attr<unsigned>("device_core_id", device_core_id);
  rs->set_attr<std::string>("model_dir", model_dir);
  rs->set_attr<std::string>("model_name", model_name);
  rs->set_attr<unsigned>("num_sequences", num_sequences);
  rs->set_attr<unsigned>("input_seq_dim", input_seq_dim);
  rs->set_attr<unsigned>("output_seq_dim", output_seq_dim);
  return fakegraph;
}
