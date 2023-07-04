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
#include <glog/logging.h>

#include <iostream>
#include <xir/graph/graph.hpp>
#include <xir/util/tool_function.hpp>

#include "vitis/ai/xmodel_postprocessor.hpp"
//
#include "../samples/xmodel_result_to_string.hpp"
#include "vart/mm/host_flat_tensor_buffer.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/path_util.hpp"
#include "vitis/ai/xmodel_jit.hpp"

static void jit(xir::Graph* graph, std::string filename) {
  std::string realpath = filename;
  auto dirname = vitis::ai::file_name_directory(realpath);
  auto basename = vitis::ai::file_name_basename(realpath);
  graph->set_attr<std::string>("__file__", filename);
  graph->set_attr<std::string>("__dir__", dirname);
  graph->set_attr<std::string>("__basename__", basename);
  auto jit = vitis::ai::XmodelJit::create(graph);
  auto ok = jit->jit();
  CHECK(ok == 0);
};

static std::map<std::string, std::vector<std::string>> get_graph_outputs(
    const xir::Graph* graph) {
  auto ret = std::map<std::string, std::vector<std::string>>();
  if (graph->has_attr("xmodel_outputs")) {
    ret = graph->get_attr<std::map<std::string, std::vector<std::string>>>(
        "xmodel_outputs");
  } else {
    LOG(WARNING) << "cannot find graph attr xmodel_outputs";
  }
  return ret;
}

static const xir::Op* find_op(const std::string& name,
                              const xir::Graph* graph) {
  return graph->get_tensor(name)->get_producer();
}

static std::string to_string(
    const std::map<std::string, std::vector<std::string>>& x) {
  std::ostringstream str;
  str << "{";
  for (auto& i : x) {
    str << i.first << ":[";
    for (auto& j : i.second) {
      str << " " << j;
    }
    str << "]";
  }
  str << "}";
  return str.str();
}

static std::string replace(const std::string& s) {
  const std::string pat = "/";
  std::ostringstream str;
  for (auto c : s) {
    if (pat.find(c) != std::string::npos) {
      str << "_";
    } else {
      str << c;
    }
  }
  return str.str();
}

static vitis::ai::XmodelPostprocessorInputs build_post_processor_inputs(
    xir::Graph* graph, const xir::OpDef& opdef) {
  auto graph_outputs = get_graph_outputs(graph);
  auto ret = vitis::ai::XmodelPostprocessorInputs{};
  ret.inputs.resize(opdef.input_args().size());
  for (auto i = 0u; i < ret.inputs.size(); ++i) {
    auto& input_arg_def = opdef.input_args()[i];
    auto& input = ret.inputs[i];
    auto it = graph_outputs.find(input_arg_def.name);
    CHECK(it != graph_outputs.end())
        << "cannot find input op!"               //
        << "name=" << input_arg_def.name << ";"  //
        << "graph_outputs: " << to_string(graph_outputs);
    input.name = input_arg_def.name;
    input.args.resize(it->second.size());
    for (auto i = 0u; i < it->second.size(); ++i) {
      input.args[i].op = find_op(it->second[i], graph);
      auto tensor = input.args[i].op->get_output_tensor();
      input.args[i].tensor_buffer =
          vart::alloc_cpu_flat_tensor_buffer(tensor).release();
      // std::make_unique<vart::mm::HostFlatTensorBuffer>(tensor).release();
      auto tb = vart::simple_tensor_buffer_t<void>::create(
          input.args[i].tensor_buffer);
      auto refdir = std::string("ref");
      auto filename = refdir + "/" +
                      replace(xir::remove_xfix(tb.tensor->get_name())) + ".bin";
      CHECK(std::ifstream(filename).read((char*)tb.data, tb.mem_size).good())
          << "fail to read! filename=" << filename
          << ";tensor=" << tb.tensor->get_name() << std::endl;
      LOG(INFO) << "read " << filename << " to " << tb.data
                << " size=" << tb.mem_size;
    }
  }
  return ret;
}

int main(int argc, char* argv[]) {
  std::string g_xmodel_file = argv[1];
  auto graph = xir::Graph::deserialize(g_xmodel_file);
  jit(graph.get(), g_xmodel_file);
  vitis::ai::XmodelPostprocessorInputs post_processor_inputs_;
  auto postprocessor_ = vitis::ai::XmodelPostprocessorBase::create(graph.get());
  post_processor_inputs_ =
      build_post_processor_inputs(graph.get(), postprocessor_->get_def());
  auto attrs_ = xir::Attrs::create();
  auto ops = graph->get_head_ops();
  const xir::Op* input_op = nullptr;
  for (auto& op : ops) {
    if (op->get_type() == "data-fix") {
      input_op = op;
      break;
    }
  }
  CHECK(input_op != nullptr);
  postprocessor_->initialize(vitis::ai::XmodelPostprocessorInitializationArgs{
      graph.get(), input_op->get_output_tensor(), post_processor_inputs_,
      attrs_.get()});
  auto results = postprocessor_->process(post_processor_inputs_);
  for (const auto& r : results) {
    LOG(INFO) << "xmodel result:\n" << to_string(r);
  }

  return 0;
}
