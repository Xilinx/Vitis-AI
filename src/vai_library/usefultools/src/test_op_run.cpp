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
#include <fstream>
#include "vart/op_imp.h"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "xir/graph/graph.hpp"
#include "xir/util/tool_function.hpp"
#include <xir/tensor/tensor.hpp>

using namespace std;

extern std::unique_ptr<vart::OpImp> create_op_imp(const xir::Op* op,
                                                  xir::Attrs* attrs);

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

static void fillin_input(const std::string refdir, vart::TensorBuffer* tb) {
  auto data = vart::get_tensor_buffer_data(tb, 0u);
  CHECK_EQ(data.size, tb->get_tensor()->get_data_size())
      << "must be continous tensor buffer";
  auto filename = refdir + "/" +
                  replace(xir::remove_xfix(tb->get_tensor()->get_name())) +
                  ".bin";
  CHECK(std::ifstream(filename).read((char*)data.data, data.size).good())
      << "fail to read! filename=" << filename
      << ";tensor=" << tb->get_tensor()->get_name() << endl;
  LOG(INFO) << "read " << filename << " to " << data.data
            << " size=" << data.size;
}

static void dump_output(const std::string dir, vart::TensorBuffer* tb) {
  auto data = vart::get_tensor_buffer_data(tb, 0u);
  CHECK_EQ(data.size, tb->get_tensor()->get_data_size())
      << "must be continous tensor buffer";
  auto filename =
      dir + "/" +
      replace(xir::remove_xfix(tb->get_tensor()->get_name()) + ".bin");
  CHECK(std::ofstream(filename).write((char*)data.data, data.size).good())
      << "failed to write: " << filename;
  LOG(INFO) << "write output to " << filename << " from " << data.data
            << " size=" << data.size;
}

bool test_op_run(const std::string& xmodel_name, 
		 const std::string& op_name,
                 const std::string& ref_dir,
                 const std::string& dump_dir) {
  auto graph = xir::Graph::deserialize(xmodel_name);
  auto op = graph->get_op(op_name);
  CHECK(op != nullptr) << "cannot find op: " << op_name;
  LOG(INFO) << "try to test op: " << op->get_name();
  auto op_def = op->get_opdef();
  auto tensor_buffer_holder =
      std::vector<std::unique_ptr<vart::TensorBuffer>>();
  auto inputs = vitis::ai::vec_map(
      op_def->input_args(),
      [op, &tensor_buffer_holder](
          const xir::OpArgDef& op_arg_def) -> vart::OpImpArg {
        auto name = op_arg_def.name;
        auto input_ops = op->get_input_ops(name);
        return vart::OpImpArg{
            name, vitis::ai::vec_map(
                      input_ops,
                      [&tensor_buffer_holder](
                          const xir::Op* input_op) -> vart::TensorBuffer* {
                        auto b1 = vart::alloc_cpu_flat_tensor_buffer(
                            input_op->get_output_tensor());
                        auto ret = b1.get();
                        LOG(INFO) << " input op: " << input_op->get_name()
                                  << " tensor: "
                                  << input_op->get_output_tensor()->get_name();
                        tensor_buffer_holder.emplace_back(std::move(b1));
                        return ret;
                      })};
      });
  for (auto& input_arg : inputs) {
    for (auto tb : input_arg.args) {
      fillin_input(ref_dir, tb);
    }
  }
  auto attrs = xir::Attrs::create();
  auto op_imp = create_op_imp(op, attrs.get());
  auto output_tensor_buffer =
      vart::alloc_cpu_flat_tensor_buffer(op->get_output_tensor());
  CHECK(op != nullptr) << "cannot find op: " << op_name;
  LOG(INFO) << "graph name:" << graph->get_name()
            << "testing op: " << vart::to_string(inputs);

  op_imp->calculate(inputs, output_tensor_buffer.get());
  dump_output(dump_dir, output_tensor_buffer.get());
  return true;
}
