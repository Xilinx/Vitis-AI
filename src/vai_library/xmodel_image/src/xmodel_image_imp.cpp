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
#include "./xmodel_image_imp.hpp"

#include <cmath>
#include <iterator>
#include <mutex>
#include <stdexcept>

#include "vart/batch_tensor_buffer_view.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
#include "vitis/ai/xmodel_postprocessor.hpp"
#include "vitis/ai/xmodel_preprocessor.hpp"
#include "xir/util/tool_function.hpp"

DEF_ENV_PARAM(DEBUG_XMODEL_IMAGE, "0")

namespace vitis {
namespace ai {

static std::shared_ptr<GraphHolder> load_graph(const std::string& filename) {
  static std::mutex mtx;
  std::lock_guard<std::mutex> lock(mtx);
  return vitis::ai::WeakStore<std::string, GraphHolder>::create(filename,
                                                                filename);
}

static std::unique_ptr<xir::Attrs> create_attrs() {
  auto attrs = xir::Attrs::create();
  attrs->set_attr("lib", std::map<std::string, std::string>{
                             {"graph", "libvitis_ai_library-graph_runner.so"}});
  return attrs;
}

static std::unique_ptr<vart::RunnerExt> create_runner(xir::Graph* graph,
                                                      xir::Attrs* attrs) {
  xir::Subgraph* s = graph->get_root_subgraph();
  if (!s->has_attr("device")) {
    s->set_attr<std::string>("device", "graph");
  }
  if (!s->has_attr("runner")) {
    s->set_attr<std::map<std::string, std::string>>(
        "runner", {{"ref", "libvitis_ai_library-graph_runner.so.3"},
                   {"sim", "libvitis_ai_library-graph_runner.so.3"},
                   {"run", "libvitis_ai_library-graph_runner.so.3"}});
  }
  auto r1 = vart::Runner::create_runner_with_attrs(s, attrs).release();
  auto r2 = dynamic_cast<vart::RunnerExt*>(r1);
  CHECK(r2 != nullptr) << "cannot create runner!";
  return std::unique_ptr<vart::RunnerExt>(r2);
}

static std::map<std::string, std::vector<std::string>> get_graph_outputs(
    const xir::Graph* graph) {
  auto ret = std::map<std::string, std::vector<std::string>>();
  if (graph->has_attr("xmodel_outputs")) {
    ret = graph->get_attr<std::map<std::string, std::vector<std::string>>>(
        "xmodel_outputs");
  } else {
    LOG(WARNING) << "cannot filload graph attr xmodel_outputs";
  }
  return ret;
}

static vart::TensorBuffer* find_tensor_buffer(
    const std::string& name,
    const std::vector<vart::TensorBuffer*>& tensor_buffers) {
  vart::TensorBuffer* ret = nullptr;
  for (auto i = 0u; i < tensor_buffers.size(); ++i) {
    auto tensor_name = tensor_buffers[i]->get_tensor()->get_name();
    auto origin_name = xir::remove_xfix(tensor_name);
    LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_IMAGE) >= 3)
        << "searching for name=" << name << ";"
        << " tensor_name=" << tensor_name << " origin_name=" << origin_name;
    if (tensor_name == name || origin_name == name) {
      ret = tensor_buffers[i];
      break;
    }
  }
  CHECK(ret != nullptr) << "cannot find tensor buffer: name=" << name
                        << "; tensor buffers: " << to_string(tensor_buffers);
  return ret;
}

static const xir::Op* find_op(const std::string& name,
                              const xir::Graph* graph) {
  auto tensor = graph->get_tensor(name);
  CHECK(tensor != nullptr) << "cannot find tensor! name:" << name;
  return tensor->get_producer();
}

XmodelImageImp::XmodelImageImp(const std::string& filename)
    : XmodelImage{},
      graph_{load_graph(filename)},
      attrs_{create_attrs()},
      runner_{create_runner(graph_->graph.get(), attrs_.get())} {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_IMAGE))
      << "XmodelImageImp create @" << (void*)this << " ";
  auto tensors = runner_->get_input_tensors();
  CHECK_EQ(tensors.size(), 1u) << "only support xmodel with a single input";
  auto shape = tensors[0]->get_shape();
  CHECK_EQ(shape.size(), 4u) << "only support xmodel with shape [NHWC]";
  CHECK_EQ(tensors[0]->get_data_type().bit_width, 8) << "must be 8bits image";
  batch_ = (size_t)shape[0];
  height_ = (size_t)shape[1];
  width_ = (size_t)shape[2];
  depth_ = (size_t)shape[3];
  input_tensor_buffers_ = runner_->get_inputs();
  CHECK_EQ(input_tensor_buffers_.size(), 1u);
  output_tensor_buffers_ = runner_->get_outputs();
  auto graph = graph_->graph.get();
  preprocessor_ = XmodelPreprocessor::create(graph, tensors[0]);
  postprocessor_ = XmodelPostprocessorBase::create(graph);
  post_processor_inputs_ =
      build_post_processor_inputs(postprocessor_->get_def());
  postprocessor_->initialize(XmodelPostprocessorInitializationArgs{
      graph, tensors[0], post_processor_inputs_, attrs_.get()});
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

XmodelPostprocessorInputs XmodelImageImp::build_post_processor_inputs(
    const xir::OpDef& opdef) {
  auto graph = graph_->graph.get();
  auto graph_outputs = get_graph_outputs(graph_->graph.get());
  auto ret = XmodelPostprocessorInputs();
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
      input.args[i].tensor_buffer =
          find_tensor_buffer(it->second[i], this->output_tensor_buffers_);
    }
  }
  return ret;
}

XmodelImageImp::~XmodelImageImp() {
  LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_IMAGE))
      << "XmodelImageImp destroyed @" << (void*)this << " ";
}

size_t XmodelImageImp::get_batch() const { return batch_; }
size_t XmodelImageImp::get_width() const { return width_; }
size_t XmodelImageImp::get_height() const { return height_; }
size_t XmodelImageImp::get_depth() const { return depth_; }

static std::vector<std::unique_ptr<vart::TensorBuffer>> map_to_single_batch(
    const std::vector<vart::TensorBuffer*>& tensor_buffers, size_t batch_index,
    size_t batch) {
  return vitis::ai::vec_map(
      tensor_buffers,
      [batch_index, batch](const vart::TensorBuffer* tensor_buffer) {
        return std::unique_ptr<vart::TensorBuffer>(
            std::make_unique<vart::BatchTensorBufferView>(
                const_cast<vart::TensorBuffer*>(tensor_buffer), batch_index,
                batch));
      });
}

static XmodelPostprocessorInputs map_to_single_batch(
    XmodelPostprocessorInputs& post_processor_inputs_, size_t batch) {
  XmodelPostprocessorInputs ret;
  ret.inputs = vitis::ai::vec_map(
      post_processor_inputs_.inputs,
      [batch](const XmodelPostprocessorArg& arg) {
        XmodelPostprocessorArg ret;
        ret.name = arg.name;
        ret.args = vitis::ai::vec_map(
            arg.args, [batch](const XmodelPostprocessorArgOpAndTensorBuffer&
                                  op_and_tensor_buffer) {
              XmodelPostprocessorArgOpAndTensorBuffer ret;
              ret.op = op_and_tensor_buffer.op;
              ret.tensor_buffer = new vart::BatchTensorBufferView(
                  const_cast<vart::TensorBuffer*>(
                      op_and_tensor_buffer.tensor_buffer),
                  0u, batch);
              return ret;
            });
        return ret;
      });
  return ret;
}

static void destroy_batch_view(
    XmodelPostprocessorInputs& post_processor_inputs) {
  XmodelPostprocessorInputs ret;
  for (auto& input : post_processor_inputs.inputs) {
    for (auto& arg : input.args) {
      delete arg.tensor_buffer;
    }
  }
}

std::vector<vitis::ai::proto::DpuModelResult> XmodelImageImp::run(
    const std::vector<Mat>& images) {
  CHECK_LE(images.size(), get_batch())
      << " the number of input images must be as same as the HW batch size";
  LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_IMAGE))
      << "running a xmodel: " << graph_->graph->get_name();
  auto user_batch = images.size();
  auto inputs = map_to_single_batch(input_tensor_buffers_, 0u, user_batch);
  auto outputs = map_to_single_batch(output_tensor_buffers_, 0u, user_batch);
  preprocessor_->process(images, inputs[0].get());
  auto status =
      runner_->execute_async(vitis::ai::vector_unique_ptr_get(inputs),
                             vitis::ai::vector_unique_ptr_get(outputs));
  auto ok = runner_->wait(status.first, -1);
  CHECK_EQ(ok, 0) << "cannot execute runner";
  auto args = map_to_single_batch(post_processor_inputs_, user_batch);
  auto ret = postprocessor_->process(args);
  destroy_batch_view(args);
  return ret;
}

}  // namespace ai
}  // namespace vitis
