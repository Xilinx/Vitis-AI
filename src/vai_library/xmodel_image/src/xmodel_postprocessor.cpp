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
#include "vitis/ai/xmodel_postprocessor.hpp"

#include <dlfcn.h>

#include <cmath>
#include <new>
#include <tuple>
#include <utility>

#include "vart/batch_tensor_buffer_view.hpp"
#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/image_util.hpp"

DEF_ENV_PARAM(DEBUG_XMODEL_IMAGE, "0")

namespace vitis {
namespace ai {

static std::string get_so_name(const xir::Graph* graph) {
  auto ret = std::string("libxmodel_postprocessor_common.so.3");
  if (graph->has_attr("xmodel_postprocessor")) {
    ret = graph->get_attr<std::string>("xmodel_postprocessor");
  } else {
    LOG(INFO) << "graph attrs:" << xir::to_string(graph->get_attrs().get());
  }
  return ret;
}

// python binding does not support vector<float> , only support vector<double>

XmodelPostprocessorBase::XmodelPostprocessorBase() {
  /*  auto tensor = graph_input_tensor;
    auto shape = tensor->get_shape();
    CHECK_EQ(shape.size(), 4u) << "only support NHWC";
    batch_ = (size_t)shape[0];
    height_ = (size_t)shape[1];
    width_ = (size_t)shape[2];
    depth_ = (size_t)shape[3];*/
}

std::unique_ptr<XmodelPostprocessorBase> XmodelPostprocessorBase::create(
    const xir::Graph* graph) {
  auto so_name = get_so_name(graph);
  //add RTLD_GLOBAL, dlopen default RTLD_LOCAL. if RTLD_LOCAL:
  //[libprotobuf ERROR google/protobuf/descriptor_database.cc:644] File already
  //exists in database: vitis/ai/proto/dpu_model_param.proto
  //[libprotobuf FATAL google/protobuf/descriptor.cc:1371] CHECK failed:
  //GeneratedDatabase()->Add(encoded_file_descriptor, size):
  //terminate called after throwing an instance of
  //'google::protobuf::FatalException'
  auto handle = dlopen(so_name.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  if (!handle) {
    LOG(FATAL) << "cannot open plugin: name=" << so_name;
  };
  typedef std::unique_ptr<XmodelPostprocessorBase> (*fm_type)();
  auto factory_method_p = (fm_type)dlsym(handle, "create_xmodel_postprocessor");
  if (factory_method_p == nullptr) {
    LOG(FATAL) << "not a valid plugin, cannot find symbol "
                  "\"create_xmodel_postprocessor\": name="
               << so_name;
  }
  auto ret = (*factory_method_p)();
  CHECK(ret != nullptr) << "plugin return a nullptr."
                           ";name="
                        << so_name;
  return ret;
}

XmodelPostprocessorSingleBatch::XmodelPostprocessorSingleBatch()
    : XmodelPostprocessorBase() {}

static std::vector<std::pair<
    std::string, std::vector<std::pair<const xir::Op*,
                                       std::unique_ptr<vart::TensorBuffer>>>>>
map_to_single_batch(
    const XmodelPostprocessorInputs& graph_output_tensor_buffers,
    size_t batch_index, size_t batch) {
  auto ret = std::vector<
      std::pair<std::string,
                std::vector<std::pair<const xir::Op*,
                                      std::unique_ptr<vart::TensorBuffer>>>>>();
  for (auto& input : graph_output_tensor_buffers.inputs) {
    ret.emplace_back(std::make_pair(
        input.name,
        vitis::ai::vec_map(
            input.args,
            [batch_index,
             batch](const XmodelPostprocessorArgOpAndTensorBuffer& op_and_tb) {
              return std::make_pair(
                  op_and_tb.op,
                  std::unique_ptr<vart::TensorBuffer>(
                      std::make_unique<vart::BatchTensorBufferView>(
                          const_cast<vart::TensorBuffer*>(
                              op_and_tb.tensor_buffer),
                          batch_index, batch)));
            })));
  }
  return ret;
}

static XmodelPostprocessorInputs unique_ptr_get(
    const std::vector<
        std::pair<std::string,
                  std::vector<std::pair<const xir::Op*,
                                        std::unique_ptr<vart::TensorBuffer>>>>>&
        graph_output_tensor_buffers) {
  auto ret = XmodelPostprocessorInputs();
  ret.inputs.resize(graph_output_tensor_buffers.size());
  int c = 0;
  for (auto& tb : graph_output_tensor_buffers) {
    auto& v = ret.inputs[c];
    v.name = tb.first;
    v.args.resize(tb.second.size());
    for (auto i = 0u; i < v.args.size(); ++i) {
      v.args[i].op = tb.second[i].first;
      v.args[i].tensor_buffer = tb.second[i].second.get();
    }
    c = c + 1;
  }
  return ret;
}

std::vector<vitis::ai::proto::DpuModelResult>
XmodelPostprocessorSingleBatch::process(
    const XmodelPostprocessorInputs& graph_output_tensor_buffers) {
  int batch = -1;
  for (auto& input : graph_output_tensor_buffers.inputs) {
    for (auto& arg : input.args) {
      if (batch == -1) {
        batch = arg.tensor_buffer->get_tensor()->get_shape()[0];
      } else {
        CHECK_EQ(batch, arg.tensor_buffer->get_tensor()->get_shape()[0])
            << "all input tensor buffer should have the same batch size."  //
            << "name: " << input.name << ";"                               //
            << "batch=" << batch << ";"                                    //
            << "tb=" << arg.tensor_buffer->to_string() << ";"              //
            ;
      }
    }
  }
  // clear coverity complain about below lines
  if (batch == -1) { 
    batch = 1;
  }
  auto ret = std::vector<vitis::ai::proto::DpuModelResult>((size_t)batch);
  for (auto i = 0; i < batch; ++i) {
    auto tbs = map_to_single_batch(graph_output_tensor_buffers, i, 1);
    auto tbs2 = unique_ptr_get(tbs);
    ret[i] = process_single_batch(tbs2);
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
