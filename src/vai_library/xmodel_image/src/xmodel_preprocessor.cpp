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
#include "vitis/ai/xmodel_preprocessor.hpp"

#include <dlfcn.h>

#include <cmath>

#include "vart/runner_helper.hpp"
#include "vitis/ai/collection_helper.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/image_util.hpp"

DEF_ENV_PARAM(DEBUG_XMODEL_PREPROCESSOR, "0")

namespace vitis {
namespace ai {

static std::string get_so_name(const xir::Graph* graph) {
  auto ret = std::string("libxmodel_preprocessor_common.so.3");
  if (graph->has_attr("xmodel_preprocessor")) {
    ret = graph->get_attr<std::string>("xmodel_preprocessor");
  } else {
    // LOG(INFO) << "graph attrs:" << xir::to_string(graph->get_attrs().get());
  }
  return ret;
}

// python binding does not support vector<float> , only support vector<double>
static std::vector<float> read_vector_double(const xir::Graph* graph,
                                             const std::string& name,
                                             size_t size, float default_value) {
  auto ret = std::vector<float>(size, default_value);
  if (graph->has_attr(name)) {
    ret = vitis::ai::vec_map(graph->get_attr<std::vector<double>>(name),
                             [](const double& x) { return (float)x; });
  }
  return ret;
}

XmodelPreprocessor::XmodelPreprocessor(const xir::Graph* graph,
                                       const xir::Tensor* tensor) {
  auto shape = tensor->get_shape();
  CHECK_EQ(shape.size(), 4u) << "only support xmodel with shape [NHWC]";
  CHECK_EQ(tensor->get_data_type().bit_width, 8) << "must be 8bits image";
  batch_ = (size_t)shape[0];
  height_ = (size_t)shape[1];
  width_ = (size_t)shape[2];
  depth_ = (size_t)shape[3];
  mean_ = read_vector_double(graph, "mean", depth_, 0.0f);
  scale_ = read_vector_double(graph, "scale", depth_, 1.0f);
  do_mean_scale_ = false;
  if (graph->has_attr("need_preprocess")) {
    do_mean_scale_ = graph->get_attr<bool>("need_preprocess");
  }
  is_rgb_input_ = false;
  if (graph->has_attr("is_rgb_input")) {
    is_rgb_input_ = graph->get_attr<bool>("is_rgb_input");
  }
}

size_t XmodelPreprocessor::get_batch() const { return batch_; }
size_t XmodelPreprocessor::get_width() const { return width_; }
size_t XmodelPreprocessor::get_height() const { return height_; }
size_t XmodelPreprocessor::get_depth() const { return depth_; }

static int get_fix_point(const xir::Tensor* tensor) {
  CHECK(tensor->has_attr("fix_point"))
      << "get tensor fix_point error! has no fix_point attr, tensor name is "
      << tensor->get_name();
  return tensor->template get_attr<int>("fix_point");
}

template <typename T>
static void copy_line_by_line(T* data, int rows, int cols, int channels,
                              int stride, const uint8_t* input) {
  for (int row = 0; row < rows; ++row) {
    memcpy(data + row * cols * channels, input + row * stride, cols * channels);
  }
}

void XmodelPreprocessor::set_input_image(const void* input_data,
                                         size_t batch_index,
                                         vart::TensorBuffer* tensor_buffer) {
  CHECK_LT(batch_index, get_batch());
  int fixpos = get_fix_point(tensor_buffer->get_tensor());
  float input_fixed_scale = std::exp2f(1.0f * (float)fixpos);
  vector<float> real_scale = vitis::ai::vec_map(
      scale_,
      [input_fixed_scale](const float& x) { return x * input_fixed_scale; });
  auto rows = get_height();
  auto cols = get_width();
  auto channels = get_depth();
  auto stride = cols * channels;
  auto image_size = get_width() * get_height() * get_depth();

  {
    uint64_t data;
    size_t size;
    auto idx = vart::get_index_zeros(tensor_buffer->get_tensor());
    idx[0] = (int)batch_index;
    std::tie(data, size) = tensor_buffer->data(idx);
    CHECK_GE(size, image_size)
        << "the tensor buffer must be accessible by host";
    LOG_IF(INFO, ENV_PARAM(DEBUG_XMODEL_PREPROCESSOR))
        << "copy input: size=" << size;
    if (do_mean_scale_) {
      if (is_rgb_input_) {
        NormalizeInputDataRGB((const uint8_t*)input_data, rows, cols, channels,
                              stride, mean_, real_scale, (int8_t*)data);
      } else {
        NormalizeInputData((const uint8_t*)input_data, rows, cols, channels,
                           stride, mean_, real_scale, (int8_t*)data);
      }
    } else {
      copy_line_by_line((uint8_t*)data, rows, cols, channels, stride,
                        (const uint8_t*)input_data);
    }
  }
}

std::unique_ptr<XmodelPreprocessor> XmodelPreprocessor::create(
    const xir::Graph* graph, const xir::Tensor* tensor) {
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
  typedef std::unique_ptr<XmodelPreprocessor> (*fm_type)(
      const xir::Graph* graph, const xir::Tensor* tensor);
  auto factory_method_p = (fm_type)dlsym(handle, "create_xmodel_preprocessor");
  if (factory_method_p == nullptr) {
    LOG(FATAL) << "not a valid plugin, cannot find symbol "
                  "\"create_xmodel_preprocessor\": name="
               << so_name;
  }
  auto ret = (*factory_method_p)(graph, tensor);
  CHECK(ret != nullptr) << "plugin return a nullptr."
                           ";name="
                        << so_name;
  return ret;
}

}  // namespace ai
}  // namespace vitis
