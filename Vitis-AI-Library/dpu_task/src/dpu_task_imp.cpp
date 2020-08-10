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
#include "dpu_task_imp.hpp"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vart/tensor_buffer.hpp>  // for vitis
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/time_measure.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/tensor/tensor.hpp>  // for xir

using namespace vitis::ai;
using namespace std;

DEF_ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME, "0");
DEF_ENV_PARAM(DEBUG_DPBASE, "0");

//# Disable for DPUV1, as DPUV1 dont have xmodel
#ifndef ENABLE_DPUCADX8G_RUNNER
static std::string get_full_filename(const std::string& filename) {
  if (filename[0] == '/') {
    return filename;
  }
  std::string current_p(getcwd(NULL, 0));
  return current_p + "/" + filename;
}

static std::string get_parent_path(const std::string& path) {
  return path.substr(0, path.find_last_of("/"));
}

static std::shared_ptr<GraphHolder> create_graph_holder(
    const std::string& filename) {
  auto ret = vitis::ai::WeakStore<std::string, GraphHolder>::create(filename,
                                                                    filename);
  auto graph = ret->get_graph();
  auto full_filename = get_full_filename(filename);
  const_cast<xir::Graph*>(graph)->set_attr<string>("filename", full_filename);
  auto dirname = get_parent_path(full_filename);
  const_cast<xir::Graph*>(graph)->set_attr<string>("dirname", dirname);

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "filename " << filename << " "            //
      << "full_filename " << full_filename << " "  //
      << "dirname " << dirname << " "              //
      << "ret.get() " << (void*)ret.get() << " "   //
      << std::endl;
  return ret;
}

static int get_batch_size_of_runner(vart::Runner* r) {
  CHECK(!r->get_input_tensors().empty());
  return r->get_input_tensors()[0]->get_dim_size(0);
}
static std::vector<std::unique_ptr<vart::Runner>> create_runners(
    std::shared_ptr<GraphHolder> graph_holder) {
  auto subgraphs = (graph_holder.get())->get_subgraphs();
  CHECK_GT(subgraphs.size(), 0);
  auto runners = std::vector<std::unique_ptr<vart::Runner>>();
  runners.reserve(subgraphs.size());
  auto batch_size = 0;
  for (auto subgraph : subgraphs) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "create runner for " << subgraph->get_name();
    if (batch_size == 0) {
      // create the very first runner
      auto r = vart::Runner::create_runner(subgraph, "run");
      batch_size = get_batch_size_of_runner(r.get());
      runners.emplace_back(std::move(r));
    } else {
      // the following runners must have the batch size as same as the
      // first runner's.
      auto is_same = false;
      int num_of_tries = 0;
      do {
        auto r = vart::Runner::create_runner(subgraph, "run");
        is_same = get_batch_size_of_runner(r.get()) == batch_size;
        if (is_same) {
          runners.emplace_back(std::move(r));
        }
        num_of_tries++;
      } while (!is_same && num_of_tries < 100);
      CHECK_LT(num_of_tries, 100) << "too many tries...";
    }
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "create runner for " << subgraph->get_name()
        << " done. batch_size=" << batch_size;
  }
  CHECK_EQ(runners.size(), subgraphs.size());
  return runners;
}

DpuTaskImp::DpuTaskImp(const std::string& model_name)
    : model_name_{model_name},
      graph_holder_{create_graph_holder(model_name)},
      runners_{create_runners(graph_holder_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}

#else
//# Enable for DPUV1 Runner and it need meta json file
static vector<string> fine_module_search_path() {
  auto ret = vector<string>{};
  ret.push_back(".");
  ret.push_back("/usr/share/vitis_ai_library/models");
  ret.push_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

//# Utility functions for DPUV1
static size_t filesize(const string& filename) {
  size_t ret = 0;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = statbuf.st_size;
  }
  return ret;
}

static string find_module_dir_name(const string& name) {
  if (filesize(name + "/" + "meta.json") > 0) {
    return name;
  }
  auto ret = std::string();
  for (const auto& p : fine_module_search_path()) {
    ret = p + "/" + name;
    const auto fullname = ret + "/" + "meta.json";
    if (filesize(fullname) > 0) {
      return ret;
    }
  }
  stringstream str;
  str << "cannot find kernel <" << name << "> after checking following files:";
  for (const auto& p : fine_module_search_path()) {
    ret = p + "/" + name;
    const auto fullname = ret + "/" + "meta.json";
    str << "\n\t" << fullname;
  }
  LOG(FATAL) << str.str();
  return string{""};
}

//# call create_runner with directory name for DPUV1
DpuTaskImp::DpuTaskImp(const std::string& model_name)
    : model_name_{model_name},
      dirname_{find_module_dir_name(model_name)},
      runners_{vart::Runner::create_runner(dirname_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}
#endif

DpuTaskImp::~DpuTaskImp() {  //
}

void DpuTaskImp::run(size_t idx) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "running dpu task " << model_name_ << "[" << idx << "]";
  auto inputs =
      dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())->get_inputs();
  auto outputs = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())
                     ->get_outputs();
  std::pair<uint32_t, int> v;

  if (ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME)) {
    auto start = std::chrono::steady_clock::now();
    v = runners_[idx]->execute_async(inputs, outputs);
    auto end = std::chrono::steady_clock::now();
    auto time =
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count();
    TimeMeasure::getThreadLocalForDpu().add(int(time));
  } else {
    v = runners_[idx]->execute_async(inputs, outputs);
  }
  runners_[idx]->wait((int)v.first, -1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "dpu task " << model_name_ << "[" << idx << "]";
}

void DpuTaskImp::setMeanScaleBGR(const std::vector<float>& mean,
                                 const std::vector<float>& scale) {
  mean_ = mean;
  scale_ = scale;
  do_mean_scale_ = true;
}

void DpuTaskImp::setImageBGR(const cv::Mat& img) {
  setImageBGR(img.data, img.step);
}

void DpuTaskImp::setImageRGB(const cv::Mat& img) {
  setImageRGB(img.data, img.step);
}

//# Templatized the input data type
template <typename T>
static void copy_line_by_line(T* data, int rows, int cols, int channels,
                              int stride, const uint8_t* input) {
  for (int row = 0; row < rows; ++row) {
    memcpy(data + row * cols * channels, input + row * stride, cols * channels);
  }
}

void DpuTaskImp::setImageBGR(const uint8_t* input, int stride) {
  set_num_of_inputs(1u);
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  // assuming the first input
  const auto& layer_data = inputs[0];
  float input_fixed_scale = tensor_scale(inputs[0]);
  vector<float> real_scale{scale_[0] * input_fixed_scale,
                           scale_[1] * input_fixed_scale,
                           scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;

//# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
  auto data = (float*)layer_data.get_data(0);
#else
  auto data = (int8_t*)layer_data.get_data(0);
#endif

  if (do_mean_scale_) {
    NormalizeInputData(input, rows, cols, channels, stride, mean_, real_scale,
                       data);
  } else {
    copy_line_by_line(data, rows, cols, channels, stride, input);
  }
}

void DpuTaskImp::setImageRGB(const uint8_t* input, int stride) {
  set_num_of_inputs(1u);
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  const auto& layer_data = inputs[0];
  float input_fixed_scale = tensor_scale(inputs[0]);
  vector<float> real_scale{scale_[0] * input_fixed_scale,
                           scale_[1] * input_fixed_scale,
                           scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;

  //# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
  auto data = (float*)layer_data.get_data(0);
#else
  auto data = (int8_t*)layer_data.get_data(0);
#endif
  if (ENV_PARAM(DEBUG_DPBASE)) {
    LOG(INFO) << "rows " << rows << " "  //
              << "cols " << cols;
    LOG(INFO) << "write before_setinput_image.bmp from " << (void*)input;
    auto img = cv::Mat((int)rows, (int)cols, CV_8UC3, (void*)input);
    cv::imwrite(std::string("before_setinput_image.bmp"), img);
  }
  if (do_mean_scale_) {
    NormalizeInputDataRGB(input, rows, cols, channels, stride, mean_,
                          real_scale, data);
  } else {
    assert(false && "not implement");
  }
  if (ENV_PARAM(DEBUG_DPBASE)) {
    LOG(INFO) << "write after_setinput_image.bmp from " << (void*)data;
    auto img = cv::Mat((int)rows, (int)cols, CV_8UC3, (void*)data);
    cv::imwrite(std::string("after_setinput_image.bmp"), img);
  }
}

void DpuTaskImp::setImageBGR(const std::vector<cv::Mat>& imgs) {
  set_num_of_inputs(imgs.size());
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  // assuming the first input
  const auto& layer_data = inputs[0];
  CHECK_EQ(imgs.size(), layer_data.batch);
  CHECK_LE(imgs.size(), get_input_batch(0, 0));
  float input_fixed_scale = tensor_scale(inputs[0]);
  vector<float> real_scale{scale_[0] * input_fixed_scale,
                           scale_[1] * input_fixed_scale,
                           scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;
  // auto data = (int8_t*)layer_data.data;

  for (auto i = 0; i < (signed)imgs.size(); i++) {
//# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
    auto data = (float*)layer_data.get_data(i);
#else
    auto data = (int8_t*)layer_data.get_data(i);
#endif

    if (do_mean_scale_) {
      NormalizeInputData(imgs[i].data, rows, cols, channels, imgs[i].step,
                         mean_, real_scale, data);
    } else {
      copy_line_by_line(data, rows, cols, channels, imgs[i].step, imgs[i].data);
    }
    // data += rows * cols * channels;
  }
}

void DpuTaskImp::setImageRGB(const std::vector<cv::Mat>& imgs) {
  set_num_of_inputs(imgs.size());
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  const auto& layer_data = inputs[0];
  // set num of inputs before getInputTensor, so that the size should
  // be same.
  CHECK_EQ(imgs.size(), layer_data.batch);
  // the num of inputs must be less than the batch size which is the
  // hardware limitation.
  CHECK_LE(imgs.size(), get_input_batch(0, 0));
  float input_fixed_scale = tensor_scale(inputs[0]);
  vector<float> real_scale{scale_[0] * input_fixed_scale,
                           scale_[1] * input_fixed_scale,
                           scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;

  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))  //
      << "rows " << rows << " "          //
      << "cols " << cols;

  for (auto i = 0; i < (signed)imgs.size(); i++) {
//# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
    auto data = (float*)layer_data.get_data(i);
#else
    auto data = (int8_t*)layer_data.get_data(i);
#endif
    if (do_mean_scale_) {
      NormalizeInputDataRGB(imgs[i].data, rows, cols, channels, imgs[i].step,
                            mean_, real_scale, data);
    } else {
      assert(false && "not implement");
    }
    //    data += rows * cols * channels;
  }
}

std::vector<float> DpuTaskImp::getMean() { return mean_; }

std::vector<float> DpuTaskImp::getScale() { return scale_; }

//# Add tensor format argument (NHWC / NCHW)
static vitis::ai::library::InputTensor convert_tensor_buffer_to_input_tensor(
    vart::TensorBuffer* tb, float scale, int num_of_input,
    vart::Runner::TensorFormat fmt) {
  auto ret = vitis::ai::library::InputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_dim_num();
  ret.size =
      tensor->get_element_num() * std::ceil(tensor->get_bit_width() / 8.f);
  ret.batch = dim_num <= 0 ? 1 : tensor->get_dim_size(0);
  if (num_of_input != -1) {
    ret.batch = (unsigned)num_of_input;
  }
  //# Store the params as per format
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    ret.height = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
    ret.width = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
    ret.channel = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
    ret.fixpos = (int8_t)log2f(scale);
    ret.dtype = library::DT_INT8;
  } else {
#ifdef ENABLE_DPUCADX8G_RUNNER
    //# DPUV1 has datatype float
    ret.size =
        tensor->get_element_num() * std::ceil(tensor->get_bit_width() / 32.f);
#endif
    ret.height = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
    ret.width = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
    ret.channel = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
    ret.fixpos = (int8_t)log2f(scale);
    ret.dtype = library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto dims = tensor->get_dims();
  auto index = dims;
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    auto data = tb->data({batch_idx, 0, 0, 0});
    ret.get_data(batch_idx) = (void*)data.first;
    size = data.second;
    //    std::tie(ret.get_data(batch_idx), size) = tb->data({batch_idx, 0,
    //    0, 0});
    CHECK_GE(size, ret.height * ret.width * ret.channel);
  }

  return ret;
}

//# Add tensor format argument (NHWC / NCHW)
static vitis::ai::library::OutputTensor convert_tensor_buffer_to_output_tensor(
    vart::TensorBuffer* tb, float scale,
    /* num of real input images, it might be less than or equal to batch size */
    int num_of_input, vart::Runner::TensorFormat fmt) {
  auto ret = vitis::ai::library::OutputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_dim_num();
  ret.size =
      tensor->get_element_num() * std::ceil(tensor->get_bit_width() / 8.f);
  ret.batch = dim_num <= 0 ? 1 : tensor->get_dim_size(0);
  if (num_of_input != -1) {
    CHECK_LE((unsigned)num_of_input, ret.batch) << "logical error";
    ret.batch = (unsigned)num_of_input;
  }
  //# Store the params as per format
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    if (dim_num == 2) {
      ret.height = 1;
      ret.width = 1;
      ret.channel = tensor->get_dim_size(1);
    } else {
      ret.height = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
      ret.width = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
      ret.channel = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
    }
    ret.fixpos = -(int8_t)log2f(scale);
    ret.dtype = library::DT_INT8;
  } else {
#ifdef ENABLE_DPUCADX8G_RUNNER
    //# DPUV1 has datatype float
    ret.size =
        tensor->get_element_num() * std::ceil(tensor->get_bit_width() / 32.f);
#endif
    ret.height = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
    ret.width = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
    ret.channel = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
    ret.fixpos = -(int8_t)log2f(scale);
    ret.dtype = library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto dims = tensor->get_dims();
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    auto idx = std::vector<int32_t>(dims.size());
    idx[0] = batch_idx;
    auto data = tb->data(idx);
    ret.get_data(batch_idx) = (void*)data.first;
    size = data.second;
    CHECK_GE(size, ret.height * ret.width * ret.channel);
  }

  return ret;
}

std::vector<vitis::ai::library::InputTensor> DpuTaskImp::getInputTensor(
    size_t idx) {
  auto dpu_runner_ext =
      dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get());
  auto inputs = dpu_runner_ext->get_inputs();
  //# Get the current format
  auto fmt = runners_[idx]->get_tensor_format();
  auto scales = dpu_runner_ext->get_input_scale();
  auto ret = std::vector<vitis::ai::library::InputTensor>{};
  ret.reserve(inputs.size());
  int c = 0;
  for (auto& t : inputs) {
    ret.emplace_back(convert_tensor_buffer_to_input_tensor(
        t, scales[c], num_of_inputs_, fmt));
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "input tensor[" << c << "]: " << ret.back();
    c++;
  }
  return ret;
}

std::vector<vitis::ai::library::OutputTensor> DpuTaskImp::getOutputTensor(
    size_t idx) {
  auto outputs = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())
                     ->get_outputs();
  //# Get the current format
  auto fmt = runners_[idx]->get_tensor_format();
  auto scales = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())
                    ->get_output_scale();

  auto ret = std::vector<vitis::ai::library::OutputTensor>{};
  ret.reserve(outputs.size());
  int c = 0;
  for (auto& t : outputs) {
    ret.emplace_back(convert_tensor_buffer_to_output_tensor(
        t, scales[c], num_of_inputs_, fmt));
    LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
        << "output tensor[" << c << "]: " << ret.back();
    c++;
  }
  return ret;
}

size_t DpuTaskImp::get_input_batch(size_t kernel_idx, size_t node_idx) const {
  return dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[kernel_idx].get())
      ->get_inputs()[node_idx]
      ->get_tensor()
      ->get_dim_size(0);
}

size_t DpuTaskImp::get_num_of_kernels() const {  //
  return runners_.size();
}

const xir::Graph* DpuTaskImp::get_graph() const {
  return graph_holder_->get_graph();
}
void DpuTaskImp::set_num_of_inputs(size_t n) {
  // TODO it is too much to call clear_num_of_inputs
  // CHECK_EQ(num_of_inputs_, -1)
  //     << "LOGICAL ERROR. you cannot set num input twices";
  CHECK_LT(n, 100) << "with current DPU design, it is not possible for very "
                      "large batch size.";
  num_of_inputs_ = (int)n;
}

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
