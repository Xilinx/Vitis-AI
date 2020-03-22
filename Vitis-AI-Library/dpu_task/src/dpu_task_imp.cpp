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
#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <cassert>
#include <chrono>
#include <iostream>

using namespace std;
#include <vart/dpu/tensor.hpp>         // for vitis
#include <vart/dpu/tensor_buffer.hpp>  // for vitis
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/weak.hpp>
#include "./dpu_task_imp.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "vitis/ai/time_measure.hpp"
DEF_ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME, "0");
DEF_ENV_PARAM(DEBUG_DPBASE, "0");
using namespace vitis::ai;
using namespace std;
static vector<string> fine_module_search_path() {
  auto ret = vector<string>{};
  ret.push_back(".");
  ret.push_back("/usr/share/vitis_ai_library/models");
  ret.push_back("/usr/share/vitis_ai_library/.models");
  return ret;
}
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

DpuTaskImp::DpuTaskImp(const std::string& model_name)
    : model_name_{model_name},
      dirname_{find_module_dir_name(model_name)},
      runners_{vitis::ai::DpuRunner::create_dpu_runner(dirname_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}

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
static void copy_line_by_line(int8_t* data, int rows, int cols, int channels,
                              int stride, const uint8_t* input) {
  for (int row = 0; row < rows; ++row) {
    memcpy(data + row * cols * channels, input + row * stride, cols * channels);
  }
}

void DpuTaskImp::setImageBGR(const uint8_t* input, int stride) {
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
  auto data = (int8_t*)layer_data.get_data(0);

  if (do_mean_scale_) {
    NormalizeInputData(input, rows, cols, channels, stride, mean_, real_scale,
                       data);
  } else {
    copy_line_by_line(data, rows, cols, channels, stride, input);
  }
}

void DpuTaskImp::setImageRGB(const uint8_t* input, int stride) {
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
  auto data = (int8_t*)layer_data.get_data(0);
  LOG_IF(INFO, false) << "rows " << rows << " "  //
                      << "cols " << cols << " "  //
      ;
  if (ENV_PARAM(DEBUG_DPBASE)) {
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
  // setImageBGR(img.data, img.step);
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  // assuming the first input
  const auto& layer_data = inputs[0];
  // CHECK_EQ(inputs.size(), layer_data.batch);
  float input_fixed_scale = tensor_scale(inputs[0]);
  vector<float> real_scale{scale_[0] * input_fixed_scale,
                           scale_[1] * input_fixed_scale,
                           scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;
  // auto data = (int8_t*)layer_data.data;

  for (auto i = 0; i < (signed)imgs.size(); i++) {
    auto data = (int8_t*)layer_data.get_data(i);
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
  // setImageRGB(img.data, img.step);
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
  // auto data = (int8_t*)layer_data.data;
  LOG_IF(INFO, false) << "rows " << rows << " "  //
                      << "cols " << cols << " "  //
      ;

  for (auto i = 0; i < (signed)imgs.size(); i++) {
    auto data = (int8_t*)layer_data.get_data(i);
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

static vitis::ai::library::InputTensor convert_tensor_buffer_to_input_tensor(
    vitis::ai::TensorBuffer* tb, float scale) {
  auto ret = vitis::ai::library::InputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_dim_num();
  ret.size =
      tensor->get_element_num() * vitis::ai::size_of(tensor->get_data_type());
  ret.batch = dim_num <= 0 ? 1 : tensor->get_dim_size(0);
  ret.height = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
  ret.width = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
  ret.channel = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
  ret.fixpos = (int8_t)log2f(scale);
  ret.dtype = library::DT_INT8;
  ret.name = tensor->get_name();
  auto dims = tensor->get_dims();
  auto index = dims;
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    std::tie(ret.get_data(batch_idx), size) = tb->data({batch_idx, 0, 0, 0});
    CHECK_GE(size, ret.height * ret.width * ret.channel);
  }

  return ret;
}

static vitis::ai::library::OutputTensor convert_tensor_buffer_to_output_tensor(
    vitis::ai::TensorBuffer* tb, float scale) {
  auto ret = vitis::ai::library::OutputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_dim_num();
  ret.size =
      tensor->get_element_num() * vitis::ai::size_of(tensor->get_data_type());
  ret.batch = dim_num <= 0 ? 1 : tensor->get_dim_size(0);
  ret.height = dim_num <= 1 ? 1 : tensor->get_dim_size(1);
  ret.width = dim_num <= 2 ? 1 : tensor->get_dim_size(2);
  ret.channel = dim_num <= 3 ? 1 : tensor->get_dim_size(3);
  ret.fixpos = -(int8_t)log2f(scale);
  ret.dtype = library::DT_INT8;
  ret.name = tensor->get_name();
  auto dims = tensor->get_dims();
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    std::tie(ret.get_data(batch_idx), size) = tb->data({batch_idx, 0, 0, 0});
    CHECK_GE(size, ret.height * ret.width * ret.channel);
  }

  return ret;
}

std::vector<vitis::ai::library::InputTensor> DpuTaskImp::getInputTensor(
    size_t idx) {
  auto inputs =
      dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())->get_inputs();
  auto scales = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())
                    ->get_input_scale();
  auto ret = std::vector<vitis::ai::library::InputTensor>{};
  ret.reserve(inputs.size());
  int c = 0;
  for (auto& t : inputs) {
    ret.emplace_back(convert_tensor_buffer_to_input_tensor(t, scales[c]));
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
  auto scales = dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[idx].get())
                    ->get_output_scale();

  auto ret = std::vector<vitis::ai::library::OutputTensor>{};
  ret.reserve(outputs.size());
  int c = 0;
  for (auto& t : outputs) {
    ret.emplace_back(convert_tensor_buffer_to_output_tensor(t, scales[c]));
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

size_t DpuTaskImp::get_num_of_kernels() const { return runners_.size(); }

const vitis::ai::DpuMeta& DpuTaskImp::get_dpu_meta_info() const {
  return dynamic_cast<vart::dpu::DpuRunnerExt*>(runners_[0].get())->get_meta();
}

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
