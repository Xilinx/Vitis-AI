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
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied. See the License for the specific language governing
 * permissions and limitations under the License.
 */
#include "dpu_task_imp.hpp"

#include <glog/logging.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <UniLog/UniLog.hpp>
#include <cassert>
#include <chrono>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vart/assistant/batch_tensor_buffer.hpp>
#include <vart/assistant/xrt_bo_tensor_buffer.hpp>
#include <vart/experimental/runner_helper.hpp>
#include <vart/tensor_buffer.hpp>  // for vitis
#include <vart/zero_copy_helper.hpp>
#include <vitis/ai/collection_helper.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/graph_runner.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/time_measure.hpp>
#include <vitis/ai/weak.hpp>
#include <xir/tensor/tensor.hpp>  // for xir
using namespace vitis::ai;
using namespace std;

DEF_ENV_PARAM(DEEPHI_DPU_CONSUMING_TIME, "0");
DEF_ENV_PARAM(DEBUG_DPBASE, "0");

int GLOBAL_ENABLE_C_SOFTMAX = 0;

//# Disable for DPUV1, as DPUV1 dont have xmodel
#ifndef ENABLE_DPUCADX8G_RUNNER
static std::string get_full_filename(const std::string& filename) {
  if (filename[0] == '/') {
    return filename;
  }
  auto cwd = getcwd(NULL, 0);
  std::string current_p(cwd);
  free(cwd);
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
  // CHECK(!r->get_input_tensors().empty());
  UNI_LOG_CHECK(!r->get_input_tensors().empty(), VAILIB_DPU_TASK_TENSORS_EMPTY);
  return r->get_input_tensors()[0]->get_shape().at(0);
}

//# Enable software softmax for DPU's which uses rt-engine
static void enable_sw_softmax(const xir::Subgraph* subgraph) {
  auto libs = subgraph->get_attr<std::map<std::string, std::string>>("runner");
  auto iter_lib = libs.find("run");
  auto lib_name_ = iter_lib->second;

  if (lib_name_.compare("librt-engine.so") == 0) GLOBAL_ENABLE_C_SOFTMAX = 2;

  return;
}

static std::vector<std::unique_ptr<vart::Runner>> create_runners_with_attrs(
    std::shared_ptr<GraphHolder> graph_holder, xir::Attrs* attrs) {
  auto subgraphs = (graph_holder.get())->get_subgraphs();
  // CHECK_GT(subgraphs.size(), 0);
  UNI_LOG_CHECK(subgraphs.size() > 0, VAILIB_DPU_TASK_SUBGRAPHS_EMPTY);
  enable_sw_softmax(subgraphs[0]);
  auto runners = std::vector<std::unique_ptr<vart::Runner>>();
  runners.reserve(subgraphs.size());
  auto batch_size = 0;
  auto use_graph_runner = attrs->has_attr("use_graph_runner") &&
                          attrs->get_attr<bool>("use_graph_runner");
  if (use_graph_runner) {
    runners.push_back(vitis::ai::GraphRunner::create_graph_runner(
        graph_holder->get_graph(), attrs));
  } else {
    for (auto subgraph : subgraphs) {
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
          << "create runner for " << subgraph->get_name();
      if (batch_size == 0) {
        // create the very first runner
        auto r = vart::Runner::create_runner_with_attrs(subgraph, attrs);
        batch_size = get_batch_size_of_runner(r.get());
        runners.emplace_back(std::move(r));
      } else {
        // the following runners must have the batch size as same as the
        // first runner's.
        auto r = vart::Runner::create_runner_with_attrs(subgraph, attrs);
        CHECK_EQ(get_batch_size_of_runner(r.get()), batch_size)
            << "batch size not same as first runner";
        runners.emplace_back(std::move(r));
      }
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
          << "create runner for " << subgraph->get_name()
          << " done. batch_size=" << batch_size;
    }
    CHECK_EQ(runners.size(), subgraphs.size());
  }
  return runners;
}

static std::vector<std::vector<vitis::ai::library::InputTensor>>
get_all_input_tensors(std::vector<std::unique_ptr<vart::Runner>>& runners);

static std::vector<std::vector<vitis::ai::library::OutputTensor>>
get_all_output_tensors(std::vector<std::unique_ptr<vart::Runner>>& runners);

DpuTaskImp::DpuTaskImp(const std::string& model_name)
    : model_name_{model_name},
      graph_holder_{create_graph_holder(model_name)},
      default_attrs_{xir::Attrs::create()},
      runners_{create_runners_with_attrs(graph_holder_, default_attrs_.get())},
      all_input_tensors_{get_all_input_tensors(runners_)},
      all_output_tensors_{get_all_output_tensors(runners_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}

DpuTaskImp::DpuTaskImp(const std::string& model_name, xir::Attrs* attrs)
    : model_name_{model_name},
      graph_holder_{create_graph_holder(model_name)},
      default_attrs_{xir::Attrs::create()},
      runners_{create_runners_with_attrs(graph_holder_, attrs)},
      all_input_tensors_{get_all_input_tensors(runners_)},
      all_output_tensors_{get_all_output_tensors(runners_)},
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
  // LOG(FATAL) << str.str();
  UNI_LOG_FATAL(VAILIB_DPU_TASK_NOT_FIND) << str.str();
  return string{""};
}

//# call create_runner with directory name for DPUV1
DpuTaskImp::DpuTaskImp(const std::string& model_name)
    : model_name_{model_name},
      dirname_{find_module_dir_name(model_name)},
      runners_{vart::Runner::create_runner(dirname_)},
      all_input_tensors_{get_all_input_tensors(runners_)},
      all_output_tensors_{get_all_output_tensors(runners_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}

DpuTaskImp::DpuTaskImp(const std::string& model_name, xir::Attrs* attrs)
    : model_name_{model_name},
      dirname_{find_module_dir_name(model_name)},
      runners_{vart::Runner::create_runner(dirname_)},
      all_input_tensors_{get_all_input_tensors(runners_)},
      all_output_tensors_{get_all_output_tensors(runners_)},
      mean_{std::vector<float>(3, 0.f)},   //
      scale_{std::vector<float>(3, 1.f)},  //
      do_mean_scale_{false} {}

#endif

DpuTaskImp::~DpuTaskImp() {  //
}

void DpuTaskImp::run(size_t idx) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "running dpu task " << model_name_ << "[" << idx << "]";
  auto vin = dynamic_cast<vart::RunnerExt*>(runners_[idx].get());
  auto vout = dynamic_cast<vart::RunnerExt*>(runners_[idx].get());
  if (vin && vout ) {
     auto inputs = vin->get_inputs();
     auto outputs = vout->get_outputs();
     std::pair<uint32_t, int> v;
     for (auto input : inputs) {
       input->sync_for_write(0, input->get_tensor()->get_data_size() /
                                    input->get_tensor()->get_shape()[0]);
     }
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
     for (auto output : outputs) {
       output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                    output->get_tensor()->get_shape()[0]);
     }
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "dpu task " << model_name_ << "[" << idx << "]";
}

void DpuTaskImp::run_with_xrt_bo(const std::vector<vart::xrt_bo_t>& input_bos) {
  // TODO: refactor them to prepresessing and post processing.
  auto idx = 0;
  LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
      << "running dpu task " << model_name_ << "[" << idx << "]";

  auto input_tensors = runners_[idx].get()->get_input_tensors();
  // we assumethe order of input_bos is as same as the input tensors.
  CHECK_EQ(input_tensors.size(), 1u);
  auto the_input_tensor = input_tensors.front();
  auto the_input_shape = the_input_tensor->get_shape();
  CHECK(!the_input_shape.empty());
  auto the_input_batch = the_input_shape[0];
  CHECK_LE(input_bos.size(), (size_t)the_input_batch);
  auto the_xrt_bo_tensor_buffers = vitis::ai::vec_map(
      input_bos, [the_input_tensor](const vart::xrt_bo_t& xrt_bo) {
        return vart::assistant::XrtBoTensorBuffer::create(xrt_bo,
                                                          the_input_tensor);
      });
  auto the_input_tensor_buffer = vart::assistant::BatchTensorBuffer::create(
      vitis::ai::vector_unique_ptr_get(the_xrt_bo_tensor_buffers));
  auto inputs = std::vector<vart::TensorBuffer*>{the_input_tensor_buffer.get()};
  auto outputs =
      dynamic_cast<vart::RunnerExt*>(runners_[idx].get())->get_outputs();
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
  for (auto output : outputs) {
    output->sync_for_read(0, output->get_tensor()->get_data_size() /
                                 output->get_tensor()->get_shape()[0]);
  }
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

void DpuTaskImp::setImageRGB(const cv::Mat& img, size_t ind) {
  setImageRGB(img.data, img.step, ind);
}

//# Templatized the input data type
template <typename T>
static void copy_line_by_line(T* data, int rows, int cols, int channels,
                              int stride, const uint8_t* input) {
  for (int row = 0; row < rows; ++row) {
    memcpy(data + row * cols * channels, input + row * stride, cols * channels);
  }
}

template <typename T>
static void copy_line_by_line(T* data, int rows, int cols, int channels,
                              int stride, const int8_t* input) {
  for (int row = 0; row < rows; ++row) {
    memcpy(data + row * cols * channels, input + row * stride, cols * channels);
  }
}

void DpuTaskImp::setInputDataArray(const std::vector<int8_t> input,
                                   size_t ind) {
  set_num_of_inputs(1u);
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  // assuming the first input
  const auto& layer_data = inputs[ind];
  float input_fixed_scale = tensor_scale(inputs[ind]);
  // vector<float> real_scale{scale_[0] * input_fixed_scale,
  //                         scale_[1] * input_fixed_scale,
  //                         scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;
  auto stride = cols * channels;
  vector<float> real_scale(channels);
  for (size_t c = 0; c < channels; c++) {
    real_scale[c] = scale_[c] * input_fixed_scale;
  }
//# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
  auto data = (float*)layer_data.get_data(0);
#else
  auto data = (int8_t*)layer_data.get_data(0);
#endif

  copy_line_by_line(data, rows, cols, channels, stride, input.data());
}

void DpuTaskImp::setInputDataArray(const std::vector<std::vector<int8_t>> input,
                                   size_t ind) {
  set_num_of_inputs(input.size());
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  // assuming the first input
  const auto& layer_data = inputs[ind];
  CHECK_EQ(input.size(), layer_data.batch);
  CHECK_LE(input.size(), get_input_batch(0, 0));
  float input_fixed_scale = tensor_scale(inputs[ind]);
  // vector<float> real_scale{scale_[0] * input_fixed_scale,
  //                         scale_[1] * input_fixed_scale,
  //                         scale_[2] * input_fixed_scale};
  auto rows = layer_data.height;
  auto cols = layer_data.width;
  auto channels = layer_data.channel;
  auto stride = cols * channels;
  vector<float> real_scale(channels);
  for (size_t c = 0; c < channels; c++) {
    real_scale[c] = scale_[c] * input_fixed_scale;
  }
  for (auto i = 0; i < (signed)input.size(); i++) {
//# For DPUV1 the datatype is float
#ifdef ENABLE_DPUCADX8G_RUNNER
    auto data = (float*)layer_data.get_data(i);
#else
    auto data = (int8_t*)layer_data.get_data(i);
#endif

    copy_line_by_line(data, rows, cols, channels, stride, input[i].data());
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

  if (ENV_PARAM(DEBUG_DPBASE) >= 5) {
    LOG(INFO) << "rows " << rows << " "  //
              << "cols " << cols;
    LOG(INFO) << "write before_setinput_image.bmp from " << (void*)input;
    auto img = cv::Mat((int)rows, (int)cols, CV_8UC3, (void*)input);
    cv::imwrite(std::string("before_setinput_image.bmp"), img);
  }

  if (do_mean_scale_) {
    NormalizeInputData(input, rows, cols, channels, stride, mean_, real_scale,
                       data);
  } else {
    copy_line_by_line(data, rows, cols, channels, stride, input);
  }

  if (ENV_PARAM(DEBUG_DPBASE) >= 5) {
    LOG(INFO) << "write after_setinput_image.bmp from " << (void*)data;
    auto img = cv::Mat((int)rows, (int)cols, CV_8UC3, (void*)data);
    cv::imwrite(std::string("after_setinput_image.bmp"), img);
  }
}

void DpuTaskImp::setImageRGB(const uint8_t* input, int stride, size_t ind) {
  set_num_of_inputs(1u);
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  const auto& layer_data = inputs[ind];
  float input_fixed_scale = tensor_scale(inputs[ind]);
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
  if (ENV_PARAM(DEBUG_DPBASE) >= 5) {
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
    copy_line_by_line(data, rows, cols, channels, stride, input);
  }
  if (ENV_PARAM(DEBUG_DPBASE) >= 5) {
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

void DpuTaskImp::setImageRGB(const std::vector<cv::Mat>& imgs, size_t ind) {
  set_num_of_inputs(imgs.size());
  auto inputs = getInputTensor(0u);
  CHECK_GT(inputs.size(), 0u);
  const auto& layer_data = inputs[ind];
  // set num of inputs before getInputTensor, so that the size should
  // be same.
  CHECK_EQ(imgs.size(), layer_data.batch);
  // the num of inputs must be less than the batch size which is the
  // hardware limitation.
  CHECK_LE(imgs.size(), get_input_batch(0, 0));
  float input_fixed_scale = tensor_scale(inputs[ind]);
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
      copy_line_by_line(data, rows, cols, channels, imgs[i].step, imgs[i].data);
    }
    //    data += rows * cols * channels;
  }
}

std::vector<float> DpuTaskImp::getMean() { return mean_; }

std::vector<float> DpuTaskImp::getScale() { return scale_; }

//# Add tensor format argument (NHWC / NCHW)
static vitis::ai::library::InputTensor convert_tensor_buffer_to_input_tensor(
    vart::TensorBuffer* tb, vart::Runner::TensorFormat fmt) {
  float scale = 1.0f;
  auto ret = vitis::ai::library::InputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_shape().size();
  auto batch = dim_num <= 0 ? 1 : tensor->get_shape().at(0);
  ret.batch = batch;
  ret.size = tensor->get_element_num() *
             std::ceil(tensor->get_data_type().bit_width / 8.f);
  //# Store the params as per format
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    ret.height = dim_num <= 1 ? 1 : tensor->get_shape().at(1);
    ret.width = dim_num <= 2 ? 1 : tensor->get_shape().at(2);
    ret.channel = dim_num <= 3 ? 1 : tensor->get_shape().at(3);
    if (tensor->get_data_type().type == xir::DataType::XINT) {
      ret.fixpos = (int8_t)log2f(scale);
      ret.dtype = library::DT_INT8;
      ret.fixpos = tensor->template get_attr<int>("fix_point");
    } else if (tensor->get_data_type().type == xir::DataType::FLOAT) {
      ret.fixpos = 0;
      ret.dtype = library::DT_FLOAT;
    } else {
      // LOG(FATAL) << "unsupported";
      UNI_LOG_FATAL(VAILIB_DPU_TASK_NOT_SUPPORT) << "unsupported";
    }
  } else {
#ifdef ENABLE_DPUCADX8G_RUNNER
    //# DPUV1 has datatype float
    ret.size = tensor->get_element_num() *
               std::ceil(tensor->get_data_type().bit_width / 32.f);
    ret.fixpos = 0;
#else
    ret.fixpos = (int8_t)log2f(scale);
#endif
    ret.height = dim_num <= 2 ? 1 : tensor->get_shape().at(2);
    ret.width = dim_num <= 3 ? 1 : tensor->get_shape().at(3);
    ret.channel = dim_num <= 1 ? 1 : tensor->get_shape().at(1);
    ret.dtype = library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto dims = tensor->get_shape();
  auto index = dims;
  auto size = 0ul;
  auto idx = std::vector<int>(tb->get_tensor()->get_shape().size(), 0);
  // CHECK_LT(dims[0], ret.data.size());
  auto tb_ext = dynamic_cast<vart::TensorBufferExt*>(tb);
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    idx[0] = batch_idx;
    auto data = tb->data(idx);
    ret.get_data(batch_idx) = (void*)data.first;
    size = data.second;
    //    std::tie(ret.get_data(batch_idx), size) = tb->data({batch_idx, 0,
    //    0, 0});
    CHECK_GE(size, ret.height * ret.width * ret.channel);
    ret.xcl_bo[batch_idx] = vitis::ai::library::XclBoInfo{0, nullptr, 0u};
    if (tb_ext) {
      auto bo = tb_ext->get_xcl_bo(batch_idx);
      ret.xcl_bo[batch_idx].xcl_handle = bo.xcl_handle;
      ret.xcl_bo[batch_idx].bo_handle = bo.bo_handle;
      ret.xcl_bo[batch_idx].offset =
          (unsigned int)tensor->template get_attr<int>("ddr_addr");
    }
  }

  return ret;
}

//# Add tensor format argument (NHWC / NCHW)
static vitis::ai::library::OutputTensor convert_tensor_buffer_to_output_tensor(
    vart::TensorBuffer* tb, vart::Runner::TensorFormat fmt) {
  float scale = 1.0f;
  auto ret = vitis::ai::library::OutputTensor{};
  auto tensor = tb->get_tensor();
  auto dim_num = tensor->get_shape().size();
  auto batch = dim_num <= 0 ? 1 : tensor->get_shape().at(0);
  ret.batch = batch;

  ret.size = tensor->get_element_num() *
             std::ceil(tensor->get_data_type().bit_width / 8.f);
  //# Store the params as per format
  if (fmt == vart::Runner::TensorFormat::NHWC) {
    if (dim_num == 2) {
      ret.height = 1;
      ret.width = 1;
      ret.channel = tensor->get_shape().at(1);
    } else {
      ret.height = dim_num <= 1 ? 1 : tensor->get_shape().at(1);
      ret.width = dim_num <= 2 ? 1 : tensor->get_shape().at(2);
      ret.channel = dim_num <= 3 ? 1 : tensor->get_shape().at(3);
    }
    if (tensor->get_data_type().type == xir::DataType::XINT) {
      ret.fixpos = (int8_t)log2f(scale);
      ret.dtype = library::DT_INT8;
      ret.fixpos = tensor->template get_attr<int>("fix_point");
    } else if (tensor->get_data_type().type == xir::DataType::FLOAT) {
      ret.fixpos = 0;
      ret.dtype = library::DT_FLOAT;
    } else {
      // LOG(FATAL) << "unsupported";
      UNI_LOG_FATAL(VAILIB_DPU_TASK_NOT_SUPPORT) << "unsupported";
    }
  } else {
#ifdef ENABLE_DPUCADX8G_RUNNER
    //# DPUV1 has datatype float
    ret.size = tensor->get_element_num() *
               std::ceil(tensor->get_data_type().bit_width / 32.f);
    ret.fixpos = 0;
#else
    ret.fixpos = -(int8_t)log2f(scale);
#endif
    ret.height = dim_num <= 2 ? 1 : tensor->get_shape().at(2);
    ret.width = dim_num <= 3 ? 1 : tensor->get_shape().at(3);
    ret.channel = dim_num <= 1 ? 1 : tensor->get_shape().at(1);
    ret.dtype = library::DT_FLOAT;
  }
  ret.name = tensor->get_name();
  auto dims = tensor->get_shape();
  auto size = 0ul;
  // CHECK_LT(dims[0], ret.data.size());
  auto tb_ext = dynamic_cast<vart::TensorBufferExt*>(tb);
  for (auto batch_idx = 0; batch_idx < dims[0]; ++batch_idx) {
    auto idx = std::vector<int32_t>(dims.size());
    idx[0] = batch_idx;
    auto data = tb->data(idx);
    ret.get_data(batch_idx) = (void*)data.first;
    size = data.second;
    CHECK_GE(size, ret.height * ret.width * ret.channel);
    ret.xcl_bo[batch_idx] = vitis::ai::library::XclBoInfo{0, nullptr, 0u};
    if (tb_ext) {
      auto bo = tb_ext->get_xcl_bo(batch_idx);
      ret.xcl_bo[batch_idx].xcl_handle = bo.xcl_handle;
      ret.xcl_bo[batch_idx].bo_handle = bo.bo_handle;
      ret.xcl_bo[batch_idx].offset =
          (unsigned int)tensor->template get_attr<int>("ddr_addr");
    }
  }

  return ret;
}

static std::vector<std::vector<vitis::ai::library::InputTensor>>
get_all_input_tensors(std::vector<std::unique_ptr<vart::Runner>>& runners) {
  auto input_tensors =
      std::vector<std::vector<vitis::ai::library::InputTensor>>();
  input_tensors.reserve(runners.size());
  for (auto& runner : runners) {
    auto dpu_runner_ext = dynamic_cast<vart::RunnerExt*>(runner.get());
    if (dpu_runner_ext ) {
      auto inputs = dpu_runner_ext->get_inputs();
      //# Get the current format
      auto fmt = runner->get_tensor_format();
      auto ret = std::vector<vitis::ai::library::InputTensor>{};
      ret.reserve(inputs.size());
      int c = 0;
      for (auto& t : inputs) {
        ret.emplace_back(convert_tensor_buffer_to_input_tensor(t, fmt));
        LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
            << "input tensor[" << c << "]: " << ret.back();
        c++;
      }
      input_tensors.emplace_back(ret);
    }
  }
  return input_tensors;
}

std::vector<vitis::ai::library::InputTensor> DpuTaskImp::getInputTensor(
    size_t idx) {
  auto input_tensors = all_input_tensors_[idx];
  if (num_of_inputs_ != -1) {
    // run batch modified to the number of incoming images
    // copy InputTensor and modify batch
    auto ret = std::vector<vitis::ai::library::InputTensor>();
    ret.reserve(input_tensors.size());
    for (auto& t : input_tensors) {
      auto tensor = vitis::ai::library::InputTensor(t);
      // num_of_inputs_ : num of real input images, it might be less than or
      // equal to batch size
      CHECK_LE((unsigned)num_of_inputs_, t.batch) << "logical error";
      auto hw_batch = tensor.batch;
      tensor.batch = (unsigned)num_of_inputs_;
      tensor.size = tensor.size * tensor.batch / hw_batch;
      ret.emplace_back(tensor);
    }
    return ret;
  } else {
    return input_tensors;
  }
}

static std::vector<std::vector<vitis::ai::library::OutputTensor>>
get_all_output_tensors(std::vector<std::unique_ptr<vart::Runner>>& runners) {
  auto output_tensors =
      std::vector<std::vector<vitis::ai::library::OutputTensor>>();
  output_tensors.reserve(runners.size());
  for (auto& runner : runners) {
    auto dpu_runner_ext = dynamic_cast<vart::RunnerExt*>(runner.get());
    auto outputs = dpu_runner_ext->get_outputs();
    //# Get the current format
    auto fmt = runner->get_tensor_format();
    auto ret = std::vector<vitis::ai::library::OutputTensor>{};
    ret.reserve(outputs.size());
    int c = 0;
    for (auto& t : outputs) {
      ret.emplace_back(convert_tensor_buffer_to_output_tensor(t, fmt));
      LOG_IF(INFO, ENV_PARAM(DEBUG_DPBASE))
          << "output tensor[" << c << "]: " << ret.back();
      c++;
    }
    output_tensors.emplace_back(ret);
  }
  return output_tensors;
}

std::vector<vitis::ai::library::OutputTensor> DpuTaskImp::getOutputTensor(
    size_t idx) {
  auto output_tensors = all_output_tensors_[idx];
  if (num_of_inputs_ != -1) {
    auto ret = std::vector<vitis::ai::library::OutputTensor>();
    ret.reserve(output_tensors.size());
    for (auto& t : output_tensors) {
      auto tensor = vitis::ai::library::OutputTensor(t);
      // num_of_inputs_ : num of real input images, it might be less than or
      // equal to batch size
      CHECK_LE((unsigned)num_of_inputs_, t.batch) << "logical error";
      auto hw_batch = tensor.batch;
      tensor.batch = (unsigned)num_of_inputs_;
      tensor.size = tensor.size * tensor.batch / hw_batch;
      ret.emplace_back(tensor);
    }
    return ret;
  } else {
    return output_tensors;
  }
}

size_t DpuTaskImp::get_input_batch(size_t kernel_idx, size_t node_idx) const {
  auto v = dynamic_cast<vart::RunnerExt*>(runners_[kernel_idx].get());
  return v ? v
      ->get_inputs()[node_idx]
      ->get_tensor()
      ->get_shape()
      .at(0)
    :  0 ;
}

size_t DpuTaskImp::get_num_of_kernels() const {  //
  return runners_.size();
}

const xir::Graph* DpuTaskImp::get_graph() const {
  return graph_holder_->get_graph();
}

std::unique_ptr<xir::Attrs> DpuTaskImp::get_attrs() const {
  return graph_holder_->get_graph()->get_attrs();
}

void DpuTaskImp::set_num_of_inputs(size_t n) {
  // TODO it is too much to call clear_num_of_inputs
  // CHECK_EQ(num_of_inputs_, -1)
  //     << "LOGICAL ERROR. you cannot set num input twices";
  CHECK_LT(n, 100) << "with current DPU design, it is not possible for very "
                      "large batch size.";
  num_of_inputs_ = (int)n;
}

int DpuTaskImp::get_input_buffer_size() const {
  auto subgraphs = graph_holder_->get_subgraphs();
  CHECK_NE(subgraphs.size(), 0u);
  auto subgraph = subgraphs[0];
  return vart::get_input_buffer_size(subgraph);
}

size_t DpuTaskImp::get_input_offset() const {
  auto subgraphs = graph_holder_->get_subgraphs();
  CHECK_NE(subgraphs.size(), 0u);
  auto subgraph = subgraphs[0];
  auto offsets = vart::get_input_offset(subgraph);
  CHECK_EQ(offsets.size(), 1u);
  return offsets.front();
}

int DpuTaskImp::get_input_fix_point() const {
  CHECK(!runners_.empty());
  auto the_first_runner = runners_.front().get();
  auto tensors = the_first_runner->get_input_tensors();
  CHECK_EQ(tensors.size(), 1u);
  auto tensor = tensors[0];
  CHECK(tensor->has_attr("fix_point"));
  int fixpos = tensor->get_attr<int>("fix_point");
  return fixpos;
}

// Local Variables:
// mode:c++
// c-basic-offset: 2
// coding: undecided-unix
// End:
