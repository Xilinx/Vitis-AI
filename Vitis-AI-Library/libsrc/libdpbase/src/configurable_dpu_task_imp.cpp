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
#include "./configurable_dpu_task_imp.hpp"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>
#include <xilinx/ai/env_config.hpp>
using namespace std;
DEF_ENV_PARAM(DEBUG_DPBASE, "0");
namespace xilinx {
namespace ai {
static xilinx::ai::proto::DpuModelParam get_config(
    const vitis::ai::DpuMeta& meta_info);

static std::vector<float> get_means(
    const xilinx::ai::proto::DpuKernelParam& c) {
  return std::vector<float>(c.mean().begin(), c.mean().end());
}

static std::vector<float> get_scales(
    const xilinx::ai::proto::DpuKernelParam& c) {
  return std::vector<float>(c.scale().begin(), c.scale().end());
}

static std::unique_ptr<DpuTask> init_tasks(const std::string& model_name) {
  return DpuTask::create(model_name);
}

ConfigurableDpuTaskImp::ConfigurableDpuTaskImp(const std::string& model_name,
                                               bool need_preprocess)
    : tasks_{init_tasks(model_name)},  //
      model_{get_config(tasks_->get_dpu_meta_info())} {
  if (need_preprocess) {
    auto mean = get_means(model_.kernel(0));
    auto scale = get_scales(model_.kernel(0));
    tasks_->setMeanScaleBGR(mean, scale);
  }
}
ConfigurableDpuTaskImp::~ConfigurableDpuTaskImp() {}

const xilinx::ai::proto::DpuModelParam& ConfigurableDpuTaskImp::getConfig()
    const {
  return model_;
}

int ConfigurableDpuTaskImp::getInputWidth() const {
  return tasks_->getInputTensor(0u)[0].width;
}

int ConfigurableDpuTaskImp::getInputHeight() const {
  return tasks_->getInputTensor(0u)[0].height;
}

const vitis::ai::DpuMeta& ConfigurableDpuTaskImp::get_dpu_meta_info() const {
  return tasks_->get_dpu_meta_info();
}

std::vector<std::vector<xilinx::ai::InputTensor>>
ConfigurableDpuTaskImp::getInputTensor() const {
  auto ret = std::vector<std::vector<xilinx::ai::InputTensor>>{};
  auto size = tasks_->get_num_of_tasks();
  ret.reserve(size);
  for (auto idx = 0u; idx < size; ++idx) {
    ret.emplace_back(tasks_->getInputTensor(idx));
  }
  return ret;
}
std::vector<std::vector<xilinx::ai::OutputTensor>>
ConfigurableDpuTaskImp::getOutputTensor() const {
  auto ret = std::vector<std::vector<xilinx::ai::OutputTensor>>{};
  auto size = tasks_->get_num_of_tasks();
  ret.reserve(size);
  for (auto idx = 0u; idx < size; ++idx) {
    ret.emplace_back(tasks_->getOutputTensor(idx));
  }
  if (ENV_PARAM(DEBUG_DPBASE)) {
    for (auto idx = 0u; idx < size; ++idx) {
      for (auto x = 0u; x < ret[idx].size(); ++x) {
        LOG(INFO) << "kernel[" << idx << "], output[" << x
                  << "] = " << ret[idx][x];
      }
    }
  }
  return ret;
}

void ConfigurableDpuTaskImp::setInputImageBGR(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  tasks_->setImageBGR(image);
}

void ConfigurableDpuTaskImp::setInputImageRGB(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0);
  } else {
    image = input_image;
  }
  tasks_->setImageRGB(image);
}

void ConfigurableDpuTaskImp::run(int task_index) {
  if (ENV_PARAM(DEBUG_DPBASE)) {
    LOG(INFO) << "running task " << task_index;
    for (auto input : tasks_->getInputTensor(task_index)) {
      LOG(INFO) << "input " << input;
    }
    for (auto output : tasks_->getOutputTensor(task_index)) {
      LOG(INFO) << "output " << output;
    }
  }
  tasks_->run(task_index);
}

static std::string slurp(const char* filename);
static xilinx::ai::proto::DpuModelParam get_config(
    const vitis::ai::DpuMeta& meta_info) {
  auto config_file = meta_info.config_file;
  if (config_file[0] != '/') {
    config_file = meta_info.dirname + "/" + config_file;
  }
  xilinx::ai::proto::DpuModelParamList mlist;
  auto text = slurp(config_file.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);
  CHECK(ok) << "cannot parse config file. config_file=" << config_file;
  CHECK_EQ(mlist.model_size(), 1)
      << "only support one model per config file."
      << "config_file " << config_file << " "       //
      << "content: " << mlist.DebugString() << " "  //
      ;
  return mlist.model(0);
}
static std::string slurp(const char* filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  CHECK(in.good()) << "failed to read config file. filename=" << filename;
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

}  // namespace ai
}  // namespace xilinx
