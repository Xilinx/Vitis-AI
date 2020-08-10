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
#include "configurable_dpu_task_imp.hpp"
#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <fstream>
#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <sstream>
#include <vector>
#include <vitis/ai/env_config.hpp>

using namespace std;

DEF_ENV_PARAM(DEBUG_DPBASE, "0");

namespace vitis {
namespace ai {

static vector<string> find_model_search_path() {
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

											  
static string find_model(const string& name) {
//# Disable the unused functions when DPUV1 Enable
#ifndef ENABLE_DPUCADX8G_RUNNER 
  if (filesize(name) > 4096u) {
    return name;
  }

  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto xmodel_name = ret + ".xmodel";
    if (filesize(xmodel_name) > 0u) {
      return xmodel_name;
    }
    const auto elf_name = ret + ".elf";
    if (filesize(elf_name) > 0u) {
      return elf_name;
    }
  }
#else
  //# Get the config prototxt from dir path
  std::string tmp_name = name;
  while(tmp_name.back() == '/') {
	  tmp_name.pop_back();
  }
  std::string last_element(tmp_name.substr(tmp_name.rfind("/") + 1));
  auto config_file = name + "/" + last_element + ".prototxt";  
  
  if (filesize(config_file) > 0u) {
    return config_file;
  }
  
  //# Get model path from standard path
  auto ret = std::string();
  for (const auto& p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto config_file = ret + ".prototxt";
    if (filesize(config_file) > 0u) {
      return config_file;
    }
  }
#endif

  stringstream str;
  str << "cannot find model <" << name << "> after checking following dir:";
  for (const auto& p : find_model_search_path()) {
    str << "\n\t" << p;
  }
  LOG(FATAL) << str.str();
  return string{""};
}


static string find_config_file(const string& name) {
  auto model = find_model(name);
  std::string pre_name = model.substr(0, model.rfind("."));
  auto config_file = pre_name + ".prototxt";
  if (filesize(config_file) > 0u) {
    return config_file;
  }
  LOG(FATAL) << "cannot find " << config_file;
  return string{""};
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

static vitis::ai::proto::DpuModelParam get_config(
    const std::string& model_name) {
#ifdef ENABLE_DPUCADX8G_RUNNER
  //# skip xmodel reading for DPUV1
  auto config_file = find_config_file(model_name);
#else
  auto config_file = find_config_file(find_model(model_name));
#endif
  vitis::ai::proto::DpuModelParamList mlist;
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

static std::vector<float> get_means(const vitis::ai::proto::DpuKernelParam& c) {
  return std::vector<float>(c.mean().begin(), c.mean().end());
}

static std::vector<float> get_scales(
    const vitis::ai::proto::DpuKernelParam& c) {
  return std::vector<float>(c.scale().begin(), c.scale().end());
}

#ifdef ENABLE_DPUCADX8G_RUNNER
//# Skip xmodel reding for DPUV1
static std::unique_ptr<DpuTask> init_tasks(const std::string& model_name) {
  return DpuTask::create(model_name);
}
#else
static std::unique_ptr<DpuTask> init_tasks(const std::string& model_name) {
  return DpuTask::create(find_model(model_name));
}
#endif

ConfigurableDpuTaskImp::ConfigurableDpuTaskImp(const std::string& model_name,
                                               bool need_preprocess)
    : tasks_{init_tasks(model_name)},  //
      model_{get_config(model_name)} {
  if (need_preprocess) {
    auto mean = get_means(model_.kernel(0));
    auto scale = get_scales(model_.kernel(0));
    tasks_->setMeanScaleBGR(mean, scale);
  }
}

ConfigurableDpuTaskImp::~ConfigurableDpuTaskImp() {}

const vitis::ai::proto::DpuModelParam& ConfigurableDpuTaskImp::getConfig()
    const {
  return model_;
}

int ConfigurableDpuTaskImp::getInputWidth() const {
  return tasks_->getInputTensor(0u)[0].width;
}

int ConfigurableDpuTaskImp::getInputHeight() const {
  return tasks_->getInputTensor(0u)[0].height;
}

size_t ConfigurableDpuTaskImp::get_input_batch() const {
  // TODO: assume all kernels and feature map have same batch size:
  return tasks_->get_input_batch(0, 0);
}

const xir::Graph* ConfigurableDpuTaskImp::get_graph() const {
  return tasks_->get_graph();
}
std::vector<std::vector<vitis::ai::library::InputTensor>>
ConfigurableDpuTaskImp::getInputTensor() const {
  auto ret = std::vector<std::vector<vitis::ai::library::InputTensor>>{};
  auto size = tasks_->get_num_of_kernels();
  ret.reserve(size);
  for (auto idx = 0u; idx < size; ++idx) {
    ret.emplace_back(tasks_->getInputTensor(idx));
  }
  return ret;
}
std::vector<std::vector<vitis::ai::library::OutputTensor>>
ConfigurableDpuTaskImp::getOutputTensor() const {
  auto ret = std::vector<std::vector<vitis::ai::library::OutputTensor>>{};
  auto size = tasks_->get_num_of_kernels();
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

void ConfigurableDpuTaskImp::setInputImageBGR(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto i = 0u; i < input_images.size(); i++) {
    if (size != input_images[i].size()) {
      cv::Mat img;
      cv::resize(input_images[i], img, size, 0);
      images.push_back(img);
    } else {
      images.push_back(input_images[i]);
    }
  }
  tasks_->setImageBGR(images);
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

void ConfigurableDpuTaskImp::setInputImageRGB(
    const std::vector<cv::Mat>& input_images) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto i = 0u; i < input_images.size(); i++) {
    if (size != input_images[i].size()) {
      cv::Mat img;
      cv::resize(input_images[i], img, size, 0);
      images.push_back(img);
    } else {
      images.push_back(input_images[i]);
    }
  }
  tasks_->setImageRGB(images);
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

}  // namespace ai
}  // namespace vitis
