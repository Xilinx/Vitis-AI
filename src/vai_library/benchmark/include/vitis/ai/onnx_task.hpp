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
#pragma once
#include <core/session/experimental_onnxruntime_cxx_api.h>
#include <vitis/ai/env_config.hpp>
#include "getbatch.hpp"
#include "vitis/ai/profiling.hpp"

DEF_ENV_PARAM(DEBUG_ONNX_TASK, "0")

#if 0
static void CheckStatus(OrtStatus* status) {
  if (status != NULL) {
    const char* msg = Ort::GetApi().GetErrorMessage(status);
    fprintf(stderr, "%s\n", msg);
    Ort::GetApi().ReleaseStatus(status);
    exit(1);
  }
}
#endif
// pretty prints a shape dimension vector
static std::string print_shape(const std::vector<int64_t>& v) {
  std::stringstream ss("");
  for (size_t i = 0; i < v.size() - 1; i++) ss << v[i] << "x";
  ss << v[v.size() - 1];
  return ss.str();
}

class OnnxTask {
 public:
  explicit OnnxTask(const std::string& model_name)
      : model_name_(model_name),
        env_(ORT_LOGGING_LEVEL_WARNING, model_name_.c_str()),
        session_options_(Ort::SessionOptions()) {
    ::get_batch();
    
    auto options = std::unordered_map<std::string,std::string>({});
    options["config_file"] = "/usr/bin/vaip_config.json";
    // optional, eg: cache path and cache key: /tmp/my_cache/abcdefg
    // options["CacheDir"] = "/tmp/my_cache";
    // options["CacheKey"] = "abcdefg";
    
    session_options_.AppendExecutionProvider("VitisAI", options );
    
    session_.reset(
        new Ort::Experimental::Session(env_, model_name_, session_options_));
    input_shapes_ = session_->GetInputShapes();
    output_shapes_ = session_->GetOutputShapes();
    if (input_shapes_[0][0] == -1) {
      input_shapes_[0][0] = g_batchnum;
      output_shapes_[0][0] = g_batchnum;
    }
  }

  OnnxTask(const OnnxTask&) = delete;

  virtual ~OnnxTask() {}

  size_t getInputWidth() const { return input_shapes_[0][3]; };
  size_t getInputHeight() const { return input_shapes_[0][2]; };
  size_t get_input_batch() const { return input_shapes_[0][0]; }

  std::vector<std::vector<int64_t>> get_input_shapes() { return input_shapes_; }

  std::vector<std::string> get_input_names() {
    return session_->GetInputNames();
  }

  std::vector<std::string> get_output_names() {
    return session_->GetOutputNames();
  }

  std::vector<std::vector<int64_t>> get_output_shapes() {
    return output_shapes_;
  }

  void set_input_image_rgb(const cv::Mat& image, float* data, const std::vector<float>& mean, const  std::vector<float>& scale) {
     return set_input_image_internal(image, data, mean, scale, true);
  }
  void set_input_image_bgr(const cv::Mat& image, float* data, const std::vector<float>& mean, const  std::vector<float>& scale) {
     return set_input_image_internal(image, data, mean, scale, false);
  }
  void set_input_image_internal(const cv::Mat& image, float* data, const std::vector<float>& mean, const  std::vector<float>& scale, bool btrans) {
    // BGR->RGB (maybe) and HWC->CHW
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < image.rows; h++) {
        for (int w = 0; w < image.cols; w++) {
          auto c_t = btrans? abs(c - 2): c;
          auto image_data = (image.at<cv::Vec3b>(h, w)[c_t] - mean[c_t]) * scale[c_t];
          data[c * image.rows * image.cols + h * image.cols + w] = (float)image_data;
        }
      }
    }
  }

  std::vector<Ort::Value> convert_input(
      std::vector<float>& input_values, size_t size,
      const std::vector<int64_t> input_tensor_shape) {
    std::vector<Ort::Value> input_tensors;
    input_tensors.push_back(Ort::Experimental::Value::CreateTensor<float>(
        input_values.data(), size, input_tensor_shape));
    return input_tensors;
  }

  void run_task(const std::vector<Ort::Value>& input_tensors,
                std::vector<Ort::Value>& output_tensors) {
    std::vector<std::string> input_names = session_->GetInputNames();
    if (ENV_PARAM(DEBUG_ONNX_TASK)) {
      auto input_shapes = get_input_shapes();
      std::cout << "Input Node Name/Shape (" << input_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < input_names.size(); i++) {
        std::cout << "\t" << input_names[i] << " : "
                  << print_shape(input_shapes[i]) << std::endl;
      }
    }
    std::vector<std::string> output_names = session_->GetOutputNames();
    if (ENV_PARAM(DEBUG_ONNX_TASK)) {
      auto output_shapes = get_output_shapes();
      std::cout << "Output Node Name/Shape (" << output_names.size()
                << "):" << std::endl;
      for (size_t i = 0; i < output_names.size(); i++) {
        std::cout << "\t" << output_names[i] << " : "
                  << print_shape(output_shapes[i]) << std::endl;
      }
    }

    output_tensors = session_->Run(session_->GetInputNames(), input_tensors,
                                   session_->GetOutputNames());
  }

 protected:
  std::string model_name_;
  Ort::Env env_;
  Ort::SessionOptions session_options_;
  std::unique_ptr<Ort::Experimental::Session> session_;
  std::vector<std::vector<int64_t>> input_shapes_;
  std::vector<std::vector<int64_t>> output_shapes_;
  // std::vector<Ort::Value> input_tensors_;
  // std::vector<Ort::Value> output_tensors_;
};

