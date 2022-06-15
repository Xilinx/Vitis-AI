/*
 * Copyright 2019 xilinx Inc.
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
#include "./polypsegmentation_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;
namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_POLYPSEGMENTATION_DEBUG, "0");

PolypSegmentationImp::PolypSegmentationImp(const std::string& model_name,
                                           bool need_preprocess)
    : PolypSegmentation(model_name, need_preprocess) {}

PolypSegmentationImp::~PolypSegmentationImp() {}

inline int8_t fix(float data) {
  auto data_max = 127.0;
  auto data_min = -128.0;
  if (data > data_max) {
    data = data_max;
  } else if (data < data_min) {
    data = data_min;
  } else if (data < 0 && (data - floor(data)) == 0.5) {
    data = static_cast<float>(ceil(data));
  } else {
    data = static_cast<float>(round(data));
  }
  return (int8_t)data;
}

std::vector<SegmentationResult> PolypSegmentationImp::run(
    const std::vector<cv::Mat>& input_images) {
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  std::vector<cv::Mat> images;

  auto size = cv::Size(sWidth, sHeight);

  for (size_t i = 0; i < input_images.size(); i++) {
    cv::Mat image;
    if (size != input_images[i].size()) {
      cv::resize(input_images[i], image, size, 0, 0, cv::INTER_LINEAR);
    } else {
      image = input_images[i];
    }
    images.push_back(image);
  }
  __TIC__(POLYPSEGMENTATION_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(POLYPSEGMENTATION_SET_IMG)

  __TIC__(POLYPSEGMENTATION_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(POLYPSEGMENTATION_DPU)

  __TIC__(POLYPSEGMENTATION_POST_ARM)
  std::vector<SegmentationResult> ret;
  for (size_t i = 0; i < input_images.size(); i++) {
    auto output_tensor = configurable_dpu_task_->getOutputTensor()[0][0];
    float scale = vitis::ai::library::tensor_scale(output_tensor);
    int8_t* output_data = (int8_t*)output_tensor.get_data(i);

    vector<int8_t> v_data(output_tensor.height * output_tensor.width);
    if (output_tensor.name.find("sigmoid") == std::string::npos) {
      // do sigmoid
      auto ops = configurable_dpu_task_->get_graph()->get_ops();
      for (auto op : ops) {
        if (op->get_name().find("MergeMultoHsigmoid") != std::string::npos) {
          auto sigmoid_in =
              3.0f * 2731.0f * pow(2, op->get_attr<int>("hsigmoid_in"));
          auto pow_32 = pow(2, 32);
          scale = 1.0 / pow(2, op->get_attr<int>("shift_hsigmoid"));

          LOG_IF(INFO, ENV_PARAM(ENABLE_POLYPSEGMENTATION_DEBUG))
              << string(op->get_name());
          for (size_t j = 0; j < v_data.size(); j++) {
            auto tmp =
                min(pow_32, max(0.0, (2731.0f * output_data[j] + sigmoid_in))) *
                scale;
            v_data[j] = fix(tmp);
          }
          output_data = v_data.data();
          break;
        }
      }
    }
    // fix to float
    cv::Mat segMat(output_tensor.height, output_tensor.width, CV_8UC1,
                   output_data);
    segMat.forEach<uint8_t>([&](uint8_t& val, const int* position) {
      val = scale * 255.0f * int8_t(val);
    });
    // resize
    cv::Mat image;
    if (input_images[i].size() != segMat.size()) {
      cv::resize(segMat, image, input_images[i].size(), 0, 0, cv::INTER_LINEAR);
    } else {
      image = segMat;
    }
    ret.push_back(SegmentationResult{(int)input_images[i].cols,
                                     (int)input_images[i].rows, image});
  }
  __TOC__(POLYPSEGMENTATION_POST_ARM)
  return ret;
}

SegmentationResult PolypSegmentationImp::run(const cv::Mat& input_image) {
  std::vector<cv::Mat> input_images;
  input_images.push_back(input_image);
  return run(input_images)[0];
}
}  // namespace ai
}  // namespace vitis
