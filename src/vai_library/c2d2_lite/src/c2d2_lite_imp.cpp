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
#include "./c2d2_lite_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {
DEF_ENV_PARAM(ENABLE_C2D2_lite_DEBUG, "0");

class C2D2_liteImp1 : public ConfigurableDpuTaskBase {
 public:
  explicit C2D2_liteImp1(const std::string& model_name, bool need_preprocess)
      : ConfigurableDpuTaskBase(model_name, need_preprocess) {}
  ConfigurableDpuTask* get_dpu_task() { return configurable_dpu_task_.get(); }

 public:
  virtual ~C2D2_liteImp1() {}

 protected:
  C2D2_liteImp1(const C2D2_liteImp1&) = delete;
};

// C2D2_liteImp
C2D2_liteImp::C2D2_liteImp(const std::string& model_name0,
                           const std::string& model_name1, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<C2D2_lite>(model_name0, need_preprocess),
      imp1(new C2D2_liteImp1(model_name1, need_preprocess)) {}

C2D2_liteImp::~C2D2_liteImp() { delete imp1; }
size_t C2D2_liteImp::get_input_batch() const {
  return std::max(configurable_dpu_task_->get_input_batch(),
                  imp1->get_input_batch());
}

float C2D2_liteImp::run(const std::vector<cv::Mat>& input_image) {
  return run(std::vector<std::vector<cv::Mat>>(1, input_image))[0];
}

std::vector<float> C2D2_liteImp::run(
    const std::vector<std::vector<cv::Mat>>& input_images) {
  auto kernel = configurable_dpu_task_->getConfig().kernel(0);
  auto mean = kernel.mean(0);
  auto input_tensor = configurable_dpu_task_->getInputTensor()[0][0];
  auto output_tensor = configurable_dpu_task_->getOutputTensor()[0][0];
  auto scale = kernel.scale(0) * tensor_scale(input_tensor);
  size_t batch0 = configurable_dpu_task_->get_input_batch();
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);

  auto dpu_task1 = imp1->get_dpu_task();
  auto input_tensor1 = dpu_task1->getInputTensor()[0][0];
  std::vector<float> res;
  for (auto i = 0u; i < input_images.size() && i < imp1->get_input_batch();
       i++) {
    const auto& inputs = input_images[i];
    auto data1 = (int8_t*)input_tensor1.get_data(i);

    // for 300
    __TIC__(C2D2_lite_RUN_300)
    for (size_t j = 0; j < inputs.size(); j += batch0) {
      // copy_input
      __TIC__(C2D2_lite_RUN_0)
      __TIC__(C2D2_lite_0_COPY_INPUT)
      for (size_t k = 0; k < batch0 && j + k < inputs.size(); k++) {
        cv::Mat image;
        if (size != inputs[j + k].size()) {
          cv::resize(inputs[j + k], image, size, 0, 0, cv::INTER_LINEAR);
        } else {
          image = inputs[j + k];
        }
        auto img_data = (int8_t*)input_tensor.get_data(k);
        for (auto h = 0; h < sHeight; h++) {
          auto img_row = image.ptr(h);
          auto didx = h * sWidth;
          for (auto w = 0; w < sWidth; w++) {
            img_data[didx + w] = std::round((float(img_row[w]) - mean) * scale);
          }
        }
      }
      __TOC__(C2D2_lite_0_COPY_INPUT)

      __TIC__(C2D2_lite_0_DPU)
      configurable_dpu_task_->run(0);
      __TOC__(C2D2_lite_0_DPU)
      // from output_tensor copy output to input_tensor1
      __TIC__(C2D2_lite_0_COPY_OUTPUT)
      for (size_t k = 0; k < batch0 && j + k < inputs.size(); k++) {
        memcpy(data1 + (k + j) * output_tensor.size / batch0,
               (int8_t*)output_tensor.get_data(k), output_tensor.size / batch0);
      }
      __TOC__(C2D2_lite_0_COPY_OUTPUT)
      __TOC__(C2D2_lite_RUN_0)
    }
    __TOC__(C2D2_lite_RUN_300)
  }

  __TIC__(C2D2_lite_1_DPU)
  dpu_task1->run(0);
  __TOC__(C2D2_lite_1_DPU)
  for (size_t k = 0; k < dpu_task1->get_input_batch(); k++) {
    res.push_back(((float*)dpu_task1->getOutputTensor()[0][0].get_data(k))[0]);
  }
  return res;
}

}  // namespace ai
}  // namespace vitis
