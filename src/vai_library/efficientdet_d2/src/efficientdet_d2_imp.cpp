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
#include "./efficientdet_d2_imp.hpp"

#include <fstream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
namespace vitis {
namespace ai {

DEF_ENV_PARAM(DEBUG_EFFICIENTDET_D2, "0");
// DEF_ENV_PARAM(DEBUG_SAVE_IMG, "0");

EfficientDetD2Imp::EfficientDetD2Imp(const std::string& model_name,
                                     bool need_preprocess)
    : EfficientDetD2(model_name, need_preprocess),
      processor_{vitis::ai::EfficientDetD2PostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

EfficientDetD2Imp::EfficientDetD2Imp(const std::string& model_name,
                                     xir::Attrs* attrs, bool need_preprocess)
    : EfficientDetD2(model_name, attrs, need_preprocess),
      processor_{vitis::ai::EfficientDetD2PostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

EfficientDetD2Imp::~EfficientDetD2Imp() {}

std::vector<float> EfficientDetD2Imp::preprocess(
    const std::vector<cv::Mat>& batch_image_bgr, size_t batch_size) {
  int width = this->getInputWidth();
  int height = this->getInputHeight();
  auto num = std::min(batch_image_bgr.size(), batch_size);
  auto input_tensor = configurable_dpu_task_->getInputTensor()[0][0];
  std::vector<cv::Mat> results(num);
  std::vector<int> batch_rect_w(num);
  std::vector<int> batch_rect_h(num);
  std::vector<float> batch_scales(num);
  for (auto i = 0u; i < num; ++i) {
    auto& image_bgr = batch_image_bgr[i];
    float scale_w = ((float)width) / image_bgr.cols;
    float scale_h = ((float)height) / image_bgr.rows;
    float scale = std::min(scale_w, scale_h);
    results[i] = cv::Mat::zeros(height, width, CV_8UC3);
    int rect_w = std::round(image_bgr.cols * scale);
    int rect_h = std::round(image_bgr.rows * scale);
    auto rect = cv::Rect{0, 0, rect_w, rect_h};
    auto image_rect = results[i](rect);
    cv::Mat resized_image;
    cv::resize(image_bgr, resized_image, cv::Size(rect_w, rect_h), 0, 0,
               cv::INTER_LINEAR);
    resized_image.copyTo(image_rect);
    batch_rect_w[i] = rect_w;
    batch_rect_h[i] = rect_h;
    batch_scales[i] = scale;
  }
  configurable_dpu_task_->setInputImageRGB(results);
  __TIC__(PREPROCESS_RESET)
  auto line = input_tensor.width * input_tensor.channel;
  for (auto i = 0u; i < num; ++i) {
    int8_t* input_ptr = (int8_t*)input_tensor.get_data(i);
    auto rect_w = batch_rect_w[i];
    auto rect_h = batch_rect_h[i];
    for (auto h = 0u; h < input_tensor.height; ++h) {
      if (h < (unsigned int)rect_h &&
          (unsigned int)rect_w >= input_tensor.width) {
        continue;
      }
      auto start = input_ptr + h * line;
      auto end = start + line;
      if (h < (unsigned int)rect_h) {
        start += rect_w * input_tensor.channel;
      }
      auto len = end - start;
      std::memset(start, 0, len);
    }
    // if (ENV_PARAM(DEBUG_SAVE_IMG)) {
    //  cv::imwrite(std::string("./batch_") + std::to_string(i) + ".jpg",
    //              results[i]);
    //}
  }
  __TOC__(PREPROCESS_RESET)
  return batch_scales;
}

EfficientDetD2Result EfficientDetD2Imp::run(const cv::Mat& input_img) {
  __TIC__(EfficientDetD2_total)

  __TIC__(EfficientDetD2_preprocess)
  // cv::Mat img;
  // auto mAP =
  //    configurable_dpu_task_->getConfig().efficientdet_d2_param().test_map();

  auto input_tensor = configurable_dpu_task_->getInputTensor()[0][0];
  // int8_t* input_ptr = (int8_t*)input_tensor.get_data(0);
  // int size = input_tensor.size / input_tensor.batch;
  // std::memset(input_ptr, 0, size);

  std::vector<cv::Mat> batch_input(1, input_img);
  std::vector<int> vwidth(1, input_img.cols);
  std::vector<int> vheight(1, input_img.rows);
  auto vscale = preprocess(batch_input, 1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
      << "input image scale:" << vscale[0];
  __TOC__(EfficientDetD2_preprocess)

  __TIC__(EfficientDetD2_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(EfficientDetD2_dpu)

  __TIC__(EfficientDetD2_post)
  auto results = processor_->postprocess(1u, vwidth, vheight, vscale);
  __TOC__(EfficientDetD2_post)

  __TOC__(EfficientDetD2_total)
  return results[0];
}

std::vector<EfficientDetD2Result> EfficientDetD2Imp::run(
    const std::vector<cv::Mat>& input_img) {
  auto batch = get_input_batch();
  __TIC__(EfficientDetD2_total)

  auto num = std::min(input_img.size(), batch);

  __TIC__(EfficientDetD2_preprocess)
  // std::vector<cv::Mat> vimg(num);
  // std::vector<float> vscale(num);
  std::vector<int> vwidth(num);
  std::vector<int> vheight(num);

  auto vscale = this->preprocess(input_img, num);

  for (auto i = 0ul; i < num; i++) {
    vwidth[i] = input_img[i].cols;
    vheight[i] = input_img[i].rows;
    // std::memset(input_ptr, 0, size);
    LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
        << "input image scale:" << vscale[i];
    // LOG_IF(INFO, ENV_PARAM(DEBUG_EFFICIENTDET_D2))
    //    << "vimg width:" << vimg[i].cols << ", height:" << vimg[i].rows;
    // if (ENV_PARAM(DEBUG_SAVE_IMG)) {
    //  auto input_tensor = configurable_dpu_task_->getInputTensor()[0][0];
    //  int8_t* input_ptr = (int8_t*)input_tensor.get_data(i);
    //  int size = input_tensor.size / input_tensor.batch;
    //  cv::Mat img_to_write =
    //      cv::Mat::zeros(input_tensor.height, input_tensor.width, CV_8UC3);
    //  for (auto idx = 0; idx < size; ++idx) {
    //    img_to_write.data[idx] = input_ptr[idx];
    //  }
    //  cv::imwrite(std::string("./dump_") + std::to_string(i) + ".jpg",
    //              img_to_write);
    //}
  }
  __TOC__(EfficientDetD2_preprocess)

  __TIC__(EfficientDetD2_dpu)
  configurable_dpu_task_->run(0);
  __TOC__(EfficientDetD2_dpu)

  __TIC__(EfficientDetD2_post)
  auto results = processor_->postprocess(num, vwidth, vheight, vscale);
  __TOC__(EfficientDetD2_post)

  __TOC__(EfficientDetD2_total)
  return results;
}

}  // namespace ai
}  // namespace vitis
