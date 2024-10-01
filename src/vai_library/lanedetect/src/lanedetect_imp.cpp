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
#include "./lanedetect_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

using namespace std;
using namespace cv;
namespace vitis {
namespace ai {

RoadLineImp::RoadLineImp(const std::string& model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<RoadLine>(model_name, need_preprocess),
      model_name_(model_name),
      processor_{vitis::ai::RoadLinePostProcess::create(
          configurable_dpu_task_->getInputTensor()[0],
          configurable_dpu_task_->getOutputTensor()[0],
          configurable_dpu_task_->getConfig())} {}

RoadLineImp::~RoadLineImp() {}

RoadLineResult RoadLineImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }

  __TIC__(ROADLINE_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(ROADLINE_SET_IMG)
  __TIC__(ROADLINE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(ROADLINE_DPU)
  __TIC__(ROADLINE_POST_PROCESS)

  auto results = processor_->road_line_post_process(
      vector<int>{input_image.cols}, vector<int>{input_image.rows}, 1u)[0];
  __TOC__(ROADLINE_POST_PROCESS)
  return results;
}

std::vector<RoadLineResult> RoadLineImp::run(
    const std::vector<cv::Mat>& input_img) {
  auto size = cv::Size(getInputWidth(), getInputHeight());

  std::vector<cv::Mat> vimg(input_img.size());
  std::vector<int> vcols, vrows;

  for (auto i = 0ul; i < input_img.size(); i++) {
    if (size != input_img[i].size()) {
      cv::resize(input_img[i], vimg[i], size, 0, 0, cv::INTER_LINEAR);

    } else {
      vimg[i] = input_img[i];
    }
    vcols.push_back(input_img[i].cols);
    vrows.push_back(input_img[i].rows);
  }

  __TIC__(ROADLINE_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(vimg);
  __TOC__(ROADLINE_SET_IMG)
  __TIC__(ROADLINE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(ROADLINE_DPU)
  __TIC__(ROADLINE_POST_PROCESS)
  auto results = processor_->road_line_post_process(vcols, vrows, vimg.size());
  __TOC__(ROADLINE_POST_PROCESS)
  return results;
}

}  // namespace ai
}  // namespace vitis
