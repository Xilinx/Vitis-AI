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
#include "./fcos_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>
using namespace std;

DEF_ENV_PARAM(ENABLE_FCOS_DEBUG, "0");

namespace vitis {
namespace ai {

void image_resize(const cv::Mat input_image, cv::Mat& output_image,
                  const int min_size, const int max_size) {
  cv::Mat img_res;
  cv::resize(input_image, img_res, cv::Size(max_size, min_size));
  output_image = img_res;
}

FCOSImp::FCOSImp(const std::string& model_name, bool need_preprocess)
    : FCOS(model_name, need_preprocess) {}
FCOSImp::FCOSImp(const std::string& model_name, xir::Attrs* attrs,
                 bool need_preprocess)
    : FCOS(model_name, attrs, need_preprocess) {}

FCOSImp::~FCOSImp() {}

FCOSResult FCOSImp::run(const cv::Mat& input_images) {
  return run(vector<cv::Mat>(1, input_images))[0];
}

vector<FCOSResult> FCOSImp::run(const vector<cv::Mat>& input_images) {
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  cv::Mat image;
  vector<cv::Mat> images(input_images.size());
  // std::ofstream outfile;
  // outfile.open("image2.txt");
  vector<FCOSResult> ret;
  // outfile<<"size="<<input_images[0].size()<<" input image="<<input_images[0];
  // outfile<<"\n----------------------------------\n";
  __TIC__(FCOS_RESIZE)
  for (auto i = 0u; i < input_images.size(); i++) {
    image_resize(input_images[i], images[i], sHeight, sWidth);
  }
  __TOC__(FCOS_RESIZE)
  // outfile<<"size="<<images[0].size()<<"after resize image="<<images[0];fcos_post_process
  // outfile<<"\n----------------------------------\n";

  __TIC__(FCOS_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(FCOS_SET_IMG)

  // outfile.close();
  __TIC__(FCOS_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(FCOS_DPU)

  vector<int> cols, rows;
  for (auto& input_image : input_images) {
    cols.push_back(input_image.cols);
    rows.push_back(input_image.rows);
  }

  __TIC__(FCOS_POST_ARM)
  ret = vitis::ai::fcos_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), cols, rows);
  __TOC__(FCOS_POST_ARM)

  return ret;
}

}  // namespace ai
}  // namespace vitis
