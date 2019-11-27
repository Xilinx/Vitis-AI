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
#include <vector>
#include "./roadline_imp.hpp"
#include <xilinx/ai/profiling.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace std;
using namespace cv;
namespace  xilinx {
namespace  ai {

RoadLineImp::RoadLineImp(const std::string& model_name, bool need_preprocess)
    : xilinx::ai::TConfigurableDpuTask<RoadLine>(model_name, need_preprocess),
	model_name_(model_name),
        processor_{xilinx::ai::RoadLinePostProcess::create(
        configurable_dpu_task_->getInputTensor()[0], configurable_dpu_task_->getOutputTensor()[0],
        configurable_dpu_task_->getConfig())} {
   }

RoadLineImp::~RoadLineImp() {}

RoadLineResult RoadLineImp::run(const cv::Mat& input_image)
{
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }

/*  if(model_name_ == "roadline"){
      if (size != input_image.size()) {
        cv::resize(input_image, image, size);
      } else {
        image = input_image;
      }
  }
  else if(model_name_ == "roadline_deephi"){
    int resize_w = configurable_dpu_task_->getConfig().roadline_dp_param().resize_w();
    int resize_h = configurable_dpu_task_->getConfig().roadline_dp_param().resize_h();
    int crop_x = configurable_dpu_task_->getConfig().roadline_dp_param().crop_x();
    int crop_y = configurable_dpu_task_->getConfig().roadline_dp_param().crop_y();
    int crop_w = configurable_dpu_task_->getConfig().roadline_dp_param().crop_w();
    int crop_h = configurable_dpu_task_->getConfig().roadline_dp_param().crop_h();

    if(input_image.size() == size)
      image = input_image;
    else if(input_image.size() == Size(crop_w, crop_h))
      cv::resize(input_image, image, size);
    else if(input_image.size() != Size(crop_w, crop_h)) {
      Mat img_res0, image_crop;
      cv::resize(input_image, img_res0, Size(resize_w, resize_h));
      image_crop = img_res0(Rect(crop_x, crop_y, crop_w, crop_h));
      cv::resize(image_crop, image, size);
    }
  }
*/
  __TIC__(ROADLINE_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(ROADLINE_SET_IMG)
  __TIC__(ROADLINE_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(ROADLINE_DPU)
  __TIC__(ROADLINE_POST_PROCESS)
  auto results = processor_->road_line_post_process(input_image.cols, input_image.rows );
  __TOC__(ROADLINE_POST_PROCESS)
  return results;
}

}
}

