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
#include "./platenum_imp.hpp"

#include <iostream>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/globalavepool.hpp>
#include <vitis/ai/image_util.hpp>
#include <vitis/ai/profiling.hpp>
#include <vitis/softmax.hpp>

using namespace std;
namespace vitis {
namespace ai {


static int find_sub_x1 (const std::string & output , const std::vector<std::vector<vitis::ai::library::OutputTensor>>& outputs) {
        auto ret = -1;
        for (auto i = 0u; i < outputs.size(); ++i) {
            for(auto j = 0u; j < outputs[i].size(); ++j) {
                if( outputs[i][j].name.find( output) != std::string::npos) {
                    ret = i;
                    return ret;
                }
            }
        }
        LOG(FATAL) << "cannot found output tensor: " << output;
        return -1;
}
static std::vector<int> find_sub_x (const google::protobuf::RepeatedPtrField<std::string>& output , const std::vector<std::vector<vitis::ai::library::OutputTensor>>& outputs) {
        auto ret = std::vector<int>();
        ret.reserve(output.size());
        for(auto i = 0; i < output.size(); ++i) {
            ret.push_back (find_sub_x1 (output[i], outputs));
        }
        return ret;
}

static int find_sub_y1(const std::string &output, const std::vector<std::vector<vitis::ai::library::OutputTensor>> &outputs)
{
  auto ret = -1;
  for (auto i = 0u; i < outputs.size(); ++i){
    for (auto j = 0u; j < outputs[i].size(); ++j) {
      if( outputs[i][j].name.find( output) != std::string::npos) {
         ret = j;
         return ret;
      }
    }
  }
  return -1;
}
static std::vector<int> find_sub_y(const google::protobuf::RepeatedPtrField<std::string> &output, 
const std::vector<std::vector<vitis::ai::library::OutputTensor>> &outputs)
{
  auto ret = std::vector<int>();
  ret.reserve(output.size());
  for (auto i = 0; i < output.size(); ++i){
    ret.push_back(find_sub_y1(output[i], outputs));
  }
  return ret;
}

static void check_all_zero(vector<int> & array) {
  size_t count = 0;
  for (auto &a: array) {
    if(a == 0) {
      count++;
    }
  }
  if(count == array.size()) {
    array = {0};
  }
}

PlateNumImp::PlateNumImp(const std::string &model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PlateNum>(model_name, need_preprocess) {
  auto param = configurable_dpu_task_->getConfig().platenum_param();
  sub_x_ = find_sub_x(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
  sub_y_ = find_sub_y(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
  check_all_zero(sub_y_);
}
PlateNumImp::PlateNumImp(const std::string &model_name, xir::Attrs *attrs,bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<PlateNum>(model_name,attrs, need_preprocess) {
  auto param = configurable_dpu_task_->getConfig().platenum_param();
  sub_x_ = find_sub_x(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
  sub_y_ = find_sub_y(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
}

PlateNumImp::~PlateNumImp() {
  sub_x_.clear();        
  sub_y_.clear();        
}

PlateNumResult PlateNumImp::run(const cv::Mat &input_image) {
  cv::Mat image;
  auto size = cv::Size(configurable_dpu_task_->getInputTensor()[0][0].width,
                       configurable_dpu_task_->getInputTensor()[0][0].height);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size, 0, 0, cv::INTER_LINEAR);
  } else {
    image = input_image;
  }
  __TIC__(PLATENUM_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(image);
  __TOC__(PLATENUM_SET_IMG)

  __TIC__(PLATENUM_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATENUM_DPU)
  __TIC__(PLATENUM_AVG)
  vitis::ai::globalAvePool(
      (int8_t *)configurable_dpu_task_->getOutputTensor()[0][0].get_data(0),
      1024, 9, 3,
      (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(0), 4);
  for (size_t k = 0; k < sub_x_.size() - 1; k++) {
    memcpy((int8_t *)configurable_dpu_task_->getInputTensor()[sub_x_[k]][0].get_data(0), 
    (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(0), sizeof(int8_t) * 1024);
  }
  __TOC__(PLATENUM_AVG)
  __TIC__(PLATENUM_DPU2)
  for (size_t k = 0; k < sub_x_.size(); k++) {
    configurable_dpu_task_->run(sub_x_[k]);
  }
  __TOC__(PLATENUM_DPU2)
  return plate_num_post_process(configurable_dpu_task_->getInputTensor(),
                                configurable_dpu_task_->getOutputTensor(), sub_x_, sub_y_)[0];
}

std::vector<PlateNumResult> PlateNumImp::run(const std::vector<cv::Mat> &imgs) {
  std::vector<cv::Mat> images;
  auto size = cv::Size(getInputWidth(), getInputHeight());
  for (auto i = 0u; i < imgs.size(); i++) {
    if (size != imgs[i].size()) {
      cv::Mat img;
      cv::resize(imgs[i], img, size, 0);
      images.push_back(img);
    } else {
      images.push_back(imgs[i]);
    }
  }
  __TIC__(PLATENUM_SET_IMG)
  configurable_dpu_task_->setInputImageBGR(images);
  __TOC__(PLATENUM_SET_IMG)

  __TIC__(PLATENUM_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(PLATENUM_DPU)
  __TIC__(PLATENUM_AVG)
  auto batch_size =configurable_dpu_task_->getInputTensor()[0][0].batch;
  for (auto batch_ind = 0u; batch_ind < batch_size; batch_ind++) {
    vitis::ai::globalAvePool(
      (int8_t *)configurable_dpu_task_->getOutputTensor()[0][0].get_data(batch_ind),
      1024, 9, 3,
      (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(batch_ind), 4);

    for (size_t k = 0; k < sub_x_.size() - 1; k++) {
      memcpy((int8_t *)configurable_dpu_task_->getInputTensor()[sub_x_[k]][0].get_data(batch_ind), 
      (int8_t *)configurable_dpu_task_->getInputTensor()[1][0].get_data(batch_ind), sizeof(int8_t) * 1024);
    }
  }
  __TOC__(PLATENUM_AVG)
  __TIC__(PLATENUM_DPU2)
  for (size_t k = 0; k < sub_x_.size(); k++) {
    configurable_dpu_task_->run(sub_x_[k]);
  }
  __TOC__(PLATENUM_DPU2)
  return plate_num_post_process(configurable_dpu_task_->getInputTensor(),
                                configurable_dpu_task_->getOutputTensor(), sub_x_, sub_y_);
}

}  // namespace ai
}  // namespace vitis
