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
#include "./solo_imp.hpp"

#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>


extern int GLOBAL_ENABLE_ROUND_SETINPUT;

DEF_ENV_PARAM(DEBUG_SOLO, "0");

using namespace std;
namespace vitis {
namespace ai {
/*
void getBinSize(std::string path)
{
    int size = 0;
    std::ifstream infile(path, std::ifstream::binary);
    
    infile.seekg(0, infile.end);
    int size= infile.tellg();
    infile.seekg(0, infile.beg);
    
    infile.close();
    return size;
}

void readBin(std::string path)
{
  auto size = getBinSize(path);
  std::vector<int8_t> data(size);
  std::ifstream infile(path, std::ifstream::binary);
  infile.read(data.data(), size);
  infile.close();
}
*/

static int find_sub_x1 (const std::string & input , const std::vector<std::vector<vitis::ai::library::InputTensor>>& outputs) {
        auto ret = -1;
        for (auto i = 0u; i < outputs.size(); ++i) {
            for(auto j = 0u; j < outputs[i].size(); ++j) {
                if( outputs[i][j].name.find(input) != std::string::npos) {
                    ret = i;
                    return ret;
                }
            }
        }
        LOG(FATAL) << "cannot found output tensor: " << input;
        return -1;
}
static std::vector<int> find_sub_x (const google::protobuf::RepeatedPtrField<std::string>& input , const std::vector<std::vector<vitis::ai::library::InputTensor>>& outputs) {
        auto ret = std::vector<int>();
        ret.reserve(input.size());
        for(auto i = 0; i < input.size(); ++i) {
            ret.push_back (find_sub_x1 (input[i], outputs));
        }
        return ret;
}

static int find_sub_y1(const std::string &input, const std::vector<std::vector<vitis::ai::library::InputTensor>> &outputs)
{
  auto ret = -1;
  for (auto i = 0u; i < outputs.size(); ++i){
    for (auto j = 0u; j < outputs[i].size(); ++j) {
      if( outputs[i][j].name.find(input) != std::string::npos) {
         ret = j;
         return ret;
      }
    }
  }
  return -1;
}
static std::vector<int> find_sub_y(const google::protobuf::RepeatedPtrField<std::string> &input, 
const std::vector<std::vector<vitis::ai::library::InputTensor>> &outputs)
{
  auto ret = std::vector<int>();
  ret.reserve(input.size());
  for (auto i = 0; i < input.size(); ++i){
    ret.push_back(find_sub_y1(input[i], outputs));
  }
  return ret;
}

/////
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


SoloImp::SoloImp(const std::string& model_name, bool need_preprocess)
    : Solo(model_name, need_preprocess) {
  GLOBAL_ENABLE_ROUND_SETINPUT = 1;
  auto param = configurable_dpu_task_->getConfig().solo_param();
  sub_x_in_ = find_sub_x(param.input_tensor_name(), configurable_dpu_task_->getInputTensor());
  sub_y_in_ = find_sub_y(param.input_tensor_name(), configurable_dpu_task_->getInputTensor());
  sub_x_out_ = find_sub_x(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
  sub_y_out_ = find_sub_y(param.output_tensor_name(), configurable_dpu_task_->getOutputTensor());
  const_input_data_.resize(5);
  float_input_data_.push_back(input0);
  float_input_data_.push_back(input1);
  float_input_data_.push_back(input2);
  float_input_data_.push_back(input3);
  float_input_data_.push_back(input4);

  auto bs = get_input_batch();
  const_input_data_batch_.resize(5);
  for(auto i = 1u; i < sub_y_in_.size(); i++) {
    auto t_size = configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].height * configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].width * configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].channel;
    auto in_scale = vitis::ai::library::tensor_scale(configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]]);
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << i<< " layer_name " << in_scale << " " << configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].name; 
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << i<< " layer_height " << in_scale << " " << configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].height; 
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << i<< " layer_width " << in_scale << " " << configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].width; 
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << i<< " layer_channel " << in_scale << " " << configurable_dpu_task_->getInputTensor()[0][sub_y_in_[i]].channel; 
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO)) << i << " layer size"<< t_size;
    CHECK_EQ(t_size, float_input_data_[i-1].size());
    const_input_data_[i-1].resize(t_size);
    for(auto j = 0u; j < t_size; j++) {
      const_input_data_[i-1][j] = (int8_t)std::min(std::max(((int)std::round(float_input_data_[i-1][j] * in_scale)), -128), 127);
    }
    for (auto k = 0u; k < bs; k++) {
      const_input_data_batch_[i - 1].push_back(const_input_data_[i - 1]);
    }
  }
}

SoloImp::~SoloImp() {}

SoloResult SoloImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  auto size = cv::Size(640, 640);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(SOLO_SET_IMG)
  for(auto i = 1u; i < sub_y_in_.size(); i++) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_SOLO))  << const_input_data_[i - 1].size();
    configurable_dpu_task_->setInputDataArray(const_input_data_[i - 1], sub_y_in_[i]);
  }
  //auto d = (int8_t*)configurable_dpu_task_->getInputTensor()[0][3].get_data(0); 
  configurable_dpu_task_->setInputImageRGB(image, sub_y_in_[0]);
  __TOC__(SOLO_SET_IMG)
  __TIC__(SOLO_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SOLO_DPU)
  __TIC__(SOLO_POST_PROCESS)
  auto ret = vitis::ai::solo_post_process(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0], 
      configurable_dpu_task_->getConfig(),{(int)input_image.cols, (int)input_image.rows}, sub_y_out_, 0);
  __TOC__(SOLO_POST_PROCESS)
  return ret;
}

std::vector<SoloResult> SoloImp::run(const std::vector<cv::Mat>& input_images) {
  vector<cv::Mat> images;
  auto size = cv::Size(640, 640);
  std::vector<std::vector<int>> ori_shape;
  for (auto& input_image : input_images) {
    ori_shape.push_back({(int)input_image.cols, (int)input_image.rows});
    Mat image;
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    images.push_back(image);
  }

    for(auto i = 1u; i < sub_y_in_.size(); i++) {
      configurable_dpu_task_->setInputDataArray(const_input_data_batch_[i - 1], sub_y_in_[i]);
    }
  __TIC__(SOLO_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images, sub_y_in_[0]);
  __TOC__(SOLO_SET_IMG)
  __TIC__(SOLO_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(SOLO_DPU)
  __TIC__(SOLO_POST_PROCESS)
  auto ret = vitis::ai::solo_post_process_batch(
      configurable_dpu_task_->getInputTensor()[0],
      configurable_dpu_task_->getOutputTensor()[0],
      configurable_dpu_task_->getConfig(), ori_shape, sub_y_out_);
  __TOC__(SOLO_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
