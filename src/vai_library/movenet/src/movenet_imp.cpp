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
#include "./movenet_imp.hpp"

#include <glog/logging.h>
#include <math.h>
#include <numeric>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/profiling.hpp>

using namespace std;
using namespace cv;
namespace vitis {
namespace ai {

MovenetImp::MovenetImp(const std::string& model_name, bool need_preprocess)
    : vitis::ai::TConfigurableDpuTask<Movenet>(model_name, need_preprocess) {}

MovenetImp::~MovenetImp() {}

static float sigmoid(float p) { return 1.0 / (1 + exp(-p * 1.0)); }
static vector<int> getMaxPoint(vector<float> center, vector<float> weights,
                               int width = 48) {
  int cx = -1, cy = -1;
  float max = -100.0f;
  for (auto i = 0u; i < weights.size(); ++i) {
    float val = center[i] * weights[i];
    if (val > max) {
      cy = i / width;
      cx = i % width;
      max = val;
    }
  }
  return vector<int>{cx, cy};
}
static vector<int> getMaxPoint(vector<float> center, int width = 48) {
  int cx = -1, cy = -1;
  float max = -100.0f;
  for (auto i = 0u; i < center.size(); ++i) {
    float val = center[i];
    if (val > max) {
      cy = i / width;
      cx = i % width;
      max = val;
    }
  }
  return vector<int>{cx, cy};
}

MovenetResult movenet_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors_unsorted,
    const vitis::ai::proto::DpuModelParam& config, int iw_in, int ih_in,
    size_t batch_idx) {
  int sWidth = input_tensors[0][0].width;
  int sHeight = input_tensors[0][0].height;

  const float hm_th = config.movenet_param().conf_threshold();
  auto config_weights =
      std::vector<float>(config.movenet_param().center_weight().begin(),
                         config.movenet_param().center_weight().end());
  auto layername =
      std::vector<std::string>(config.movenet_param().layer_name().begin(),
                               config.movenet_param().layer_name().end());
  std::vector<vitis::ai::library::OutputTensor> output_tensors;
  for (auto i = 0u; i < layername.size(); i++) {
    for (auto j = 0u; j < output_tensors_unsorted[0].size(); j++) {
      if (output_tensors_unsorted[0][j].name.find(layername[i]) !=
          std::string::npos) {
        output_tensors.push_back(output_tensors_unsorted[0][j]);
        break;
      }
    }
  }

  auto* data_heatmap =
      (int8_t*)output_tensors[0].get_data(batch_idx);  // heatmap 48x48x17
  auto* data_center =
      (int8_t*)output_tensors[1].get_data(batch_idx);  // center 48x48
  auto* data_reg =
      (int8_t*)output_tensors[2].get_data(batch_idx);  // ret 48x48x34
  auto* data_offset =
      (int8_t*)output_tensors[3].get_data(batch_idx);  // offset 48x48x34
  float scale_heatmap = vitis::ai::library::tensor_scale(output_tensors[0]);
  float scale_center = vitis::ai::library::tensor_scale(output_tensors[1]);
  float scale_reg = vitis::ai::library::tensor_scale(output_tensors[2]);
  float scale_offset = vitis::ai::library::tensor_scale(output_tensors[3]);
  int channel = output_tensors[0].channel;
  int width = output_tensors[0].width;
  int height = output_tensors[0].height;
  auto size = width * height;
  float float_data_heatmap[size * channel];
  vector<float> float_data_center(size);
  float float_data_reg[size * channel * 2];
  float float_data_offset[size * channel * 2];
  // hwc -> chw int8->float
  for (int ih = 0; ih < height; ++ih) {
    for (int iw = 0; iw < width; ++iw) {
      for (int ic = 0; ic < channel; ++ic) {
        int offset = ic * width * height + ih * width + iw;
        float val = data_heatmap[ih * width * channel + iw * channel + ic] *
                    scale_heatmap;
        float_data_heatmap[offset] = sigmoid(val);
      }
    }
  }
  for (int ih = 0; ih < height; ++ih) {
    for (int iw = 0; iw < width; ++iw) {
      for (int ic = 0; ic < 2 * channel; ++ic) {
        int offset = ic * width * height + ih * width + iw;
        float_data_reg[offset] =
            data_reg[ih * width * 2 * channel + iw * 2 * channel + ic] *
            scale_reg;
        float_data_offset[offset] =
            data_offset[ih * width * 2 * channel + iw * 2 * channel + ic] *
            scale_offset;
      }
    }
  }
  for (int i = 0; i < size; ++i) {
    float val = data_center[i] * scale_center;
    float_data_center[i] = sigmoid(val);
  }
  std::vector<cv::Point2f> poses;
  auto maxPoint = getMaxPoint(float_data_center, config_weights);
  vector<int32_t> range_width(width);
  std::iota(range_width.begin(), range_width.end(), 0);  // form 0 to 47
  for (auto i = 0; i < channel; ++i) {
    int start = size * i;
    int start_x = size * 2 * i;
    int start_y = size * 2 * i + size;
    auto reg_x_ori =
        (float_data_reg[start_x + width * maxPoint[1] + maxPoint[0]] + 0.5);
    auto reg_y_ori =
        (float_data_reg[start_y + width * maxPoint[1] + maxPoint[0]] + 0.5);
    auto reg_x = reg_x_ori + maxPoint[0];
    auto reg_y = reg_y_ori + maxPoint[1];
    vector<int> map_reg_x(width);
    vector<int> map_reg_y(width);
    for (auto iw = 0; iw < width; ++iw) {
      map_reg_x[iw] = (range_width[iw] - reg_x) * (range_width[iw] - reg_x);
      map_reg_y[iw] = (range_width[iw] - reg_y) * (range_width[iw] - reg_y);
    }
    vector<float> float_reg_map(width * width);
    for (auto iy = 0; iy < width; ++iy) {
      for (auto ix = 0; ix < width; ++ix) {
        auto val = sqrt(map_reg_x[ix] + map_reg_y[iy]) + 1.8;
        float_reg_map[iy * width + ix] = val;
      }
    }
    vector<float> float_tem_reg(width * width);
    for (auto idx = 0; idx < size; ++idx) {
      float_tem_reg[idx] = float_data_heatmap[start + idx] / float_reg_map[idx];
    }
    auto reg_max_point = getMaxPoint(float_tem_reg);
    auto score =
        float_data_heatmap[start + reg_max_point[1] * width + reg_max_point[0]];
    auto offset_x = float_data_offset[start_x + reg_max_point[1] * width +
                                      reg_max_point[0]];
    auto offset_y = float_data_offset[start_y + reg_max_point[1] * width +
                                      reg_max_point[0]];
    auto ret_x = (reg_max_point[0] + offset_x) / width;
    auto ret_y = (reg_max_point[1] + offset_y) / width;
    if (score < hm_th) {
      ret_x = -1;
      ret_y = -1;
    }
    poses.push_back(Point2f(ret_x * iw_in, ret_y * ih_in));
  }
  MovenetResult result{sWidth, sHeight, poses};
  return result;
}

std::vector<MovenetResult> movenet_post_process(
    const std::vector<std::vector<vitis::ai::library::InputTensor>>&
        input_tensors,
    const std::vector<std::vector<vitis::ai::library::OutputTensor>>&
        output_tensors,
    const vitis::ai::proto::DpuModelParam& config, vector<int> ws,
    vector<int> hs) {
  auto batch = input_tensors[0][0].batch;
  auto ret = std::vector<MovenetResult>{};
  ret.reserve(batch);
  for (auto i = 0u; i < batch; i++) {
    ret.emplace_back(movenet_post_process(input_tensors, output_tensors, config,
                                          ws[i], hs[i], i));
  }
  return ret;
}

MovenetResult MovenetImp::run(const cv::Mat& input_image) {
  cv::Mat image;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  __TIC__(MOVENET_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(image);
  __TOC__(MOVENET_SET_IMG)
  __TIC__(MOVENET_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MOVENET_DPU)
  __TIC__(MOVENET_POST_PROCESS)
  auto ret =
      vitis::ai::movenet_post_process(configurable_dpu_task_->getInputTensor(),
                                      configurable_dpu_task_->getOutputTensor(),
                                      configurable_dpu_task_->getConfig(),
                                      input_image.cols, input_image.rows, 0);
  __TOC__(MOVENET_POST_PROCESS)
  return ret;
}
vector<MovenetResult> MovenetImp::run(const vector<cv::Mat>& input_images) {
  vector<cv::Mat> images;
  vector<int> ws, hs;
  int sWidth = getInputWidth();
  int sHeight = getInputHeight();
  auto size = cv::Size(sWidth, sHeight);
  for (auto& input_image : input_images) {
    Mat image;
    if (size != input_image.size()) {
      cv::resize(input_image, image, size);
    } else {
      image = input_image;
    }
    images.push_back(image.clone());
    ws.push_back(input_image.cols);
    hs.push_back(input_image.rows);
  }
  __TIC__(MOVENET_SET_IMG)
  configurable_dpu_task_->setInputImageRGB(images);
  __TOC__(MOVENET_SET_IMG)
  __TIC__(MOVENET_DPU)
  configurable_dpu_task_->run(0);
  __TOC__(MOVENET_DPU)
  __TIC__(MOVENET_POST_PROCESS)
  auto ret = vitis::ai::movenet_post_process(
      configurable_dpu_task_->getInputTensor(),
      configurable_dpu_task_->getOutputTensor(),
      configurable_dpu_task_->getConfig(), ws, hs);
  __TOC__(MOVENET_POST_PROCESS)
  return ret;
}

}  // namespace ai
}  // namespace vitis
