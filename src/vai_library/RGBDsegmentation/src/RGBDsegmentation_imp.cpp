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
#include "RGBDsegmentation_imp.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <vitis/ai/profiling.hpp>

#include "utils.hpp"
DEF_ENV_PARAM(DEBUG_RGBDSEGMENTATION, "0")

using namespace std;
namespace vitis {
namespace ai {

RGBDsegmentationImp::RGBDsegmentationImp(const string& model_name,
                                         bool need_preprocess)
    : RGBDsegmentation(model_name, need_preprocess) {}

RGBDsegmentationImp::~RGBDsegmentationImp() {}

static vector<float> get_means(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.mean().begin(), c.mean().end());
}
static vector<float> get_scales(const vitis::ai::proto::DpuKernelParam& c) {
  return vector<float>(c.scale().begin(), c.scale().end());
}

SegmentationResult RGBDsegmentationImp::run(const cv::Mat& image_bgr,
                                            const cv::Mat& image_hha) {
  cv::Mat image;
  size_t crop_size = configurable_dpu_task_->getInputWidth();

  vector<size_t> margin;
  auto disp_pad = vitis::ai::rgbdsegmentation::pad_image_to_shape(
      image_hha, crop_size, cv::BORDER_CONSTANT, margin);
  auto img_pad = vitis::ai::rgbdsegmentation::pad_image_to_shape(
      image_bgr, crop_size, cv::BORDER_CONSTANT, margin);

  size_t image_height = image_bgr.rows;
  size_t image_width = image_bgr.cols;
  bool is_le_crop_size =
      (image_height <= crop_size) && (image_width <= crop_size);
  vector<pair<size_t, size_t>> img_map;
  // create img map
  if (is_le_crop_size) {
    img_map.emplace_back(0, 0);
  } else {
    float stride_rate = 2.0 / 3.0;

    size_t pad_height = img_pad.rows;
    size_t pad_width = img_pad.cols;
    size_t stride = ceil(crop_size * stride_rate);

    size_t r_grid = ceil(float(pad_height - crop_size) / float(stride)) + 1;
    size_t c_grid = ceil(float(pad_width - crop_size) / float(stride)) + 1;

    for (size_t grid_yidx = 0; grid_yidx < r_grid; grid_yidx++) {
      for (size_t grid_xidx = 0; grid_xidx < c_grid; grid_xidx++) {
        size_t s_x = min(grid_xidx * stride, pad_width - crop_size);
        size_t s_y = min(grid_yidx * stride, pad_height - crop_size);
        img_map.emplace_back(s_x, s_y);
      }
    }
  }
  // run
  auto input_tensor = configurable_dpu_task_->getInputTensor()[0];
  auto mean = get_means(configurable_dpu_task_->getConfig().kernel(0));
  float img_fixed_scale = tensor_scale(input_tensor[0]);
  float disp_fixed_scale = tensor_scale(input_tensor[1]);
  auto img_scale = get_scales(configurable_dpu_task_->getConfig().kernel(0));
  auto disp_scale = get_scales(configurable_dpu_task_->getConfig().kernel(0));
  for (auto& s : img_scale) {
    s *= img_fixed_scale;
  }
  for (auto& s : disp_scale) {
    s *= disp_fixed_scale;
  }
  size_t batch = configurable_dpu_task_->get_input_batch();

  size_t score_height = image_height;
  size_t score_width = image_width;
  size_t score_channels =
      configurable_dpu_task_->getOutputTensor()[0][0].channel;
  size_t score_stride = score_width * score_channels;
  vector<float> score(score_height * score_stride, 0);

  size_t output_height = configurable_dpu_task_->getOutputTensor()[0][0].height;
  size_t output_width = configurable_dpu_task_->getOutputTensor()[0][0].width;
  size_t output_channels = score_channels;
  size_t output_stride = output_width * score_channels;
  float output_scale =
      tensor_scale(configurable_dpu_task_->getOutputTensor()[0][0]);
  for (size_t i = 0; i < img_map.size(); i += batch) {
    // copy input
    for (size_t j = i; j < img_map.size() && j - i < batch; j++) {
      auto img_data = (int8_t*)input_tensor[0].get_data(j - i);
      auto disp_data = (int8_t*)input_tensor[1].get_data(j - i);
      LOG_IF(INFO, ENV_PARAM(DEBUG_RGBDSEGMENTATION))
          << "j: " << j << " Rect x,y : " << img_map[j].first << " "
          << img_map[j].second;

      vitis::ai::rgbdsegmentation::process_image_rgbd(
          img_pad(cv::Rect(img_map[j].first, img_map[j].second, crop_size,
                           crop_size)),
          disp_pad(cv::Rect(img_map[j].first, img_map[j].second, crop_size,
                            crop_size)),
          mean, img_scale, disp_scale, img_data, disp_data);
      if (ENV_PARAM(DEBUG_RGBDSEGMENTATION)) {
        ofstream(to_string(j) + "_debug_rgbdsegmentation_" + to_string(j - i) +
                     "_input_0.bin",
                 ios::out | ios::binary)
            .write((char*)img_data, crop_size * crop_size * 3);
        ofstream(to_string(j) + "_debug_rgbdsegmentation_" + to_string(j - i) +
                     "_input_1.bin",
                 ios::out | ios::binary)
            .write((char*)disp_data, crop_size * crop_size * 3);
      }
    }

    __TIC__(RGBDsegmentation_DPU)
    configurable_dpu_task_->run(0);
    __TOC__(RGBDsegmentation_DPU);

    // copy output
    for (size_t j = i; j < img_map.size() && j - i < batch; j++) {
      int8_t* output_data =
          (int8_t*)configurable_dpu_task_->getOutputTensor()[0][0].get_data(j -
                                                                            i);
      if (ENV_PARAM(DEBUG_RGBDSEGMENTATION)) {
        ofstream(to_string(j) + "_debug_rgbdsegmentation_" + to_string(j - i) +
                     "_output.bin",
                 ios::out | ios::binary)
            .write((char*)output_data, output_stride * output_height);
      }
      for (size_t h = img_map[j].second, th = 0; th < output_height;
           h++, th++) {
        for (size_t w = img_map[j].first, tw = 0; tw < output_width;
             w++, tw++) {
          if (h < margin[0] || h > score_height + margin[0]  //
              || w < margin[2] || w > score_width + margin[2]) {
            continue;
          }
          auto sidx =
              (h - margin[0]) * score_stride + (w - margin[2]) * score_channels;
          auto oidx = th * output_stride + tw * output_channels;

          for (size_t c = 0; c < output_channels; c++) {
            score[sidx + c] += float(output_data[oidx + c]) * output_scale;
          }
        }
      }
    }
  }

  // post processing

  cv::Mat segMat(image_bgr.size(), CV_8UC1);
  LOG_IF(INFO, ENV_PARAM(DEBUG_RGBDSEGMENTATION))
      << "score_height: " << score_height << " score_width : " << score_width;
  for (size_t h = 0; h < score_height; h++) {
    for (size_t w = 0; w < score_width; w++) {
      auto begin = score.data() + h * score_stride + w * score_channels;
      auto max_ind = max_element(begin, begin + score_channels);
      segMat.at<uchar>(h, w) = distance(begin, max_ind);
    }
  }

  return SegmentationResult{image_bgr.cols, image_bgr.rows, segMat};
}

}  // namespace ai
}  // namespace vitis
