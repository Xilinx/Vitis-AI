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

#include "utils.hpp"

#include <glog/logging.h>

#include <cmath>
#include <iostream>

using cv::Mat;
using cv::Size;

namespace vitis {
namespace ai {
namespace rgbdsegmentation {

// pytorch utils img_utils
cv::Mat pad_image_to_shape(const cv::Mat& img, size_t shape, int border_mode,
                           std::vector<size_t>& margin) {
  margin.resize(4);
  size_t img_height = img.rows;
  size_t img_width = img.cols;
  size_t pad_height = (shape > img_height) ? shape - img_height : 0;
  size_t pad_width = (shape > img_width) ? shape - img_width : 0;

  margin[0] = floor(pad_height / 2.0);
  margin[1] = floor(pad_height / 2.0) + pad_height % 2;
  margin[2] = floor(pad_width / 2.0);
  margin[3] = floor(pad_width / 2.0) + pad_width % 2;

  Mat dst;
  cv::copyMakeBorder(img, dst, margin[0], margin[1], margin[2], margin[3],
                     border_mode);

  return dst;
}

//# for rgbd segmentation
void process_image_rgbd(const cv::Mat& img, const cv::Mat& disp,
                        const std::vector<float>& mean,
                        const std::vector<float>& img_scale,
                        const std::vector<float>& disp_scale, int8_t* img_data,
                        int8_t* disp_data) {
  // new m = 255*mean ,new scale=1/255/std*tensor_scale
  CHECK_EQ(img.channels(), 3u)
      << "Currently only 3 channels are implemented. img channels="
      << img.channels();
  CHECK_EQ(disp.channels(), 3u)
      << "Currently only 3 channels are implemented. disp channels="
      << disp.channels();

  int channels = img.channels();
  int rows = img.rows;
  int cols = img.cols;
  const uchar *img_row, *disp_row;
  for (auto h = 0; h < rows; h++) {
    img_row = img.ptr(h);
    disp_row = disp.ptr(h);
    for (auto w = 0; w < cols; w++) {
      for (auto c = 0; c < channels; c++, img_row++, disp_row++) {
        auto didx = h * channels * cols + w * channels + abs(2 - c);

        img_data[didx] =
            std::round((float(img_row[0]) - mean[c]) * img_scale[c]);
        disp_data[didx] =
            std::round((float(disp_row[0]) - mean[c]) * disp_scale[c]);
      }
    }
  }
}
/*
cv::Mat scale_process_rgbdepth(const cv::Mat& img, const cv::Mat& disp,
                               const cv::Size& ori_shape,
                               size_t crop_size,
                               size_t stride_rate,  //, device = Noneconst
                               const std::vector<float>& mean,
                               const std::vector<float>& img_scale,
                               const std::vector<float>& disp_scale,
                               float output_scale) {
  size_t new_rows = img.rows;
  size_t new_cols = img.cols;
  size_t class_num = 40;  //输入参数 待确认
  std::vector<size_t> margin;
  auto disp_pad =
      pad_image_to_shape(disp, crop_size, cv::BORDER_CONSTANT, margin);
  auto img_pad =
      pad_image_to_shape(img, crop_size, cv::BORDER_CONSTANT, margin);
  std::vector<size_t> score_size = {img.rows, img.cols, class_num};
  auto drows = img.cols;
  auto dcols = class_num;
  Mat score = cv::Mat::zeros(3, score_size.data(), CV_32F);
  float* pscore = (float*)score.data;

  size_t long_size = (new_cols > new_rows) ? new_cols : new_rows;
  if (long_size <= crop_size) {
    std::vector<int8_t> img_data, disp_data;
    process_image_rgbd(img_pad, disp_pad, mean, img_scale, disp_scale,
                       img_data.data(), disp_data.data());
    int8_t* temp_score =
        img_data.data();  //= dpurun(input_data, input_disp, device);  //
                          // exp(xi)&后处理
    for (auto c = 0; c < class_num; c++) {
      for (size_t h = 0; h < img.rows; h++) {
        for (auto w = 0; w < img.cols; w++) {
          pscore[h * drows * dcols + w * dcols + c] +=
              std::exp(temp_score[c * crop_size * crop_size +
                                  (h + margin[0]) * crop_size + w + margin[2]] *
                       output_scale);
        }
      }
    }
  } else {
    size_t stride = std::ceil(crop_size * stride_rate);

    auto pad_rows = img_pad.rows;
    auto pad_cols = img_pad.cols;
    size_t r_grid = size_t(std::ceil((pad_rows - crop_size) / stride)) + 1;
    size_t c_grid = size_t(std::ceil((pad_cols - crop_size) / stride)) + 1;
    std::vector<size_t> dist_size = {pad_rows, pad_cols, class_num};
    Mat data_scale = cv::Mat::zeros(3, dist_size.data(), CV_32F);

    for (size_t grid_yidx = 0; grid_yidx < r_grid; grid_yidx++) {
      for (size_t grid_xidx = 0; grid_xidx < c_grid; grid_xidx++) {
        size_t s_x = min(grid_xidx * stride, pad_cols - crop_size);
        size_t s_y = min(grid_yidx * stride, pad_rows - crop_size);
        std::vector<int8_t> img_data, disp_data;
        process_image_rgbd(img_pad(cv::Rect(s_x, s_y, crop_size, crop_size)),
                           disp_pad(cv::Rect(s_x, s_y, crop_size, crop_size)),
                           mean, img_scale, disp_scale, img_data.data(),
                           disp_data.data());
        int8_t* temp_score =
            img_data.data();  //= dpurun(input_data, input_disp, device);  //
                              // exp(xi)&后处理
        for (auto c = 0; c < class_num; c++) {
          for (size_t h = s_y, th = 0; th < crop_size; h++, th++) {
            for (size_t w = s_x, tw = 0; tw < crop_size; w++, tw++) {
              if (h < margin[0] || h > pad_rows - margin[1]  //
                  || w < margin[2] || w > pad_cols - margin[3]) {
                continue;
              }
              pscore[(h - margin[0]) * drows * dcols + (w - margin[2]) * dcols +
                     c] += std::exp(temp_score[c * crop_size * crop_size +
                                               th * crop_size + tw] *
                                    output_scale);
            }
          }
        }
      }
    }
  }
  // resize
  Mat data_output;
  cv::resize(score, data_output, ori_shape, 0, 0, cv::INTER_LINEAR);
  return data_output;
}
*/
}  // namespace rgbdsegmentation
}  // namespace ai
}  // namespace vitis
