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
#pragma once
#include <eigen3/Eigen/Dense>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  // vector<uchar> te(imsize, 2);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar*>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar*>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}

static cv::Mat process_result(cv::Mat& m1,
                              const vitis::ai::MultiTaskResult& result,
                              bool is_jpeg) {
  cv::Mat image;
  cv::resize(m1, image, result.segmentation.size());
  // for (auto row_ind = 0; row_ind < image.size().height; ++row_ind) {
  //   for (auto col_ind = 0; col_ind < image.size().width; ++col_ind) {
  //     image.at<cv::Vec3b>(row_ind, col_ind) =
  //         image.at<cv::Vec3b>(row_ind, col_ind) * 0.5 +
  //         result.segmentation.at<cv::Vec3b>(row_ind, col_ind) * 0.5;
  //   }
  // }
  overLay1(image, result.segmentation);
  for (auto& r : result.vehicle) {
    LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
                          << r.width << " " << r.height << " " << r.angle;
    int xmin = r.x * result.segmentation.cols;
    int ymin = r.y * result.segmentation.rows;

    int width = r.width * result.segmentation.cols;
    int height = r.height * result.segmentation.rows;
    cv::rectangle(image, cv::Rect_<int>(xmin, ymin, width, height),
                  cv::Scalar(185, 181, 178), 2, 1, 0);
  }

  return image;
}
