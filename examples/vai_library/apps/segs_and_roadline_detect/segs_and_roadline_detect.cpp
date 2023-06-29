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

#include <glog/logging.h>

#include <eigen3/Eigen/Dense>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_b1.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/ssd.hpp>

extern std::vector<cv::Rect> GLOBAL_SET_RECT_MANUAL;

// Overlay the original image with the result
// Eigen Optimized version
static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar*>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar*>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}

// This function is used to process the multitask result and show on the image
static cv::Mat process_result_multitask(
    cv::Mat& m1, const vitis::ai::MultiTaskResult& result, bool is_jpeg) {
  (void)process_result_multitask;
  cv::Mat image;
  // Overlay segmentation result to the original image
  if (false) {
    cv::resize(m1, image, result.segmentation.size());
    overLay1(image, result.segmentation);
  } else {
    cv::resize(result.segmentation, image, m1.size());
    overLay1(image, m1);
  }
  // Draw detection results
  for (auto& r : result.vehicle) {
    LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
                          << r.width << " " << r.height << " " << r.angle;
    int xmin = r.x * image.cols;
    int ymin = r.y * image.rows;

    int width = r.width * image.cols;
    int height = r.height * image.rows;
    cv::rectangle(image, cv::Rect_<int>(xmin, ymin, width, height),
                  cv::Scalar(185, 181, 178), 2, 1, 0);
  }
  return image;
}

std::vector<cv::Mat> process_result_multitask_batch(
    std::vector<cv::Mat>& images,
    const std::vector<vitis::ai::MultiTaskResult>& results, bool is_jpeg) {
  size_t size = std::min(images.size(), results.size());
  std::vector<cv::Mat> image_results(size);
  for (auto i = 0u; i < size; ++i) {
    image_results[i] = process_result_multitask(images[i], results[i], is_jpeg);
  }
  return image_results;
}

using namespace cv;

// This function is used to process the roadline result and show on the image
cv::Mat process_result_roadline(cv::Mat& image,
                                const vitis::ai::RoadLineResult& result,
                                bool is_jpeg) {
  std::vector<int> color1 = {0, 255, 0, 0, 100, 255};
  std::vector<int> color2 = {0, 0, 255, 0, 100, 255};
  std::vector<int> color3 = {0, 0, 0, 255, 100, 255};

  LOG_IF(INFO, is_jpeg) << "lines.size " << result.lines.size() << " ";
  for (auto& line : result.lines) {
    LOG_IF(INFO, is_jpeg) << "line.points_cluster.size() "
                          << line.points_cluster.size() << " ";
    std::vector<cv::Point> points_poly = line.points_cluster;
    int type = line.type < 5 ? line.type : 5;
    if (type == 2 && points_poly[0].x < image.rows * 0.5) continue;
    cv::polylines(image, points_poly, false,
                  cv::Scalar(color1[type], color2[type], color3[type]), 3,
                  cv::LINE_AA, 0);
  }
  return image;
}

std::vector<cv::Mat> process_result_roadline_batch(
    std::vector<cv::Mat>& images,
    const std::vector<vitis::ai::RoadLineResult>& results, bool is_jpeg) {
  size_t size = std::min(images.size(), results.size());
  std::vector<cv::Mat> image_results(size);
  for (auto i = 0u; i < size; ++i) {
    image_results[i] = process_result_roadline(images[i], results[i], is_jpeg);
  }
  return image_results;
}

int main(int argc, char* argv[]) {
  // set the layout
  //
  int seg_px = 100;
  int seg_py = 252;
  // assign to Lvalue : static std::vector<cv::Rect> rects, the coordinates of
  // each window
  gui_layout() = {{seg_px, seg_py, 512, 288},
                  {seg_px + 512, seg_py, 512, 288},
                  {seg_px, seg_py + 288, 512, 288},
                  {seg_px + 512, seg_py + 288, 512, 288},
                  {1200, 252, 640, 480}};

  // init each dpu filter and process instance, using video demo framework
  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv,
      {
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
                process_result_multitask);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
                process_result_multitask);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
                process_result_multitask);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
                process_result_multitask);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] {
                  return vitis::ai::RoadLine::create("vpgnet_pruned_0_99");
                },
                process_result_roadline);
          },
      });
}
