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

#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vitis/ai/demo_bx.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/ssd.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/facedetect.hpp>
using namespace std;
using namespace cv; 

#ifndef HAVE_EIGEN
#define HAVE_EIGEN 0
#endif
#if HAVE_EIGEN
#include <eigen3/Eigen/Dense>
// Overlay the original image with the result
// Eigen Optimized version
static void overLay1(cv::Mat &src1, const cv::Mat &src2) {
  const int imsize = src1.cols * src2.rows * 3;
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar *>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar *>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}
#else
// c version
static void overLay1(cv::Mat &src1, const cv::Mat &src2) {
  const int imsize = src1.cols * src2.rows * 3;
  for (int i = 0; i < imsize; ++i) {
    src1.data[i] = src1.data[i] / 2 + src2.data[i] / 2;
  }
}
#endif

// This function is used to process the multitask result and show on the image
static std::vector<cv::Mat> process_result_multitask(
    std::vector<cv::Mat> &m1s, const std::vector<vitis::ai::MultiTaskResult> &results, bool is_jpeg) {
  (void)process_result_multitask;
  std::vector<cv::Mat> images(m1s.size());
  for (auto i = 0u; i < m1s.size(); i++) {
    cv::Mat m1 = m1s[i];
    auto result = results[i];
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
    for (auto &r : result.vehicle) {
      LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
                          << r.width << " " << r.height << " " << r.angle;
      int xmin = r.x * image.cols;
      int ymin = r.y * image.rows;

      int width = r.width * image.cols;
      int height = r.height * image.rows;
      cv::rectangle(image, cv::Rect_<int>(xmin, ymin, width, height),
                   cv::Scalar(185, 181, 178), 2, 1, 0);
    }
    images[i] = image;
  }
  return images;
}

int main(int argc, char *argv[]) {
  // set the layout
  //
  int seg_px = 0; //100 his
  int seg_py = 0; //252 his
  // assign to Lvalue : static std::vector<cv::Rect> rects, the coordinates of
  // each window
  gui_layout() = {{seg_px, seg_py, 512, 288},
                  {seg_px + 512, seg_py, 512, 288},
                  {seg_px, seg_py + 288*2, 512, 288},
                  {seg_px + 1024, seg_py + 288*2, 512, 288},
  		};
//  gui_layout() = {{seg_px, seg_py, 512, 288},
//                  {seg_px + 1024, seg_py, 512, 288},
//                  {seg_px, seg_py + 288*2, 512, 288},
//                  {seg_px + 1024, seg_py + 288*2, 512, 288},
//                  {1024, 0, 640, 480},
//  		  {1024, 480, 640, 360},
//  		};
  // init each dpu filter and process instance, using video demo framework
  //return vitis::ai::main_for_video_demo_multiple_channel(
  if (!ENV_PARAM(ENABLE_MULTI_BATCH)) {
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
      });

  } else {
    return vitis::ai::main_for_video_demo_batch(
      argc, argv,
      {
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
                process_result_multitask);
          },
      });
  }
}
