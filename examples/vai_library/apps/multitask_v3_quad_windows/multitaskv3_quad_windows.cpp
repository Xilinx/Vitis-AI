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
#include <vitis/ai/multitaskv3.hpp>

static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  // vector<uchar> te(imsize, 2);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar*>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar*>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}

static void overLay2(cv::Mat& src1, const cv::Mat& src2, uint8_t ig_value) {
  for (auto row_ind = 0; row_ind < src1.rows; row_ind++) {
    for (auto col_ind = 0; col_ind < src1.cols; col_ind++) {
      if (src1.at<cv::Vec3b>(row_ind, col_ind)[0] == ig_value) {
        src1.at<cv::Vec3b>(row_ind, col_ind) =
            src2.at<cv::Vec3b>(row_ind, col_ind);
      }
    }
  }
}

using namespace cv;
using namespace std;
static cv::Mat process_result_multitaskv3(
    cv::Mat& m1, const vitis::ai::MultiTaskv3Result& result, bool is_jpeg) {
  (void)process_result_multitaskv3;
  cv::Mat image_seg;
  cv::Mat image_dr;
  cv::Mat image_ln;
  cv::Mat image_dp(result.depth.rows, result.depth.cols, CV_8UC3);
  if (false) {
    // cv::resize(m1, image, result.segmentation.size());
    // overLay1(image, result.segmentation);
  } else {
    cv::resize(result.segmentation, image_seg, m1.size());
    cv::resize(result.drivable, image_dr, m1.size());
    cv::resize(result.lane, image_ln, m1.size());
    // cvtColor(result.depth,image_dp, CV_GRAY2RGB);
    cv::applyColorMap(result.depth, image_dp, cv::COLORMAP_PARULA);
    cv::resize(image_dp, image_dp, m1.size());
    overLay1(image_seg, m1);
    overLay1(image_dr, m1);
    overLay2(image_ln, m1, 128);
  }
  for (auto& r : result.vehicle) {
    LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
                          << r.width << " " << r.height << " " << r.angle;
    int xmin = r.x * image_seg.cols;
    int ymin = r.y * image_seg.rows;

    int width = r.width * image_seg.cols;
    int height = r.height * image_seg.rows;
    cv::rectangle(image_seg, cv::Rect_<int>(xmin, ymin, width, height),
                  cv::Scalar(185, 181, 178), 2, 1, 0);
  }

  cv::Mat canvas(cv::Size(m1.cols * 2, m1.rows * 2), CV_8UC3);
  image_dr.copyTo(canvas(Rect(0, m1.rows, m1.cols, m1.rows)));
  image_seg.copyTo(canvas(Rect(0, 0, m1.cols, m1.rows)));
  image_dp.copyTo(canvas(Rect(m1.cols, m1.rows, m1.cols, m1.rows)));
  image_ln.copyTo(canvas(Rect(m1.cols, 0, m1.cols, m1.rows)));
  return canvas;
}

[[maybe_unused]]  static std::vector<cv::Mat> process_result_multitaskv3_batch(
    std::vector<cv::Mat>& images,
    const std::vector<vitis::ai::MultiTaskv3Result>& results, bool is_jpeg) {
  size_t size = std::min(images.size(), results.size());
  std::vector<cv::Mat> image_results(size);
  for (auto i = 0u; i < size; ++i) {
    image_results[i] =
        process_result_multitaskv3(images[i], results[i], is_jpeg);
  }
  return image_results;
}

int main(int argc, char* argv[]) {
  gui_layout() = {{320, 180, 1280, 720}};

  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv, {[] {
        return vitis::ai::create_dpu_filter(
            [] {
              return vitis::ai::MultiTaskv38UC3::create("multi_task_v3_pt");
            },
            process_result_multitaskv3);
            //process_result_multitaskv3_batch);
      }});
}
