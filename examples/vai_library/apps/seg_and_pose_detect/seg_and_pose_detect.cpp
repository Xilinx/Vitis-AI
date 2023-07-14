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
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/ssd.hpp>

namespace vitis {
namespace ai {

typedef std::vector<vitis::ai::PoseDetectResult> SSDPoseDetectResult;
struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  // std::vector<vitis::ai::PoseDetectResult> run(const cv::Mat& input_image);
  SSDPoseDetectResult run(const cv::Mat& input_image);
  std::vector<SSDPoseDetectResult> run(
      const std::vector<cv::Mat>& input_images);
  int getInputWidth();
  int getInputHeight();

 private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PoseDetect> pose_detect_;
};
// A factory function to get a instance of derived classes of class
std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97", true)},
      pose_detect_{vitis::ai::PoseDetect::create("sp_net")} {}

// std::vector<vitis::ai::PoseDetectResult> SSDPoseDetect::run(
SSDPoseDetectResult SSDPoseDetect::run(const cv::Mat& input_image) {
  // std::vector<vitis::ai::PoseDetectResult> mt_results;
  SSDPoseDetectResult mt_results;
  cv::Mat image;
  auto size = cv::Size(ssd_->getInputWidth(), ssd_->getInputHeight());
  if (size != input_image.size()) {
    cv::resize(input_image, image, size);
  } else {
    image = input_image;
  }
  // run ssd
  auto ssd_results = ssd_->run(image);

  for (auto& box : ssd_results.bboxes) {
      //DLOG(INFO) << "box.x " << box.x << " "            //
      //           << "box.y " << box.y << " "            //
      //           << "box.width " << box.width << " "    //
      //           << "box.height " << box.height << " "  //
      //           << "box.score " << box.score << " "    //
      //    ;
    // int label = box.label;
    int xmin = box.x * input_image.cols;
    int ymin = box.y * input_image.rows;
    int xmax = xmin + box.width * input_image.cols;
    int ymax = ymin + box.height * input_image.rows;
    float confidence = box.score;
    if (confidence < 0.55) continue;
    xmin = std::min(std::max(xmin, 0), input_image.cols);
    xmax = std::min(std::max(xmax, 0), input_image.cols);
    ymin = std::min(std::max(ymin, 0), input_image.rows);
    ymax = std::min(std::max(ymax, 0), input_image.rows);
    cv::Rect roi = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
    cv::Mat sub_img = input_image(roi);
    // process each result of ssd detection
    auto single_result = pose_detect_->run(sub_img);
    for (size_t i = 0; i < 28; i = i + 2) {
      ((float*)&single_result.pose14pt)[i] =
          ((float*)&single_result.pose14pt)[i] * sub_img.cols;
      ((float*)&single_result.pose14pt)[i] =
          (((float*)&single_result.pose14pt)[i] + xmin) / input_image.cols;
      ((float*)&single_result.pose14pt)[i + 1] =
          ((float*)&single_result.pose14pt)[i + 1] * sub_img.rows;
      ((float*)&single_result.pose14pt)[i + 1] =
          (((float*)&single_result.pose14pt)[i + 1] + ymin) / input_image.rows;
    }
    mt_results.emplace_back(single_result);
  }
  return mt_results;
}

std::vector<SSDPoseDetectResult> SSDPoseDetect::run(
    const std::vector<cv::Mat>& input_images) {
  std::vector<SSDPoseDetectResult> batch_mt_results(input_images.size());
  for (auto i = 0u; i < input_images.size(); ++i) {
    batch_mt_results[i] = run(input_images[i]);
  }
  return batch_mt_results;
}

}  // namespace ai
}  // namespace vitis

static void overLay1(cv::Mat& src1, const cv::Mat& src2) {
  const int imsize = src1.cols * src2.rows * 3;
  // vector<uchar> te(imsize, 2);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data1(const_cast<uchar*>(src1.data),
                                                imsize);
  Eigen::Map<Eigen::Matrix<uchar, -1, 1>> data2(const_cast<uchar*>(src2.data),
                                                imsize);
  data1 = data1 / 2 + data2 / 2;
}

// static cv::Mat process_result_multitask(
//     cv::Mat &m1, const vitis::multitask::MultiTaskResult &result,
//     bool is_jpeg) {
//   (void)process_result_multitask;
//   cv::Mat image;
//   if (false) {
//     cv::resize(m1, image, result.segmentation.size());
//     overLay1(image, result.segmentation);
//   } else {
//     cv::resize(result.segmentation, image, m1.size());
//     overLay1(image, m1);
//   }
//   for (auto &r : result.vehicle) {
//     LOG_IF(INFO, is_jpeg) << r.label << " " << r.x << " " << r.y << " "
//                           << r.width << " " << r.height << " " << r.angle;
//     int xmin = r.x * image.cols;
//     int ymin = r.y * image.rows;

//     int width = r.width * image.cols;
//     int height = r.height * image.rows;
//     cv::rectangle(image, cv::Rect_<int>(xmin, ymin, width, height),
//                   cv::Scalar(185, 181, 178), 2, 1, 0);
//   }
//   return image;
// }
using namespace cv;
static cv::Mat process_result_multitask_with_original(
    cv::Mat& m1, const vitis::ai::MultiTaskResult& result, bool is_jpeg) {
  (void)process_result_multitask_with_original;
  cv::Mat image;
  if (false) {
    cv::resize(m1, image, result.segmentation.size());
    overLay1(image, result.segmentation);
  } else {
    cv::resize(result.segmentation, image, m1.size());
    overLay1(image, m1);
  }
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

  cv::Mat canvas(cv::Size(image.cols, image.rows * 2), CV_8UC3);
  m1.copyTo(canvas(Rect(0, image.rows, image.cols, image.rows)));
  image.copyTo(canvas(Rect(0, 0, image.cols, image.rows)));
  return canvas;
}

[[maybe_unused]] static std::vector<cv::Mat> process_result_multitask_with_original_batch(
    std::vector<cv::Mat>& images,
    const std::vector<vitis::ai::MultiTaskResult>& results, bool is_jpeg) {
  size_t size = std::min(images.size(), results.size());
  std::vector<cv::Mat> image_results(size);

  for (auto i = 0u; i < size; ++i) {
    image_results[i] =
        process_result_multitask_with_original(images[i], results[i], is_jpeg);
  }
  return image_results;
}

static inline void DrawLine(Mat& img, Point2f point1, Point2f point2,
                            Scalar colour, int thickness, float scale_w,
                            float scale_h) {
  if ((point1.x * img.cols > scale_w || point1.y * img.rows > scale_h) &&
      (point2.x * img.cols > scale_w || point2.y * img.rows > scale_h))
    cv::line(img, Point2f(point1.x * img.cols, point1.y * img.rows),
             Point2f(point2.x * img.cols, point2.y * img.rows), colour,
             thickness);
}

static void DrawLines(Mat& img,
                      const vitis::ai::PoseDetectResult::Pose14Pt& results) {
  float scale_w = 1;
  float scale_h = 1;

  float mark = 5.f;
  float mark_w = mark * scale_w;
  float mark_h = mark * scale_h;
  std::vector<Point2f> pois(14);
  for (size_t i = 0; i < pois.size(); ++i) {
    pois[i].x = ((float*)&results)[i * 2] * img.cols;
    // std::cout << ((float*)&results)[i * 2] << " " << ((float*)&results)[i * 2
    // + 1] << std::endl;
    pois[i].y = ((float*)&results)[i * 2 + 1] * img.rows;
  }
  for (size_t i = 0; i < pois.size(); ++i) {
    circle(img, pois[i], 3, Scalar::all(255));
  }
  DrawLine(img, results.right_shoulder, results.right_elbow, Scalar(255, 0, 0),
           2, mark_w, mark_h);
  DrawLine(img, results.right_elbow, results.right_wrist, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.right_knee, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_knee, results.right_ankle, Scalar(255, 0, 0), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_elbow, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_elbow, results.left_wrist, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_hip, results.left_knee, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_knee, results.left_ankle, Scalar(0, 0, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.head, results.neck, Scalar(0, 255, 255), 2, mark_w,
           mark_h);
  DrawLine(img, results.right_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.neck, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_shoulder, results.right_hip, Scalar(0, 255, 255),
           2, mark_w, mark_h);
  DrawLine(img, results.left_shoulder, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
  DrawLine(img, results.right_hip, results.left_hip, Scalar(0, 255, 255), 2,
           mark_w, mark_h);
}

cv::Mat process_result_pose_detect(cv::Mat& image,
                                   const vitis::ai::PoseDetectResult& results,
                                   bool is_jpeg) {
  std::vector<float> pose14pt_arry((float*)&results.pose14pt,
                                   (float*)&results.pose14pt + 28);
  for (size_t i = 0; i < pose14pt_arry.size(); i = i + 2) {
    LOG_IF(INFO, is_jpeg) << "(" << pose14pt_arry[i] << ","
                          << pose14pt_arry[i + 1] << ")";
  }
  DrawLines(image, results.pose14pt);
  return image;
}

static cv::Mat process_result_pose_detect_with_ssd(
    cv::Mat& image, const vitis::ai::SSDPoseDetectResult& results,
    bool is_jpeg) {
  (void)process_result_pose_detect_with_ssd;
  cv::Mat canvas(cv::Size(image.cols, image.rows * 2), CV_8UC3);
  image.copyTo(canvas(Rect(0, image.rows, image.cols, image.rows)));
  for (auto& result : results) {
    process_result_pose_detect(image, result, is_jpeg);
  }
  image.copyTo(canvas(Rect(0, 0, image.cols, image.rows)));
  return canvas;
}

[[maybe_unused ]] static std::vector<cv::Mat> process_result_pose_detect_with_ssd_batch(
    std::vector<cv::Mat>& images,
    const std::vector<vitis::ai::SSDPoseDetectResult>& batch_results,
    bool is_jpeg) {
  size_t size = std::min(images.size(), batch_results.size());
  std::vector<cv::Mat> image_results(size);
  for (auto i = 0u; i < size; ++i) {
    image_results[i] = process_result_pose_detect_with_ssd(
        images[i], batch_results[i], is_jpeg);
  }
  return image_results;
}

int main(int argc, char* argv[]) {
  gui_layout() = {{0, 0, 960, 540 * 2}, {960, 0, 960, 540 * 2}};
  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv,
      {[] {
         return vitis::ai::create_dpu_filter(
             [] { return vitis::ai::MultiTask8UC3::create("multi_task"); },
             process_result_multitask_with_original);
             //process_result_multitask_with_original_batch);
         //    }});
       },
       [] {
         return vitis::ai::create_dpu_filter(
             [] { return vitis::ai::SSDPoseDetect::create(); },
             process_result_pose_detect_with_ssd);
             //process_result_pose_detect_with_ssd_batch);
       }});
}
