/*
 * Copyright 2019 Xilinx Inc.
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
#include <vitis/ai/demo5.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/multitask.hpp>
#include <vitis/ai/ssd.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/openpose.hpp>

using namespace std;
using namespace cv; 

namespace vitis {
namespace ai {
struct SSDPoseDetect {
  static std::unique_ptr<SSDPoseDetect> create();
  SSDPoseDetect();
  std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>, vitis::ai::FaceDetectResult>>  run(const std::vector<cv::Mat> &input_image);
  int getInputWidth();
  int getInputHeight();

 private:
  std::unique_ptr<vitis::ai::SSD> ssd_;
  std::unique_ptr<vitis::ai::PoseDetect> pose_detect_;
  std::unique_ptr<vitis::ai::FaceDetect> face_detect_;
};
// A factory function to get a instance of derived classes of class
std::unique_ptr<SSDPoseDetect> SSDPoseDetect::create() {
  return std::unique_ptr<SSDPoseDetect>(new SSDPoseDetect());
}
int SSDPoseDetect::getInputWidth() { return ssd_->getInputWidth(); }
int SSDPoseDetect::getInputHeight() { return ssd_->getInputHeight(); }

SSDPoseDetect::SSDPoseDetect()
    : ssd_{vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97", true)}, face_detect_{vitis::ai::FaceDetect::create("densebox_640_360", true)},
      pose_detect_{vitis::ai::PoseDetect::create("sp_net")} {}

std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>, vitis::ai::FaceDetectResult>> SSDPoseDetect::run(
    const std::vector<cv::Mat> &input_images) {
  std::vector<std::pair<std::vector<vitis::ai::PoseDetectResult>, vitis::ai::FaceDetectResult>> results; 
  for (auto k = 0u; k < input_images.size(); k++) {
    std::vector<vitis::ai::PoseDetectResult> mt_results;
    cv::Mat image;
    auto size = cv::Size(ssd_->getInputWidth(), ssd_->getInputHeight());
    if (size != input_images[k].size()) {
      cv::resize(input_images[k], image, size);
    } else {
      image = input_images[k];
    }
  // run ssd
    auto ssd_results = ssd_->run(image);
    auto face_results = face_detect_->run(image);

    for (auto &box : ssd_results.bboxes) {
      if (1)
        DLOG(INFO) << "box.x " << box.x << " "            //
                 << "box.y " << box.y << " "            //
                 << "box.width " << box.width << " "    //
                 << "box.height " << box.height << " "  //
                 << "box.score " << box.score << " "    //
          ;
    // int label = box.label;
      int xmin = box.x * input_images[k].cols;
      int ymin = box.y * input_images[k].rows;
      int xmax = xmin + box.width * input_images[k].cols;
      int ymax = ymin + box.height * input_images[k].rows;
      float confidence = box.score;
      if (confidence < 0.55) continue;
      xmin = std::min(std::max(xmin, 0), input_images[k].cols);
      xmax = std::min(std::max(xmax, 0), input_images[k].cols);
      ymin = std::min(std::max(ymin, 0), input_images[k].rows);
      ymax = std::min(std::max(ymax, 0), input_images[k].rows);
      cv::Rect roi = cv::Rect(cv::Point(xmin, ymin), cv::Point(xmax, ymax));
      cv::Mat sub_img = input_images[k](roi);
    // process each result of ssd detection
      auto single_result = pose_detect_->run(sub_img);
      for (size_t i = 0; i < 28; i = i + 2) {
        ((float *)&single_result.pose14pt)[i] =
            ((float *)&single_result.pose14pt)[i] * sub_img.cols;
        ((float *)&single_result.pose14pt)[i] =
            (((float *)&single_result.pose14pt)[i] + xmin) / input_images[k].cols;
        ((float *)&single_result.pose14pt)[i + 1] =
            ((float *)&single_result.pose14pt)[i + 1] * sub_img.rows;
        ((float *)&single_result.pose14pt)[i + 1] =
            (((float *)&single_result.pose14pt)[i + 1] + ymin) / input_images[k].rows;
      }
      mt_results.emplace_back(single_result);
    }
    results.push_back(make_pair(mt_results, face_results));
  }
  return results;
}
}  // namespace ai
}  // namespace vitis



static inline void DrawLine(Mat &img, Point2f point1, Point2f point2,
                            Scalar colour, int thickness, float scale_w,
                            float scale_h) {
  if ((point1.x * img.cols > scale_w || point1.y * img.rows > scale_h) &&
      (point2.x * img.cols > scale_w || point2.y * img.rows > scale_h))
    cv::line(img, Point2f(point1.x * img.cols, point1.y * img.rows),
             Point2f(point2.x * img.cols, point2.y * img.rows), colour,
             thickness);
}

static void DrawLines(Mat &img,
                      const vitis::ai::PoseDetectResult::Pose14Pt &results) {
  float scale_w = 1;
  float scale_h = 1;

  float mark = 5.f;
  float mark_w = mark * scale_w;
  float mark_h = mark * scale_h;
  std::vector<Point2f> pois(14);
  for (size_t i = 0; i < pois.size(); ++i) {
    pois[i].x = ((float *)&results)[i * 2] * img.cols;
    // std::cout << ((float*)&results)[i * 2] << " " << ((float*)&results)[i * 2
    // + 1] << std::endl;
    pois[i].y = ((float *)&results)[i * 2 + 1] * img.rows;
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

cv::Mat process_result_pose_detect(cv::Mat &image,
                                   const vitis::ai::PoseDetectResult &results,
                                   bool is_jpeg) {
  std::vector<float> pose14pt_arry((float *)&results.pose14pt,
                                   (float *)&results.pose14pt + 28);
  for (size_t i = 0; i < pose14pt_arry.size(); i = i + 2) {
    LOG_IF(INFO, is_jpeg) << "(" << pose14pt_arry[i] << ","
                          << pose14pt_arry[i + 1] << ")";
  }
  DrawLines(image, results.pose14pt);
  return image;
}

static std::vector<cv::Mat> process_result_pose_detect_with_ssd(
    std::vector<cv::Mat> &images, const std::vector<pair<std::vector<vitis::ai::PoseDetectResult>, vitis::ai::FaceDetectResult>> &all_results,
    bool is_jpeg) {
  (void)process_result_pose_detect_with_ssd;
//cv::Mat canvas(cv::Size(image.cols, image.rows * 2), CV_8UC3);
  //image.copyTo(canvas(Rect(0, image.rows, image.cols, image.rows)));
  
  for(auto i = 0u; i < images.size(); i++) {
    for (auto &result : all_results[i].first) {
      process_result_pose_detect(images[i], result, is_jpeg);
    }
    for (const auto &r : all_results[i].second.rects){
      cv::rectangle(images[i],
                        cv::Rect{cv::Point(r.x * images[i].cols, r.y * images[i].rows),
                                 cv::Size{(int)(r.width * images[i].cols),
                                          (int)(r.height * images[i].rows)}},
                        0xff);
    }
  }
  //image.copyTo(canvas(Rect(0, 0, image.cols, image.rows)));
  return images;
}






extern std::vector<cv::Rect> GLOBAL_SET_RECT_MANUAL;

static cv::Scalar getColor(int label) {
  int c[3];
  for (int i = 1, j = 0; i <= 9; i *= 3, j++) {
    c[j] = ((label / i) % 3) * 127;
  }
  return cv::Scalar(c[2], c[1], c[0]);
}

static std::vector<cv::Mat> process_result_ssd(std::vector<cv::Mat> &images, std::vector<vitis::ai::SSDResult> results, bool is_jpeg) {
  for(auto im = 0u; im < images.size(); im++) {
    for (const auto bbox : results[im].bboxes) {
      int label = bbox.label;
      float xmin = bbox.x * images[im].cols;
      float ymin = bbox.y * images[im].rows;
      float xmax = xmin + bbox.width * images[im].cols;
      float ymax = ymin + bbox.height * images[im].rows;
      float confidence = bbox.score;
      if (xmax > images[im].cols) xmax = images[im].cols;
      if (ymax > images[im].rows) ymax = images[im].rows;
      LOG_IF(INFO, is_jpeg) << "RESULT: " << label << "\t" << xmin << "\t" << ymin
                            << "\t" << xmax << "\t" << ymax << "\t" << confidence
                            << "\n";

      cv::rectangle(images[im], cv::Point(xmin, ymin), cv::Point(xmax, ymax),
                    getColor(label), 1, 1, 0);
    }
  }
  return images;
}

static std::vector<cv::Mat> process_result_facedetect(std::vector<cv::Mat> &images, std::vector<vitis::ai::FaceDetectResult> results, bool is_jpeg) {
  for(auto im = 0u; im < images.size(); im++) {
    cv::resize(images[im], images[im], cv::Size{results[im].width, results[im].height});
    for (const auto &r : results[im].rects) {
      LOG_IF(INFO, is_jpeg) << " " << r.score << " "  //
                            << r.x << " "             //
                            << r.y << " "             //
                            << r.width << " "         //
                            << r.height;
      cv::rectangle(images[im],
                    cv::Rect{cv::Point(r.x * images[im].cols, r.y * images[im].rows),
                             cv::Size{(int)(r.width * images[im].cols),
                                      (int)(r.height * images[im].rows)}},
      
                    cv::Scalar(255,0,0), 2, 2, 0);
    }
  }
  return images;
}


using Result = vitis::ai::OpenPoseResult::PosePoint;
std::vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 3},  {3, 4}, {1, 5},
                                 {5, 6},  {6, 7},   {1, 8},  {8, 9}, {9, 10},
                                 {1, 11}, {11, 12}, {12, 13}};
static std::vector<cv::Mat> process_result_openpose(std::vector<cv::Mat> &images, std::vector<vitis::ai::OpenPoseResult> results, bool is_jpeg) {

  for(auto im = 0u; im < images.size(); im++) {
    for (size_t k = 1; k < results[im].poses.size(); ++k) {
      for (size_t i = 0; i < results[im].poses[k].size(); ++i) {
        if (results[im].poses[k][i].type == 1) {
          cv::circle(images[im], results[im].poses[k][i].point, 5, cv::Scalar(0, 255, 0),
                     -1);
        }
      }
      for (size_t i = 0; i < limbSeq.size(); ++i) {
        Result a = results[im].poses[k][limbSeq[i][0]];
        Result b = results[im].poses[k][limbSeq[i][1]];
        if (a.type == 1 && b.type == 1) {
          cv::line(images[im], a.point, b.point, cv::Scalar(255, 0, 0), 3, 4);
        }
      }
    }
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
  gui_layout() = {{seg_px, seg_py, 640, 360},
                  {seg_px + 640, seg_py, 640, 360},
                  {seg_px, seg_py + 360, 640, 360},
                  {seg_px + 640, seg_py + 360, 640, 360},
  		};
  // assign to Lvalue : set background image
  //
  gui_background() = cv::imread("/usr/share/weston/logo.jpg");
  // init each dpu filter and process instance, using video demo framework
  return vitis::ai::main_for_video_demo_multiple_channel(
      argc, argv,
      {
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::SSDPoseDetect::create(); },
                process_result_pose_detect_with_ssd);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::FaceDetect::create("densebox_640_360"); },
                process_result_facedetect);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::SSD::create("ssd_pedestrian_pruned_0_97"); },
                process_result_ssd);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] { return vitis::ai::FaceDetect::create("densebox_640_360"); },
                process_result_facedetect);
          },
          /*[] {
            return vitis::ai::create_dpu_filter(
                [] {
                  return vitis::ai::RoadLine::create("vpgnet_pruned_0_99");
                },
                process_result_roadline);
          },
          [] {
            return vitis::ai::create_dpu_filter(
                [] {
                  return vitis::ai::SSDPoseDetect::create();
                },
                process_result_pose_detect_with_ssd);
          },*/
      });
}
