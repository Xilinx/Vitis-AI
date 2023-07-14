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
 * diLOG_IF(INFO, false)ibuted under the License is diLOG_IF(INFO, false)ibuted
 * on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
 * express or implied. See the License for the specific language governing
 * permissions and limitations under the License.
 * */

#pragma once
#include <glog/logging.h>

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::ClassificationResult& r) {
  cv::Mat ret;
  for (const auto& k : r.topk()) {
    LOG_IF(INFO, false) << "index: " << k.index()  //
                        << " score " << k.score()  //
                        << " text: " << k.name()   //
                        << "\n";
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::DetectResult& r) {
  // 0.995669 0.45625 0.109375 0.125 0.15
  // 0.99371 0.16875 0.15625 0.125 0.140625
  cv::Mat ret;
  for (const auto& k : r.bounding_box()) {
    LOG_IF(INFO, false) << " " << k.label().score()  //
                        << " " << k.top_left().x()   //
                        << " " << k.top_left().y()   //
                        << " " << k.size().width()   //
                        << " " << k.size().height()  //
                        << " " << k.label().index()  //
                        << " " << k.label().name()   //
                        << "\n";
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::PlateNumberResult& r) {
  // 0.995669 0.45625 0.109375 0.125 0.15
  // 0.99371 0.16875 0.15625 0.125 0.140625
  cv::Mat ret;
  LOG_IF(INFO, false) << r.plate_number() << "\n";
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::FaceFeatureResult& r) {
  cv::Mat ret;
  for (auto i = 0; i < r.float_vec_size() && i < 10; ++i) {
    LOG_IF(INFO, false) << " " << r.float_vec(i);
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::LandmarkResult& r) {
  cv::Mat ret;
  LOG_IF(INFO, false) << "quality=" << r.score() << "\n";
  for (auto i = 0; i < r.point_size(); ++i) {
    LOG_IF(INFO, false) << " " << r.point(i).x() << " " << r.point(i).y()
                        << "\n";
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::RoadlineResult& r) {
  cv::Mat ret = image.clone();
  std::vector<int> color1 = {0, 255, 0, 0, 100, 255};
  std::vector<int> color2 = {0, 0, 255, 0, 100, 255};
  std::vector<int> color3 = {0, 0, 0, 255, 100, 255};

  for (auto& line : r.line_attribute()) {
    std::vector<cv::Point> points_poly;
    for (auto& point : line.point()) {
      points_poly.emplace_back(cv::Point{(int)(point.x() * image.cols),
                                         (int)(point.y() * image.rows)});
    }
    int type = line.type() < 5 ? line.type() : 5;
    if (type == 2 && points_poly[0].x < image.rows * 0.5) continue;
    cv::polylines(image, points_poly, false,
                  cv::Scalar(color1[type], color2[type], color3[type]), 3,
                  cv::LINE_AA, 0);
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::PoseDetectResult& r) {
  vector<vector<int>> limbSeq = {{0, 1},  {1, 2},   {2, 3},  {3, 4}, {1, 5},
                                 {5, 6},  {6, 7},   {1, 8},  {8, 9}, {9, 10},
                                 {1, 11}, {11, 12}, {12, 13}};
  cv::Mat ret = image.clone();
  auto to_point = [&ret](const vitis::ai::proto::Point& point) {
    int x = point.x() * ret.cols;
    int y = point.y() * ret.rows;
    auto p = cv::Point(x, y);
    return p;
  };
  auto draw_point = [&ret, &to_point](bool valid,
                                      const vitis::ai::proto::Point& a) {
    if (valid) {
      cv::circle(ret, to_point(a), 5, cv::Scalar(0, 255, 0), -1);
    }
  };
  auto draw_line = [&ret, &to_point](const vitis::ai::proto::Point& a,
                                     const vitis::ai::proto::Point& b,
                                     bool valid) {
    if (valid) {
      cv::line(ret, to_point(a), to_point(b), cv::Scalar(255, 0, 0), 3, 4);
    }
  };
  for (auto& key : {r}) {
    draw_point(key.has_right_shoulder(), key.right_shoulder());
    draw_point(key.has_right_elbow(), key.right_elbow());
    draw_point(key.has_right_wrist(), key.right_wrist());
    draw_point(key.has_left_shoulder(), key.left_shoulder());
    draw_point(key.has_left_elbow(), key.left_elbow());
    draw_point(key.has_left_wrist(), key.left_wrist());
    draw_point(key.has_right_hip(), key.right_hip());
    draw_point(key.has_right_knee(), key.right_knee());
    draw_point(key.has_right_ankle(), key.right_ankle());
    draw_point(key.has_left_hip(), key.left_hip());
    draw_point(key.has_left_knee(), key.left_knee());
    draw_point(key.has_left_ankle(), key.left_ankle());
    draw_point(key.has_head(), key.head());
    draw_point(key.has_neck(), key.neck());
  }
  for (auto& key : {r}) {
    auto O = vitis::ai::proto::Point();
    draw_line(key.head(), key.neck(), key.has_head() && key.has_neck());
    draw_line(key.neck(), key.right_shoulder(),
              key.has_neck() && key.has_right_shoulder());
    draw_line(key.neck(), key.left_shoulder(),
              key.has_neck() && key.has_left_shoulder());
    draw_line(key.right_shoulder(), key.right_elbow(),
              key.has_right_shoulder() && key.has_right_elbow());
    draw_line(key.right_elbow(), key.right_wrist(),
              key.has_right_elbow() && key.has_right_wrist());
    draw_line(key.left_shoulder(), key.left_elbow(),
              key.has_left_shoulder() && key.has_left_elbow());
    draw_line(key.left_elbow(), key.left_wrist(),
              key.has_left_elbow() && key.has_left_wrist());
    draw_line(key.neck(), key.right_hip(),
              key.has_neck() && key.has_right_hip());
    draw_line(key.right_hip(), key.right_knee(),
              key.has_right_hip() && key.has_right_knee());
    draw_line(key.right_knee(), key.right_ankle(),
              key.has_right_knee() && key.has_right_ankle());
    draw_line(key.neck(), key.left_hip(), key.has_neck() && key.has_left_hip());
    draw_line(key.left_hip(), key.left_knee(),
              key.has_left_hip() && key.has_left_knee());
    draw_line(key.left_knee(), key.left_ankle(),
              key.has_left_knee() && key.has_left_ankle());
  }
  return ret;
}

static cv::Mat process_image(const cv::Mat& image,
                             const vitis::ai::proto::DpuModelResult& r) {
  cv::Mat ret;
  switch (r.dpu_model_result_case()) {
    case vitis::ai::proto::DpuModelResult::kClassificationResult:
      ret = process_image(image, r.classification_result());
      break;
    case vitis::ai::proto::DpuModelResult::kDetectResult:
      ret = process_image(image, r.detect_result());
      break;
    case vitis::ai::proto::DpuModelResult::kPlateNumberResult:
      ret = process_image(image, r.plate_number_result());
      break;
    case vitis::ai::proto::DpuModelResult::kFaceFeatureResult:
      ret = process_image(image, r.face_feature_result());
      break;
    case vitis::ai::proto::DpuModelResult::kLandmarkResult:
      ret = process_image(image, r.landmark_result());
      break;
    case vitis::ai::proto::DpuModelResult::kRoadlineResult:
      ret = process_image(image, r.roadline_result());
      break;
    case vitis::ai::proto::DpuModelResult::kPoseDetectResult:
      ret = process_image(image, r.pose_detect_result());
      break;
    default:
      break;
  }
  return ret;
}
