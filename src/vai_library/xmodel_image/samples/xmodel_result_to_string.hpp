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
 * */
#pragma once
#include <sstream>

static std::string to_string(const vitis::ai::proto::ClassificationResult& r) {
  // index: 109 score 0.992801 text: brain coral,
  std::ostringstream str;
  for (const auto& k : r.topk()) {
    str << "index: " << k.index()  //
        << " score " << k.score()  //
        << " text: " << k.name()   //
        << "\n";
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::DetectResult& r) {
  // 0.995669 0.45625 0.109375 0.125 0.15
  // 0.99371 0.16875 0.15625 0.125 0.140625
  std::ostringstream str;
  for (const auto& k : r.bounding_box()) {
    str << " " << k.label().score()  //
        << " " << k.top_left().x()   //
        << " " << k.top_left().y()   //
        << " " << k.size().width()   //
        << " " << k.size().height()  //
        << " " << k.label().index()  //
        << " " << k.label().name()   //
        << "\n";
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::PlateNumberResult& r) {
  // 0.995669 0.45625 0.109375 0.125 0.15
  // 0.99371 0.16875 0.15625 0.125 0.140625
  std::ostringstream str;
  str << r.plate_number() << "\n";
  return str.str();
}

static std::string to_string(const vitis::ai::proto::FaceFeatureResult& r) {
  std::ostringstream str;
  for (auto i = 0; i < r.float_vec_size() && i < 10; ++i) {
    str << " " << r.float_vec(i);
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::LandmarkResult& r) {
  std::ostringstream str;
  str << "quality=" << r.score() << "\n";
  for (auto i = 0; i < r.point_size(); ++i) {
    str << " " << r.point(i).x() << " " << r.point(i).y() << "\n";
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::RoadlineResult& r) {
  std::ostringstream str;
  for (auto i = 0; i < r.line_attribute_size(); ++i) {
    auto& line = r.line_attribute(i);
    str << " " << line.type() << " ";
    for (auto j = 0; j < line.point_size(); ++j) {
      str << line.point(j).x() << " " << line.point(j).y() << " ";
    }
    str << "\n";
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::PoseDetectResult& r) {
  std::ostringstream str;
  int c = 0;
  for (auto& key : {r}) {
    str << " " << c++;
    str << " "
        << "right_shoulder= (" << key.right_shoulder().x() << ","
        << key.right_shoulder().y() << ")";
    str << " "
        << "right_elbow= (" << key.right_elbow().x() << ","
        << key.right_elbow().y() << ")";
    str << " "
        << "right_wrist= (" << key.right_wrist().x() << ","
        << key.right_wrist().y() << ")";
    str << " "
        << "left_shoulder= (" << key.left_shoulder().x() << ","
        << key.left_shoulder().y() << ")";
    str << " "
        << "left_elbow= (" << key.left_elbow().x() << ","
        << key.left_elbow().y() << ")";
    str << " "
        << "left_wrist= (" << key.left_wrist().x() << ","
        << key.left_wrist().y() << ")";
    str << " "
        << "right_hip= (" << key.right_hip().x() << "," << key.right_hip().y()
        << ")";
    str << " "
        << "right_knee= (" << key.right_knee().x() << ","
        << key.right_knee().y() << ")";
    str << " "
        << "right_ankle= (" << key.right_ankle().x() << ","
        << key.right_ankle().y() << ")";
    str << " "
        << "left_hip= (" << key.left_hip().x() << "," << key.left_hip().y()
        << ")";
    str << " "
        << "left_knee= (" << key.left_knee().x() << "," << key.left_knee().y()
        << ")";
    str << " "
        << "left_ankle= (" << key.left_ankle().x() << ","
        << key.left_ankle().y() << ")";
    str << " "
        << "head= (" << key.head().x() << "," << key.head().y() << ")";
    str << " "
        << "neck= (" << key.neck().x() << "," << key.neck().y() << ")";
    str << "\n";
  }
  return str.str();
}

static std::string to_string(const vitis::ai::proto::DpuModelResult& r) {
  auto ret = std::string();
  switch (r.dpu_model_result_case()) {
    case vitis::ai::proto::DpuModelResult::kClassificationResult:
      ret = to_string(r.classification_result());
      break;
    case vitis::ai::proto::DpuModelResult::kDetectResult:
      ret = to_string(r.detect_result());
      break;
    case vitis::ai::proto::DpuModelResult::kPlateNumberResult:
      ret = to_string(r.plate_number_result());
      break;
    case vitis::ai::proto::DpuModelResult::kFaceFeatureResult:
      ret = to_string(r.face_feature_result());
      break;
    case vitis::ai::proto::DpuModelResult::kLandmarkResult:
      ret = to_string(r.landmark_result());
      break;
    case vitis::ai::proto::DpuModelResult::kRoadlineResult:
      ret = to_string(r.roadline_result());
      break;
    case vitis::ai::proto::DpuModelResult::kPoseDetectResult:
      ret = to_string(r.pose_detect_result());
      break;

    default:
      ret = r.DebugString();
  }
  return ret;
}
