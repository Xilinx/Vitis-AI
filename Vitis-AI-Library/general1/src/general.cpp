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

#include "../include/vitis/ai/general.hpp"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>

#include <fstream>
#include <iostream>
#include <sstream>
#include <vitis/ai/classification.hpp>
#include <vitis/ai/env_config.hpp>
#include <vitis/ai/facedetect.hpp>
#include <vitis/ai/facelandmark.hpp>
#include <vitis/ai/lanedetect.hpp>
#include <vitis/ai/platedetect.hpp>
#include <vitis/ai/platenum.hpp>
#include <vitis/ai/posedetect.hpp>
#include <vitis/ai/refinedet.hpp>
#include <vitis/ai/ssd.hpp>
#include <vitis/ai/tfssd.hpp>
#include <vitis/ai/yolov2.hpp>
#include <vitis/ai/yolov3.hpp>

// #include <vitis/ai/platerecog.hpp>
// #include <vitis/ai/segmentation.hpp>
// #include <vitis/ai/multitask.hpp>
// #include <vitis/ai/medicalsegmentation.hpp>
// #include <vitis/ai/reid.hpp>
// #include <vitis/ai/facefeature.hpp>

using namespace std;
DEF_ENV_PARAM(DEBUG_GENERAL, "0");
#include "./general_adapter.hpp"
namespace vitis {
namespace ai {
extern "C" vitis::ai::proto::DpuModelParam *find(const std::string &model_name);

General::General() {}
General::~General() {}

static std::vector<std::string> find_model_search_path() {
  auto ret = vector<string>{};
  ret.push_back(".");
  ret.push_back("/usr/share/vitis_ai_library/models");
  ret.push_back("/usr/share/vitis_ai_library/.models");
  return ret;
}

static size_t filesize(const string &filename) {
  size_t ret = 0;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = statbuf.st_size;
  }
  return ret;
}

static string find_model(const string &name) {
  if (filesize(name) > 4096u) {
    return name;
  }

  auto ret = std::string();
  for (const auto &p : find_model_search_path()) {
    ret = p + "/" + name + "/" + name;
    const auto xmodel_name = ret + ".xmodel";
    if (filesize(xmodel_name) > 0u) {
      return xmodel_name;
    }
    const auto elf_name = ret + ".elf";
    if (filesize(elf_name) > 0u) {
      return elf_name;
    }
  }

  stringstream str;
  str << "cannot find model <" << name << "> after checking following dir:";
  for (const auto &p : find_model_search_path()) {
    str << "\n\t" << p;
  }
  LOG(FATAL) << str.str();
  return string{""};
}

static string find_config_file(const string &name) {
  auto model = find_model(name);
  std::string pre_name = model.substr(0, model.rfind("."));
  auto config_file = pre_name + ".prototxt";
  if (filesize(config_file) > 0u) {
    return config_file;
  }
  LOG(FATAL) << "cannot find " << config_file;
  return string{""};
}

static std::string slurp(const char *filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  CHECK(in.good()) << "failed to read config file. filename=" << filename;
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

static vitis::ai::proto::DpuModelParam get_config(
    const std::string &model_name) {
  auto config_file = find_config_file(find_model(model_name));
  vitis::ai::proto::DpuModelParamList mlist;
  auto text = slurp(config_file.c_str());
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);
  CHECK(ok) << "cannot parse config file. config_file=" << config_file;
  CHECK_EQ(mlist.model_size(), 1)
      << "only support one model per config file."
      << "config_file " << config_file << " "       //
      << "content: " << mlist.DebugString() << " "  //
      ;
  return mlist.model(0);
}

template <typename... T>
struct SupportedModels {
  using types = std::tuple<T...>;
  static std::unique_ptr<General> create(
      vitis::ai::proto::DpuModelParam::ModelType type,
      const std::string &name) {
    return SupportedModels_create(SupportedModels<T...>(), type, name);
  };
};

template <vitis::ai::proto::DpuModelParam::ModelType ModelType,
          typename ModelClass>
struct ModelDef {
  static const auto type = ModelType;
  using cls = ModelClass;
};

std::unique_ptr<General> SupportedModels_create(
    SupportedModels<> _tag, vitis::ai::proto::DpuModelParam::ModelType type,
    const std::string &name) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GENERAL)) << " type = " << type       //
                                         << " mode name = " << name  //
                                         << endl;
  return nullptr;
}

template <typename T0, typename... Tn>
std::unique_ptr<General> SupportedModels_create(
    SupportedModels<T0, Tn...> _tag,
    vitis::ai::proto::DpuModelParam::ModelType type, const std::string &name) {
  LOG_IF(INFO, ENV_PARAM(DEBUG_GENERAL)) << " type = " << type          //
                                         << " T0::type = " << T0::type  //
                                         << " mode name = " << name     //
                                         << endl;
  if (T0::type == type) {
    return std::make_unique<GeneralAdapter<typename T0::cls>>(
        T0::cls::create(name, true));
  }
  return SupportedModels_create(SupportedModels<Tn...>(), type, name);
}

using ListOfSupportedModels = SupportedModels<
    /* list of supported models begin */
    ModelDef<vitis::ai::proto::DpuModelParam::DENSE_BOX, vitis::ai::FaceDetect>,
    ModelDef<vitis::ai::proto::DpuModelParam::YOLOv2, vitis::ai::YOLOv2>,
    ModelDef<vitis::ai::proto::DpuModelParam::YOLOv3, vitis::ai::YOLOv3>,
    ModelDef<vitis::ai::proto::DpuModelParam::FACELANDMARK,
             vitis::ai::FaceLandmark>,
    ModelDef<vitis::ai::proto::DpuModelParam::ROADLINE, vitis::ai::RoadLine>,
    ModelDef<vitis::ai::proto::DpuModelParam::REFINEDET, vitis::ai::RefineDet>,
    ModelDef<vitis::ai::proto::DpuModelParam::CLASSIFICATION,
             vitis::ai::Classification>,
    ModelDef<vitis::ai::proto::DpuModelParam::SSD, vitis::ai::SSD>,
    ModelDef<vitis::ai::proto::DpuModelParam::TFSSD, vitis::ai::TFSSD>,
    ModelDef<vitis::ai::proto::DpuModelParam::PLATEDETECT,
             vitis::ai::PlateDetect>,
    ModelDef<vitis::ai::proto::DpuModelParam::PLATENUM, vitis::ai::PlateNum>,
    ModelDef<vitis::ai::proto::DpuModelParam::POSEDETECT, vitis::ai::PoseDetect>
    // ModelDef<vitis::ai::proto::DpuModelParam::PLATERECOG,
    // vitis::ai::PlateRecog>
    /* list of supported models end */
    >;

std::unique_ptr<General> General::create(const std::string &model_name,
                                         bool need_preprocess) {
  auto model = get_config(model_name);
  auto model_type = model.model_type();
  return ListOfSupportedModels::create(model_type, model_name);
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::PoseDetectResult>(
    const vitis::ai::PoseDetectResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &pose_detect_result = *dpu_model_result.mutable_pose_detect_result();
  auto &right_shoulder = *pose_detect_result.mutable_right_shoulder();
  right_shoulder.set_x(result.pose14pt.right_shoulder.x);
  right_shoulder.set_y(result.pose14pt.right_shoulder.y);
  auto &right_elbow = *pose_detect_result.mutable_right_elbow();
  right_elbow.set_x(result.pose14pt.right_elbow.x);
  right_elbow.set_y(result.pose14pt.right_elbow.y);
  auto &right_wrist = *pose_detect_result.mutable_right_wrist();
  right_wrist.set_x(result.pose14pt.right_wrist.x);
  right_wrist.set_y(result.pose14pt.right_wrist.y);
  auto &left_shoulder = *pose_detect_result.mutable_left_shoulder();
  left_shoulder.set_x(result.pose14pt.left_shoulder.x);
  left_shoulder.set_y(result.pose14pt.left_shoulder.y);
  auto &left_elbow = *pose_detect_result.mutable_left_elbow();
  left_elbow.set_x(result.pose14pt.left_elbow.x);
  left_elbow.set_y(result.pose14pt.left_elbow.y);
  auto &left_wrist = *pose_detect_result.mutable_left_wrist();
  left_wrist.set_x(result.pose14pt.left_wrist.x);
  left_wrist.set_y(result.pose14pt.left_wrist.y);
  auto &right_hip = *pose_detect_result.mutable_right_hip();
  right_hip.set_x(result.pose14pt.right_hip.x);
  right_hip.set_y(result.pose14pt.right_hip.y);
  auto &right_knee = *pose_detect_result.mutable_right_knee();
  right_knee.set_x(result.pose14pt.right_knee.x);
  right_knee.set_y(result.pose14pt.right_knee.y);
  auto &right_ankle = *pose_detect_result.mutable_right_ankle();
  right_ankle.set_x(result.pose14pt.right_ankle.x);
  right_ankle.set_y(result.pose14pt.right_ankle.y);
  auto &left_hip = *pose_detect_result.mutable_left_hip();
  left_hip.set_x(result.pose14pt.left_hip.x);
  left_hip.set_y(result.pose14pt.left_hip.y);
  auto &left_knee = *pose_detect_result.mutable_left_knee();
  left_knee.set_x(result.pose14pt.left_knee.x);
  left_knee.set_y(result.pose14pt.left_knee.y);
  auto &left_ankle = *pose_detect_result.mutable_left_ankle();
  left_ankle.set_x(result.pose14pt.left_ankle.x);
  left_ankle.set_y(result.pose14pt.left_ankle.y);

  return dpu_model_result;
}

// template <>
// vitis::ai::proto::DpuModelResult process_result<vitis::ai::PlateRecogResult>(
//     const vitis::ai::PlateRecogResult &result) {
//   vitis::ai::proto::DpuModelResult dpu_model_result;

//   auto &plate_recog_result = *dpu_model_result.mutable_plate_recog_result();
//   plate_recog_result.set_plate_number(result.plate_number);
//   plate_recog_result.set_plate_color(result.plate_color);
//   auto &box = *plate_recog_result.mutable_bounding_box();
//   box.set_score(result.box.score);
//   box.set_x(result.box.x);
//   box.set_y(result.box.y);
//   box.set_height(result.box.height);
//   box.set_width(result.box.width);
//   return dpu_model_result;
// }

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::PlateNumResult>(
    const vitis::ai::PlateNumResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &plate_num_result = *dpu_model_result.mutable_plate_num_result();
  plate_num_result.set_plate_number(result.plate_number);
  plate_num_result.set_plate_color(result.plate_color);
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::PlateDetectResult>(
    const vitis::ai::PlateDetectResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &plate_detect_result = *dpu_model_result.mutable_plate_detect_result();
  auto &box = *plate_detect_result.mutable_bounding_box();
  box.set_score(result.box.score);
  box.set_x(result.box.x);
  box.set_y(result.box.y);
  box.set_height(result.box.height);
  box.set_width(result.box.width);
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::FaceLandmarkResult>(
    const vitis::ai::FaceLandmarkResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &facelandmark_result = *dpu_model_result.mutable_facelandmark_result();
  for (auto &r : result.points) {
    auto point = facelandmark_result.add_point();
    point->set_x(r.first);
    point->set_y(r.second);
  }
  LOG(INFO) << "facelandmark_result.point().size() "
            << facelandmark_result.point().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::RoadLineResult>(
    const vitis::ai::RoadLineResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &roadline_result = *dpu_model_result.mutable_roadline_result();
  for (auto &r : result.lines) {
    auto line = roadline_result.add_line_att();
    line->set_type(r.type);
    for (auto &p : r.points_cluster) {
      auto point = line->add_point();
      point->set_x(p.x);
      point->set_y(p.y);
    }
  }
  LOG(INFO) << "detect_result.line_att().size() "
            << roadline_result.line_att().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::SSDResult>(
    const vitis::ai::SSDResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &detect_result = *dpu_model_result.mutable_detect_result();
  for (auto &r : result.bboxes) {
    auto box = detect_result.add_bounding_box();
    box->set_label(r.label);
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  LOG(INFO) << "detect_result.bounding_box().size() "
            << detect_result.bounding_box().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::TFSSDResult>(
    const vitis::ai::TFSSDResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &detect_result = *dpu_model_result.mutable_detect_result();
  for (auto &r : result.bboxes) {
    auto box = detect_result.add_bounding_box();
    box->set_label(r.label);
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  LOG(INFO) << "detect_result.bounding_box().size() "
            << detect_result.bounding_box().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::RefineDetResult>(
    const vitis::ai::RefineDetResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;

  auto &refine_det_result = *dpu_model_result.mutable_refine_det_result();
  for (auto &r : result.bboxes) {
    auto box = refine_det_result.add_bounding_box();
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::YOLOv2Result>(
    const vitis::ai::YOLOv2Result &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;
  auto &detect_result = *dpu_model_result.mutable_detect_result();
  for (auto &r : result.bboxes) {
    auto box = detect_result.add_bounding_box();
    box->set_label(r.label);
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  LOG(INFO) << "detect_result.bounding_box().size() "
            << detect_result.bounding_box().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::YOLOv3Result>(
    const vitis::ai::YOLOv3Result &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;
  auto &detect_result = *dpu_model_result.mutable_detect_result();
  for (auto &r : result.bboxes) {
    auto box = detect_result.add_bounding_box();
    box->set_label(r.label);
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  LOG(INFO) << "detect_result.bounding_box().size() "
            << detect_result.bounding_box().size() << " ";
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult
process_result<vitis::ai::ClassificationResult>(
    const vitis::ai::ClassificationResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;
  auto &classification_result =
      *dpu_model_result.mutable_classification_result();
  for (auto &r : result.scores) {
    auto score = classification_result.add_score();
    score->set_index(r.index);
    score->set_score(r.score);
  }
  return dpu_model_result;
}

template <>
vitis::ai::proto::DpuModelResult process_result<vitis::ai::FaceDetectResult>(
    const vitis::ai::FaceDetectResult &result) {
  vitis::ai::proto::DpuModelResult dpu_model_result;
  auto &detect_result = *dpu_model_result.mutable_detect_result();
  for (auto &r : result.rects) {
    auto box = detect_result.add_bounding_box();
    box->set_x(r.x);
    box->set_y(r.y);
    box->set_width(r.width);
    box->set_height(r.height);
    box->set_score(r.score);
  }
  LOG(INFO) << "detect_result.bounding_box().size() "
            << detect_result.bounding_box().size() << " ";
  return dpu_model_result;
}

}  // namespace ai
}  // namespace vitis
