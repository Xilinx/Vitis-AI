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
#include <vitis/ai/refinedet.hpp>
#include <vitis/ai/yolov3.hpp>
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
    ModelDef<vitis::ai::proto::DpuModelParam::YOLOv3, vitis::ai::YOLOv3>,
    ModelDef<vitis::ai::proto::DpuModelParam::REFINEDET, vitis::ai::RefineDet>,
    ModelDef<vitis::ai::proto::DpuModelParam::CLASSIFICATION,
             vitis::ai::Classification>
    /* list of supported models end */
    >;

std::unique_ptr<General> General::create(const std::string &model_name,
                                         bool need_preprocess) {
  auto model = get_config(model_name);
  auto model_type = model.model_type();
  return ListOfSupportedModels::create(model_type, model_name);
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
