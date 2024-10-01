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

#include <dirent.h>
#include <glog/logging.h>
#include <google/protobuf/stubs/common.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <UniLog/UniLog.hpp>

#include <algorithm>
#include <cassert>
#include <fstream>
#include <mutex>

#include "vitis/ai/proto/dpu_model_param.pb.h"
namespace vitis {
namespace ai {

using namespace std;
static std::shared_ptr<vitis::ai::proto::DpuModelParamList> create();
static std::vector<std::string> collect_proto_txt_file_names();
static std::string slurp(const char* filename);
static bool merge_file(const string& filename,
                       vitis::ai::proto::DpuModelParamList* list);
static std::shared_ptr<vitis::ai::proto::DpuModelParamList> instance();

extern "C" vitis::ai::proto::DpuModelParam* find(
    const std::string& model_name) {
  auto model_list = instance();
  ostringstream names;
  auto size = model_list->model_size();
  auto models = model_list->mutable_model()->mutable_data();
  for (auto i = 0; i < size; ++i) {
    auto m = models[i];
    if (m->name() == model_name) {
      if (0) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                  << "]"                                                  //
                  << "m " << (void*)m << " "                              //
                  << "model_list.get " << (void*)model_list.get() << " "  //
                  << std::endl;

        /*
        for (const auto &k : m->ssdinfo().kernel()) {
          std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                    << "]"                            //
                    << "k.name() " << k.name() << " " //
                    << std::endl;
        }
        */
      }
      return m;
    }
  }
  // LOG(FATAL) << "cannot find model " << model_name
  //           << ", valid names are as below:" << names.str();
  UNI_LOG_FATAL(VAILIB_MODEL_CONFIG_NOT_FIND)
      << "cannot find model " << model_name
      << ", valid names are as below:" << names.str();
  // never goes here
  {
    // static auto v = vitis::ai::proto::TrainEvalPipelineConfig{};  //
    // suppression warning
    assert(false);
    return nullptr;
  }
}

extern "C" vitis::ai::proto::DpuModelParam get_config(
    const std::string& model_name) {
  auto ret = vitis::ai::proto::DpuModelParam{};
  auto m = find(model_name);
  if (0)
    std::cout << "get_config() address " << (void*)m << " "  //
              << std::endl;
  ret.CopyFrom(*m);
  if (0) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "m " << (void*)m << " "                                      //
              << "&ret " << &ret << " "                                       //
              << "name() " << ret.name() << " " << std::endl;
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "HELLO"
              << "HELLO"
              << "HELLO"
              << "sizeof(ret) " << sizeof(ret) << " "  //
              << "sizeof(*m) " << sizeof(*m) << " "    //
              << std::endl;
    /*
    for (const auto &k : ret.kernel()) {
      std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                << "]" //
                << "k.name " << &vitis::ai::proto::DpuKernelParam::name
                << " "                                        //
                << "k.name_size() " << k.name().size() << " " //
                << "k.name " << k.name() << " "               //
                << std::endl;
    } */
    // std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"
    // //
    //          << "HELLO" << std::endl;
  }
  return ret;
}

static std::shared_ptr<vitis::ai::proto::DpuModelParamList> instance() {
  static auto mutex = std::unique_ptr<std::mutex>{new std::mutex()};
  static std::shared_ptr<vitis::ai::proto::DpuModelParamList> ret;
  std::lock_guard<std::mutex> lock(*mutex.get());
  if (!ret) {
    google::protobuf::LogSilencer* s1 = new google::protobuf::LogSilencer;
    if (0) {
      std::cerr << "suppress warning of unused variable " << s1 << std::endl;
    }
    ret = create();
  }
  assert(ret != nullptr);
  return ret;
}

static std::shared_ptr<vitis::ai::proto::DpuModelParamList> create() {
  auto file_names = collect_proto_txt_file_names();
  auto ret = std::make_shared<vitis::ai::proto::DpuModelParamList>();
  auto all_failed = true;
  for (const auto& filename : file_names) {
    if (0)
      std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                << "]"                             //
                << "filename " << filename << " "  //
                << std::endl;
    all_failed = merge_file(filename, ret.get()) == false && all_failed;
    if (0) {
      for (const auto& m : ret->model()) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                  << "]"                           //
                  << "name() " << m.name() << " "  //
                  << std::endl;
        /*
        for (const auto &k : m.kernel()) {
          std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                    << "]"                            //
                    << "k.name() " << k.name() << " " //
                    << std::endl;
        } */
      }
    }
  }
  // LOG_IF(FATAL, all_failed) << "cannot parse any config files";
  if (all_failed) {
    UNI_LOG_FATAL(VAILIB_MODEL_CONFIG_CONFIG_PARSE_ERROR)
        << "cannot parse any config files";
  }
  return ret;
}

static std::vector<std::string> collect_proto_txt_file_names() {
  const char suffix[] = {".prototxt"};
  const char* dirname = "/etc/dpu_model_param.conf.d";
  auto dir = opendir(dirname);
  // CHECK(dir != nullptr) << " cannot open directory: dirname = " << dirname;
  UNI_LOG_CHECK(dir != nullptr, VAILIB_MODEL_CONFIG_OPEN_ERROR)
      << " cannot open directory: dirname = " << dirname;
  auto ret = std::vector<std::string>{};
  for (auto ent = readdir(dir); ent != nullptr; ent = readdir(dir)) {
    auto len = strlen(ent->d_name);
    auto end_with_suffix =
        len >= sizeof(suffix) &&
        strcmp(&ent->d_name[len - sizeof(suffix) + 1], suffix) == 0;
    struct stat st;
    auto filename = std::string{dirname} + "/" + ent->d_name;
    // CHECK_EQ(stat(filename.c_str(), &st), 0)
    //    << " cannot check stat of " << filename;
    UNI_LOG_CHECK(stat(filename.c_str(), &st) == 0,
                  VAILIB_MODEL_CONFIG_OPEN_ERROR)
        << " cannot check stat of " << filename;
    auto is_normal_file = S_ISREG(st.st_mode);
    if (is_normal_file && end_with_suffix) {
      ret.emplace_back(std::string(dirname) + "/" + std::string{ent->d_name});
    }
  }
  closedir(dir);
  std::sort(ret.begin(), ret.end());
  return ret;
}

static bool merge_file(const string& filename,
                       vitis::ai::proto::DpuModelParamList* list) {
  auto text = slurp(filename.c_str());
  vitis::ai::proto::DpuModelParamList mlist;
  auto ok = google::protobuf::TextFormat::ParseFromString(text, &mlist);

  if (!ok) {
    return ok;
  }

  bool bOK = google::protobuf::TextFormat::MergeFromString(text, list);

  if (0) {
    std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__ << "]"  //
              << "ok " << ok << " "                                           //
              << "mlist " << mlist.DebugString() << " "                       //
              << std::endl;
    for (const auto& m : mlist.model()) {
      std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                << "]"                           //
                << "name() " << m.name() << " "  //
                << std::endl;
      /*
      for (const auto &k : m.kernel()) {
        std::cerr << __FILE__ << ":" << __LINE__ << ": [" << __FUNCTION__
                  << "]"                            //
                  << "k.name() " << k.name() << " " //
                  << std::endl;
      } */
    }
  }

  LOG_IF(WARNING, !bOK) << "parse configuration file failed. filename = "
                        << filename;
  return bOK;
}

static std::string slurp(const char* filename) {
  std::ifstream in;
  in.open(filename, std::ifstream::in);
  std::stringstream sstr;
  sstr << in.rdbuf();
  in.close();
  return sstr.str();
}

}  // namespace ai
}  // namespace vitis
