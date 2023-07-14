/*
Copyright 2019 Xilinx Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

#include "vitis/ai/target_factory.hpp"

#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#endif
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>

#include <UniLog/UniLog.hpp>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>

#include "config.hpp"
#include "target_list.hpp"

namespace vitis {
namespace ai {

static uint64_t type2int(const std::string& type) {
  uint32_t ret = 0U;
  if (type == "DPUCZDX8G")
    ret = 1U;
  else if (type == "DPUCAHX8H")
    ret = 2U;
  else if (type == "DPUCAHX8L")
    ret = 3U;
  else if (type == "DPUCZDI4G")
    ret = 4U;
  else if (type == "DPUCVDX8H")
    ret = 5U;
  else if (type == "DPUCVDX8G")
    ret = 6U;
  else if (type == "DPUCADF8H")
    ret = 7U;
  else if (type == "IPU_PHX")
    ret = 8U;
  else if (type == "DPUCV2DX8G")
    ret = 9U;
  else
    UNI_LOG_FATAL(TARGET_FACTORY_INVALID_TYPE) << type;
  return ret;
}

static std::string int2type(uint64_t type) {
  std::string ret = "";
  if (type == 1U)
    ret = "DPUCZDX8G";
  else if (type == 2U)
    ret = "DPUCAHX8H";
  else if (type == 3U)
    ret = "DPUCAHX8L";
  else if (type == 4U)
    ret = "DPUCZDI4G";
  else if (type == 5U)
    ret = "DPUCVDX8H";
  else if (type == 6U)
    ret = "DPUCVDX8G";
  else if (type == 7U)
    ret = "DPUCADF8H";
  else if (type == 8U)
    ret = "IPU_PHX";
  else if (type == 9U)
    ret = "DPUCV2DX8G";
  else
    UNI_LOG_FATAL(TARGET_FACTORY_INVALID_TYPE) << type;
  return ret;
}

std::string to_string(const std::map<std::string, std::uint64_t>& value) {
  std::ostringstream str;
  int c = 0;
  str << "{";
  for (auto& x : value) {
    if (c++ != 0) {
      str << ",";
    }
    str << x.first << "=>" << std::hex << "0x" << x.second << std::dec;
  }
  str << "}";
  return str.str();
};

const Target create_target_v2(const std::uint64_t fingerprint);
const Target create_target_DPUCZDX8G_ISA1(const std::uint64_t fingerprint);
const Target create_target_DPUCVDX8G_ISA2(const std::uint64_t fingerprint);
const Target create_target_DPUCVDX8G_ISA3(const std::uint64_t fingerprint);
const Target create_target_DPUCV2DX8G_ISA0(const std::uint64_t fingerprint);
const Target create_target_DPUCV2DX8G_ISA1(const std::uint64_t fingerprint);

class TargetFactoryImp : public TargetFactory {
 public:
  const Target create(const std::string& name) const override {
    return this->create(this->get_fingerprint(name));
  }

  const Target create(const std::string& type, std::uint64_t isa_version,
                      std::uint64_t feature_code) const override {
    return this->create(this->get_fingerprint(type, isa_version, feature_code));
  }

  const Target create(const std::uint64_t fingerprint) const override {
    if (map_fingerprint_target_.count(fingerprint) != 0) {
      return map_fingerprint_target_.at(fingerprint);
    } else {
      auto type = int2type(fingerprint >> 56);
      auto isa_version = ((fingerprint & 0x00ff000000000000) >> 48);
             if (type == "DPUCZDX8G" && isa_version == 0) {
        return create_target_v2(fingerprint);
      } else if (type == "DPUCZDX8G" && isa_version == 1) {
        return create_target_DPUCZDX8G_ISA1(fingerprint);
      } else if (type == "DPUCVDX8G" && isa_version == 2) {
        return create_target_DPUCVDX8G_ISA2(fingerprint);
      } else if (type == "DPUCVDX8G" && isa_version == 3) {
        return create_target_DPUCVDX8G_ISA3(fingerprint);
      } else if (type == "DPUCV2DX8G" && isa_version == 0) {
        return create_target_DPUCV2DX8G_ISA0(fingerprint);
      } else if (type == "DPUCV2DX8G" && isa_version == 1) {
        return create_target_DPUCV2DX8G_ISA1(fingerprint);
      } else {
        UNI_LOG_FATAL(TARGET_FACTORY_UNREGISTERED_TARGET)
            << "Cannot find or create target with fingerprint=0x" << std::hex
            << std::setfill('0') << std::setw(16) << fingerprint;
        Target target;
        return target;
      }
    }
  }

  const std::uint64_t get_fingerprint(const std::string& name) const override {
    UNI_LOG_CHECK(map_name_fingerprint_.count(name) != 0,
                  TARGET_FACTORY_UNREGISTERED_TARGET)
        << "Cannot find target with name " << name
        << ", valid names are: " << to_string(map_name_fingerprint_);
    return map_name_fingerprint_.at(name);
  }

  const std::uint64_t get_fingerprint(
      const std::string& type, std::uint64_t isa_version,
      std::uint64_t feature_code) const override {
    uint64_t fingureprint = 0U;
    UNI_LOG_CHECK((feature_code & 0xffff000000000000) == 0,
                  TARGET_FACTORY_INVALID_ISA_VERSION)
        << "0x" << std::hex << std::setfill('0') << std::setw(16)
        << feature_code;
    UNI_LOG_CHECK((isa_version & 0xffffffffffffff00) == 0,
                  TARGET_FACTORY_INVALID_FEATURE_CODE)
        << "0x" << std::hex << std::setfill('0') << std::setw(16)
        << isa_version;
    fingureprint |= feature_code;
    fingureprint |= isa_version << 48;
    fingureprint |= type2int(type) << 56;
    return fingureprint;
  }

  void dump(const Target& target, const std::string& file) const override {
    auto fd = open(file.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 00644);
    google::protobuf::io::FileOutputStream fstream(fd);
    google::protobuf::TextFormat::Print(target, &fstream);
  }


  bool is_registered_target(const std::string& name) const override {
    return map_name_fingerprint_.count(name) != 0;
  }

 public:
  void register_h(const Target& target) {
    auto name = target.name();
    auto fingerprint = this->get_fingerprint(
        target.type(), target.isa_version(), target.feature_code());
    UNI_LOG_CHECK(map_fingerprint_target_.count(fingerprint) == 0,
                  TARGET_FACTORY_MULTI_REGISTERED_TARGET)
        << "fingerprint=0x" << std::hex << std::setw(16) << std::setfill('0')
        << fingerprint;
    UNI_LOG_CHECK(map_name_fingerprint_.count(name) == 0,
                  TARGET_FACTORY_MULTI_REGISTERED_TARGET)
        << "name=" << name;
    map_fingerprint_target_.emplace(fingerprint, target);
    map_name_fingerprint_.emplace(name, fingerprint);
  }

 public:
  TargetFactoryImp() = default;
  TargetFactoryImp(const TargetFactoryImp&) = delete;
  TargetFactoryImp& operator=(const TargetFactoryImp&) = delete;

 private:
  std::map<std::uint64_t, const Target> map_fingerprint_target_;
  std::map<std::string, std::uint64_t> map_name_fingerprint_;
};

static std::unique_ptr<std::vector<std::string>> get_target_prototxt_list() {
  auto ret = std::make_unique<std::vector<std::string>>();
  for (auto target_proto : TARGET_PROTOTXTS) {
    ret->emplace_back(std::string(target_proto));
  }
  return ret;
}

static void register_targets(TargetFactoryImp* factory) {
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  auto target_prototxt_list = get_target_prototxt_list();
  for (auto& target_prototxt : *target_prototxt_list) {
    Target target;
    UNI_LOG_CHECK(
        google::protobuf::TextFormat::ParseFromString(target_prototxt, &target),
        TARGET_FACTORY_PARSE_TARGET_FAIL)
        << "Cannot parse prototxt: \n"
        << target_prototxt;
    UNI_LOG_CHECK(target.name() != "" && target.type() != "",
                  TARGET_FACTORY_PARSE_TARGET_FAIL)
        << "Uninitialized name or type";
    factory->register_h(target);
  }
}

const TargetFactory* target_factory() {
  static std::once_flag register_once_flag;
  static TargetFactoryImp self;
  std::call_once(register_once_flag, register_targets, &self);
  return &self;
}

const std::string TargetFactory::get_lib_name() {
  const auto ret =
      std::string{PROJECT_NAME} + "." + std::string{PROJECT_VERSION};
  return ret;
}

const std::string TargetFactory::get_lib_id() {
  const auto ret = std::string{PROJECT_GIT_COMMIT_ID};
  return ret;
}

}  // namespace ai
}  // namespace vitis
