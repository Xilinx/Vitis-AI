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

#pragma once
#if defined(_WIN32)
#if TARGET_FACTORY_USE_DLL == 1
#ifdef TARGET_FACTORY_EXPORT
#define TARGET_FACTORY_DLLESPEC __declspec(dllexport)
#else
#define TARGET_FACTORY_DLLESPEC __declspec(dllimport)
#endif
#else
#define TARGET_FACTORY_DLLESPEC
#endif
#else
#define TARGET_FACTORY_DLLESPEC __attribute__((visibility("default")))
#endif

#include <string>

#include "target.pb.h"

namespace vitis {
namespace ai {

class TARGET_FACTORY_DLLESPEC TargetFactory {
 public:
  virtual const Target create(const std::string& name) const = 0;
  virtual const Target create(const std::uint64_t fingerprint) const = 0;
  virtual const Target create(const std::string& type,
                              std::uint64_t isa_version,
                              std::uint64_t feature_code) const = 0;

  virtual const std::uint64_t get_fingerprint(
      const std::string& name) const = 0;
  virtual const std::uint64_t get_fingerprint(
      const std::string& type, std::uint64_t isa_version,
      std::uint64_t feature_code) const = 0;

  virtual void dump(const Target& target, const std::string& file) const = 0;
  virtual bool is_registered_target(const std::string& name) const = 0;
  static const std::string get_lib_name();
  static const std::string get_lib_id();

 public:
  virtual ~TargetFactory() = default;
};

TARGET_FACTORY_DLLESPEC
const TargetFactory* target_factory();

}  // namespace ai
}  // namespace vitis
