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

#include <dlfcn.h>
#include <json-c/json.h>

#include <UniLog/UniLog.hpp>
#include <xir/graph/graph.hpp>
#include <xir/graph/subgraph.hpp>

#include "vart/runner.hpp"
#include "vitis/ai/env_config.hpp"

DEF_ENV_PARAM(DEBUG_RUNNER, "0");

namespace vart {

//# Bring back older meta json read utility functions
static std::string safe_read_string_with_default(
    json_object* value, const std::string& key,
    const std::string& default_value) {
  json_object* field = nullptr;
  if ((json_object_object_get_ex(value, key.c_str(), &field)) &&
      (json_object_is_type(field, json_type_string))) {
    return json_object_get_string(field);
  }
  return default_value;
}

static std::string safe_read_string(json_object* value,
                                    const std::string& key) {
  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(value, key.c_str(), &field))
      << "no such field! key=" << key
      << ", value=" << json_object_to_json_string(value);
  CHECK(json_object_is_type(field, json_type_string))
      << "not a string! key=" << key
      << ", value=" << json_object_to_json_string(value);
  return json_object_get_string(field);
}

static std::vector<std::string> safe_read_string_or_vec_string(
    json_object* value, const std::string& key) {
  json_object* field = nullptr;
  CHECK(json_object_object_get_ex(value, key.c_str(), &field))
      << "no such field! key=" << key
      << ", value=" << json_object_to_json_string(value);
  CHECK(json_object_is_type(field, json_type_string) ||
        json_object_is_type(field, json_type_array))
      << "not a string or array of string ! key=" << key
      << ", value=" << json_object_to_json_string(value);
  if (json_object_is_type(field, json_type_string)) {
    return std::vector<std::string>{std::string(json_object_get_string(field))};
  }
  // must be an array
  auto ret = std::vector<std::string>{};
  auto size = json_object_array_length(field);
  ret.reserve(size);
  for (decltype(size) idx = 0; idx < size; ++idx) {
    auto elt = json_object_array_get_idx(field, idx);
    CHECK(json_object_is_type(elt, json_type_string))
        << "element is not a string or array of string ! key="
        << ", idx=" << idx << ", value=" << json_object_to_json_string(value);
    ret.emplace_back(json_object_get_string(elt));
  }
  return ret;
}
static json_object* read_json_from_directory(
    const std::string& model_directory) {
  auto meta_filename = model_directory + "/" + "meta.json";
  json_object* value = json_object_from_file(meta_filename.c_str());
  CHECK(value != nullptr) << "failed to read meta file! filename="
                          << meta_filename;
  CHECK(json_object_is_type(value, json_type_object))
      << "not a json object. value=" << json_object_to_json_string(value);
  return value;
}

static std::string get_target(json_object* value) {
  return safe_read_string_with_default(value, "target", "DPUv2");
}

static std::string safe_read_string_as_file_name(json_object* value,
                                                 const std::string& key,
                                                 const std::string& dirname) {
  auto filename = safe_read_string(value, key);
  return (filename[0] == '/') ? filename : dirname + "/" + filename;
}

static std::string safe_read_string_as_file_name_with_default_value(
    json_object* value, const std::string& key, const std::string& dirname,
    const std::string& default_value) {
  auto filename = safe_read_string_with_default(value, key, default_value);
  return (filename[0] == '/') ? filename : dirname + "/" + filename;
}

static DpuMeta read_dpu_meta_from_value(json_object* value,
                                        const std::string& dirname) {
  std::string target = get_target(value);
  std::string lib = safe_read_string(value, "lib");
  DpuMeta ret;
  ret.target = target;
  ret.lib = lib;
  ret.dirname = dirname;
  ret.filename = safe_read_string_as_file_name(value, "filename", dirname);
  ret.kernels = safe_read_string_or_vec_string(value, "kernel");
  ret.config_file = safe_read_string_as_file_name_with_default_value(
      value, "config_file", dirname, "config.prototxt");
  return ret;
}

//# Read meta info from json and call to DPUV1 runner
static std::vector<std::unique_ptr<vart::Runner>>* create_dpu_runner_by_meta(
    const DpuMeta& dpuMeta) {
  typedef std::vector<std::unique_ptr<vart::Runner>>* (*INIT_FUN)(
      const DpuMeta& dpuMeta);
  INIT_FUN init_fun = NULL;
  auto handle = dlopen(dpuMeta.lib.c_str(), RTLD_LAZY);
  CHECK(handle != NULL) << "cannot open library!"
                        << " lib=" << dpuMeta.lib << ";error=" << dlerror();
  dlerror();
  init_fun = (INIT_FUN)dlsym(handle, "create_runner");
  CHECK(init_fun != NULL) << "cannot load symbol 'create_runner'!"
                          << "! lib=" << dpuMeta.lib << ";error=" << dlerror();
  return init_fun(dpuMeta);
}

//# Runner DPUV2
std::unique_ptr<Runner> Runner::create_runner(const xir::Subgraph* subgraph,
                                              const std::string& mode) {
  UNI_LOG_CHECK(subgraph != nullptr, VART_RUNNER_CONSTRUCTION_FAIL)
      << "Invalid subgraph!";
  UNI_LOG_CHECK(subgraph->has_attr("runner"), VART_RUNNER_CONSTRUCTION_FAIL)
      << "Cannot find runner attr, this subgraph may not be compiled! subgraph "
         "name: "
      << subgraph->get_name();
  auto libs = subgraph->get_attr<std::map<std::string, std::string>>("runner");
  auto iter_lib = libs.find(mode);
  UNI_LOG_CHECK(iter_lib != libs.end(), VART_RUNNER_CONSTRUCTION_FAIL)
      << "Cannot find runner for mode " << mode
      << "! subgraph name: " << subgraph->get_name();
  typedef vart::Runner* (*INIT_FUN)(const xir::Subgraph* subgraph);
  INIT_FUN init_fun = NULL;
  auto handle = dlopen(iter_lib->second.c_str(), RTLD_LAZY);
  UNI_LOG_CHECK(handle != NULL, VART_RUNNER_CONSTRUCTION_FAIL)
      << "cannot open library!"
      << " lib=" << iter_lib->second << ", error=" << dlerror();
  init_fun = (INIT_FUN)dlsym(handle, "create_runner");
  UNI_LOG_CHECK(init_fun != NULL, VART_RUNNER_CONSTRUCTION_FAIL)
      << "cannot load symbol 'create_runner'!"
      << " lib=" << iter_lib->second << ", error=" << dlerror();
  return std::unique_ptr<vart::Runner>(init_fun(subgraph));
}

//# Runner

std::unique_ptr<Runner> Runner::create_runner_with_attrs(
    const xir::Subgraph* subgraph, xir::Attrs* attrs) {
  UNI_LOG_CHECK(subgraph != nullptr, VART_RUNNER_CONSTRUCTION_FAIL)
      << "Invalid subgraph!";
  UNI_LOG_CHECK(attrs != nullptr, VART_RUNNER_CONSTRUCTION_FAIL)
      << "Invalid attrs!";
  UNI_LOG_CHECK(subgraph->has_attr("runner"), VART_RUNNER_CONSTRUCTION_FAIL)
      << "Cannot find runner attr, this subgraph may not be compiled! subgraph "
         "name: "
      << subgraph->get_name();
  auto mode = std::string("run");
  if (attrs->has_attr("mode")) {
    mode = attrs->get_attr<std::string>("mode");
  }
  auto libs = subgraph->get_attr<std::map<std::string, std::string>>("runner");
  // "runner" is a map, e.g.
  // key: "runner"
  // value {
  //   map_string_2_string_value {
  //     value {
  //       key: "ref"
  //       value: "libvart-cpu-runner.so"
  //     }
  //     value {
  //       key: "run"
  //       value: "libvart-dpu-runner.so"
  //     }
  //     value {
  //       key: "sim"
  //       value: "libvart-sim-runner.so"
  //     }
  //   }
  // }
  auto iter_lib = libs.find(mode);
  UNI_LOG_CHECK(iter_lib != libs.end(), VART_RUNNER_CONSTRUCTION_FAIL)
      << "Cannot find runner for mode " << mode
      << "! subgraph name: " << subgraph->get_name();
  typedef vart::Runner* (*INIT_FUN)(const xir::Subgraph* subgraph,
                                    xir::Attrs* attrs);
  INIT_FUN init_fun = NULL;
  auto libname = iter_lib->second;
  // override the default runner defined in the subgraph via ATTRS
  // [code]
  //   attrs[rr]->set_attr("lib", std::map<std::string, std::string>{
  //                                  {"DPU", "libvart-dummy-runner.so"}});
  // }
  // [/code]
  if (attrs->has_attr("lib")) {
    auto override_libs =
        attrs->get_attr<std::map<std::string, std::string>>("lib");
    auto device = subgraph->get_attr<std::string>("device");
    auto override_iter_lib = override_libs.find(device);
    if (override_iter_lib != override_libs.end()) {
      auto new_libname = override_iter_lib->second;
      LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
          << "override default runner " << libname << " with new runner "
          << new_libname;
      libname = new_libname;
    } else {
      LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
          << "cannot find lib[" << device
          << "] in attrs, use default lib in the subgraph, i.e. " << libname;
    }
  }
  auto handle = dlopen(libname.c_str(), RTLD_LAZY | RTLD_GLOBAL);
  UNI_LOG_CHECK(handle != NULL, VART_RUNNER_CONSTRUCTION_FAIL)
      << "cannot open library!"
      << " lib=" << libname << ", error=" << dlerror();
  // finally we look up for the init function.
  init_fun = (INIT_FUN)dlsym(handle, "create_runner_with_attrs");
  UNI_LOG_CHECK(init_fun != NULL, VART_RUNNER_CONSTRUCTION_FAIL)
      << "cannot load symbol 'create_runner'!"
      << " lib=" << iter_lib->second << ", error=" << dlerror();
  // attrs
  if (attrs->has_attr("interception")) {
    auto interception_lib = attrs->get_attr<std::string>("interception");
    typedef vart::Runner* (*INTERCEPT_INIT_FUN)(
        INIT_FUN fun, const xir::Subgraph* subgraph, xir::Attrs* attrs);
    auto interception_handle =
        dlopen(interception_lib.c_str(), RTLD_LAZY | RTLD_GLOBAL);
    UNI_LOG_CHECK(interception_handle != NULL, VART_RUNNER_CONSTRUCTION_FAIL)
        << "cannot open library!"
        << " lib=" << interception_lib << ", error=" << dlerror();
    auto interception_fun = (INTERCEPT_INIT_FUN)dlsym(
        interception_handle, "create_runner_with_attrs");
    LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER))
        << "create runner via interception lib " << interception_lib;
    return std::unique_ptr<vart::Runner>(
        interception_fun(init_fun, subgraph, attrs));
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_RUNNER)) << "create runner via " << libname;
  return std::unique_ptr<vart::Runner>(init_fun(subgraph, attrs));
}

//# Method overload for DPUV1
std::vector<std::unique_ptr<Runner>> Runner::create_runner(
    const std::string& model_directory) {
  auto value = read_json_from_directory(model_directory);
  auto target = get_target(value);
  auto dpu_meta = read_dpu_meta_from_value(value, model_directory);
  dpu_meta.dirname = model_directory;
  auto ret = std::unique_ptr<std::vector<std::unique_ptr<vart::Runner>>>(
      create_dpu_runner_by_meta(dpu_meta));
  json_object_put(value);
  return std::move(*ret.get());
}

// default implements
Runner::TensorFormat Runner::get_tensor_format() {
  return Runner::TensorFormat::NHWC;
}
}  // namespace vart
