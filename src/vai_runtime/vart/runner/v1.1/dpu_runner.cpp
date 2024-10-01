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
#ifndef USE_JSON_C
#define USE_JSON_C 1
#endif
#if USE_JSON_C
#include <glog/logging.h>
#include <json-c/json.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <vitis/ai/plugin.hpp>
#include <xir/attrs/attrs.hpp>
#include <xir/graph/graph.hpp>

#include "runner_adaptor.hpp"
#include "vart/runner.hpp"
#include "vitis/ai/env_config.hpp"
#include "vitis/ai/weak.hpp"
DEF_ENV_PARAM(XLNX_ENABLE_ASYNC_RUNNER, "-1");
DEF_ENV_PARAM(XLNX_NUM_OF_DPU_RUNNERS, "-1");
DEF_ENV_PARAM(DEBUG_USE_DUMMY_RUNNER, "0");
DEF_ENV_PARAM_2(XLNX_RUNNER_INTERCEPTION, "", std::string);
namespace vitis {
namespace ai {
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
  auto meta_filename = std::filesystem::path(model_directory) / "meta.json";
  // Why (const char*)?: json-c might have a bad design in API, it is wchar_t *
  // on Windows.
  json_object* value =
      json_object_from_file((const char*)meta_filename.c_str());
  LOG_IF(WARNING, value != nullptr)
      << "failed to read meta file! filename=" << meta_filename
      << " use default value. {}";
  if (value == nullptr) {
    value = json_object_new_object();
  }
  CHECK(json_object_is_type(value, json_type_object))
      << "not a json object. value=" << json_object_to_json_string(value);
  return value;
}

static std::string basename(const std::string& filename) {
  std::string ret;
  CHECK(!filename.empty());
  if (filename.back() == std::filesystem::path::preferred_separator) {
    ret.assign(filename.begin(), filename.begin() + filename.size() - 1);
  } else {
    ret.assign(filename.begin(), filename.end());
  }
  auto pos = ret.rfind(std::filesystem::path::preferred_separator);
  if (pos == std::string::npos) {
    pos = 0;
  } else {
    pos = pos + 1;
  }
  return ret.substr(pos);
}

static std::string safe_read_string_as_file_name(json_object* value,
                                                 const std::string& key,
                                                 const std::string& dirname) {
  auto filename = safe_read_string(value, key);
  return (filename[0] == std::filesystem::path::preferred_separator)
             ? filename
             : (std::filesystem::path(dirname) / filename).string();
}

static std::string safe_read_string_as_file_name_with_default_value(
    json_object* value, const std::string& key, const std::string& dirname,
    const std::string& default_value) {
  auto filename = safe_read_string_with_default(value, key, default_value);
  return (filename[0] == std::filesystem::path::preferred_separator)
             ? filename
             : (std::filesystem::path(dirname) / filename).string();
}

static std::unique_ptr<xir::Attrs> create_attrs_from_meta_dot_json(
    json_object* value, const std::string& model_directory) {
  (void)safe_read_string_as_file_name_with_default_value;
  (void)safe_read_string_as_file_name;
  (void)safe_read_string_or_vec_string;
  auto ret = xir::Attrs::create();
  auto dirname = basename(model_directory);
  LOG(INFO) << "debug "
            << "dirname " << dirname << " "                  //
            << "model_directory " << model_directory << " "  //
      ;
  ret->set_attr<std::string>(
      "filename", safe_read_string_as_file_name_with_default_value(
                      value, "filename", model_directory, dirname + ".xmodel"));
  // used attrs:
  //
  //  filename: string, optional, the xmodel filename, as same as the directory
  //  name
  //
  // async: bool = true, optional
  //
  // lib: string = "libvart-dpu-runner.so", optional, read from subgraph
  // attribute.
  //
  // mode: string = "run", optional.

  //
  if (!ret->has_attr("async")) {
    ret->set_attr<bool>("async", true);
  }
  if (ENV_PARAM(XLNX_ENABLE_ASYNC_RUNNER) >= 0) {
    ret->set_attr<bool>("async", ENV_PARAM(XLNX_ENABLE_ASYNC_RUNNER) != 0);
  }
  if (ret->get_attr<bool>("async")) {
    ret->set_attr<std::string>("interception",
                               std::string("libvart-async-runner.so"));
  }
  if (!ENV_PARAM(XLNX_RUNNER_INTERCEPTION).empty()) {
    ret->set_attr<std::string>("interception",
                               ENV_PARAM(XLNX_RUNNER_INTERCEPTION));
  }
  if (!ret->has_attr("num_of_dpu_runners")) {
    ret->set_attr<size_t>("num_of_dpu_runners", 4u);
  }
  if (ENV_PARAM(XLNX_NUM_OF_DPU_RUNNERS) >= 0) {
    ret->set_attr<size_t>("num_of_dpu_runners",
                          (size_t)ENV_PARAM(XLNX_NUM_OF_DPU_RUNNERS));
  }
  if (ENV_PARAM(DEBUG_USE_DUMMY_RUNNER)) {
    ret->set_attr<std::string>("lib", "libvart-dummy-runner.so");
  }
  return ret;
}

std::vector<std::unique_ptr<vitis::ai::DpuRunner>> DpuRunner::create_dpu_runner(
    const std::string& model_directory) {
  auto value = read_json_from_directory(model_directory);
  std::shared_ptr<xir::Attrs> attrs =
      create_attrs_from_meta_dot_json(value, model_directory);
  auto filename = attrs->get_attr<std::string>("filename");
  auto graph = vitis::ai::WeakStore<std::string, GraphHolder>::create(filename,
                                                                      filename);
  attrs->set_attr<std::string>("dirname", model_directory);
  auto ret = std::vector<std::unique_ptr<vitis::ai::DpuRunner>>();
  ret.reserve(10);
  auto root = graph->graph_->get_root_subgraph();
  auto children = root->children_topological_sort();
  for (auto child : children) {
    if (child->get_attr<std::string>("device") == "DPU") {
      ret.emplace_back(new RunnerAdaptor(graph, attrs, child));
    }
  }
  json_object_put(value);
  return ret;
}
}  // namespace ai
}  // namespace vitis
#endif
