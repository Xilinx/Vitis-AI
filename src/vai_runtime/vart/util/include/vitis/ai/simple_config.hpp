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
#pragma once
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
namespace vitis {
namespace ai {

class SimpleConfig {
 public:
  static std::shared_ptr<SimpleConfig> getOrCreateSimpleConfig(
      const std::string& filename);

  SimpleConfig(const std::string& filename);

  template <typename T>
  T as(const std::string& name) const;

  struct SimpleConfigViewer {
    SimpleConfigViewer(const SimpleConfig& cfg, const std::string& name);

    template <class T>
    SimpleConfigViewer operator[](const T& name) const;

    SimpleConfigViewer operator()(const std::string& name) const;

    SimpleConfigViewer operator()(int index) const;
    std::vector<SimpleConfigViewer> fields() const;

    bool has(const std::string& name) const;
    bool has(size_t idx) const;
    template <typename T>
    T as() const;

    const SimpleConfig& cfg_;
    std::string name_;
  };

  struct SimpleConfigViewer operator()(
      const std::string& name = std::string()) const;

  bool has(const std::string& name) const;

 private:
  std::map<std::string, std::string> values_;
  std::vector<std::string> fields_;
  friend struct SimpleConfigViewer;
  /* following are private help functions*/
 private:
  void Initialize(const std::string& filename);

  template <typename T>
  static void ParseValue(const std::string& text, T& value);
};
}  // namespace ai
}  // namespace vitis
