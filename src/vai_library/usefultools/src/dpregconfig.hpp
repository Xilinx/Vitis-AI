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
#include <memory>
#include <string>
#include <vector>

namespace xilinx {
namespace dpregconfig {

struct Config {
  std::string keyword;
  uint32_t offset;
  int bit_from;
  int bit_to;
  std::string name;
  std::string privilege;
  uint32_t value;
  std::string function;
  std::string unit;
  std::vector<std::pair<int, std::string> > dicts;
  std::string desc;
};

class RegConfig {
 public:
  static std::unique_ptr<RegConfig> create(const char* config_file);

 public:
  explicit RegConfig();
  RegConfig(const RegConfig&) = delete;
  virtual ~RegConfig();

 public:
  virtual void run() = 0;
  virtual std::vector<Config> getConfigs() = 0;
};
}  // namespace dpregconfig
}  // namespace xilinx
