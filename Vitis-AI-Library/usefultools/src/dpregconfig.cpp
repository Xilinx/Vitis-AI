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
#include "./dpregconfig.hpp"
#include "./dpregconfig_imp.hpp"

namespace  xilinx {namespace dpregconfig {
RegConfig::RegConfig() {}
RegConfig::~RegConfig() {}

std::unique_ptr<RegConfig> RegConfig::create(const char* config_file) {
    return std::unique_ptr<RegConfig>(new RegConfigImp(config_file));
}
}}
