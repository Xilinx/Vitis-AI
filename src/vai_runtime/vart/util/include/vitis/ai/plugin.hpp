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
#include <string>

namespace vitis {
namespace ai {
using plugin_t = void*;

enum class scope_t { PUBLIC, PRIVATE };
plugin_t open_plugin(const std::string& name, scope_t scope);
void* plugin_sym(plugin_t plugin, const std::string& name);
std::string plugin_error(plugin_t plugin);
void close_plugin(plugin_t plugin);
void register_plugin(const std::string& name, const std::string& symbol,
                     void* addr);
class StaticPluginRegister {
 public:
  StaticPluginRegister(const char* name, const char* symbol, void* addr);
};
}  // namespace ai
}  // namespace vitis
