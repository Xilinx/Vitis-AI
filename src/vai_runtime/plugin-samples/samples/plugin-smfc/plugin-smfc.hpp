/* Copyright 2019 Xilinx Inc.
**
** Licensed under the Apache License, Version 2.0 (the "License");
** you may not use this file except in compliance with the License.
** You may obtain a copy of the License at
**
**     http://www.apache.org/licenses/LICENSE-2.0
**
** Unless required by applicable law or agreed to in writing, software
** distributed under the License is distributed on an "AS IS" BASIS,
** WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
** See the License for the specific language governing permissions and
** limitations under the License.
*/

#include "plugin/Plugin.hpp"

using namespace xcompiler;

class plugin_smfc : public Plugin {
 public:
  std::string get_plugin_name() override;

  std::string get_device() override;

  std::map<std::string, std::string> get_runner() override;

  std::set<xir::Subgraph*> partition(xir::Graph* graph) override;

  void compile(xir::Subgraph* subgraph) override;
};

extern "C" Plugin* get_plugin() {
  UNI_LOG_INFO << "\033[32m"
               << "get plugin_smfc"
               << "\033[0m";
  return new plugin_smfc();
}