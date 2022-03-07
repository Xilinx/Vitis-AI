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

#include "cmd_dump_code.hpp"

#include <glog/logging.h>
#include <google/protobuf/text_format.h>
#include <sys/stat.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>

using namespace std;
static void mkdir_minus_p(const std::string& dirname) {
  CHECK(std::filesystem::create_directories(dirname))
      << "cannot create directories: " << dirname;
}

static bool is_exist_path(const std::string& filename) {
  return std::filesystem::exists(filename);
}

static std::string get_parent_path(const std::string& path) {
  return path.substr(0, path.find_last_of("/"));
}

static void create_parent_path(const std::string& path) {
  if (is_exist_path(path)) {
    return;
  }
  auto parent_path = get_parent_path(path);
  if (!is_exist_path(parent_path)) {
    create_parent_path(parent_path);
  }
  mkdir_minus_p(path);
}
static std::string mk_file_name(const std::string& dirname,
                                const std::string& file) {
  return dirname + "/" + file;
}

static void dump_subgraph1(const xir::Subgraph* sg,
                           const std::string& dirname) {
  const auto mode =
      std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  if (sg->has_attr("mc_code")) {
    auto mc_code = sg->get_attr<vector<char>>("mc_code");
    auto mc_code_file = mk_file_name(dirname, sg->get_name() + ".mc");
    CHECK(std::ofstream(mc_code_file, mode)
              .write((char*)&mc_code[0], mc_code.size())
              .good())
        << " faild to dump code to " << mc_code_file;
    cout << "dump mc code to " << mc_code_file << endl;
  }
  if (sg->has_attr("ac_code")) {
    auto mc_code = sg->get_attr<vector<string>>("ac_code");
    auto ac_code_file = mk_file_name(dirname, sg->get_name() + ".ac");
    auto str = std::ofstream(ac_code_file);
    for (const auto& line : mc_code) {
      str << line << "\n";
      CHECK(str.good()) << " faild to dump code to " << ac_code_file;
    }
    str.close();
    cout << "dump ac code to " << ac_code_file << endl;
  }
  if (sg->has_attr("reg_id_to_parameter_value")) {
    auto reg_id_to_parameter_value =
        sg->get_attr<std::map<std::string, std::vector<char>>>(
            "reg_id_to_parameter_value");
    for (const auto& c : reg_id_to_parameter_value) {
      if (!c.second.empty()) {
        auto file = mk_file_name(dirname, c.first + ".bin");
        CHECK(std::ofstream(file, mode)
                  .write(&c.second[0], c.second.size())
                  .good())
            << " faild to dump code to " << file;
        cout << "dump to " << file << " from subgraph " << sg->get_name()
             << endl;
      }
    }
  }
}
static const xir::Subgraph* dump_subgraph(const xir::Subgraph* s,
                                          const std::string& dirname) {
  dump_subgraph1(s, dirname);
  auto children = s->get_children();
  for (auto c : children) {
    dump_subgraph(c, dirname);
  }
  return nullptr;
}

int CmdDumpCode::main(int argc, char* argv[]) {
  if (argc < 3) {
    cout << help() << endl;
    return 1;
  }
  auto xmodel = std::string(argv[1]);
  auto dirname = std::string(argv[2]);
  mkdir_minus_p(dirname);
  auto graph = xir::Graph::deserialize(xmodel);
  dump_subgraph(graph->get_root_subgraph(), dirname);
  return 0;
}

CmdDumpCode::CmdDumpCode(const std::string& name) : Cmd(name) {}

std::string CmdDumpCode::help() const {
  std::ostringstream str;
  str << "xir " << get_name()
      << " <xmodel> <dir>\n\t"
         "dump binary data from a xmodel to a directory including machine\n\t"
         "codes and parameters. Machine codes are written to \n\t"
         "<dir>/<subgraph_name>.mc, assembly code are written to \n\t"
         "<dir>/<subgraph_name>.ac and parameters are written to "
         "<dir>/<reg>.bin\n\t";
  return str.str();
}
