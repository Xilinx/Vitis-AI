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
#include "./xir_util.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
using namespace std;

int main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << "usage: " << argv[0] << " <subcommand> or help" << std::endl;
    return 1;
  }
  auto cmd = Cmd::create(argv[1]);
  cmd->main(argc - 1, &argv[1]);
  return 0;
}

Cmd::Cmd(const std::string& name) : name_{name} {}

static void usage(const std::vector<std::unique_ptr<Cmd>>& cmds) {
  ostringstream str;
  str << "usage: xir <subcommand> <args>\n";
  for (auto& c : cmds) {
    str << "\n";
    str << c->help();
  }
  str << endl;
  cout << str.str();
  return;
}

std::unique_ptr<Cmd> Cmd::create(const std::string& name) {
  auto all_cmds = std::vector<std::unique_ptr<Cmd>>();
  all_cmds.push_back(std::make_unique<CmdDumpTxt>("dump_txt"));
  all_cmds.push_back(std::make_unique<CmdSubgraph>("subgraph"));
  all_cmds.push_back(std::make_unique<CmdDumpCode>("dump_bin"));
  all_cmds.push_back(std::make_unique<CmdGraph>("graph"));
  all_cmds.push_back(std::make_unique<CmdPng>("png"));
  all_cmds.push_back(std::make_unique<CmdSvg>("svg"));
  all_cmds.push_back(std::make_unique<CmdDumpReg>("dump_reg"));
  auto it = std::find_if(
      all_cmds.begin(), all_cmds.end(),
      [=](const std::unique_ptr<Cmd>& cmd) { return cmd->get_name() == name; });
  if (it == all_cmds.end()) {
    if (name != "help") {
      cout << "no such command: " << name << "! try \"xir help\"\n"
           << std::endl;
    }
    usage(all_cmds);
    exit(1);
  }
  return std::move(*it);
}
