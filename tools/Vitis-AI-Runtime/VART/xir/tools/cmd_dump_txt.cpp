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

#include "cmd_dump_txt.hpp"
// must include this first to define XIR_DLLSPEC;
#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "graph_proto_v2.pb.h"
#include "xir/XirExport.hpp"
#include "xir/util/tool_function.hpp"
static inline std::string xxd(const unsigned char* p, int size, int column,
                              int group) {
  std::ostringstream str;
  char buf[128];
  for (int i = 0; i < size; ++i) {
    if (i % column == 0) {
      snprintf(buf, sizeof(buf), "\n%p %08x:", p + i, i);
      str << buf;
    }
    if (i % group == 0) {
      snprintf(buf, sizeof(buf), " ");
      str << buf;
    }
    snprintf(buf, sizeof(buf), "%02x", p[i]);
    str << buf;
  }
  snprintf(buf, sizeof(buf), "\n");
  str << buf;
  return str.str();
}

static std::string dump_string(const std::string& val, const std::string& md5) {
  std::ostringstream str;
  const std::string::size_type max_size = 160u;
  auto xxd_size = std::min(max_size, val.size());
  str << "bytes = " << val.size() << " md5sum = " + md5;
  str << "\nhead: " << xxd((unsigned char*)&val[0], xxd_size, 16, 2);
  str << "tail: "
      << xxd((unsigned char*)&val[val.size() - xxd_size], xxd_size, 16, 2);
  return str.str();
}

struct MyPrinter : public google::protobuf::TextFormat::FastFieldValuePrinter {
  MyPrinter(bool dump) : dump_(dump) {}
  virtual ~MyPrinter() {}
  virtual void PrintBytes(const std::string& val,
                          google::protobuf::TextFormat::BaseTextGenerator*
                              generator) const override {
    auto md5value = xir::get_md5_of_buffer(&val[0], val.size());
    std::string n = dump_string(val, md5value);
    generator->PrintString(n);
    if (dump_) {
      auto mode =
          std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
      auto filename = std::string("") + md5value + ".bin";
      CHECK(std::ofstream(filename, mode).write(&val[0], val.size()).good())
          << " faild to dump code to " << filename;
      LOG(INFO) << "dump parameter to " << filename;
    }
  }
  bool dump_;
};

static std::string DebugString(const google::protobuf::Message& message) {
  google::protobuf::TextFormat::Printer printer;
  std::string debug_string;
  auto enable_dump = getenv("ENABLE_DUMP");
  auto my_printer =
      new MyPrinter(enable_dump != nullptr && strcmp(enable_dump, "1") == 0);
  printer.SetDefaultFieldValuePrinter(my_printer);
  printer.SetExpandAny(true);
  printer.PrintToString(message, &debug_string);
  return debug_string;
}

CmdDumpTxt::CmdDumpTxt(const std::string& name) : Cmd(name) {}

std::string CmdDumpTxt::help() const {
  std::ostringstream str;
  str << "xir " << get_name() << " <xmodel> [<txt>]"
      << "\n\t"
      << "e.g. xir " << get_name() << " a.xmodel a.txt"
      << "\n\t"
      << "when <txt> is missing, it dumps to standard output." << std::endl;
  return str.str();
}

#include <iostream>
using namespace std;
int CmdDumpTxt::main(int argc, char* argv[]) {
  if (argc < 2) {
    cout << help();
    return 1;
  }
  auto xmodel = std::string(argv[1]);
  ostream* txt_stream = &cout;
  auto text_fstream = std::unique_ptr<std::ofstream>();
  if (argc >= 3) {
    text_fstream = std::make_unique<std::ofstream>(std::string(argv[2]));
    txt_stream = text_fstream.get();
  }
  std::ifstream ifs(xmodel, std::ios::binary);
  serial_v2::Graph pb_graph2;
  if (!pb_graph2.ParseFromIstream(&ifs)) {
    LOG(ERROR) << "[ReadErr] Read graph from protobuf error!" << endl;
    abort();
  }
  CHECK(((*txt_stream) << DebugString(pb_graph2)).good());
  return 0;
}
