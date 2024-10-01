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

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include "xir/util/tool_function.hpp"
#include "tools_extra_ops.hpp"
#include "xir_graph_proto_v2.pb.h"
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

static std::string md5sum(const std::string& val) {
  return xir::get_md5_of_buffer((const unsigned char*)&val[0], val.size());
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
    auto md5value = md5sum(val);
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

std::string xmodel_to_txt(std::string xmodel) {
  std::ifstream ifs(xmodel, std::ios::binary);
  serial_v2::Graph pb_graph2;
  if (!pb_graph2.ParseFromIstream(&ifs)) {
    LOG(ERROR) << "[ReadErr] Read graph from protobuf error!";
    abort();
  }
  return DebugString(pb_graph2);
}
std::map<std::string, std::string> get_reg_id_to_parameter(
    const xir::Subgraph* s) {
  std::map<std::string, std::string> md5s;
  if (s->has_attr("reg_id_to_parameter_value")) {
    auto values = s->get_attr<std::map<std::string, std::vector<char>>>(
        "reg_id_to_parameter_value");
    for (auto& it : values) {
      md5s[it.first] = md5sum(std::string(it.second.begin(), it.second.end()));
    }
  }
  return md5s;
}
