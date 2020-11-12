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

#include <glog/logging.h>
#include <google/protobuf/text_format.h>

#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include "graph_proto_v1.pb.h"
#include "graph_proto_v2.pb.h"
#include "xir/graph/graph.hpp"
using namespace std;
#include <openssl/md5.h>
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
  std::vector<unsigned char> result((size_t)MD5_DIGEST_LENGTH, '0');
  std::ostringstream str;
  MD5((const unsigned char*)&val[0], val.size(), (unsigned char*)&result[0]);
  str << std::hex << std::setfill('0') << std::setw(2);
  for (const auto x : result) {
    str << ((unsigned int)x);
  }
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
  LOG(INFO) << "eanble " << (void*)enable_dump << " value:" << enable_dump;
  auto my_printer =
      new MyPrinter(enable_dump != nullptr && strcmp(enable_dump, "1") == 0);
  printer.SetDefaultFieldValuePrinter(my_printer);
  printer.SetExpandAny(true);
  printer.PrintToString(message, &debug_string);
  return debug_string;
}

static std::string g_output_txt = "";
static std::string g_output_png = "";
static std::string g_model_name = "a.model";
static void parse_opt(int argc, char* argv[]) {
  int opt = 0;
  while ((opt = getopt(argc, argv, "p:t:")) != -1) {
    switch (opt) {
      case 'p':
        g_output_png = optarg;
        break;
      case 't':
        g_output_txt = optarg;
        break;
      default:
        // usage();
        exit(1);
    }
  }
  if (optind >= argc) {
    cerr << "Expected argument after options\n";
    exit(EXIT_FAILURE);
  }
  g_model_name = argv[optind];
  return;
}

int main(int argc, char* argv[]) {
  parse_opt(argc, argv);
  // auto pb_fname = std::string{argv[1]};
  std::ifstream ifs(g_model_name, std::ios::binary);
  serial_v2::Graph pb_graph2;
  if (true) {
    if (!pb_graph2.ParseFromIstream(&ifs)) {
      LOG(ERROR) << "[ReadErr] Read graph from protobuf error!" << endl;
      abort();
    }
    if (g_output_txt != "") {
      CHECK((std::ofstream(g_output_txt) << DebugString(pb_graph2)).good());
    }
  }
  if (g_output_png != "") {
    auto g = xir::Graph::deserialize(g_model_name);
    LOG(INFO) << "convt " << g->get_name() << " to " << g_output_png;
    g->visualize(g_output_png, "png");
  }
  return 0;
}
