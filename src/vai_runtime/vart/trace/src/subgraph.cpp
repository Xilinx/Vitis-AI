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

#include <fstream>
#include <ostream>
#include <sstream>
#include <string>
#include <vitis/ai/traceclass.hpp>
#include <xir/graph/graph.hpp>
#if _WIN32
#  include <windows.h>
#endif
// MSVC NOTE: must not using namespace std; it trigger an error, 'byte':
// ambiguous symbol, because c++17 introduce std::byte and MSVC use byte
// internally
//
// using namespace std;
using std::ios;
using std::make_pair;
using std::map;
using std::ofstream;
using std::ostringstream;
using std::pair;
using std::string;
using std::stringstream;
using std::vector;
using trace_entry_t = map<string, string>;

#define _j(x) (j_format(make_pair(#x, x)))

string _q(string s) {
  s.insert(0, "\"").append("\"");
  return s;
}

template <typename T>
std::ostream& operator<<(std::ostream& os, const vector<T>& vec) {
  os << "[";
  for (size_t i = 0; i < vec.size(); i++) {
    os << vec[i];
    if (i != vec.size() - 1) os << ',';
  };
  os << "]";

  return os;
};

template <typename T>
string j_format(pair<const char*, T> x) {
  stringstream ret;
  ret << _q(x.first) << ":" << x.second;
  return ret.str();
}

template <>
string j_format(pair<const char*, string> x) {
  stringstream ret;
  ret << _q(x.first) << ":" << _q(x.second);
  return ret.str();
}

class subgraph_info {
 public:
  subgraph_info() = delete;
  subgraph_info(const xir::Subgraph* subg);
  void to_json(ofstream& o);

 private:
  string subgraph_name;
  string device;
  string dpu_name;
  uint32_t depth;
  uint64_t workload;
  uint32_t op_num;
  string op_list;
  vector<vector<int>> i_tensors_shape;
  vector<vector<int>> o_tensors_shape;
  ostringstream mc_code_sstr;
};

subgraph_info::subgraph_info(const xir::Subgraph* subg)
    : subgraph_name(""),
      device(""),
      dpu_name(""),
      depth(0),
      workload(0),
      op_num(0),
      op_list(""),
      mc_code_sstr("") {
  vector<char> mc_code;

  subgraph_name = subg->get_name();
  depth = subg->get_depth();
  op_num = subg->get_op_num();
  for (auto op : subg->get_ops()) {
    string op_desc;
    op_desc = op->get_name() + "@" + op->get_type() + "|";
    op_list += op_desc;
  }

  if (subg->has_attr("device"))
    device = subg->get_attr<decltype(device)>("device");
  if (subg->has_attr("dpu_name"))
    dpu_name = subg->get_attr<decltype(dpu_name)>("dpu_name");
  if (subg->has_attr("workload"))
    workload = subg->get_attr<decltype(workload)>("workload");
  if (subg->has_attr("mc_code"))
    mc_code = subg->get_attr<decltype(mc_code)>("mc_code");

  for (unsigned char m : mc_code) {
    static char buf[4] = {0};
    snprintf(buf, sizeof(buf), "%02x", m);
    mc_code_sstr << buf;
  }

  auto i_tensors = subg->get_input_tensors();
  for (auto tensor : i_tensors) {
    i_tensors_shape.push_back(tensor->get_shape());
  };
  auto o_tensors = subg->get_output_tensors();
  for (auto tensor : o_tensors) {
    o_tensors_shape.push_back(tensor->get_shape());
  };
};

void subgraph_info::to_json(ofstream& o) {
  vector<string> rec;
  rec.push_back(_j(subgraph_name));
  rec.push_back(_j(dpu_name));
  rec.push_back(_j(device));
  rec.push_back(_j(depth));
  rec.push_back(_j(workload));
  rec.push_back(_j(op_num));
  rec.push_back(_j(op_list));
  rec.push_back(_j(i_tensors_shape));
  rec.push_back(_j(o_tensors_shape));
  rec.push_back(_j(mc_code_sstr.str()));

  o << "{";
  for (size_t i = 0; i < rec.size(); i++) {
    o << rec[i];
    if (i != rec.size() - 1) o << ",";
  }
  o << "}\n";
}

namespace vitis::ai::trace {
string add_subgraph_raw(const xir::Subgraph* subg) {
  static uint32_t subgraph_id = 0;
  auto dir = vitis::ai::my_getenv_s("VAI_TRACE_DIR", "/temp");
  auto pid =
#if _WIN32
      GetCurrentProcessId();
#else
      getpid();
#endif
  string path = dir + "vaitrace_subgraph_info_" + to_string(pid) + "_" +
                to_string(subgraph_id++);

  ofstream output_f;
  output_f.open(path, ios::out | ios::trunc);

  // 0 for root
  auto root = subgraph_info(subg);

  root.to_json(output_f);
  for (auto c : subg->children_topological_sort()) {
    auto s_info = subgraph_info(c);
    s_info.to_json(output_f);
  }

  output_f.close();

  return path;
};

}  // namespace vitis::ai::trace
