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

#pragma once
#include <memory>
#include <ostream>
#include <set>
#include <string>
#include <vector>

#include "xir/graph/graph.hpp"
class Cmd {
 public:
  static std::unique_ptr<Cmd> create(const std::string& name);

 public:
  Cmd(const std::string& name);
  virtual ~Cmd() = default;

 public:
  virtual int main(int argc, char* argv[]) = 0;
  virtual std::string help() const = 0;

 public:
  std::string get_name() const { return name_; }

 private:
  const std::string name_;
};

template <typename T>
static inline std::ostream& operator<<(std::ostream& out,
                                       const std::set<T>& v) {
  int c = 0;
  out << "[";
  for (const auto& x : v) {
    if (c++ != 0) {
      out << ",";
    }
    out << x;
  }
  out << "]";
  return out;
}
inline std::ostream& operator<<(std::ostream& out,
                                const xir::Tensor* xir_tensor) {
  out << "xir_tensor{";
  out << xir_tensor->get_name() << ":(";
  int fixpos = xir_tensor->template get_attr<int>("fix_point");
  auto dims = xir_tensor->get_shape();
  for (auto i = 0u; i < dims.size(); ++i) {
    if (i != 0) {
      out << ",";
    }
    out << dims[i];
  }
  out << "), fixpos=" << fixpos
      << " # of elements= " << xir_tensor->get_element_num();
  out << "}";
  return out;
}

#include "./cmd_dump_code.hpp"
#include "./cmd_dump_reg.hpp"
#include "./cmd_dump_txt.hpp"
#include "./cmd_graph.hpp"
#include "./cmd_png.hpp"
#include "./cmd_subgraph.hpp"
#include "./cmd_svg.hpp"
