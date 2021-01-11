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

#include "xir/graph/graph.hpp"

#include "UniLog/UniLog.hpp"
#include "xir/graph/graph_imp.hpp"
#include "xir/graph/serialize_v2.hpp"

namespace xir {

std::unique_ptr<Graph> Graph::create(std::string name) {
  return std::unique_ptr<Graph>{static_cast<Graph*>(new GraphImp{name})};
}

extern int g_use_proto_version;  // defined in GraphImp.cpp
static bool is_elf(const std::string& filename) {
  std::string magic;
  magic.resize(4);
  CHECK(std::ifstream(filename).read(&magic[0], magic.size()).good())
      << "read fail! filename=" << filename;
  return magic[0] == 0x7f && magic[1] == 0x45 && magic[2] == 0x4c &&
         magic[3] == 0x46;
}
// defined in elf2xir.cpp
extern std::unique_ptr<Graph> my_elf2xir(const std::string&);
std::unique_ptr<Graph> Graph::deserialize(const std::string& pb_fname) {
  if (!is_elf(pb_fname)) {
    v2::Serialize s;
    auto g = s.read(pb_fname);
    return g;
  }
  return my_elf2xir(pb_fname);
}

std::unique_ptr<Graph> Graph::deserialize_from_string(const std::string& str) {
  v2::Serialize s;
  auto g = s.read_from_string(str);
  return g;
}

}  // namespace xir
