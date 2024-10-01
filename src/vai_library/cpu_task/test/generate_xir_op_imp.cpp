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

#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <unordered_map>
#include <xir/graph/graph.hpp>
#include <xir/op/op_def.hpp>

namespace xir {

class OpDefFactoryImp : public OpDefFactory {
 public:
  void register_h(const OpDef& def) override;
  const OpDef* create(const std::string& type) const;
  const std::vector<std::string> get_registered_ops() const;

 private:
  std::unordered_map<std::string, OpDef> store_;
  friend int generate(const char* op_type);
};

const OpDefFactoryImp* op_def_factory();

}  // namespace xir

namespace xir {
static std::string cpp_id_name(const std::string& name) {
  const std::string pat = "/():[]{}\\?%*|\"'><;=-";
  std::ostringstream str;
  for (auto& c : name) {
    if (pat.find(c) != std::string::npos) {
      str << "_";
    } else {
      str << c;
    }
  }
  return str.str();
}

static std::string cpp_class_name(const OpDef& opdef) {
  return cpp_id_name(opdef.name());
};

static std::string cpp_comment(const std::string& comment, size_t indent = 2u) {
  const std::string pat = "\n";
  std::ostringstream str;
  std::string spaces(indent, ' ');
  str << spaces << "// ";
  for (auto& c : comment) {
    if (pat.find(c) != std::string::npos) {
      str << "\n" << spaces << "// ";
    } else {
      str << c;
    }
  }
  return str.str();
}

static std::string cpp_op_arg_type(const OpArgDef& arg_def) {
  std::ostringstream str;
  std::string type;
  if (arg_def.data_type == DataType::FLOAT) {
    type = "vart::simple_tensor_buffer_t<float>";
  } else if (arg_def.data_type == DataType::INT) {
    // TODO:
    type = "vart::simple_tensor_buffer_t<int8_t>";
  } else if (arg_def.data_type == DataType::UINT) {
    type = "vart::simple_tensor_buffer_t<uint8_t>";
  } else {
    type = "vart::simple_tensor_buffer_t<void>";
  }
  if (arg_def.occur_type == OpArgDef::REQUIRED) {
    str << type;
  } else if (arg_def.occur_type == OpArgDef::OPTIONAL) {
    str << "std::unique_ptr<" << type << ">";
  } else if (arg_def.occur_type == OpArgDef::REPEATED ||
             arg_def.occur_type == OpArgDef::REQUIRED_AND_REPEATED) {
    str << "std::vector<" << type << ">";
  }
  return str.str();
}

static std::string cpp_op_args(const OpDef& opdef) {
  std::ostringstream str;
  for (auto& arg : opdef.input_args()) {
    // clang-format off
    str << cpp_comment(arg.annotation, 16) << "\n";
    str << "                ," << cpp_op_arg_type(arg) << cpp_id_name(arg.name)
        << "\n";
    // clang-format on
  }
  return str.str();
};

int generate(const char* op_type) {
  auto f = xir::op_def_factory();
  auto it = f->store_.find(std::string(op_type));
  if (it == f->store_.end()) {
    std::cerr << "op type not found. op_type=" << op_type << std::endl;
    return 1;
  }
  auto& opdef = it->second;
  // clang-format off
  std::cout <<  //
      "/*\n"
      " * Copyright 2022-2023 Advanced Micro Devices Inc.\n"
      " *\n"
      " * Licensed under the Apache License, Version 2.0 (the \"License\");\n"
      " * you may not use this file except in compliance with the License.\n"
      " * You may obtain a copy of the License at\n"
      " *\n"
      " *     http://www.apache.org/licenses/LICENSE-2.0\n"
      " *\n"
      " * Unless required by applicable law or agreed to in writing, software\n"
      " * distributed under the License is distributed on an \"AS IS\" BASIS,\n"
      " * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or "
      "implied.\n"
      " * See the License for the specific language governing permissions and\n"
      " * limitations under the License.\n"
      " */\n"
      "#include <vart/op_imp.h>\n"
      "class " << cpp_class_name(opdef) << " {\n"
      " public:\n"
      "  " << cpp_class_name(opdef) << "(const xir::Op* op1, xir::Attrs* attrs) : op{op1} {\n"
      "    // op and attrs is not in use.\n"
      "  }\n"
      "\n" << cpp_comment(opdef.annotation()) << "\n"
      "  int calculate(vart::simple_tensor_buffer_t<float> output\n"
      << cpp_op_args(opdef) <<
      ") {\n"
      "    return 0;\n"
      "  }\n"
      "\n"
      " public:\n"
      "  const xir::Op* const op;\n"
      "};\n"
      "\n"
      "DEF_XIR_OP_IMP(" << cpp_class_name(opdef) << ")\n"
      "\n";
  // clang-format on
  return 0;
}
}  // namespace xir
static void usage(const char* program) {
  auto f = xir::op_def_factory();
  std::cout << "usage: " << program << " <op_type> \n"  //
       << " where <op_type> could be one of following: ";
  auto types = f->get_registered_ops();
  std::sort(types.begin(), types.end());
  for (auto& type : types) {
    std::cout << "\n\t" << type;
  }
  std::cout << std::endl;
}

int main(int argc, char* argv[]) {
  auto graph = xir::Graph::create("simple");
  graph = nullptr;  // make sure all op def are loaded.
  if (argc < 2) {
    usage(argv[0]);
    return 0;
  }
  return xir::generate(argv[1]);
}
