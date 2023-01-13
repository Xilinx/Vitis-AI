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

#include "xir/op/op_def_factory_imp.hpp"

#include <iostream>
#include <iterator>
#include <mutex>
#include <sstream>

#include "UniLog/UniLog.hpp"
#include "xir/attrs/attrs.hpp"
#include "xir/op/built_in_ops.hpp"
#include "xir/op/shape_inference.hpp"
#include "xir/util/dynamic_load.hpp"
namespace xir {
using namespace std;
static vector<string> str_split(const string& str) {
  auto ret = vector<string>{};
  istringstream iss(str);
  std::copy(istream_iterator<string>(iss), istream_iterator<string>(),
            back_inserter(ret));
  return ret;
}

static void load_ops_library(OpDefFactoryImp* self,
                             const std::string& file_name) {
  typedef void (*INIT_FUN)(OpDefFactory*);
  INIT_FUN reg_func = NULL;
  auto handle = dlopen(file_name.c_str(), RTLD_LAZY);
  UNI_LOG_CHECK(handle, XIR_OPERATION_FAILED)
      << "Cannot open library " << file_name;

  reg_func = (INIT_FUN)dlsym(handle, "register_ops");

  UNI_LOG_CHECK(dlerror() == NULL, XIR_OPERATION_FAILED)
      << "Cannot load symbol 'register_ops' from " << file_name;

  UNI_LOG_INFO << "Load expanded ops from " << file_name;
  reg_func(self);
}

std::mutex factory_mutex;
OpDefFactoryImp* op_def_factory() {
  std::lock_guard<std::mutex> gaurd(factory_mutex);
  static unique_ptr<OpDefFactoryImp> self;
  if (!self) {
    self = make_unique<OpDefFactoryImp>();
    register_built_in_ops(self.get());
    auto ops_library_list = getenv("OPS_LIBRARY");
    if (ops_library_list != nullptr) {
      for (const auto& ops_library : str_split(string{ops_library_list})) {
        load_ops_library(self.get(), ops_library);
      }
    }
  }
  return self.get();
}

void OpDefFactoryImp::register_h(const OpDef& def) {
  UNI_LOG_CHECK(store_.count(def.name()) == 0, XIR_MULTI_REGISTERED_OP)
      << def.name();
  // check the shape, if this op has no input
  UNI_LOG_CHECK(
      def.input_args().size()  //
          || (std::any_of(def.attrs().begin(), def.attrs().end(),
                          [](const AttrDef& adef) {
                            return ((adef.name == "shape") &&
                                    (adef.data_type == TYPE_INDEX_INT32_VEC));
                          })  //
              && std::any_of(def.attrs().begin(), def.attrs().end(),
                             [](const AttrDef& adef) {
                               return ((adef.name == "data_type") &&
                                       (adef.data_type == TYPE_INDEX_STRING));
                             })  //
              ),
      XIR_OP_DEF_SHAPE_HINT_MISSING)
      << "The definition of \"" << def.name()
      << "\" op has no input arguments, so it requires some hints, which "
         "required as "
         "attributes {shape, std::vector<std::int32_t>}, {data_type, "
         "std::string} and {bit_width, std::int32_t>.";
  store_.emplace(def.name(), def);
}

const OpDef* OpDefFactoryImp::create(const string& type) const {
  auto it = store_.find(type);
  if (it == store_.end()) {
    UNI_LOG_FATAL(XIR_UNREGISTERED_OP) << type;
  }
  return &(it->second);
}

const vector<string> OpDefFactoryImp::get_registered_ops() const {
  auto ret = vector<string>{};
  std::transform(store_.begin(), store_.end(), std::back_inserter(ret),
                 [](unordered_map<string, OpDef>::value_type it) -> string {
                   return it.second.name();
                 });
  return ret;
}

const OpDef* OpDefFactoryImp::get_op_def(
    const std::string& type, bool register_custome_op_if_not_exists) {
  auto it = this->store_.find(type);
  if (it == this->store_.end()) {
    if (register_custome_op_if_not_exists) {
      register_customized_operator_definition(type);
    }
  }
  it = this->store_.find(type);
  CHECK(it != this->store_.end());
  return &it->second;
}

const OpDef& OpDefFactoryImp::get_const_op_def(const std::string& type) const {
  auto it = store_.find(type);
  CHECK(it != store_.end()) << "cannot find op def " << type;
  return it->second;
}

void OpDefFactoryImp::register_customized_operator_definition(
    const std::string& type) {
  auto new_operator =
      xir::OpDef(type)
          .add_input_arg(xir::OpArgDef{"input", OpArgDef::REPEATED,
                                       xir::DataType::FLOAT, ""})
          .add_attr(xir::AttrDefBuilder<std::vector<std::int32_t>>::build(
              "shape", AttrDef::REQUIRED, 0,
              "`Datatype`: `vector<int>`\n\n"
              "The shape of the output tensor"))
          .add_attr(xir::AttrDefBuilder<std::string>::build(
              "data_type", AttrDef::REQUIRED,
              "`Datatype`: `string`\n\n"
              "The data type of the data of output feature maps, "
              "we use FLOAT32 as the default."))
          .set_annotation("This operator is not defined by XIR.")
          .set_shape_infer(xir::shape_infer_data);
  register_h(new_operator);
}

}  // namespace xir
