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
#include "xir/attrs/attrs.hpp"

#include <type_traits>

#include "xir/attrs/attrs_imp.hpp"

namespace xir {

std::unique_ptr<Attrs> Attrs::create() { return std::make_unique<AttrsImp>(); }

std::unique_ptr<Attrs> Attrs::clone(Attrs* param) {
  return std::make_unique<AttrsImp>(*static_cast<AttrsImp*>(param));
}

template <typename T>
static bool cmp(const any& a, const any& b) {
  return stdx::any_cast<const T&>(a) == stdx::any_cast<const T&>(b);
}

template <typename... T>
static std::unordered_map<std::type_index,
                          std::function<bool(const any&, const any&)>>
make_cmp_fun() {
  return {
      make_pair(std::type_index(typeid(T)), &cmp<T>)...,
  };
}

static std::unordered_map<std::type_index,
                          std::function<bool(const any&, const any&)>>
get_cmp_functions() {
  using bytes_t = std::vector<int8_t>;
  return make_cmp_fun<  //
      bool, int8_t,
      uint8_t,                                                          //
      int16_t, uint16_t,                                                //
      int32_t, uint32_t,                                                //
      int64_t, uint64_t,                                                //
      float, double,                                                    //
      std::string,                                                      //
      bytes_t,                                                          //
      std::vector<bool>,                                                //
      std::vector<int8_t>, std::vector<uint8_t>,                        //
      std::vector<int16_t>, std::vector<uint16_t>,                      //
      std::vector<int32_t>, std::vector<uint32_t>,                      //
      std::vector<int64_t>, std::vector<uint64_t>,                      //
      std::vector<float>, std::vector<double>,                          //
      std::vector<std::string>,                                         //
      std::vector<bytes_t>,                                             //
      std::map<std::string, int8_t>, std::map<std::string, uint8_t>,    //
      std::map<std::string, int16_t>, std::map<std::string, uint16_t>,  //
      std::map<std::string, int32_t>, std::map<std::string, uint32_t>,  //
      std::map<std::string, int64_t>, std::map<std::string, uint64_t>,  //
      std::map<std::string, float>, std::map<std::string, double>,      //
      std::map<std::string, std::string>,                               //
      std::map<std::string, bytes_t>,                                   //
      std::map<std::string, std::vector<bool>>,                         //
      std::map<std::string, std::vector<int8_t>>,                       //
      std::map<std::string, std::vector<uint8_t>>,                      //
      std::map<std::string, std::vector<int16_t>>,                      //
      std::map<std::string, std::vector<uint16_t>>,                     //
      std::map<std::string, std::vector<int32_t>>,                      //
      std::map<std::string, std::vector<uint32_t>>,                     //
      std::map<std::string, std::vector<int64_t>>,                      //
      std::map<std::string, std::vector<uint64_t>>,                     //
      std::map<std::string, std::vector<float>>,                        //
      std::map<std::string, std::vector<double>>,                       //
      std::map<std::string, std::vector<std::string>>,                  //
      std::map<std::string, std::vector<bytes_t>>,                      //
      nullptr_t>();
}

int Attrs::cmp(const any& a, const any& b) {
  auto t1 = std::type_index(a.type());
  auto t2 = std::type_index(b.type());
  if (t1 != t2) {
    return 0;
  }
  auto cmp_functions = get_cmp_functions();
  auto it_cmp = cmp_functions.find(t1);
  if (it_cmp == cmp_functions.end()) {
    return -1;  // uncertain
  }
  return it_cmp->second(a, b);
}
}  // namespace xir
