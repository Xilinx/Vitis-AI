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
#pragma once
#include <glog/logging.h>
#include <json-c/json.h>

#include <string>
#include <vector>

namespace vitis {
namespace ai {

class JsonObjectVisitor {
 public:
  explicit JsonObjectVisitor(const json_object* json) : json_{json} {};
  virtual ~JsonObjectVisitor() = default;
  JsonObjectVisitor(const JsonObjectVisitor& other) = default;
  JsonObjectVisitor& operator=(const JsonObjectVisitor& rhs) = default;

 public:
  void visit(bool& value);
  void visit(int& value);
  void visit(float& value);
  void visit(std::string& value);
  template <typename T>
  void visit(T& x);
  template <typename T>
  void visit(std::vector<T>& value);
  JsonObjectVisitor operator[](const char* field);

 private:
  const json_object* json_;
};

inline void JsonObjectVisitor::visit(bool& value) {
  CHECK(json_object_is_type(json_, json_type_boolean))
      << "not a boolean type! json="
      << json_object_to_json_string(const_cast<json_object*>(json_));
  auto v1 = json_object_get_boolean(json_);
  value = (bool)v1;
}
inline void JsonObjectVisitor::visit(int& value) {
  CHECK(json_object_is_type(json_, json_type_int))
      << "not a int type! json="
      << json_object_to_json_string(const_cast<json_object*>(json_));
  value = json_object_get_int(json_);
}
inline void JsonObjectVisitor::visit(float& value) {
  CHECK(json_object_is_type(json_, json_type_double))
      << "not a double type! json="
      << json_object_to_json_string(const_cast<json_object*>(json_));
  auto v1 = json_object_get_double(json_);
  value = (float)v1;
}
inline void JsonObjectVisitor::visit(std::string& value) {
  CHECK(json_object_is_type(json_, json_type_string))
      << "not a string type! json="
      << json_object_to_json_string(const_cast<json_object*>(json_));
  value = json_object_get_string(const_cast<json_object*>(json_));
}

inline JsonObjectVisitor JsonObjectVisitor::operator[](const char* field) {
  return JsonObjectVisitor(json_object_object_get(json_, field));
}

template <typename T>
inline void JsonObjectVisitor::visit(T& x) {
  x.VisitAttrs(*this);
}
template <typename T>
inline void JsonObjectVisitor::visit(std::vector<T>& value) {
  CHECK(json_object_is_type(json_, json_type_array))
      << "not a array type! json="
      << json_object_to_json_string(const_cast<json_object*>(json_));
  auto size = json_object_array_length(json_);
  value.reserve(size);
  for (decltype(size) idx = 0; idx < size; ++idx) {
    auto elt = JsonObjectVisitor(json_object_array_get_idx(json_, idx));
    T res;
    elt.visit(res);
    value.emplace_back(res);
  }
}

}  // namespace ai
}  // namespace vitis
