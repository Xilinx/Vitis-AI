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
#include <stdlib.h>

#include <sstream>
#include <string>
#include <vector>

#include "./parse_value.hpp"
namespace vitis {
namespace ai {
template <typename T>
struct env_config_helper {
  static inline T from_string(const std::string& s);
};
std::string my_getenv_s(const char* name,
                        const std::string& default_value = "");
template <typename T, typename env_name>
struct env_config {
  static T init() {
    const char* name = env_name::get_name();
    const char* defvalue = env_name::get_default_value();
    auto p = my_getenv_s(name, defvalue);
    const T tmp_value = env_config_helper<T>::from_string(p);
    return tmp_value;
  }
  static T value;
};
template <typename T, typename env_name>
T env_config<T, env_name>::value = env_config<T, env_name>::init();

template <typename T>
inline T env_config_helper<T>::from_string(const std::string& s) {
  T ret = T();
  parse_value(s, ret);
  return ret;
}

template <>
inline std::string env_config_helper<std::string>::from_string(
    const std::string& s) {
  return s;
}

template <typename T>
struct env_config_helper<std::vector<T>> {
  static inline std::vector<T> from_string(const std::string& s);
};

template <typename T>
inline std::vector<T> env_config_helper<std::vector<T>>::from_string(
    const std::string& s) {
  const char delim = ',';
  auto list = std::vector<T>();
  auto ss = std::istringstream(std::string(s));
  std::string item;
  while (std::getline(ss, item, delim)) {
    list.push_back(env_config_helper<T>::from_string(&item[0]));
  }
  return list;
}

}  // namespace ai
}  // namespace vitis

#define DEF_ENV_PARAM_2(param_name, defvalue1, type)                           \
  struct ENV_PARAM_##param_name                                                \
      : public ::vitis::ai::env_config<type, ENV_PARAM_##param_name> {         \
    static const char* get_name() { return #param_name; }                      \
    static const char* get_default_value() { return defvalue1; }               \
  };

#define ENV_PARAM(param_name) (ENV_PARAM_##param_name::value)

#define DEF_ENV_PARAM(param_name, defvalue1)                                   \
  DEF_ENV_PARAM_2(param_name, defvalue1, int)
