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
#include "../include/vitis/ai/simple_config.hpp"
#include <sys/stat.h>
#include <sys/types.h>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <sstream>
#include <typeinfo>
#include <utility>

namespace vitis {
namespace ai {
static size_t filesize(const std::string& filename) {
  size_t ret = 0;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = statbuf.st_size;
  }
  return ret;
}

SimpleConfig::SimpleConfig(const std::string& filename) {
  Initialize(filename);
}

std::shared_ptr<SimpleConfig> SimpleConfig::getOrCreateSimpleConfig(
    const std::string& filename) {
  static std::unordered_map<std::string, std::weak_ptr<SimpleConfig> >
      simple_configs_;
  if (filesize(filename) == 0) {
    return nullptr;
  }
  std::shared_ptr<SimpleConfig> ret;
  if (simple_configs_[filename].expired()) {
    ret = std::make_shared<SimpleConfig>(filename);
    simple_configs_[filename] = ret;
  } else {
    ret = simple_configs_[filename].lock();
  }
  return ret;
}

void SimpleConfig::Initialize(const std::string& filename) {
  static std::regex string_value("^ *([^: ]+) *: *(.*?) *(#.*)*$");
  static std::regex comment_value("^ *#.*");
  static std::regex white_line_value("^ *$");

  std::ifstream fs(filename.c_str());
  std::string line;
  while (getline(fs, line)) {
    std::smatch pieces_match;
    if (std::regex_match(line, pieces_match, string_value)) {
      if (values_.find(pieces_match[1].str()) != values_.end()) {
        std::cerr << pieces_match[1].str() << " is redefined!" << std::endl;
        abort();
      }
      values_[pieces_match[1].str()] = pieces_match[2].str();
      fields_.push_back(pieces_match[1].str());
    } else if (std::regex_match(line, pieces_match, comment_value)) {
      // igore comment
    } else if (std::regex_match(line, pieces_match, white_line_value)) {
      // igore comment
    } else {
      std::cerr << "unknown config: " << line << std::endl;
    }
  }
  fs.close();
}

template <typename T>
T SimpleConfig::as(const std::string& name) const {
  T ret;
  if (values_.find(name) == values_.end()) {
    std::cerr << "cannot find config. name = '" << name << "'" << std::endl;
    abort();
  }
  try {
    ParseValue(values_.at(name), ret);
  } catch (std::exception& e) {
    std::cerr << "cannot parse " << name << "\n" << e.what() << std::endl;
    abort();
  }

  return ret;
}

static bool begin_with(const std::string& a, const std::string& b) {
  auto pos = a.find(b);
  auto ret = pos == 0u;
  return ret;
}

bool SimpleConfig::has(const std::string& name) const {
  return std::find_if(fields_.begin(), fields_.end(),
                      [&name](const std::string& f) {
                        return begin_with(f, name);
                      }) != fields_.end();
}

template <typename T>
void SimpleConfig::ParseValue(const std::string& text, T& value) {
  std::istringstream is(text);
  if (!(is >> value)) {
    // throw SimpleConfigException(text);
    std::cerr << "cannot parse string: " << text << " as " << typeid(T).name()
              << std::endl;
    abort();
  }

  if (is.rdbuf()->in_avail() != 0) {
    std::cerr << "parse error for string: " << text << " as "
              << typeid(T).name() << std::endl;
    abort();
  }
}

SimpleConfig::SimpleConfigViewer::SimpleConfigViewer(const SimpleConfig& cfg,
                                                     const std::string& name)
    : cfg_(cfg), name_(name) {}

template <class T>
SimpleConfig::SimpleConfigViewer SimpleConfig::SimpleConfigViewer::operator[](
    const T& name) const {
  return operator()(name);
}

SimpleConfig::SimpleConfigViewer SimpleConfig::SimpleConfigViewer::operator()(
    const std::string& name) const {
  return SimpleConfig::SimpleConfigViewer(cfg_, name_ + "." + name);
}

SimpleConfig::SimpleConfigViewer SimpleConfig::SimpleConfigViewer::operator()(
    int index) const {
  return SimpleConfig::SimpleConfigViewer(
      cfg_, name_ + "[" + std::to_string(index) + "]");
}
std::vector<SimpleConfig::SimpleConfigViewer>
SimpleConfig::SimpleConfigViewer::fields() const {
  auto guess_field = [](const std::string& s,
                        const std::string& prefix) -> std::string {
    auto field_name = [](const std::string& full_name) -> std::string {
      auto pos = full_name.find_first_of(".[");
      std::string ret;
      if (pos != std::string::npos) {
        ret = full_name.substr(0, pos);
      } else {
        ret = full_name;
      }
      return ret;
    };
    if (s.find(prefix) != 0) {
      return "";
    }
    auto pos_begin = prefix.size();
    auto field_full_name = s.substr(pos_begin);
    return field_name(field_full_name);
  };
  auto prefix = name_ + ".";
  auto ret = std::vector<SimpleConfigViewer>();
  ret.reserve(cfg_.fields_.size());
  std::set<std::string> fields;
  for (const auto& s : cfg_.fields_) {
    auto field = guess_field(s, prefix);
    if (!field.empty() && fields.find(field) == fields.end()) {
      ret.emplace_back(cfg_, prefix + field);
      fields.insert(field);
    }
  }
  return ret;
}
bool SimpleConfig::SimpleConfigViewer::has(const std::string& field) const {
  return cfg_.has(name_ + "." + field);
}

bool SimpleConfig::SimpleConfigViewer::has(size_t idx) const {
  return cfg_.has(name_ + "[" + std::to_string(idx) + "]");
}

template <typename T>
T SimpleConfig::SimpleConfigViewer::as() const {
  return cfg_.as<T>(name_);
}

struct SimpleConfig::SimpleConfigViewer SimpleConfig::operator()(
    const std::string& name) const {
  return SimpleConfigViewer{*this, name};
}

template <>
inline void SimpleConfig::ParseValue<unsigned long long>(
    const std::string& text, unsigned long long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoull(text.substr(2), 0, 16);
  } else {
    value = stoull(text, 0, 10);
  }
}
template <>
inline void SimpleConfig::ParseValue<long long>(const std::string& text,
                                                long long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoll(text.substr(2), 0, 16);
  } else {
    value = stoll(text, 0, 10);
  }
}
template <>
inline void SimpleConfig::ParseValue<unsigned long>(const std::string& text,
                                                    unsigned long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoul(text.substr(2), 0, 16);
  } else {
    value = stoul(text, 0, 10);
  }
}
template <>
inline void SimpleConfig::ParseValue<long>(const std::string& text,
                                           long& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stol(text.substr(2), 0, 16);
  } else {
    value = stol(text, 0, 10);
  }
}

template <>
inline void SimpleConfig::ParseValue<float>(const std::string& text,
                                            float& value) {
  value = stof(text);
}

template <>
inline void SimpleConfig::ParseValue<double>(const std::string& text,
                                             double& value) {
  value = stod(text);
}

template <>
inline void SimpleConfig::ParseValue<unsigned int>(const std::string& text,
                                                   unsigned int& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = (unsigned int)stoul(text.substr(2), 0, 16);
  } else {
    value = (unsigned int)stoul(text, 0, 10);
  }
}
template <>
inline void SimpleConfig::ParseValue<int>(const std::string& text, int& value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = (int)stol(text.substr(2), 0, 16);
  } else {
    value = (int)stol(text, 0, 10);
  }
}
template <>
inline void SimpleConfig::ParseValue<bool>(const std::string& text,
                                           bool& value) {
  if (text == "yes" || text == "on" || text == "enable" || text == "true") {
    value = true;
  } else {
    value = false;
  }
}
template <>
inline void SimpleConfig::ParseValue<std::string>(const std::string& text,
                                                  std::string& value) {
  value = text;
}

template unsigned long long
SimpleConfig::SimpleConfigViewer::as<unsigned long long>() const;

template long long SimpleConfig::SimpleConfigViewer::as<long long>() const;

template unsigned long SimpleConfig::SimpleConfigViewer::as<unsigned long>()
    const;

template long SimpleConfig::SimpleConfigViewer::as<long>() const;

template double SimpleConfig::SimpleConfigViewer::as<double>() const;

template float SimpleConfig::SimpleConfigViewer::as<float>() const;

template unsigned SimpleConfig::SimpleConfigViewer::as<unsigned int>() const;

template int SimpleConfig::SimpleConfigViewer::as<int>() const;

template bool SimpleConfig::SimpleConfigViewer::as<bool>() const;

template std::string SimpleConfig::SimpleConfigViewer::as<std::string>() const;

template SimpleConfig::SimpleConfigViewer SimpleConfig::SimpleConfigViewer::
operator[]<int>(const int&) const;

template SimpleConfig::SimpleConfigViewer SimpleConfig::SimpleConfigViewer::
operator[]<const std::string>(const std::string&) const;
}  // namespace ai
}  // namespace vitis
