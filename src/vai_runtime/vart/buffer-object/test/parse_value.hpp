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
#include <cassert>
#include <sstream>

template <typename T> void parse_value(const std::string &text, T &value) {
  std::istringstream is(text);
  if (!(is >> value)) {
    assert(false);
  }

  if (is.rdbuf()->in_avail() != 0) {
    assert(false);
  }
}

inline void parse_value(const std::string &text, unsigned long long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoull(text.substr(2), 0, 16);
  } else {
    value = stoull(text, 0, 10);
  }
}
inline void parse_value(const std::string &text, long long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoll(text.substr(2), 0, 16);
  } else {
    value = stoll(text, 0, 10);
  }
}

inline void parse_value(const std::string &text, unsigned long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stoul(text.substr(2), 0, 16);
  } else {
    value = stoul(text, 0, 10);
  }
}
inline void parse_value(const std::string &text, long &value) {
  if (text.size() > 2 && text[0] == '0' && text[1] == 'x') {
    value = stol(text.substr(2), 0, 16);
  } else {
    value = stol(text, 0, 10);
  }
}

inline void parse_value(const std::string &text, bool &value) {
  if (text == "yes" || text == "on" || text == "enable" || text == "true") {
    value = true;
  } else {
    value = false;
  }
}
