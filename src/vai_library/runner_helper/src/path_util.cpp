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
#include <limits.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

namespace vitis {
namespace ai {
const std::string sep = "/";
std::string file_name_realpath(const std::string& filename) {
  std::vector<char> buf(PATH_MAX);
  auto r = realpath(filename.c_str(), &buf[0]);
  CHECK(r != nullptr) << "cannot resolve filename";
  return std::string(r);
}

std::string file_name_directory(const std::string& path) {
  if (path == sep) {
    return path;
  }
  auto pos = path.find_last_of(sep);
  if (pos == std::string::npos) {
    return ".";
  }
  return path.substr(0, pos);
}

bool is_directory(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

bool is_regular_file(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISREG(buffer.st_mode));
}

std::string file_name_basename(const std::string& filename) {
  std::string buf = filename.c_str();
  return std::string(basename(&buf[0]));
}

std::string file_name_basename_no_ext(const std::string& filename) {
  auto path = file_name_basename(filename);
  auto pos = path.find_last_of(".");
  return path.substr(0, pos);
}

std::string file_name_ext(const std::string& filename) {
  auto path = file_name_basename(filename);
  auto pos = path.find_last_of(".");
  if (pos == std::string::npos) {
    return "";
  }
  return path.substr(pos + 1);
}

static void my_mkdir(const std::string& dirname) {
  struct stat st = {0};
  if (stat(dirname.c_str(), &st) == -1) {
    PCHECK(mkdir(dirname.c_str(), 0777) == 0)
        << "mkdir error; dirname=" << dirname;
  }
  PCHECK(stat(dirname.c_str(), &st) == 0)
      << "stat dir error: dirname=" << dirname;
  CHECK(S_ISDIR(st.st_mode)) << "error not a directory: dirname=" << dirname;
}

void create_parent_path(const std::string& path) {
  if (is_directory(path)) {
    return;
  }
  auto parent_path = file_name_directory(path);
  if (!is_directory(parent_path)) {
    create_parent_path(parent_path);
  }
  my_mkdir(path);
}

std::string to_valid_file_name(const std::string& filename) {
  const std::string pat = "/():[]{}\\?%*|\"'><;=";
  std::ostringstream str;
  for (auto c : filename) {
    if (pat.find(c) != std::string::npos) {
      str << "_";  // << std::hex << (int)c << "_";
    } else {
      str << c;
    }
  }
  return str.str();
};

size_t file_size(const std::string& filename) {
  size_t ret = 0u;
  struct stat statbuf;
  const auto r_stat = stat(filename.c_str(), &statbuf);
  if (r_stat == 0) {
    ret = S_ISREG(statbuf.st_mode) ? statbuf.st_size : 0u;
  }
  return ret;
}

}  // namespace ai
}  // namespace vitis
