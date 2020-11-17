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

#include "xir/device_memory.hpp"
#include <dirent.h>
#include <glog/logging.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <map>

#include "vitis/ai/env_config.hpp"
DEF_ENV_PARAM(DEBUG_DEVICE_MEMORY, "0");

namespace xir {

static void mkdir_minus_p(const std::string& dirname) {
  struct stat st = {0};
  if (stat(dirname.c_str(), &st) == -1) {
    PCHECK(mkdir(dirname.c_str(), 0777) == 0)
        << "mkdir error; dirname=" << dirname;
  }
  PCHECK(stat(dirname.c_str(), &st) == 0)
      << "stat dir error: dirname=" << dirname;
  CHECK(S_ISDIR(st.st_mode)) << "error not a directory: dirname=" << dirname;
}

bool is_exist_path(const std::string& filename) {
  struct stat buffer;
  return (stat(filename.c_str(), &buffer) == 0 && S_ISDIR(buffer.st_mode));
}

static std::string get_full_filename(const std::string& filename) {
  if (filename[0] == '/') {
    return filename;
  }
  std::string current_p(getcwd(NULL, 0));
  return current_p + "/" + filename;
}

static std::string get_parent_path(const std::string& path) {
  return path.substr(0, path.find_last_of("/"));
}

static void create_parent_path(const std::string& path) {
  if (is_exist_path(path)) {
    LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
        << path << " is exist!" << std::endl;
    return;
  }
  auto parent_path = get_parent_path(path);
  if (!is_exist_path(parent_path)) {
    create_parent_path(parent_path);
  }
  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "create dir : " << path << std::endl;
  mkdir_minus_p(path);
}
bool DeviceMemory::save(const std::string& filename, uint64_t offset,
                        size_t size) {
  auto data = std::vector<char>(size);
  CHECK_EQ(data.size(), size);

  auto ok = download(&data[0], offset, size);
  if (!ok) {
    return false;
  }
  auto path = get_full_filename(filename);
  auto full_filename = path;
  LOG_IF(INFO, false) << "full_filename " << full_filename << " "  //
                      << "is_exist_path(full_filename) "
                      << is_exist_path(full_filename) << " "  //
                      << std::endl;
  /// check file or directory
  auto parent_path = get_parent_path(path);
  create_parent_path(parent_path);

  LOG_IF(INFO, ENV_PARAM(DEBUG_DEVICE_MEMORY))
      << "filename " << filename << " "  //
      << "offset " << offset << " "      //
      << "size " << size << " "          //
      << std::endl;
  auto mode = std::ios_base::out | std::ios_base::binary | std::ios_base::trunc;
  CHECK(std::ofstream(full_filename, mode).write(&data[0], size).good())
      << " faild to write to " << filename;
  return true;
}
/*
{
  LOG_IF(INFO, false) << "filename " << filename << " \n"  //
                      << "fs::current_path.string() "
                      << fs::current_path().string() << " \n"        //
                      << "path.string() " << path.string() << " \n"  //
                      << "path.has_parent_path() " << path.has_parent_path()
                      << " \n "  //
                      << "path.parent_path() " << path.parent_path()
                      << " \n"                                           //
                      << "path.filename() " << path.filename() << " \n"  //
                      << "path.has_root_path() " << path.has_root_path()
                      << " "                                               //
                      << "path.root_path() " << path.root_path() << " \n"  //
                      << "path.root_name() " << path.root_name() << " \n"  //
                      << "path.relative_path() " << path.relative_path()
                      << " \n"  //
}
*/
}  // namespace xir
