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
#include <sys/stat.h>
#include <filesystem>
#include <fstream>
#include <map>
#include "vitis/ai/env_config.hpp"
#include "xir/device_memory.hpp"
DEF_ENV_PARAM(DEBUG_DEVICE_MEMORY, "0");

namespace xir {

static void mkdir_minus_p(const std::string& dirname) {
  CHECK(std::filesystem::create_directories(dirname))
      << "cannot create directories: " << dirname;
}

bool is_exist_path(const std::string& filename) {
  return std::filesystem::exists(filename);
}

static std::string get_full_filename(const std::string& filename) {
  if (filename[0] == std::filesystem::path::preferred_separator) {
    return filename;
  }
  return (std::filesystem::current_path() / filename).string();
}

static std::string get_parent_path(const std::string& path) {
  return path.substr(
      0, path.find_last_of(std::filesystem::path::preferred_separator));
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

std::unique_ptr<DeviceMemory> DeviceMemory::create(size_t v) {
  return DeviceMemory::create0(v);
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
// ON MSVC: no implementation yet. so that is possible that no factory method is
// defined. cause runtime error.
DECLARE_INJECTION_NULLPTR(xir::DeviceMemory, size_t&);
