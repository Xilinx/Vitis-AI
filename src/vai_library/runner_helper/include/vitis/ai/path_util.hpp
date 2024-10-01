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
#include <string>
namespace vitis {
namespace ai {
std::string file_name_realpath(const std::string& filename);
std::string file_name_directory(const std::string& filename);
std::string file_name_basename(const std::string& filename);
std::string file_name_basename_no_ext(const std::string& filename);
std::string file_name_ext(const std::string& filename);
std::string to_valid_file_name(const std::string& filename);
bool is_directory(const std::string& filename);
bool is_regular_file(const std::string& filename);
void create_parent_path(const std::string& path);
size_t file_size(const std::string& filename);
}  // namespace ai
}  // namespace vitis
