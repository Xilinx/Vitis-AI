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

#include <map>
#include <string>
#include <typeindex>
#include <typeinfo>
#include <vector>

namespace xir {

using namespace std;

std::type_index TYPE_INDEX_BOOL = std::type_index(typeid(bool));
std::type_index TYPE_INDEX_INT8 = std::type_index(typeid(int8_t));
std::type_index TYPE_INDEX_UINT8 = std::type_index(typeid(uint8_t));
std::type_index TYPE_INDEX_INT16 = std::type_index(typeid(int16_t));
std::type_index TYPE_INDEX_UINT16 = std::type_index(typeid(uint16_t));
std::type_index TYPE_INDEX_INT32 = std::type_index(typeid(int32_t));
std::type_index TYPE_INDEX_UINT32 = std::type_index(typeid(uint32_t));
std::type_index TYPE_INDEX_INT64 = std::type_index(typeid(int64_t));
std::type_index TYPE_INDEX_UINT64 = std::type_index(typeid(uint64_t));
std::type_index TYPE_INDEX_FLOAT = std::type_index(typeid(float));
std::type_index TYPE_INDEX_DOUBLE = std::type_index(typeid(double));
std::type_index TYPE_INDEX_STRING = std::type_index(typeid(std::string));
std::type_index TYPE_INDEX_BYTES = std::type_index(typeid(vector<char>));

std::type_index TYPE_INDEX_BOOL_VEC = std::type_index(typeid(vector<bool>));
std::type_index TYPE_INDEX_INT8_VEC = std::type_index(typeid(vector<int8_t>));
std::type_index TYPE_INDEX_UINT8_VEC = std::type_index(typeid(vector<uint8_t>));
std::type_index TYPE_INDEX_INT16_VEC = std::type_index(typeid(vector<int16_t>));
std::type_index TYPE_INDEX_UINT16_VEC =
    std::type_index(typeid(vector<uint16_t>));
std::type_index TYPE_INDEX_INT32_VEC = std::type_index(typeid(vector<int32_t>));
std::type_index TYPE_INDEX_UINT32_VEC =
    std::type_index(typeid(vector<uint32_t>));
std::type_index TYPE_INDEX_INT64_VEC = std::type_index(typeid(vector<int64_t>));
std::type_index TYPE_INDEX_UINT64_VEC =
    std::type_index(typeid(vector<uint64_t>));
std::type_index TYPE_INDEX_FLOAT_VEC = std::type_index(typeid(vector<float>));
std::type_index TYPE_INDEX_DOUBLE_VEC = std::type_index(typeid(vector<double>));
std::type_index TYPE_INDEX_STRING_VEC =
    std::type_index(typeid(vector<std::string>));
std::type_index TYPE_INDEX_BYTES_VEC =
    std::type_index(typeid(vector<vector<char>>));

// special, map type
std::type_index TYPE_INDEX_MAP_STR_2_INT32 =
    std::type_index(typeid(map<string, int32_t>));
std::type_index TYPE_INDEX_MAP_STR_2_VEC_CHAR =
    std::type_index(typeid(map<string, vector<char>>));
std::type_index TYPE_INDEX_MAP_STR_2_STR =
    std::type_index(typeid(map<string, string>));

}  // namespace xir
