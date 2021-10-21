/*
 * Copyright 2021 Xilinx, Inc.
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
#ifndef _DUP_MATCH_COMMON_HPP_
#define _DUP_MATCH_COMMON_HPP_

#include <vector>
#include <map>
#include <string>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>

namespace dup_match {
namespace internal {

// typedef std::unordered_map<std::string, int> MT;
typedef std::map<std::string, uint32_t> MT;
typedef std::map<uint32_t, uint32_t> uuMT;
typedef std::map<uint32_t, double> udMT;
typedef std::pair<uint32_t, double> udPT;

union DTConvert64 {
    uint64_t dt0;
    double dt1;
};

std::string to_preprocess(std::string str);
void splitWord(std::string& inStr, std::vector<std::string>& terms, std::string& outStr);
int checkAlphaNum(char in);
void addMerge(std::vector<udPT>& in1, std::vector<udPT>& in2, std::vector<udPT>& out);

} // internal
} // dup_match
#endif
