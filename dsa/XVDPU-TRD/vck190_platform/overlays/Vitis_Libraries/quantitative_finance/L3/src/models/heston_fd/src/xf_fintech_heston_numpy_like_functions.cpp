/*
 * Copyright 2019 Xilinx, Inc.
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

#include <vector>

#include "xf_fintech_heston_types.hpp"

namespace xf {
namespace fintech {
namespace hestonfd {

void Diff(std::vector<double> inputArray, std::vector<double>* diffArray) {
    unsigned int i;

    for (i = 0; i < inputArray.size() - 1; i++) {
        diffArray->push_back(inputArray.at(i + 1) - inputArray.at(i));
    }
}

void Concatenate(std::vector<double> firstInputArray,
                 std::vector<double> secondInputArray,
                 std::vector<double>* concatenatedArray) {
    unsigned int i;

    for (i = 0; i < firstInputArray.size(); i++) {
        concatenatedArray->push_back(firstInputArray.at(i));
    }

    for (i = 0; i < secondInputArray.size(); i++) {
        concatenatedArray->push_back(secondInputArray.at(i));
    }
}

} // namespace hestonfd
} // namespace fintech
} // namespace xf
