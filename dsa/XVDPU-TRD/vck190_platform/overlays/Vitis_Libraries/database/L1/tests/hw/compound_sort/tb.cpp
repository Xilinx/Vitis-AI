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
#include <math.h>
#include <iostream>
#include <algorithm>
#include "dut.hpp"

int main() {
    int nerr = 0;

    std::vector<KEY_TYPE> v(SORT_LEN);
    std::vector<KEY_TYPE> inKey(SORT_LEN);
    std::vector<KEY_TYPE> outKey(SORT_LEN);
    for (unsigned i = 0; i < v.size(); i++) {
        v[i] = rand();
        inKey[i] = v[i];
    }

    std::sort(v.begin(), v.end());

    hls::stream<KEY_TYPE> inKeyStrm;
    hls::stream<KEY_TYPE> outKeyStrm;
    hls::stream<bool> inEndStrm;
    hls::stream<bool> outEndStrm;

    for (int i = 0; i < SORT_LEN; i++) {
        inKeyStrm.write(inKey[i]);
        inEndStrm.write(false);
    }
    inEndStrm.write(true);

    dut(1, inKeyStrm, inEndStrm, outKeyStrm, outEndStrm);

    for (int i = 0; i < SORT_LEN; i++) {
        outEndStrm.read();
        KEY_TYPE outKey = outKeyStrm.read();
        bool cmp_key = (outKey == v[i]) ? 1 : 0;
        if (!cmp_key) {
            std::cout << "v[" << i << "]=" << v[i] << ",key[" << i << "]=" << outKey << std::endl;
            std::cout << "\nthe sort key is incorrect" << std::endl;
            nerr++;
        }
    }
    outEndStrm.read();
    return nerr;
}
