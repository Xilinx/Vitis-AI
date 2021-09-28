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
#include <iostream>
#include <stdlib.h>

#include "xf_utils_hw/stream_discard.hpp"

#define NS 1024 * 8
#define NSTRM 4

void test_core_discard_share_e(hls::stream<float> istrms[NSTRM], hls::stream<bool>& e_istrm) {
    xf::common::utils_hw::streamDiscard<float, NSTRM>(istrms, e_istrm);
}

int test_discard_share_e() {
    hls::stream<float> data_istrms[NSTRM];
    hls::stream<bool> e_data_istrm;

    std::cout << std::dec << "NSTRM     = " << NSTRM << std::endl;
    std::cout << std::dec << "NS        = " << NS << std::endl;
    for (int d = 1; d <= NS; ++d) {
        int id = (d - 1) % NSTRM;
        data_istrms[id].write(d + 0.5);
        if (id == 0) e_data_istrm.write(false);
    }

    e_data_istrm.write(true);

    test_core_discard_share_e(data_istrms, e_data_istrm);
    std::cout << "\nPASS: no error found.\n";
    return 0;
}

int main() {
    return test_discard_share_e();
}
