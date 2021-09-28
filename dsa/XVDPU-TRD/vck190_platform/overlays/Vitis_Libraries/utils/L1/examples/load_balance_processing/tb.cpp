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
#include "code.hpp"

// extract the meaningful data from the input data, then calculate.
ap_uint<W_PU> calculate(ap_uint<W_PU> data) {
#pragma HLS inline
    ap_uint<W_PU> p = data.range(W_PRC - 1, 0);
    ap_uint<W_PU> d = data.range(W_PRC + W_DSC - 1, 0);
    ap_uint<W_PU> nd = p * d;
    return nd;
}

int test() {
    hls::stream<ap_uint<W_STRM> > istrm;
    hls::stream<bool> e_istrm;
    hls::stream<ap_uint<W_STRM> > ostrm;
    hls::stream<bool> e_ostrm;

    std::cout << std::dec << "W_STRM  = " << W_STRM << std::endl;
    std::cout << std::dec << "W_PU    = " << W_PU << std::endl;
    std::cout << std::dec << "NPU     = " << NPU << std::endl;
    std::cout << std::dec << "NS      = " << NS << std::endl;
    /*
     *  1. Generate test data and merege them to a wide width stream
     *  2. Updata the data in hardware
     *  3. Fetch back the updated data and calculate their dot product.
     *
     *  Distribution on load balance is NOT order-preserving, so calculate their dot product to verify the result.
     */

    // Generate data
    ap_uint<W_STRM> gld = 0;
    for (int i = 0; i < NS; ++i) {
        ap_uint<W_STRM> bd = 0;
        // generate test data
        for (int j = 0; j < W_STRM / W_PU; ++j) {
            ap_uint<W_PRC> p = i;
            ap_uint<W_DSC> d = i % 10;
            ap_uint<W_PU> data = 0;
            data.range(W_PRC - 1, 0) = p;
            data.range(W_DSC + W_PRC - 1, W_PRC) = d;
            // combine to a wide width data
            bd.range((j + 1) * W_PU - 1, j * W_PU) = data;
            // calculate the golden result
            ap_uint<W_PU> nd = update_data(data);
            gld += calculate(nd);
        }
        istrm.write(bd);
        e_istrm.write(false);
    }
    e_istrm.write(true);
    // Update data in hardware
    test_core(istrm, e_istrm, ostrm, e_ostrm);

    // Fetch bach the updated data and calculate their dot product.
    int nerror = 0;
    int count = 0;
    ap_uint<W_STRM> total = 0;
    while (!e_ostrm.read()) {
        ap_uint<W_STRM> bd = ostrm.read();
        // calculate the test result
        for (int j = 0; j < W_STRM / W_PU; ++j) {
            ap_uint<W_PU> data = bd.range((j + 1) * W_PU - 1, j * W_PU);
            total += calculate(data);
        }
        count++;
    }
    std::cout << "\n  total=" << total << "   gld=" << gld << std::endl;
    if (total != gld) {
        nerror = 1;
        std::cout << "\n error: the test result is not equal to glden result." << std::endl;
    }

    std::cout << "\n read: " << count << std::endl;
    if (count != NS) {
        nerror = 1;
        std::cout << "\n error  read: " << count << std::endl;
    }
    if (nerror) {
        std::cout << "\nFAIL: " << nerror << "the order is wrong.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

int main() {
    return test();
}
