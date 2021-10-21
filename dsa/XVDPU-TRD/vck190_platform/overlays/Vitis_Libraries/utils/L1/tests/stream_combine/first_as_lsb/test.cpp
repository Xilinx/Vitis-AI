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
#include <stdlib.h>
#include <iostream>
#include <vector>

#include "xf_utils_hw/stream_combine.hpp"

#include "ap_int.h"
#include "hls_stream.h"
#define WIN 32
#define NSTRM 16

extern "C" void dut(hls::stream<ap_uint<NSTRM> >& cfg,
                    hls::stream<ap_uint<WIN> > in[NSTRM],
                    hls::stream<bool>& ein,
                    hls::stream<ap_uint<WIN * NSTRM> >& out,
                    hls::stream<bool>& eout) {
    xf::common::utils_hw::streamCombine(cfg, in, ein, out, eout, xf::common::utils_hw::LSBSideT());
}

#ifndef __SYNTHESIS__

// generate a random integer sequence between specified limits a and b (a<b);
unsigned rand_uint(unsigned a, unsigned b) {
    return rand() % (b - a + 1) + a;
}

// generate test data
template <int _WIn, int _NStrm>
void generate_test_data(ap_uint<_NStrm>& testcfg, uint64_t len, std::vector<std::vector<ap_uint<_WIn> > >& testvector) {
    for (int j = 0; j < _NStrm; j++) {
        testcfg[j] = (ap_uint<1>)(rand_uint(0, 1));
        std::cout << testcfg[j];
    }
    std::cout << std::endl;
    for (int i = 0; i < len; i++) {
        std::vector<ap_uint<_WIn> > a;
        for (int j = 0; j < _NStrm; j++) {
            unsigned randnum = rand_uint(1, 9);
            a.push_back((ap_uint<_WIn>)randnum);
            std::cout << randnum << " ";
        }
        testvector.push_back(a);
        std::cout << std::endl;
        a.clear();
    }
    std::cout << " random test data generated! " << std::endl;
}

int main(int argc, const char* argv[]) {
    int err = 0; // 0 for pass, 1 for error
                 // TODO prepare cfg, in, ein; decl out, eout

    int len = 100;
    ap_uint<NSTRM> testcfg;
    std::vector<std::vector<ap_uint<WIN> > > testvector;
    hls::stream<ap_uint<NSTRM> > incfg;
    hls::stream<ap_uint<WIN> > in[NSTRM];
    hls::stream<ap_uint<WIN * NSTRM> > out;
    hls::stream<bool> ein;
    hls::stream<bool> eout;

    // reference vector
    std::vector<ap_uint<WIN * NSTRM> > refvec;
    // generate test data
    generate_test_data<WIN, NSTRM>(testcfg, len, testvector);
    // prepare data to stream
    for (std::string::size_type i = 0; i < len; i++) {
        int count = 0;
        ap_uint<WIN * NSTRM> tmp;
        if (i == 0) {
            incfg.write(testcfg);
        }
        for (int j = 0; j < NSTRM; j++) {
            in[j].write(testvector[i][j]);
            if (testcfg[j] == 1) {
                tmp((count + 1) * WIN - 1, count * WIN) = testvector[i][j];
                count++;
            }
        }
        if (count < NSTRM)
            for (int j = count; j < NSTRM; j++) {
                tmp((j + 1) * WIN - 1, j * WIN) = (ap_uint<WIN>)(0);
            }
        refvec.push_back(tmp);

        ein.write(0);
    }

    ein.write(1);

    // run hls::func
    dut(incfg, in, ein, out, eout);
    // compare hls::func and reference result
    for (std::string::size_type i = 0; i < len; i++) {
        ap_uint<WIN* NSTRM> out_res = out.read();
        for (int n = 0; n < NSTRM; n++) {
            std::cout << refvec[i].range(WIN * (n + 1) - 1, WIN * n) << "   ";
        }
        for (int n = 0; n < NSTRM; n++) {
            std::cout << out_res.range(WIN * (n + 1) - 1, WIN * n) << "   ";
        }
        std::cout << err << std::endl;
        bool comp_res = (refvec[i] == out_res) ? 1 : 0;
        if (!comp_res) {
            err++;
        }
    }
    // compare e flag
    for (std::string::size_type i = 0; i < len; i++) {
        bool estrm = eout.read();
        if (estrm) {
            err++;
        }
    }
    bool estrm = eout.read();
    if (!estrm) {
        err++;
    }

    // TODO check out, eout
    if (err) {
        std::cout << "\nFAIL: nerror= " << err << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return err;
}

#endif
