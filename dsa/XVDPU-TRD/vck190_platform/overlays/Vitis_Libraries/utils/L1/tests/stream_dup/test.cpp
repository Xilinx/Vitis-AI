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
#include <stdint.h>

#include "xf_utils_hw/stream_dup.hpp"

#include "ap_int.h"
#include "hls_stream.h"

typedef uint32_t TYPE;

#define LEN_STRM 10
#define NUM_ISTRM 8
#define NUM_DSTRM 4
#define NUM_COPY 16

extern "C" void dut0(hls::stream<TYPE>& istrm,
                     hls::stream<bool>& e_istrm,
                     hls::stream<TYPE> ostrms[NUM_COPY],
                     hls::stream<bool> e_ostrms[NUM_COPY]) {
    xf::common::utils_hw::streamDup<TYPE, NUM_COPY>(istrm, e_istrm, ostrms, e_ostrms);
}

extern "C" void dut1(hls::stream<TYPE> istrm[NUM_ISTRM],
                     hls::stream<bool>& e_istrm,
                     hls::stream<TYPE> ostrms[NUM_ISTRM],
                     hls::stream<TYPE> dstrms[NUM_COPY][NUM_DSTRM],
                     hls::stream<bool>& e_ostrms) {
    const unsigned int choose[4] = {0, 1, 2, 3};
    xf::common::utils_hw::streamDup<TYPE, NUM_ISTRM, NUM_DSTRM, NUM_COPY>(choose, istrm, e_istrm, ostrms, dstrms,
                                                                          e_ostrms);
}

#ifndef __SYNTHESIS__

int test_dut0() {
    int nerr = 0;
    int i;

    //================generate test data=========================

    TYPE testdata[NUM_ISTRM][LEN_STRM];
    TYPE glddata[NUM_COPY * NUM_DSTRM][LEN_STRM];

    for (i = 0; i < NUM_DSTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            testdata[i][j] = i * 10 + j; // rand()%1000;
            for (int k = 0; k < 16; k++) {
                glddata[i * 16 + k][j] = testdata[i][j];
            }
        }
    }

    for (; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            testdata[i][j] = i * 10 + j; // rand()%1000;
        }
    }

    hls::stream<TYPE> istrm[NUM_ISTRM];
    hls::stream<bool> e_istrm[NUM_ISTRM];
    hls::stream<TYPE> ostrms[NUM_DSTRM][NUM_COPY];
    hls::stream<bool> e_ostrms[NUM_DSTRM][NUM_COPY];

    for (i = 0; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            istrm[i].write(testdata[i][j]);
            e_istrm[i].write(0);
        }
        e_istrm[i].write(1);
    }

    //================test module===============================

    for (i = 0; i < NUM_DSTRM; i++) {
        dut0(istrm[i], e_istrm[i], ostrms[i], e_ostrms[i]);
    }

    //================check result==============================

    bool rd_success;
    TYPE outdata;
    bool e;

    // check duplicated data
    for (i = 0; i < NUM_DSTRM; i++) {
        for (int k = 0; k < NUM_COPY; k++) {
            for (int j = 0; j < LEN_STRM; j++) {
                rd_success = ostrms[i][k].read_nb(outdata);
                if (!rd_success) {
                    nerr++;
                    std::cout << "\n error: data loss\n";
                } else if (glddata[i * NUM_COPY + k][j] != outdata)
                    nerr++;
            }
        }
    }

    // check non-duplicated data
    for (; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            rd_success = istrm[i].read_nb(outdata);
            if (!rd_success) {
                nerr++;
                std::cout << "\n error: data loss\n";
            } else if (testdata[i][j] != outdata)
                nerr++;
        }
    }

    // check duplicated data end flag
    for (i = 0; i < NUM_DSTRM; i++) {
        for (int k = 0; k < NUM_COPY; k++) {
            for (int j = 0; j < LEN_STRM; j++) {
                rd_success = e_ostrms[i][k].read_nb(e);
                if (!rd_success) {
                    nerr++;
                    std::cout << "\n error: end flag loss\n";
                } else if (e)
                    nerr++;
            }
            rd_success = e_ostrms[i][k].read_nb(e);
            if (!rd_success) {
                nerr++;
                std::cout << "\n error: end flag loss\n";
            } else if (!e)
                nerr++;
        }
    }

    // check non-duplicated data end flag
    for (; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            rd_success = e_istrm[i].read_nb(e);
            if (!rd_success) {
                nerr++;
                std::cout << "\n error: end flag loss\n";
            } else if (e)
                nerr++;
        }
        rd_success = e_istrm[i].read_nb(e);
        if (!rd_success) {
            nerr++;
            std::cout << "\n error: end flag loss\n";
        } else if (!e)
            nerr++;
    }

    return nerr;
}

int test_dut1() {
    int nerr = 0;
    int i;

    //================generate test data=========================

    TYPE testdata[NUM_ISTRM][LEN_STRM];
    TYPE glddata[NUM_COPY * NUM_DSTRM][LEN_STRM];

    for (i = 0; i < NUM_DSTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            testdata[i][j] = i * 10 + j; // rand()%1000;
            for (int k = 0; k < NUM_COPY; k++) {
                glddata[i * NUM_COPY + k][j] = testdata[i][j];
            }
        }
    }

    for (; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            testdata[i][j] = i * 10 + j; // rand()%1000;
        }
    }

    hls::stream<TYPE> istrm[NUM_ISTRM];
    hls::stream<bool> e_istrm;
    hls::stream<TYPE> ostrms[NUM_ISTRM];
    hls::stream<TYPE> dstrms[NUM_COPY][NUM_DSTRM];
    hls::stream<bool> e_ostrms;

    for (int j = 0; j < LEN_STRM; j++) {
        for (i = 0; i < NUM_ISTRM; i++) {
            istrm[i].write(testdata[i][j]);
        }
        e_istrm.write(0);
    }
    e_istrm.write(1);

    //================test module===============================

    dut1(istrm, e_istrm, ostrms, dstrms, e_ostrms);

    //================check result==============================

    bool rd_success;
    TYPE outdata;
    bool e;

    // check duplicated data
    for (i = 0; i < NUM_DSTRM; i++) {
        for (int k = 0; k < NUM_COPY; k++) {
            for (int j = 0; j < LEN_STRM; j++) {
                rd_success = dstrms[k][i].read_nb(outdata);
                if (!rd_success) {
                    nerr++;
                    std::cout << "\n error: data loss\n";
                } else if (glddata[i * NUM_COPY + k][j] != outdata)
                    nerr++;
            }
        }
    }

    // check non-duplicated data
    for (i = 0; i < NUM_ISTRM; i++) {
        for (int j = 0; j < LEN_STRM; j++) {
            rd_success = ostrms[i].read_nb(outdata);
            if (!rd_success) {
                nerr++;
                std::cout << "\n error: data loss\n";
            } else if (testdata[i][j] != outdata)
                nerr++;
        }
    }

    // check end flag
    for (int j = 0; j < LEN_STRM; j++) {
        rd_success = e_ostrms.read_nb(e);
        if (!rd_success) {
            nerr++;
            std::cout << "\n error: end flag loss\n";
        } else if (e)
            nerr++;
    }
    rd_success = e_ostrms.read_nb(e);
    if (!rd_success) {
        nerr++;
        std::cout << "\n error: end flag loss\n";
    } else if (!e)
        nerr++;

    return nerr;
}

int main(int argc, const char* argv[]) {
    int nerr = 0;

    if (argv[1][0] == '0') {
        nerr = nerr + test_dut0();
        if (nerr) {
            std::cout << "\nFAIL: nerror= " << nerr << " errors found.\n";
        } else {
            std::cout << "\nPASS: no error found.\n";
        }
    } else if (argv[1][0] == '1') {
        nerr = nerr + test_dut1();
        if (nerr) {
            std::cout << "\nFAIL: nerror= " << nerr << " errors found.\n";
        } else {
            std::cout << "\nPASS: no error found.\n";
        }
    } else {
        std::cout << "\nFAIL: test not found.\n";
        nerr++;
    }
    return nerr;
}

#endif
