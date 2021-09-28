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
#include <ap_int.h>
#include <hls_stream.h>
#include <iostream>
#include <stdlib.h>

#define INPUT_WIDTH 64
#define HASH_WIDTH 14
#define BRAM_MEM_SPACE (1 << (HASH_WIDTH - 4))
#define URAM_MEM_SPACE (1 << (HASH_WIDTH - 6))

#include "xf_database/bloom_filter.hpp"

void syn_bloom_filter_bram(hls::stream<ap_uint<INPUT_WIDTH> >& gen_msg_strm,
                           hls::stream<bool>& gen_in_e_strm,
                           hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                           hls::stream<bool>& check_in_e_strm,
                           hls::stream<bool>& res_msg_strm,
                           hls::stream<bool>& res_in_e_strm) {
    ap_uint<16> bit_vector_ptr0[BRAM_MEM_SPACE];
    ap_uint<16> bit_vector_ptr1[BRAM_MEM_SPACE];
    ap_uint<16> bit_vector_ptr2[BRAM_MEM_SPACE];

    for (int i = 0; i < BRAM_MEM_SPACE; i++) {
#pragma HLS pipeline II = 1
        bit_vector_ptr0[i] = 0;
        bit_vector_ptr1[i] = 0;
        bit_vector_ptr2[i] = 0;
    }

    xf::database::bfGen<true, INPUT_WIDTH, HASH_WIDTH>(gen_msg_strm, gen_in_e_strm, bit_vector_ptr0, bit_vector_ptr1,
                                                       bit_vector_ptr2);

    xf::database::bfCheck<true, INPUT_WIDTH, HASH_WIDTH>(check_msg_strm, check_in_e_strm, bit_vector_ptr0,
                                                         bit_vector_ptr1, bit_vector_ptr2, res_msg_strm, res_in_e_strm);
}

void load_bram_and_check(hls::stream<ap_uint<16> >& bit_vet_strm,
                         hls::stream<bool>& bit_vet_e_strm,
                         hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                         hls::stream<bool>& check_in_e_strm,
                         hls::stream<bool>& res_msg_strm,
                         hls::stream<bool>& res_in_e_strm) {
    // init
    ap_uint<16> bit_vector_ptr0[BRAM_MEM_SPACE];
    ap_uint<16> bit_vector_ptr1[BRAM_MEM_SPACE];
    ap_uint<16> bit_vector_ptr2[BRAM_MEM_SPACE];

    for (int i = 0; i < BRAM_MEM_SPACE; i++) {
#pragma HLS pipeline II = 1
        bit_vector_ptr0[i] = 0;
        bit_vector_ptr1[i] = 0;
        bit_vector_ptr2[i] = 0;
    }

    // load
    bool e;
    int cnt = 0;

    e = bit_vet_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<16> bit_vet = bit_vet_strm.read();
        bit_vector_ptr0[cnt] = bit_vet(15, 0);
        bit_vector_ptr1[cnt] = bit_vet(15, 0);
        bit_vector_ptr2[cnt] = bit_vet(15, 0);

        e = bit_vet_e_strm.read();
        cnt++;
    }

    // check
    xf::database::bfCheck<true, INPUT_WIDTH, HASH_WIDTH>(check_msg_strm, check_in_e_strm, bit_vector_ptr0,
                                                         bit_vector_ptr1, bit_vector_ptr2, res_msg_strm, res_in_e_strm);
}

void syn_bloom_filter_bram_and_strm(hls::stream<ap_uint<INPUT_WIDTH> >& gen_msg_strm,
                                    hls::stream<bool>& gen_in_e_strm,
                                    hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                                    hls::stream<bool>& check_in_e_strm,
                                    hls::stream<bool>& res_msg_strm,
                                    hls::stream<bool>& res_in_e_strm) {
#pragma HLS dataflow
    hls::stream<ap_uint<16> > bit_vet_strm;
#pragma HLS STREAM variable = bit_vet_strm depth = 64
    hls::stream<bool> bit_vet_e_strm;
#pragma HLS STREAM variable = bit_vet_e_strm depth = 64

    xf::database::bfGenStream<true, INPUT_WIDTH, HASH_WIDTH>(gen_msg_strm, gen_in_e_strm, bit_vet_strm, bit_vet_e_strm);

    load_bram_and_check(bit_vet_strm, bit_vet_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm, res_in_e_strm);
}

void syn_bloom_filter_uram(hls::stream<ap_uint<INPUT_WIDTH> >& gen_msg_strm,
                           hls::stream<bool>& gen_in_e_strm,
                           hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                           hls::stream<bool>& check_in_e_strm,
                           hls::stream<bool>& res_msg_strm,
                           hls::stream<bool>& res_in_e_strm) {
    ap_uint<72> bit_vector_ptr0[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr0 type = ram_2p impl = uram

    ap_uint<72> bit_vector_ptr1[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr1 type = ram_2p impl = uram

    ap_uint<72> bit_vector_ptr2[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr2 type = ram_2p impl = uram

    for (int i = 0; i < URAM_MEM_SPACE; i++) {
#pragma HLS pipeline II = 1
        bit_vector_ptr0[i] = 0;
        bit_vector_ptr1[i] = 0;
        bit_vector_ptr2[i] = 0;
    }
    xf::database::bfGen<false, INPUT_WIDTH, HASH_WIDTH>(gen_msg_strm, gen_in_e_strm, bit_vector_ptr0, bit_vector_ptr1,
                                                        bit_vector_ptr2);
    xf::database::bfCheck<false, INPUT_WIDTH, HASH_WIDTH>(check_msg_strm, check_in_e_strm, bit_vector_ptr0,
                                                          bit_vector_ptr1, bit_vector_ptr2, res_msg_strm,
                                                          res_in_e_strm);
}

void load_uram_and_check(hls::stream<ap_uint<64> >& bit_vet_strm,
                         hls::stream<bool>& bit_vet_e_strm,
                         hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                         hls::stream<bool>& check_in_e_strm,
                         hls::stream<bool>& res_msg_strm,
                         hls::stream<bool>& res_in_e_strm) {
    // init
    ap_uint<72> bit_vector_ptr0[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr0 type = ram_2p impl = uram

    ap_uint<72> bit_vector_ptr1[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr1 type = ram_2p impl = uram

    ap_uint<72> bit_vector_ptr2[URAM_MEM_SPACE];
#pragma HLS bind_storage variable = bit_vector_ptr2 type = ram_2p impl = uram

    for (int i = 0; i < URAM_MEM_SPACE; i++) {
#pragma HLS pipeline II = 1
        bit_vector_ptr0[i] = 0;
        bit_vector_ptr1[i] = 0;
        bit_vector_ptr2[i] = 0;
    }

    // load
    bool e;
    int cnt = 0;

    e = bit_vet_e_strm.read();
    while (!e) {
#pragma HLS pipeline II = 1
        ap_uint<72> bit_vet = bit_vet_strm.read();
        bit_vector_ptr0[cnt] = bit_vet(71, 0);
        bit_vector_ptr1[cnt] = bit_vet(71, 0);
        bit_vector_ptr2[cnt] = bit_vet(71, 0);

        e = bit_vet_e_strm.read();
        cnt++;
    }

    // check
    xf::database::bfCheck<false, INPUT_WIDTH, HASH_WIDTH>(check_msg_strm, check_in_e_strm, bit_vector_ptr0,
                                                          bit_vector_ptr1, bit_vector_ptr2, res_msg_strm,
                                                          res_in_e_strm);
}

void syn_bloom_filter_uram_and_strm(hls::stream<ap_uint<INPUT_WIDTH> >& gen_msg_strm,
                                    hls::stream<bool>& gen_in_e_strm,
                                    hls::stream<ap_uint<INPUT_WIDTH> >& check_msg_strm,
                                    hls::stream<bool>& check_in_e_strm,
                                    hls::stream<bool>& res_msg_strm,
                                    hls::stream<bool>& res_in_e_strm) {
#pragma HLS dataflow
    hls::stream<ap_uint<64> > bit_vet_strm;
#pragma HLS STREAM variable = bit_vet_strm depth = 64
    hls::stream<bool> bit_vet_e_strm;
#pragma HLS STREAM variable = bit_vet_e_strm depth = 64

    xf::database::bfGenStream<false, INPUT_WIDTH, HASH_WIDTH>(gen_msg_strm, gen_in_e_strm, bit_vet_strm,
                                                              bit_vet_e_strm);

    load_uram_and_check(bit_vet_strm, bit_vet_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm, res_in_e_strm);
}

int test_bloom_filter(bool is_bram, bool is_strm) {
    hls::stream<ap_uint<INPUT_WIDTH> > gen_msg_strm;
    hls::stream<bool> gen_in_e_strm;
    hls::stream<ap_uint<INPUT_WIDTH> > check_msg_strm;
    hls::stream<bool> check_in_e_strm;
    hls::stream<bool> res_msg_strm;
    hls::stream<bool> res_in_e_strm;

    ap_uint<INPUT_WIDTH>* gen_list = (ap_uint<INPUT_WIDTH>*)malloc(100 * sizeof(long long));
    bool* check_res = (bool*)calloc(10000, sizeof(bool));

    for (int i = 0; i < 100; i++) {
        gen_list[i] = rand() % 1000;
        gen_msg_strm.write(gen_list[i]);
        gen_in_e_strm.write(0);
    }
    gen_in_e_strm.write(1);

    for (int i = 0; i < 10000; i++) {
        ap_uint<INPUT_WIDTH> check_num = rand() % 1000;
        check_msg_strm.write(check_num);
        check_in_e_strm.write(0);
        for (int j = 0; j < 100; j++) {
            if (gen_list[j] == check_num) {
                check_res[i] = 1;
                break;
            }
        }
    }
    check_in_e_strm.write(1);

    if (is_bram && !is_strm) {
        syn_bloom_filter_bram(gen_msg_strm, gen_in_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm,
                              res_in_e_strm);
    } else if (is_bram && is_strm) {
        syn_bloom_filter_bram_and_strm(gen_msg_strm, gen_in_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm,
                                       res_in_e_strm);

    } else if (!is_bram && !is_strm) {
        syn_bloom_filter_uram(gen_msg_strm, gen_in_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm,
                              res_in_e_strm);
    } else {
        syn_bloom_filter_uram_and_strm(gen_msg_strm, gen_in_e_strm, check_msg_strm, check_in_e_strm, res_msg_strm,
                                       res_in_e_strm);
    }

    int cnt = 0;
    int nerr = 0;
    bool e = res_in_e_strm.read();
    while (!e) {
        bool res = res_msg_strm.read();
        if (res == 0 && check_res[cnt] == 1) nerr++;
        e = res_in_e_strm.read();
        cnt++;
    }

    if (cnt != 10000) nerr++;

    return nerr;
}

int main(int argc, char* argv[]) {
    int nerr = 0;

    if (argv[1][0] == '1') {
        nerr = nerr + test_bloom_filter(false, false);
        if (nerr)
            std::cout << "FAIL: ";
        else
            std::cout << "PASS: ";
        std::cout << "Bloom filter URAM and non-stream testcase." << std::endl;

    } else if (argv[1][0] == '2') {
        nerr = nerr + test_bloom_filter(false, true);
        if (nerr)
            std::cout << "FAIL: ";
        else
            std::cout << "PASS: ";
        std::cout << "Bloom filter URAM and stream testcase." << std::endl;

    } else if (argv[1][0] == '3') {
        nerr = nerr + test_bloom_filter(true, false);
        if (nerr)
            std::cout << "FAIL: ";
        else
            std::cout << "PASS: ";
        std::cout << "Bloom filter BRAM and non-stream testcase." << std::endl;

    } else if (argv[1][0] == '4') {
        nerr = nerr + test_bloom_filter(true, true);
        if (nerr)
            std::cout << "FAIL: ";
        else
            std::cout << "PASS: ";
        std::cout << "Bloom filter BRAM and stream testcase." << std::endl;

    } else {
        std::cout << "FAIL: Testcase not found." << std::endl;
        nerr++;
    }

    return nerr;
}
