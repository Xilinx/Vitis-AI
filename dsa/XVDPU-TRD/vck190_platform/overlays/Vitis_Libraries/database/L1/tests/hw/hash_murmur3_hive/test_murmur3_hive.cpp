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

#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include "xf_database/hash_murmur3_hive.hpp"

// This macro cannot be set bigger than 100,
// since the testset file only contains 100 key-hash pairs
#define TEST_NUM 100

// table to save each input key and its hash value
struct Test {
    ap_int<64> key;
    ap_int<64> hash;
    Test(ap_int<64> k, ap_int<64> h) : key(k), hash(h) {}
};

void murmur3_hive_dut(hls::stream<ap_int<64> >& key_strm, hls::stream<ap_int<64> >& hash_strm) {
    for (int i = 0; i < TEST_NUM; i++) {
#pragma HLS pipeline II = 1
        xf::database::hashMurmur3Hive(key_strm, hash_strm);
    }
}

int main(int argc, const char* argv[]) {
    // in/out stream for FPGA
    hls::stream<ap_int<64> > key_strm("key_strm");
    hls::stream<ap_int<64> > hash_strm("hash_strm");

    // number of errors in current test
    int nerror = 0;

    // vector for saving key-hash pairs
    std::vector<Test> tests;

    // input file stream
    std::ifstream infile("testset", std::ios::in);
    if (infile.is_open()) {
        for (int i = 0; i < TEST_NUM; i++) {
            std::string in;
            // read key from testset file
            getline(infile, in);
            // transform to long type
            ap_int<64> key = std::stol(in);
            // write to key stream
            key_strm.write(key);
            // read golden hash value from testset file
            getline(infile, in);
            // transform to long type
            ap_int<64> golden = std::stol(in);
            // push to key-hash table
            tests.push_back(Test(key, golden));
        }
    } else {
        std::cout << "Error in opening input testset file" << std::endl;
        return -1;
    }

    // call FPGA
    murmur3_hive_dut(key_strm, hash_strm);

    // checks result
    for (std::vector<Test>::const_iterator singletest = tests.begin(); singletest != tests.end(); singletest++) {
        ap_int<64> hash = hash_strm.read();
        if (hash != (*singletest).hash) {
            nerror++;
            std::cout << "ERROR: key    = " << std::dec << (*singletest).key << std::endl;
            std::cout << "ERROR: hash   = " << std::hex << hash << std::endl;
            std::cout << "ERROR: golden = " << std::hex << (*singletest).hash << std::endl;
        }
    }

    if (nerror) {
        std::cout << "FAIL: found " << std::dec << nerror << " errors in " << TEST_NUM << " inputs." << std::endl;
    } else {
        std::cout << "PASS: " << std::dec << TEST_NUM << " inputs verified." << std::endl;
    }
    return nerror;
}
