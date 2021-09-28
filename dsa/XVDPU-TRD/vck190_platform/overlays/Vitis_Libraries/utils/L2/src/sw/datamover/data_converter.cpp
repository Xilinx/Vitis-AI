/*
 * Copyright 2020 Xilinx, Inc.
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

#include <cassert>
#include <iostream>
#include <iomanip>
#include <fstream>

// just want the naive float2half impl, not FPO ver
#define HLS_NO_XIL_FPO_LIB
#include "hls_half.h"

// Use Argument Parser From Utility library
#include "xf_utils_sw/arg_parser.hpp"

// The widest port supported
#define MAXVEC 512

// overload for each type we support.
uint16_t get_bits(half v) {
    return v.get_bits();
}

uint32_t get_bits(float v) {
    union {
        float f;
        uint32_t u;
    } u;
    u.f = v;
    uint32_t ret = u.u;
    return ret;
}

uint64_t get_bits(double v) {
    union {
        double d;
        uint64_t u;
    } u;
    u.d = v;
    uint64_t ret = u.u;
    return ret;
}

template <typename T>
T get_bits(T v) {
    return v;
}

// for reading from input file with '>>', most native type can by default support this.
template <class T>
struct read_type {
    typedef T type;
};

// work around for half, which cannot directly read.
template <>
struct read_type<half> {
    typedef float type;
};

// read a file, and write hex string as required.
template <typename T>
void generate_hex(std::string& ifile, std::string& ofile, const int width, const int total_num) {
    assert(width < MAXVEC && "width required for output is too wide");
    const int W = 8 * sizeof(T);
    T vec[MAXVEC / W];
    const int n = width / W;
    assert(n > 0 && "output width have to be wider than input");
    int i = 0;
    int cnt = 0;
    int cur_num = 0;
    std::ifstream fin(ifile, std::ios_base::in);
    typename read_type<T>::type tmp;
    std::ofstream fout;
    fout.open(ofile);
    while (fin >> tmp) {
        cur_num++;
        vec[i++] = tmp;
        if (i == n) {
            i = 0;
            if (cnt) {
                fout << ",\n";
            }
            fout << "\"0x";
            for (int j = n - 1; j >= 0; --j) {
                // first data on LSB
                fout << std::hex << std::setfill('0') << std::setw(2 * sizeof(T)) << get_bits(vec[j]);
            }
            fout << "\"";
            cnt++;
        }
        // hit last, break the loop
        if (cur_num == total_num) {
            break;
        }
    }
    fin.close();
    // some incomplete
    if (i) {
        if (cnt) {
            fout << ",\n";
        }
        fout << "\"0x";
        for (int j = n - 1; j >= 0; --j) {
            // first data on LSB
            if (j >= i) {
                fout << std::hex << std::setfill('0') << std::setw(2 * sizeof(T)) << 0;
            } else {
                fout << std::hex << std::setfill('0') << std::setw(2 * sizeof(T)) << get_bits(vec[j]);
            }
        }
        fout << "\"";
        cnt++;
    }
    fout.close();
}

int main(int argc, const char* argv[]) {
    xf::common::utils_sw::ArgParser parser(argc, argv);
    parser.addOption("-t", "", "Type of input, half/float/double/int8/int16/int32/int64", "", true);
    parser.addOption("-i", "", "Input file path", "", true);
    parser.addOption("-o", "", "Output file path", "", true);
    parser.addOption("-w", "", "Output stream width", "", true);
    parser.addOption("-n", "", "Number of valid inputs", "", true);
    if (parser.getAs<bool>("help")) {
        parser.showUsage();
        return 0;
    }

    std::string ifile = parser.getAs<std::string>("i");
    std::string ofile = parser.getAs<std::string>("o");
    std::string type = parser.getAs<std::string>("t");
    const int width = parser.getAs<int>("w");
    const int num = parser.getAs<int>("n");

    // supported data type list
    if (type == "half") {
        generate_hex<half>(ifile, ofile, width, num);
    } else if (type == "float") {
        generate_hex<float>(ifile, ofile, width, num);
    } else if (type == "double") {
        generate_hex<double>(ifile, ofile, width, num);
    } else if (type == "int8_t") {
        generate_hex<int8_t>(ifile, ofile, width, num);
    } else if (type == "int16_t") {
        generate_hex<int16_t>(ifile, ofile, width, num);
    } else if (type == "int32_t") {
        generate_hex<int32_t>(ifile, ofile, width, num);
    } else if (type == "int64_t") {
        generate_hex<int64_t>(ifile, ofile, width, num);
    }

    return 0;
}
