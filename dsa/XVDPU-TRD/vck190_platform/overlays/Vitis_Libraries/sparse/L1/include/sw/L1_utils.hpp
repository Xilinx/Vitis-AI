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

/**
 * @file L1_utils.hpp
 * @brief header file for common functions used in L1 test code or L2/L3 host code.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_L1_UTILS_HPP
#define XF_SPARSE_L1_UTILS_HPP

#include <iostream>
#include <fstream>
#include "ap_int.h"
#include "hls_stream.h"
#include <cstdlib>
#include <cmath>
#include <vector>
using namespace std;

namespace xf {
namespace sparse {

template <typename T>
bool isClose(float p_tolRel, float p_tolAbs, T p_vRef, T p_v, bool& p_exactMatch) {
    float l_diffAbs = abs(p_v - p_vRef);
    p_exactMatch = (p_vRef == p_v);
    bool l_status = true;
    if (l_diffAbs < (p_tolAbs + p_tolRel * abs(p_vRef))) {
        l_status = true;
    } else {
        l_status = false;
    }
    return (l_status || p_exactMatch);
}
template <typename T>
bool compare(T x, T ref) {
    return x == ref;
}

template <>
bool compare<double>(double x, double ref) {
    bool l_exactMatch;
    return isClose<float>(1e-3, 3e-6, x, ref, l_exactMatch);
}
template <>
bool compare<float>(float x, float ref) {
    bool l_exactMatch;
    return isClose<float>(1e-3, 3e-6, x, ref, l_exactMatch);
}

template <typename T>
bool compare(T x, T ref, bool& p_exactMatch) {
    return isClose<float>(1e-3, 3e-6, x, ref);
}

template <typename t_DataType>
bool compareVec(vector<t_DataType> p_x, vector<t_DataType> p_ref, bool& p_exactMatch) {
    bool l_exactMatch = true;
    if (p_x.size() != p_ref.size()) {
        cout << "ERROR: vector has different size. One has " << p_x.size() << " the other hass " << p_ref.size()
             << " entries." << endl;
    }
    unsigned int l_size = p_x.size();
    bool l_res = true;
    unsigned int l_errs = 0;
    for (unsigned int i = 0; i < l_size; ++i) {
        bool l_sim = isClose<t_DataType>(1e-3, 3e-6, p_x[i], p_ref[i], l_exactMatch);
        if ((!l_sim) && (!l_exactMatch)) {
            cout << "ERROR: mismatch at " << i << " val = " << p_x[i] << " ref_val = " << p_ref[i] << endl;
            l_errs++;
        }
        l_res = l_res && l_sim;
        p_exactMatch = p_exactMatch && l_exactMatch;
    }
    if (!l_res) {
        cout << "ERROR: total " << l_errs << " mismatches." << endl;
    }
    return l_res;
}

template <typename t_DataType>
unsigned int compareVec(vector<t_DataType> p_x, vector<t_DataType> p_ref) {
    bool l_exactMatch = true;
    if (p_x.size() != p_ref.size()) {
        cout << "ERROR: vector has different size. One has " << p_x.size() << " the other hass " << p_ref.size()
             << " entries." << endl;
    }
    unsigned int l_size = p_x.size();
    unsigned int l_errs = 0;
    for (unsigned int i = 0; i < l_size; ++i) {
        bool l_sim = isClose<t_DataType>(1e-3, 3e-6, p_x[i], p_ref[i], l_exactMatch);
        if ((!l_sim) && (!l_exactMatch)) {
            cout << "ERROR: mismatch at " << i << " val = " << p_x[i] << " ref_val = " << p_ref[i] << endl;
            l_errs++;
        }
    }
    return l_errs;
}

template <typename Func>
void openInputFile(const std::string& p_filename, const Func& func) {
    std::ifstream l_if(p_filename.c_str(), std::ios::binary);
    if (!l_if.is_open()) {
        std::cout << "ERROR: Open " << p_filename << std::endl;
    }
    func(l_if);
    l_if.close();
}

template <int t_Bits>
size_t loadStream(std::ifstream& p_if, hls::stream<ap_uint<t_Bits> >& p_stream) {
    size_t l_size;
    p_if.read(reinterpret_cast<char*>(&l_size), sizeof(size_t));

    for (size_t i = 0; i < l_size; i++) {
        ap_uint<t_Bits> l_dat;
        p_if.read(reinterpret_cast<char*>(&l_dat), (t_Bits + 7) / 8);
        p_stream.write(l_dat);
    }

    return l_size;
}

template <typename T>
std::vector<T> copyStreamToVector(hls::stream<T>& p_stream) {
    std::vector<T> l_vector;
    hls::stream<T> l_stream;
    while (!p_stream.empty()) {
        auto l_dat = p_stream.read();
        l_vector.push_back(l_dat);
        l_stream.write(l_dat);
    }
    while (!l_stream.empty()) {
        p_stream.write(l_stream.read());
    }
    return l_vector;
}
} // end namespace sparse
} // end namespace xf
#endif
