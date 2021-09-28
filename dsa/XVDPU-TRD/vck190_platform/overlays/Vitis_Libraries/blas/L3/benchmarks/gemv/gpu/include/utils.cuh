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

#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include <exception>

// provide same functionality as numpy.isclose
template <typename T>
bool isClose(float p_tolRel, float p_tolAbs, T p_vRef, T p_v, bool& p_exactMatch) {
    float l_diffAbs = abs(p_v - p_vRef);
    p_exactMatch = (p_vRef == p_v);
    bool l_status = (l_diffAbs <= (p_tolAbs + p_tolRel * abs(p_vRef)));
    return (l_status);
}
template <typename T>
bool compare(T x, T ref) {
    return x == ref;
}

template <>
bool compare<double>(double x, double ref) {
    bool l_exactMatch;
    return isClose<float>(1e-4, 3e-6, x, ref, l_exactMatch);
}
template <>
bool compare<float>(float x, float ref) {
    bool l_exactMatch;
    return isClose<float>(1e-3, 3e-6, x, ref, l_exactMatch);
}

template <typename T>
bool compare(unsigned int n, T* x, T* ref) {
    bool l_ret = true;
    try {
        if (ref == nullptr) {
            if (x == nullptr) return true;
            for (unsigned int i = 0; i < n; i++) l_ret = l_ret && compare(x[i], (T)0);
        } else {
            for (unsigned int i = 0; i < n; i++) {
                l_ret = l_ret && compare(x[i], ref[i]);
            }
        }
    } catch (std::exception& e) {
        std::cout << "Exception happend: " << e.what() << std::endl;
        return false;
    }
    return l_ret;
}

template <typename T>
bool compare(unsigned int n, T* x, T* ref, int& err) {
    bool l_ret = true;
    try {
        if (ref == nullptr) {
            if (x == nullptr) return true;
            for (unsigned int i = 0; i < n; i++) {
                if (!compare(x[i], (T)0)) {
                    err++;
                    l_ret = false;
                }
            }
        } else {
            for (unsigned int i = 0; i < n; i++) {
                if (!compare(x[i], ref[i])) {
                    l_ret = false;
                    err++;
                }
            }
        }
    } catch (std::exception& e) {
        std::cout << "Exception happend: " << e.what() << std::endl;
        return false;
    }
    return l_ret;
}
#endif
