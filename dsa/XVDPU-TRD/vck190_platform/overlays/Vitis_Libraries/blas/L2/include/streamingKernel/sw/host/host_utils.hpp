/**********
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
 * **********/

#ifndef XF_BLAS_HOST_UTILS_HPP
#define XF_BLAS_HOST_UTILS_HPP

#include <stdint.h>
#include <iostream>
#include <chrono>

#define BLAS_CMP_WIDTH 11

using namespace std;

template <typename T>
bool cmpVal(float p_TolRel, float p_TolAbs, T vRef, T v, string p_Prefix, bool& p_exactMatch, unsigned int p_Verbose) {
    float l_diffAbs = abs(v - vRef);
    float l_diffRel = l_diffAbs;
    if (vRef != 0) {
        l_diffRel /= abs(vRef);
    }
    p_exactMatch = (vRef == v);
    bool l_status = p_exactMatch || (l_diffRel <= p_TolRel) || (l_diffAbs <= p_TolAbs);
    if ((p_Verbose >= 3) || ((p_Verbose >= 2) && !p_exactMatch) || ((p_Verbose >= 1) && !l_status)) {
        cout << p_Prefix << "  ValRef " << left << setw(BLAS_CMP_WIDTH) << vRef << " Val " << left
             << setw(BLAS_CMP_WIDTH) << v << "  DifRel " << left << setw(BLAS_CMP_WIDTH) << l_diffRel << " DifAbs "
             << left << setw(BLAS_CMP_WIDTH) << l_diffAbs << "  Status " << l_status << "\n";
    }
    return (l_status);
}

typedef chrono::time_point<chrono::high_resolution_clock> TimePointType;
inline void showTimeData(string p_Task, TimePointType& t1, TimePointType& t2, double* p_TimeMsOut = 0) {
    t2 = chrono::high_resolution_clock::now();
    chrono::duration<double> l_durationSec = t2 - t1;
    double l_timeMs = l_durationSec.count() * 1e3;
    if (p_TimeMsOut) {
        *p_TimeMsOut = l_timeMs;
    }
    (VERBOSE > 0) && cout << p_Task << "  " << fixed << setprecision(6) << l_timeMs << " msec\n";
}

#endif
