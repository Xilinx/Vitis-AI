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
//================================== End Lic =================================================
// File Name :  suf.hpp
#ifndef __SUF__H__
#define __SUF__H__
#include <complex>
template <int dim1, typename T_T>
int compare1DArrays(T_T arr1[dim1], T_T arr2[dim1]) {
    int mis_matches = 0;
    for (int t = 0; t < dim1; t++) {
        if (arr1[t] != arr2[t]) mis_matches++;
    }

    return mis_matches;
}

template <int dim1, int dim2, typename T_T>
int compare2DArrays(T_T arr1[dim1][dim2], T_T arr2[dim1][dim2]) {
    int mis_matches = 0;
    for (int t = 0; t < dim1; t++) {
        for (int r = 0; r < dim2; r++) {
            if (arr1[t][r] == arr2[t][r]) {
            } else
                mis_matches++;
        }
    }

    return mis_matches;
}

template <int dim1, int dim2, typename T_T>
int compare2DArrays(std::complex<T_T> arr1[dim1][dim2], std::complex<T_T> arr2[dim1][dim2]) {
    int mis_matches = 0;
    for (int t = 0; t < dim1; t++) {
        for (int r = 0; r < dim2; r++) {
            if (arr1[t][r].real() == arr2[t][r].real() && arr1[t][r].imag() == arr2[t][r].imag()) {
            } else
                mis_matches++;
        }
    }

    return mis_matches;
}

template <int dim1, int dim2>
int compare2DArraysXcomplex(std::complex<int> arr1[dim1][dim2], std::complex<int> arr2[dim1][dim2]) {
    int mis_matches = 0;
    for (int t = 0; t < dim1; t++) {
        for (int r = 0; r < dim2; r++) {
            if (arr1[t][r].real() == arr2[t][r].real() && arr1[t][r].imag() == arr2[t][r].imag()) {
            } else
                mis_matches++;
        }
    }

    return mis_matches;
}
#endif
