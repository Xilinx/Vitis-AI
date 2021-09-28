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
 * @file storeKernel.cpp
 * @brief storeKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "storeYkernel.hpp"
void storeYkernel(unsigned int p_rows, HBM_InfTyp* p_yPtr, DatStrTyp& p_yStr) {
    SCALAR(p_rows)
    POINTER(p_yPtr, p_yPtr)
    AXIS(p_yStr)
    SCALAR(return )

    static const unsigned int t_ParEntriesMinusOne = SPARSE_parEntries - 1;
    static const unsigned int t_ParamEntries = SPARSE_hbmMemBits / 32;
    unsigned int l_rowBlocks = (p_rows + SPARSE_parEntries - 1) / SPARSE_parEntries;
    unsigned int l_rows = l_rowBlocks * SPARSE_parEntries;

    xf::blas::WideType<SPARSE_dataType, SPARSE_parEntries> l_val(0);
    for (unsigned int i = 0; i < l_rows; ++i) {
#pragma HLS PIPELINE
        SPARSE_dataType l_valDat = 0;
        if (i < p_rows) {
            ap_uint<SPARSE_dataBits> l_valBits = p_yStr.read();
            l_valDat = *reinterpret_cast<SPARSE_dataType*>(&l_valBits);
        }
        (void)l_val.unshift(l_valDat);
        if (i % SPARSE_parEntries == t_ParEntriesMinusOne) {
            p_yPtr[i / SPARSE_parEntries] = l_val;
        }
    }
}
