
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
 * @file krnl_loadIdx.cpp
 * @brief krnl_loadIdx definition.
 *
 * This file is part of Vitis HPC Library.
 */

#include "krnl_loadAidx.hpp"

extern "C" void krnl_loadAidx(CG_interface* p_idxPtr,
                              CG_tkStrType& p_tkInStr,
                              CG_idxStrType& p_idxStr0,
                              CG_idxStrType& p_idxStr1,
                              CG_idxStrType& p_idxStr2,
                              CG_idxStrType& p_idxStr3,
                              CG_idxStrType& p_idxStr4,
                              CG_idxStrType& p_idxStr5,
                              CG_idxStrType& p_idxStr6,
                              CG_idxStrType& p_idxStr7,
                              CG_idxStrType& p_idxStr8,
                              CG_idxStrType& p_idxStr9,
                              CG_idxStrType& p_idxStr10,
                              CG_idxStrType& p_idxStr11,
                              CG_idxStrType& p_idxStr12,
                              CG_idxStrType& p_idxStr13,
                              CG_idxStrType& p_idxStr14,
                              CG_idxStrType& p_idxStr15) {
    POINTER(p_idxPtr, p_idxPtr)
    AXIS(p_tkInStr)
    AXIS(p_idxStr0)
    AXIS(p_idxStr1)
    AXIS(p_idxStr2)
    AXIS(p_idxStr3)
    AXIS(p_idxStr4)
    AXIS(p_idxStr5)
    AXIS(p_idxStr6)
    AXIS(p_idxStr7)
    AXIS(p_idxStr8)
    AXIS(p_idxStr9)
    AXIS(p_idxStr10)
    AXIS(p_idxStr11)
    AXIS(p_idxStr12)
    AXIS(p_idxStr13)
    AXIS(p_idxStr14)
    AXIS(p_idxStr15)
    SCALAR(return )

    xf::hpc::cg::Token<CG_dataType> l_token;
    xf::hpc::StreamInstr<sizeof(l_token)> l_cs;
    l_token.read_decode(p_tkInStr, l_cs);
    while (!l_token.getExit()) {
        xf::sparse::loadIdx<SPARSE_indexType, CG_numChannels, SPARSE_indexBits, SPARSE_hbmMemBits>(
            p_idxPtr, p_idxStr0, p_idxStr1, p_idxStr2, p_idxStr3, p_idxStr4, p_idxStr5, p_idxStr6, p_idxStr7, p_idxStr8,
            p_idxStr9, p_idxStr10, p_idxStr11, p_idxStr12, p_idxStr13, p_idxStr14, p_idxStr15);
        l_token.read_decode(p_tkInStr, l_cs);
    }
}
