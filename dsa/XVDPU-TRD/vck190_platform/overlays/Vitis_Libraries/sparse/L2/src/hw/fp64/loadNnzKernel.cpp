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
 * @file loadNnzKernel.cpp
 * @brief loadNnzKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "loadNnzKernel.hpp"
extern "C" void loadNnzKernel(HBM_InfTyp* p_nnzPtr0,
                              HBM_InfTyp* p_nnzPtr1,
                              HBM_InfTyp* p_nnzPtr2,
                              HBM_InfTyp* p_nnzPtr3,
                              HBM_InfTyp* p_nnzPtr4,
                              HBM_InfTyp* p_nnzPtr5,
                              HBM_InfTyp* p_nnzPtr6,
                              HBM_InfTyp* p_nnzPtr7,
                              HBM_InfTyp* p_nnzPtr8,
                              HBM_InfTyp* p_nnzPtr9,
                              HBM_InfTyp* p_nnzPtr10,
                              HBM_InfTyp* p_nnzPtr11,
                              HBM_InfTyp* p_nnzPtr12,
                              HBM_InfTyp* p_nnzPtr13,
                              HBM_InfTyp* p_nnzPtr14,
                              HBM_InfTyp* p_nnzPtr15,

                              HBM_StrTyp& p_nnzStr0,
                              HBM_StrTyp& p_nnzStr1,
                              HBM_StrTyp& p_nnzStr2,
                              HBM_StrTyp& p_nnzStr3,
                              HBM_StrTyp& p_nnzStr4,
                              HBM_StrTyp& p_nnzStr5,
                              HBM_StrTyp& p_nnzStr6,
                              HBM_StrTyp& p_nnzStr7,
                              HBM_StrTyp& p_nnzStr8,
                              HBM_StrTyp& p_nnzStr9,
                              HBM_StrTyp& p_nnzStr10,
                              HBM_StrTyp& p_nnzStr11,
                              HBM_StrTyp& p_nnzStr12,
                              HBM_StrTyp& p_nnzStr13,
                              HBM_StrTyp& p_nnzStr14,
                              HBM_StrTyp& p_nnzStr15) {
    POINTER(p_nnzPtr0, p_nnzPtr0)
    POINTER(p_nnzPtr1, p_nnzPtr1)
    POINTER(p_nnzPtr2, p_nnzPtr2)
    POINTER(p_nnzPtr3, p_nnzPtr3)
    POINTER(p_nnzPtr4, p_nnzPtr4)
    POINTER(p_nnzPtr5, p_nnzPtr5)
    POINTER(p_nnzPtr6, p_nnzPtr6)
    POINTER(p_nnzPtr7, p_nnzPtr7)
    POINTER(p_nnzPtr8, p_nnzPtr8)
    POINTER(p_nnzPtr9, p_nnzPtr9)
    POINTER(p_nnzPtr10, p_nnzPtr10)
    POINTER(p_nnzPtr11, p_nnzPtr11)
    POINTER(p_nnzPtr12, p_nnzPtr12)
    POINTER(p_nnzPtr13, p_nnzPtr13)
    POINTER(p_nnzPtr14, p_nnzPtr14)
    POINTER(p_nnzPtr15, p_nnzPtr15)

    AXIS(p_nnzStr0)
    AXIS(p_nnzStr1)
    AXIS(p_nnzStr2)
    AXIS(p_nnzStr3)
    AXIS(p_nnzStr4)
    AXIS(p_nnzStr5)
    AXIS(p_nnzStr6)
    AXIS(p_nnzStr7)
    AXIS(p_nnzStr8)
    AXIS(p_nnzStr9)
    AXIS(p_nnzStr10)
    AXIS(p_nnzStr11)
    AXIS(p_nnzStr12)
    AXIS(p_nnzStr13)
    AXIS(p_nnzStr14)
    AXIS(p_nnzStr15)

    SCALAR(return )

    xf::sparse::loadNnz<SPARSE_dataType, SPARSE_parEntries, SPARSE_dataBits, SPARSE_hbmMemBits>(
        p_nnzPtr0, p_nnzPtr1, p_nnzPtr2, p_nnzPtr3, p_nnzPtr4, p_nnzPtr5, p_nnzPtr6, p_nnzPtr7, p_nnzPtr8, p_nnzPtr9,
        p_nnzPtr10, p_nnzPtr11, p_nnzPtr12, p_nnzPtr13, p_nnzPtr14, p_nnzPtr15, p_nnzStr0, p_nnzStr1, p_nnzStr2,
        p_nnzStr3, p_nnzStr4, p_nnzStr5, p_nnzStr6, p_nnzStr7, p_nnzStr8, p_nnzStr9, p_nnzStr10, p_nnzStr11, p_nnzStr12,
        p_nnzStr13, p_nnzStr14, p_nnzStr15);
}
