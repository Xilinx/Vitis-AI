
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
 * @file moveXkernel.cpp
 * @brief moveXkernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "moveXkernel.hpp"

extern "C" void moveXkernel(HBM_StrTyp& p_inStr,
                            HBM_StrTyp& p_outStr0,
                            HBM_StrTyp& p_outStr1,
                            HBM_StrTyp& p_outStr2,
                            HBM_StrTyp& p_outStr3,
                            HBM_StrTyp& p_outStr4,
                            HBM_StrTyp& p_outStr5,
                            HBM_StrTyp& p_outStr6,
                            HBM_StrTyp& p_outStr7,
                            HBM_StrTyp& p_outStr8,
                            HBM_StrTyp& p_outStr9,
                            HBM_StrTyp& p_outStr10,
                            HBM_StrTyp& p_outStr11,
                            HBM_StrTyp& p_outStr12,
                            HBM_StrTyp& p_outStr13,
                            HBM_StrTyp& p_outStr14,
                            HBM_StrTyp& p_outStr15) {
    AXIS(p_inStr);
    AXIS(p_outStr0);
    AXIS(p_outStr1);
    AXIS(p_outStr2);
    AXIS(p_outStr3);
    AXIS(p_outStr4);
    AXIS(p_outStr5);
    AXIS(p_outStr6);
    AXIS(p_outStr7);
    AXIS(p_outStr8);
    AXIS(p_outStr9);
    AXIS(p_outStr10);
    AXIS(p_outStr11);
    AXIS(p_outStr12);
    AXIS(p_outStr13);
    AXIS(p_outStr14);
    AXIS(p_outStr15);
    AP_CTRL_NONE(return )

    HBM_StrTyp l_fwdStr[SPARSE_hbmChannels];
#pragma HLS STREAM variable = l_fwdStr depth = 8
#pragma HLS DATAFLOW

    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(0, p_inStr, l_fwdStr[0], p_outStr0);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(1, l_fwdStr[0], l_fwdStr[1],
                                                                                 p_outStr1);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(2, l_fwdStr[1], l_fwdStr[2],
                                                                                 p_outStr2);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(3, l_fwdStr[2], l_fwdStr[3],
                                                                                 p_outStr3);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(4, l_fwdStr[3], l_fwdStr[4],
                                                                                 p_outStr4);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(5, l_fwdStr[4], l_fwdStr[5],
                                                                                 p_outStr5);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(6, l_fwdStr[5], l_fwdStr[6],
                                                                                 p_outStr6);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(7, l_fwdStr[6], l_fwdStr[7],
                                                                                 p_outStr7);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(8, l_fwdStr[7], l_fwdStr[8],
                                                                                 p_outStr8);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(9, l_fwdStr[8], l_fwdStr[9],
                                                                                 p_outStr9);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(10, l_fwdStr[9], l_fwdStr[10],
                                                                                 p_outStr10);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(11, l_fwdStr[10], l_fwdStr[11],
                                                                                 p_outStr11);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(12, l_fwdStr[11], l_fwdStr[12],
                                                                                 p_outStr12);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(13, l_fwdStr[12], l_fwdStr[13],
                                                                                 p_outStr13);
    xf::sparse::getFwdX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(14, l_fwdStr[13], l_fwdStr[14],
                                                                                 p_outStr14);
    xf::sparse::getX<SPARSE_indexType, SPARSE_hbmChannels, SPARSE_hbmMemBits>(l_fwdStr[14], p_outStr15);
}
