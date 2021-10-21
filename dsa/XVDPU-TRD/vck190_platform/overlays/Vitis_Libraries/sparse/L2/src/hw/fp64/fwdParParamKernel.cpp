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
 * @file fwdParParamKernel.cpp
 * @brief fwdParParamKernel definition.
 *
 * This file is part of Vitis SPARSE Library.
 */

#include "kernel.hpp"

void readWideParam(WideParamStrTyp& p_inParamStr, ParamStrTyp p_outParamStr[SPARSE_hbmChannels]) {
    WideType<uint32_t, SPARSE_hbmChannels> l_val;
#pragma HLS PARTITION variable = l_val complete dim = 1
    for (unsigned int j = 0; j < 2; ++j) {
#pragma HLS PIPELINE
        l_val = p_inParamStr.read();
        for (unsigned int k = 0; k < SPARSE_hbmChannels; ++k) {
            p_outParamStr[k].write(l_val[k]);
        }
    }
}

void fwdParam(ParamStrTyp& p_inParamStr, ParamStrTyp& p_outParamStr) {
    for (unsigned int j = 0; j < 2; ++j) {
#pragma HLS PIPELINE
        p_outParamStr.write(p_inParamStr.read());
    }
}

extern "C" void fwdParParamKernel(WideParamStrTyp& p_inParamStr,
                                  ParamStrTyp& p_paramStr0,
                                  ParamStrTyp& p_paramStr1,
                                  ParamStrTyp& p_paramStr2,
                                  ParamStrTyp& p_paramStr3,
                                  ParamStrTyp& p_paramStr4,
                                  ParamStrTyp& p_paramStr5,
                                  ParamStrTyp& p_paramStr6,
                                  ParamStrTyp& p_paramStr7,
                                  ParamStrTyp& p_paramStr8,
                                  ParamStrTyp& p_paramStr9,
                                  ParamStrTyp& p_paramStr10,
                                  ParamStrTyp& p_paramStr11,
                                  ParamStrTyp& p_paramStr12,
                                  ParamStrTyp& p_paramStr13,
                                  ParamStrTyp& p_paramStr14,
                                  ParamStrTyp& p_paramStr15) {
    AXIS(p_inParamStr)
    AXIS(p_paramStr0)
    AXIS(p_paramStr1)
    AXIS(p_paramStr2)
    AXIS(p_paramStr3)
    AXIS(p_paramStr4)
    AXIS(p_paramStr5)
    AXIS(p_paramStr6)
    AXIS(p_paramStr7)
    AXIS(p_paramStr8)
    AXIS(p_paramStr9)
    AXIS(p_paramStr10)
    AXIS(p_paramStr11)
    AXIS(p_paramStr12)
    AXIS(p_paramStr13)
    AXIS(p_paramStr14)
    AXIS(p_paramStr15)
    AP_CTRL_NONE(return )

    ParamStrTyp l_paramStr[SPARSE_hbmChannels];
#pragma HLS STREAM variable = l_paramStr depth = 16

#pragma HLS DATAFLOW
    readWideParam(p_inParamStr, l_paramStr);
    fwdParam(l_paramStr[0], p_paramStr0);
    fwdParam(l_paramStr[1], p_paramStr1);
    fwdParam(l_paramStr[2], p_paramStr2);
    fwdParam(l_paramStr[3], p_paramStr3);
    fwdParam(l_paramStr[4], p_paramStr4);
    fwdParam(l_paramStr[5], p_paramStr5);
    fwdParam(l_paramStr[6], p_paramStr6);
    fwdParam(l_paramStr[7], p_paramStr7);
    fwdParam(l_paramStr[8], p_paramStr8);
    fwdParam(l_paramStr[9], p_paramStr9);
    fwdParam(l_paramStr[10], p_paramStr10);
    fwdParam(l_paramStr[11], p_paramStr11);
    fwdParam(l_paramStr[12], p_paramStr12);
    fwdParam(l_paramStr[13], p_paramStr13);
    fwdParam(l_paramStr[14], p_paramStr14);
    fwdParam(l_paramStr[15], p_paramStr15);
}
