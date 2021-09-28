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

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "blas_def.hpp"

using namespace std;
using namespace xf::blas;

extern "C" {
GenBinType* genBinNew() {
    return new GenBinType();
}

void genBinDel(GenBinType* genBin) {
    delete genBin;
}

xfblasStatus_t addB1Instr(GenBinType* genBin,
                          const char* p_opName,
                          uint32_t p_n,
                          BLAS_dataType p_alpha,
                          void* p_x,
                          void* p_y,
                          void* p_xRes,
                          void* p_yRes,
                          BLAS_resDataType p_res) {
    return genBin->addB1Instr(p_opName, p_n, p_alpha, p_x, p_y, p_xRes, p_yRes, p_res);
}
xfblasStatus_t addB2Instr(GenBinType* genBin,
                          const char* p_opName,
                          uint32_t p_m,
                          uint32_t p_n,
                          uint32_t p_kl,
                          uint32_t p_ku,
                          BLAS_dataType p_alpha,
                          BLAS_dataType p_beta,
                          void* p_a,
                          void* p_x,
                          void* p_y,
                          void* p_aRes,
                          void* p_yRes) {
    return genBin->addB2Instr(p_opName, p_m, p_n, p_kl, p_ku, p_alpha, p_beta, p_a, p_x, p_y, p_aRes, p_yRes);
}
xfblasStatus_t write2BinFile(GenBinType* genBin, const char* p_fileName) {
    return genBin->write2BinFile(p_fileName);
}
xfblasStatus_t readFromBinFile(GenBinType* genBin, const char* p_fileName) {
    return genBin->readFromBinFile(p_fileName);
}
void printProgram(GenBinType* genBin) {
    genBin->printProgram();
}
}
