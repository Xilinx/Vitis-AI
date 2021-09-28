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
#include <iostream>
#include <string>
#include <vector>
#include <cmath>
#include "blas_def.hpp"
#include "utils.hpp"
#include "uut_top.hpp"

using namespace xf::blas;
int main(int argc, char** argv) {
    if (argc < 2) {
        std::cout << "ERROR: passed %d arguments instead of %d, exiting" << argc << 2 << std::endl;
        std::cout << " Usage:" << std::endl;
        std::cout << "    test.exe testfile.bin" << std::endl;
        std::cout << " Example Usage:" << std::endl;
        std::cout << "    test.exe ./data/test_amax.bin" << std::endl;
        return EXIT_FAILURE;
    }
    std::string l_binFile(argv[1]);
    GenBinType l_gen;

    vector<Instr> l_instrs;
    bool l_return = true;
    xfblasStatus_t l_status = l_gen.readInstrs(l_binFile, l_instrs);
    for (unsigned int i = 0; i < l_instrs.size(); ++i) {
        Instr l_curInstr = l_instrs[i];
#if BLAS_L1
        if (l_curInstr.m_opClass == B1_OP_CLASS) {
            uint32_t l_n;
            BLAS_dataType l_alpha;
            BLAS_resDataType l_resGolden = 0, l_res = 0;
            BLAS_dataType *l_x = nullptr, *l_y = nullptr;
            BLAS_dataType *l_xRes = nullptr, *l_yRes = nullptr;
            BLAS_dataType *l_xResRef = nullptr, *l_yResRef = nullptr;
            l_gen.decodeB1Instr(l_curInstr, l_n, l_alpha, l_x, l_y, l_xResRef, l_yResRef, l_resGolden);
            l_xRes = new BLAS_dataType[l_n];
            l_yRes = new BLAS_dataType[l_n];
            for (int l = 0; l < l_n; l++) {
                l_xRes[l] = 0;
                l_yRes[l] = 0;
            }
            uut_top(l_n, l_alpha, l_x, l_y, l_xRes, l_yRes, l_res);
            l_return = l_return && compare(l_n, l_xRes, l_xResRef);
            l_return = l_return && compare(l_n, l_yRes, l_yResRef);
            l_return = l_return && compare(l_res, l_resGolden);
            delete[] l_xRes;
            delete[] l_yRes;
            if (!l_return) break;
        }
#elif BLAS_L2
        if (l_curInstr.m_opClass == B2_OP_CLASS) {
            uint32_t l_m, l_n, l_ku, l_kl;
            BLAS_dataType l_alpha, l_beta;
            BLAS_dataType *l_a = nullptr, *l_x = nullptr, *l_y = nullptr;
            BLAS_dataType *l_aRes = nullptr, *l_yRes = nullptr;
            BLAS_dataType *l_aResRef = nullptr, *l_yResRef = nullptr;
            l_gen.decodeB2Instr(l_curInstr, l_m, l_n, l_kl, l_ku, l_alpha, l_beta, l_a, l_x, l_y, l_aResRef, l_yResRef);
            l_aRes = new BLAS_dataType[l_m * l_n];
            l_yRes = new BLAS_dataType[l_m];
            for (int p = 0; p < l_m; p++) {
                for (int l = 0; l < l_n; l++) l_aRes[p * l_n + l] = 0;
                l_yRes[p] = 0;
            }
            uut_top(l_m, l_n, l_kl, l_ku, l_alpha, l_beta, l_a, l_x, l_y, l_aRes, l_yRes);
            l_return = l_return && compare(l_n * l_m, l_aRes, l_aResRef);
            l_return = l_return && compare(l_m, l_yRes, l_yResRef);
            delete[] l_aRes;
            delete[] l_yRes;
            if (!l_return) break;
        }
#else
        return EXIT_FAILURE;
#endif
    }
    // compute
    if (l_return)
        return 0;
    else
        return -1;
};
