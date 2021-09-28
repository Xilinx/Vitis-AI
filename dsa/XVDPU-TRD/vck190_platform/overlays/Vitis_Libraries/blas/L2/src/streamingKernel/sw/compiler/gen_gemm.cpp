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
/**
 *  @file gen_gemm.cpp
 *  @brief main function for generating binary images for GEMM operation
 *
 *  This file is part of Vits BLAS library
 */

#include <stdio.h>
#include <string>
#include <vector>
#include "gen_gemm.hpp"
#include "host_utils.hpp"

#define BLAS_memWordBytes BLAS_ddrMemBits / 8

using namespace xf::blas;
using namespace std;

typedef chrono::time_point<std::chrono::high_resolution_clock> TimePointType;

typedef MemInstr<BLAS_instrBytes> MemInstrType;
typedef ControlInstr<BLAS_instrBytes> ControlInstrType;
typedef ResInstr<BLAS_instrBytes> ResInstrType;
typedef GemmLdStInstr<BLAS_instrBytes> GemmLdStInstrType;
typedef GemmInstr<BLAS_instrBytes> GemmInstrType;
typedef GenControl<BLAS_dataType,
                   BLAS_instrOffsetBytes,
                   BLAS_resOffsetBytes,
                   BLAS_dataOffsetBytes,
                   BLAS_maxNumInstrs,
                   BLAS_pageSizeBytes,
                   BLAS_memWordBytes,
                   BLAS_instrBytes>
    GenControlType;

typedef GenGemmLdSt<BLAS_dataType,
                    BLAS_instrOffsetBytes,
                    BLAS_resOffsetBytes,
                    BLAS_dataOffsetBytes,
                    BLAS_maxNumInstrs,
                    BLAS_pageSizeBytes,
                    BLAS_memWordBytes,
                    BLAS_instrBytes,
                    BLAS_parEntries,
                    BLAS_mParWords,
                    BLAS_kParWords,
                    BLAS_nParWords>
    GenGemmLdStType;

typedef GenGemm<BLAS_dataType,
                BLAS_instrOffsetBytes,
                BLAS_resOffsetBytes,
                BLAS_dataOffsetBytes,
                BLAS_maxNumInstrs,
                BLAS_pageSizeBytes,
                BLAS_memWordBytes,
                BLAS_instrBytes,
                BLAS_parEntries,
                BLAS_mParWords,
                BLAS_kParWords,
                BLAS_nParWords>
    GenGemmType;

typedef GenControlType::ProgramType ProgramType;

int main(int argc, char** argv) {
    if (argc < 3) {
        printf("ERROR: passed %d arguments instead of %d, exiting\n", argc, 3);
        cout << "  Usage:\n    blas_gen_bin.exe  <-write | -read> app.bin [op1 arg arg ...] [op2 arg arg ...] ... "
                "| -compare tol_rel tol_abs app_gold.bin app_out.bin\n"
             << "    Ops:\n"
             << "      gemm   M K N HandleA HandleB HandleX HandleC\n"
             << "      gemmLdSt   M K N HandleA HandleB HandleX\n"
             << "    Examples:\n"
             << "      blas_gen_bin.exe -write app.bin gemmLdSt 128 128 128 A B X\n"
             << "      blas_gen_bin.exe -write app.bin gemm 128 128 128 A B X C\n"
             << "      blas_gen_bin.exe -read app_gold.bin\n"
             << "      blas_gen_bin.exe -read app_gold.bin\n"
             << "      blas_gen_bin.exe -compare 1e-3 1e-9 app_gold.bin app_out.bin\n"
             << "\n";
        return EXIT_FAILURE;
    }

    string l_mode(argv[1]);
    bool l_write = l_mode == "-write";
    bool l_read = l_mode == "-read";
    bool l_compare = l_mode == "-compare";
    float l_TolRel = 0, l_TolAbs = 0;

    string l_binFile[2];

    if (l_read || l_write) {
        l_binFile[0] = argv[2];
        l_binFile[1] = l_binFile[0].substr(0, l_binFile[0].find_last_of(".")) + "_gold.bin";

        printf("BLAS:  %s %s %s\n", argv[0], l_mode.c_str(), l_binFile[0].c_str());
    } else if (l_compare) {
        stringstream l_TolRelS(argv[2]);
        stringstream l_TolAbsS(argv[3]);
        l_TolRelS >> l_TolRel;
        l_TolAbsS >> l_TolAbs;
        l_binFile[0] = argv[4];
        l_binFile[1] = argv[5];
        printf("BLAS:  %s %s %g %g %s %s\n", argv[0], l_mode.c_str(), l_TolRel, l_TolAbs, l_binFile[0].c_str(),
               l_binFile[1].c_str());
    } else {
        assert(0);
    }

    assert(sizeof(BLAS_dataType) * BLAS_parEntries * 8 == BLAS_ddrMemBits);

    GenControlType l_control;
    GenGemmType l_gemm;
    GenGemmLdStType l_gemmLdSt;
    if (l_write) {
        ProgramType l_p[2]; // 0 - no golden, 1 with golden

        for (unsigned int wGolden = 0; wGolden < 2; ++wGolden) {
            int l_argIdx = 3;
            unsigned int l_instrCount = 0;

            while (l_argIdx < argc) {
                string l_opName(argv[l_argIdx++]);
                TimePointType l_t1 = chrono::high_resolution_clock::now(), l_t2;
                if (l_opName == "control") {
                    bool l_isLastOp = atoi(argv[l_argIdx++]);
                    bool l_noop = atoi(argv[l_argIdx++]);
                    l_control.addInstr(l_p[wGolden], l_isLastOp, l_noop);
                } else if (l_opName == "gemm") {
                    unsigned int l_m = atoi(argv[l_argIdx++]);
                    unsigned int l_k = atoi(argv[l_argIdx++]);
                    unsigned int l_n = atoi(argv[l_argIdx++]);
                    if ((l_m == 0) && (l_k == 0) && (l_n == 0)) {
                        cout << "ERROR: dimensions error" << endl;
                        return EXIT_FAILURE;
                    } else {
                        string l_handleA(argv[l_argIdx++]);
                        string l_handleB(argv[l_argIdx++]);
                        string l_handleX(argv[l_argIdx++]);
                        string l_handleC(argv[l_argIdx++]);
                        if (!l_gemm.check(l_m, l_k, l_n)) exit(1);
                        l_gemm.addInstr(l_p[wGolden], l_m, l_k, l_n, l_handleA, l_handleB, l_handleX, l_handleC,
                                        wGolden);
                    }
                } else if (l_opName == "gemmLdSt") {
                    unsigned int l_m = atoi(argv[l_argIdx++]);
                    unsigned int l_k = atoi(argv[l_argIdx++]);
                    unsigned int l_n = atoi(argv[l_argIdx++]);
                    string l_handleA(argv[l_argIdx++]);
                    string l_handleB(argv[l_argIdx++]);
                    string l_handleX(argv[l_argIdx++]);
                    if (!l_gemmLdSt.check(l_m, l_k, l_n)) exit(1);
                    l_gemmLdSt.addInstr(l_p[wGolden], l_m, l_k, l_n, l_handleA, l_handleB, l_handleX, wGolden);
                } else {
                    cerr << "ERROR: unknow op " << l_opName << "\n";
                    exit(EXIT_FAILURE);
                }
                l_instrCount++;
                assert(l_instrCount < BLAS_maxNumInstrs - 1); // 1 is for the mandatory control instruction
                assert(l_argIdx <= argc);
                if (wGolden) {
                    showTimeData("  " + l_opName + " with golden took ", l_t1, l_t2);
                } else {
                    cout << "\n";
                }
            }
            if (l_instrCount < BLAS_maxNumInstrs) {
                l_control.addInstr(l_p[wGolden], true, false);
                cout << "\n";
            }
            l_p[wGolden].writeToBinFile(l_binFile[wGolden]);
        }

    } else if (l_read) {
        // Read file
        ProgramType l_p;
        l_p.readFromBinFile(l_binFile[0]);

        // Show cycle counts
        MemInstrType l_memInstr;
        cout << "\nINFO:   format " << right << setw(4) << "op" << right << setw(12) << "start" << right << setw(12)
             << "end" << right << setw(12) << "duration" << right << setw(14) << "ms@300MHz"
             << "\n";

        l_memInstr.loadMem(l_p.getBaseResAddr());
        uint16_t l_opCode = OpCodeType::OpControl;
        l_memInstr.loadOpCode(l_opCode);
        if (l_opCode == OpCodeType::OpResult) {
            ResInstrType l_resInstr;
            l_resInstr.load(l_memInstr);
            cout << "  DATA: cycles " << setw(12) << l_resInstr.m_startTime << setw(12) << l_resInstr.m_endTime
                 << setw(12) << l_resInstr.getDuration() << setw(14) << fixed << setprecision(6)
                 << (l_resInstr.getDuration() / 250e6 * 1e3) << "\n";
        } else {
            cout << "  ERROR: Res OpCode Error!"
                 << "\n";
            return EXIT_FAILURE;
        }
        cout << "\n";

        // Show all instructions
        unsigned int l_pc = 0;
        bool l_isLastOp = false;
        do {
            l_memInstr.loadMem(l_p.getBaseInstrAddr() + l_pc * BLAS_instrBytes);
            uint16_t l_opCode = OpCodeType::OpControl;
            l_memInstr.loadOpCode(l_opCode);
            switch (l_opCode) {
                case OpCodeType::OpControl: {
                    ControlInstrType l_controlInstr;
                    l_controlInstr.load(l_memInstr);
                    l_isLastOp = l_controlInstr.getIsLastOp();
                    bool l_noop = l_controlInstr.getNoop();
                    assert(l_isLastOp || l_noop);
                    break;
                }
                case OpCodeType::OpGemm: {
                    GemmInstrType l_gemmInstr;
                    l_gemmInstr.load(l_memInstr);
                    l_gemm.show(l_p, l_gemmInstr);
                    break;
                }
                case OpCodeType::OpGemmLdSt: {
                    GemmLdStInstrType l_gemmLdStInstr;
                    l_gemmLdStInstr.load(l_memInstr);
                    l_gemmLdSt.show(l_p, l_gemmLdStInstr);
                    break;
                }
                default: { assert(false); }
            }
            l_pc++;
        } while ((!l_isLastOp) && (l_pc < BLAS_maxNumInstrs));

    } else if (l_compare) {
        // Read files
        ProgramType l_p[2];
        l_p[0].readFromBinFile(l_binFile[0]);
        l_p[1].readFromBinFile(l_binFile[1]);

        // Compare all instructions
        MemInstrType l_memInstr1, l_memInstr2;
        unsigned int l_pc = 0;
        bool l_isLastOp = false;
        bool l_compareOk = true;
        do {
            l_memInstr1.loadMem(l_p[0].getBaseInstrAddr() + l_pc * BLAS_instrBytes);
            l_memInstr2.loadMem(l_p[1].getBaseInstrAddr() + l_pc * BLAS_instrBytes);
            uint16_t l_opCode1 = OpCodeType::OpControl;
            uint16_t l_opCode2 = OpCodeType::OpControl;
            l_memInstr1.loadOpCode(l_opCode1);
            l_memInstr2.loadOpCode(l_opCode2);
            if (l_opCode1 == OpCodeType::OpResult) {
                break;
            }
            assert(l_opCode1 == l_opCode1);
            switch (l_opCode1) {
                case OpCodeType::OpControl: {
                    ControlInstrType l_controlInstr;
                    l_controlInstr.load(l_memInstr1);
                    l_isLastOp = l_controlInstr.getIsLastOp();
                    break;
                }
                case OpCodeType::OpGemm: {
                    GemmInstrType l_gemmInstr;
                    l_gemmInstr.load(l_memInstr1);
                    bool l_opOk = l_gemm.compare(l_TolRel, l_TolAbs, l_p[0], l_p[1], l_gemmInstr);
                    l_compareOk = l_compareOk && l_opOk;
                    break;
                }
                case OpCodeType::OpGemmLdSt: {
                    GemmLdStInstrType l_gemmLdStInstr;
                    l_gemmLdStInstr.load(l_memInstr1);
                    bool l_opOk = l_gemmLdSt.compare(l_TolRel, l_TolAbs, l_p[0], l_p[1], l_gemmLdStInstr);
                    l_compareOk = l_compareOk && l_opOk;
                    break;
                }
                default: { assert(false); }
            }
            l_pc++;
        } while ((!l_isLastOp) && (l_pc < BLAS_maxNumInstrs));

        // Exit status from compare
        if (!l_compareOk) {
            return EXIT_FAILURE;
        }
    } else {
        assert(0); // Unknown user command
    }

    return EXIT_SUCCESS;
}
