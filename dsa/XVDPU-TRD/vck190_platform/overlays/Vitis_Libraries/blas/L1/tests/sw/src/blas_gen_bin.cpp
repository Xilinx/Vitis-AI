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
 *  @brief xf_blas compiler
 *
 *  $DateTime: 2019/06/18 $
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include "blas_def.hpp"

using namespace std;
using namespace xf::blas;

void to_upper(string& p_str) {
    for_each(p_str.begin(), p_str.end(), [](char& c) { c = toupper(c); });
}

void initVec(const string& p_handle, size_t p_n, vector<BLAS_dataType>& p_vec, uint8_t*& p_vecPtr) {
    if (p_handle != "NULL") {
        p_vec.resize(p_n);
        for (unsigned int i = 0; i < p_n; ++i) {
            p_vec[i] = (BLAS_dataType)i / 10;
        }
        p_vecPtr = reinterpret_cast<uint8_t*>(&(p_vec[0]));
    }
}

void initMat(const string& p_handle,
             uint16_t p_opCode,
             uint32_t p_m,
             uint32_t p_n,
             uint32_t p_kl,
             uint32_t p_ku,
             vector<BLAS_dataType>& p_mat,
             uint8_t*& p_matPtr) {
    size_t l_size = 0;
    switch (p_opCode) {
        case GBMV:
            l_size = (p_kl + p_ku + 1) * p_n;
            break;
        case SPMV:
            l_size = ((p_n / BLAS_parEntries) + 1) * (p_n / BLAS_parEntries) * BLAS_parEntries * BLAS_parEntries / 2;
            break;
        case TPMV:
            l_size = ((p_n / BLAS_parEntries) + 1) * (p_n / BLAS_parEntries) * BLAS_parEntries * BLAS_parEntries / 2;
            break;
        case SBMV:
            if (p_ku == 0) {
                l_size = (p_kl + 1) * p_n;
            } else {
                l_size = (p_ku + 1) * p_n;
            }
            break;
        case TBMV:
            if (p_ku == 0) {
                l_size = (p_kl + 1) * p_n;
            } else {
                l_size = (p_ku + 1) * p_n;
            }
            break;
        default:
            l_size = p_m * p_n;
    }
    if (p_handle != "NULL") {
        p_mat.resize(l_size);
        for (unsigned int i = 0; i < l_size; ++i) {
            p_mat[i] = (BLAS_dataType)i / 10;
        }
        p_matPtr = reinterpret_cast<uint8_t*>(&(p_mat[0]));
    }
}

void outputVec(string p_str, uint32_t p_n, BLAS_dataType* p_data) {
    if (p_data != nullptr) {
        cout << "  " << p_str << endl;
        for (unsigned int i = 0; i < p_n; ++i) {
            if ((i % ENTRIES_PER_LINE) == 0) {
                cout << endl;
            }
            cout << setw(OUTPUT_WIDTH) << p_data[i] << "  ";
        }
    }
}

void outputMat(
    string p_str, uint16_t p_opCode, uint32_t p_m, uint32_t p_n, uint32_t p_kl, uint32_t p_ku, BLAS_dataType* p_data) {
    uint32_t l_rows = 0;
    switch (p_opCode) {
        case GBMV:
            l_rows = p_kl + p_ku + 1;
            break;
        case SBMV:
            if (p_ku == 0) {
                l_rows = p_kl + 1;
            } else {
                l_rows = p_ku + 1;
            }
            break;
        case TBMV:
            if (p_ku == 0) {
                l_rows = p_kl + 1;
            } else {
                l_rows = p_ku + 1;
            }
            break;
        default:
            l_rows = p_m;
    }

    if (p_data != nullptr) {
        cout << "  " << p_str << endl;
        if ((p_opCode == SPMV) || (p_opCode == TPMV)) {
            if (p_ku == 0) {
                assert(p_kl == p_n);
                unsigned int l_blocks = p_n / BLAS_parEntries;
                unsigned int l_off = 0;
                for (unsigned int b = 0; b < l_blocks; ++b) {
                    for (unsigned int i = 0; i < BLAS_parEntries; ++i) {
                        for (unsigned int j = 0; j < (b + 1) * BLAS_parEntries; ++j) {
                            cout << setw(OUTPUT_WIDTH) << p_data[l_off + j] << "  ";
                        }
                        l_off += (b + 1) * BLAS_parEntries;
                        cout << "\n";
                    }
                    cout << "\n";
                }
            } else {
                assert(p_kl == 0);
                assert(p_ku == p_n);
                unsigned int l_blocks = p_n / BLAS_parEntries;
                unsigned int l_off = 0;
                for (unsigned int b = l_blocks; b > 0; --b) {
                    for (unsigned int i = 0; i < BLAS_parEntries; ++i) {
                        for (unsigned int j = 0; j < b * BLAS_parEntries; ++j) {
                            cout << setw(OUTPUT_WIDTH) << p_data[l_off + j] << "  ";
                        }
                        l_off += b * BLAS_parEntries;
                        cout << "\n";
                    }
                    cout << "\n";
                }
            }
        } else {
            for (unsigned int i = 0; i < l_rows; ++i) {
                for (unsigned int j = 0; j < p_n; ++j) {
                    if ((j % ENTRIES_PER_LINE) == 0) {
                        cout << endl;
                    }
                    cout << setw(OUTPUT_WIDTH) << p_data[i * p_n + j] << "  ";
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "ERROR: passed " << argc << " arguments, less than least arguments number " << 3 << ", exiting" << endl;
        cout << "  Usage:\n    blas_gen_bin.exe  <-write | -read> app.bin [op1 arg arg ...] [op2 arg arg ...] ..."
             << "    Ops:\n"
             << "      opName n alpha handleX handleY handleResX handleResY resScalar\n"
             << "    Examples:\n"
             << "      blas_gen_bin.exe -write app.bin amin 8092 0 x null null null 0\n"
             << "      blas_gen_bin.exe -read app.bin\n"
             << "\n";
        return EXIT_FAILURE;
    }

    string l_mode(argv[1]);
    bool l_write = l_mode == "-write";
    bool l_read = l_mode == "-read";
    bool l_print = l_mode == "-print";

    string l_binFile;

    if (l_read || l_write || l_print) {
        l_binFile = argv[2];

        printf("XFBLAS:  %s %s %s\n", argv[0], l_mode.c_str(), l_binFile.c_str());
    } else {
        assert(0);
    }

    // Early assert for proper instruction length setting
    assert(BLAS_instrSizeBytes * BLAS_argInstrWidth == BLAS_memWidthBytes);

    ////////////////////////  TEST PROGRAM STARTS HERE  ////////////////////////
    GenBinType l_gen;
    FindOpCode l_findOp;
    if (l_write) {
        unsigned int l_argIdx = 3;
        unsigned int l_instrCount = 0;

        vector<vector<BLAS_dataType> > l_a, l_x, l_y, l_aRes, l_xRes, l_yRes;
        l_a.resize(BLAS_maxNumInstrs);
        l_x.resize(BLAS_maxNumInstrs);
        l_y.resize(BLAS_maxNumInstrs);
        l_aRes.resize(BLAS_maxNumInstrs);
        l_xRes.resize(BLAS_maxNumInstrs);
        l_yRes.resize(BLAS_maxNumInstrs);
        unsigned int l_idx = 0;
        while (l_argIdx < argc) {
            string l_opName(argv[l_argIdx++]);
            uint32_t l_opCode;
            xfblasStatus_t l_status = l_findOp.getOpCode(l_opName, l_opCode);
            if (l_status != XFBLAS_STATUS_SUCCESS) {
                return -1;
            }
            if (l_opCode <= B1_MaxOpCode) {
                uint32_t l_n = stoi(argv[l_argIdx++]);
                BLAS_dataType l_alpha = static_cast<BLAS_dataType>(atof(argv[l_argIdx++]));
                string l_handleX(argv[l_argIdx++]);
                string l_handleY(argv[l_argIdx++]);
                string l_handleXres(argv[l_argIdx++]);
                string l_handleYres(argv[l_argIdx++]);
                BLAS_resDataType l_resScalar = static_cast<BLAS_resDataType>(atof(argv[l_argIdx++]));
                uint8_t* l_xPtr = nullptr;
                uint8_t* l_yPtr = nullptr;
                uint8_t* l_xResPtr = nullptr;
                uint8_t* l_yResPtr = nullptr;
                to_upper(l_handleX);
                to_upper(l_handleY);
                to_upper(l_handleXres);
                to_upper(l_handleYres);

                initVec(l_handleX, l_n, l_x[l_idx], l_xPtr);
                initVec(l_handleY, l_n, l_y[l_idx], l_yPtr);
                initVec(l_handleXres, l_n, l_xRes[l_idx], l_xResPtr);
                initVec(l_handleYres, l_n, l_yRes[l_idx], l_yResPtr);
                xfblasStatus_t l_status =
                    l_gen.addB1Instr(l_opName, l_n, l_alpha, l_xPtr, l_yPtr, l_xResPtr, l_yResPtr, l_resScalar);
                assert(l_status == XFBLAS_STATUS_SUCCESS);
            } else if (l_opCode <= B2_MaxOpCode) {
                uint32_t l_m = stoi(argv[l_argIdx++]);
                uint32_t l_n = stoi(argv[l_argIdx++]);
                uint32_t l_kl = stoi(argv[l_argIdx++]);
                uint32_t l_ku = stoi(argv[l_argIdx++]);
                BLAS_dataType l_alpha = static_cast<BLAS_dataType>(atof(argv[l_argIdx++]));
                BLAS_dataType l_beta = static_cast<BLAS_dataType>(atof(argv[l_argIdx++]));
                string l_handleA(argv[l_argIdx++]);
                string l_handleX(argv[l_argIdx++]);
                string l_handleY(argv[l_argIdx++]);
                string l_handleAres(argv[l_argIdx++]);
                string l_handleYres(argv[l_argIdx++]);
                to_upper(l_handleA);
                to_upper(l_handleX);
                to_upper(l_handleY);
                to_upper(l_handleAres);
                to_upper(l_handleYres);
                uint8_t* l_aPtr = nullptr;
                uint8_t* l_xPtr = nullptr;
                uint8_t* l_yPtr = nullptr;
                uint8_t* l_aResPtr = nullptr;
                uint8_t* l_yResPtr = nullptr;
                initMat(l_handleA, l_opCode, l_m, l_n, l_kl, l_ku, l_a[l_idx], l_aPtr);
                initVec(l_handleX, l_n, l_x[l_idx], l_xPtr);
                initVec(l_handleY, l_m, l_y[l_idx], l_yPtr);
                initMat(l_handleAres, l_opCode, l_m, l_n, l_kl, l_ku, l_aRes[l_idx], l_aResPtr);
                initVec(l_handleYres, l_m, l_yRes[l_idx], l_yResPtr);
                xfblasStatus_t l_status = l_gen.addB2Instr(l_opName, l_m, l_n, l_kl, l_ku, l_alpha, l_beta, l_aPtr,
                                                           l_xPtr, l_yPtr, l_aResPtr, l_yResPtr);
                assert(l_status == XFBLAS_STATUS_SUCCESS);
            }
            l_idx++;
        }
        xfblasStatus_t l_status = l_gen.write2BinFile(l_binFile);
        assert(l_status == XFBLAS_STATUS_SUCCESS);
    } else if (l_print) {
        xfblasStatus_t l_status = l_gen.readFromBinFile(l_binFile);
        assert(l_status == XFBLAS_STATUS_SUCCESS);
        l_gen.printProgram();
    } else if (l_read) {
        vector<Instr> l_instrs;
        uint32_t l_m, l_n, l_kl, l_ku;
        BLAS_dataType l_alpha, l_beta;
        BLAS_resDataType l_resGolden;
        BLAS_dataType *l_a, *l_x, *l_y, *l_aRes, *l_xRes, *l_yRes;
        BLAS_dataType l_aVal = 0;
        BLAS_dataType l_xVal = 0;
        BLAS_dataType l_yVal = 0;
        BLAS_dataType l_aResVal = 0;
        BLAS_dataType l_xResVal = 0;
        BLAS_dataType l_yResVal = 0;
        l_a = &l_aVal;
        l_x = &l_xVal;
        l_y = &l_yVal;
        l_aRes = &l_aResVal;
        l_xRes = &l_xResVal;
        l_yRes = &l_yResVal;

        xfblasStatus_t l_status = l_gen.readInstrs(l_binFile, l_instrs);
        for (unsigned int i = 0; i < l_instrs.size(); ++i) {
            Instr l_curInstr = l_instrs[i];
            if (l_curInstr.m_opClass == B1_OP_CLASS) {
                l_gen.decodeB1Instr(l_curInstr, l_n, l_alpha, l_x, l_y, l_xRes, l_yRes, l_resGolden);
                cout << "  n=" << l_n << "  alpha=" << l_alpha << "  resScalar=" << l_resGolden << endl;
                outputVec("x:", l_n, l_x);
                outputVec("y:", l_n, l_y);
                outputVec("xRes:", l_n, l_xRes);
                outputVec("yRes:", l_n, l_yRes);
            } else if (l_curInstr.m_opClass == B2_OP_CLASS) {
                l_gen.decodeB2Instr(l_curInstr, l_m, l_n, l_kl, l_ku, l_alpha, l_beta, l_a, l_x, l_y, l_aRes, l_yRes);
                cout << "m=" << l_m << "  n=" << l_n << " kl=" << l_kl << " ku=" << l_ku << "  alpha=" << l_alpha
                     << "  beta=" << l_beta << endl;
                outputMat("A:", l_curInstr.m_opCode, l_m, l_n, l_kl, l_ku, l_a);
                outputVec("x:", l_n, l_x);
                outputVec("y:", l_m, l_y);
                outputMat("ARes", l_curInstr.m_opCode, l_m, l_n, l_kl, l_ku, l_aRes);
                outputVec("yRes:", l_m, l_yRes);
            }
        }
    } else {
        assert(0); // Unknown user command
    }

    return EXIT_SUCCESS;
}
