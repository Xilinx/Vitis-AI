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
#ifndef XF_BLAS_GEN_GEMM_H
#define XF_BLAS_GEN_GEMM_H

#include "gen_bin.hpp"
#include "matrix.hpp"

using namespace std;

namespace xf {
namespace blas {

template <typename t_DataType>
void gemm_ref(DenseMat<t_DataType>& p_a,
              DenseMat<t_DataType>& p_b,
              DenseMat<t_DataType>& p_x,
              DenseMat<t_DataType>& p_c) {
    assert(p_a.rows() == p_c.rows());
    assert(p_a.cols() == p_b.rows());
    assert(p_b.cols() == p_c.cols());
    assert(p_x.rows() == p_c.rows());
    assert(p_x.cols() == p_c.cols());
    for (unsigned int row = 0; row < p_c.rows(); ++row) {
        for (unsigned int col = 0; col < p_c.cols(); ++col) {
            t_DataType l_val = 0;
            for (unsigned int k = 0; k < p_a.cols(); ++k) {
                l_val += p_a.getVal(row, k) * p_b.getVal(k, col);
            }
            l_val += p_x.getVal(row, col);
            t_DataType l_entry = (t_DataType)(l_val);
            p_c.getVal(row, col) = l_entry;
        }
    }
}

template <typename t_DataType,
          unsigned int t_InstrOffsetBytes,
          unsigned int t_ResOffsetBytes,
          unsigned int t_DataOffsetBytes,
          unsigned int t_MaxNumInstrs,
          unsigned int t_PageSizeBytes,
          unsigned int t_MemWordBytes,
          unsigned int t_InstrBytes,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords>
class GenGemm {
   public:
    typedef Program<t_DataType,
                    t_InstrOffsetBytes,
                    t_ResOffsetBytes,
                    t_DataOffsetBytes,
                    t_MaxNumInstrs,
                    t_PageSizeBytes,
                    t_MemWordBytes,
                    t_InstrBytes>
        ProgramType;
    typedef MemInstr<t_InstrBytes> MemInstrType;
    typedef ResInstr<t_InstrBytes> ResInstrType;
    typedef GemmInstr<t_InstrBytes> GemmInstrType;
    typedef DenseMat<t_DataType> MatType;

   public:
    bool checkDim(string p_VarName, unsigned int p_Val, unsigned int p_mod, unsigned int p_min) {
        bool l_ok = true;
        if (p_Val % p_mod != 0) {
            cerr << "ERROR: " << p_VarName << " " << p_Val << " must be multiple of " << p_mod << "\n";
            l_ok = false;
        }
        if (p_Val < p_min) {
            cerr << "ERROR: " << p_VarName << " " << p_Val << " must be at least " << p_min << "\n";
            l_ok = false;
        }
        return (l_ok);
    }

    bool check(unsigned int p_m, unsigned int p_k, unsigned int p_n) {
        bool ok = true;
        const unsigned int l_mMin = t_MparWords * t_ParEntries;
        const unsigned int l_kMin =
            t_KparWords * t_ParEntries; // t_KparWords must >= 2, due to kernel Cout control to save area
        const unsigned int l_nMin = t_NparWords * t_ParEntries;

        ok = checkDim("M", p_m, l_mMin, l_mMin) && checkDim("K", p_k, l_kMin, l_kMin) &&
             checkDim("N", p_n, l_nMin, l_nMin);
        return (ok);
    }

    void genInstr(ProgramType& p_program,
                  unsigned int p_m,
                  unsigned int p_k,
                  unsigned int p_n,
                  string p_handleA,
                  string p_handleB,
                  string p_handleX,
                  string p_handleC,
                  bool& p_newAllocA,
                  bool& p_newAllocB,
                  bool& p_newAllocX,
                  bool& p_newAllocC,
                  unsigned int& p_pageA,
                  unsigned int& p_pageB,
                  unsigned int& p_pageX,
                  unsigned int& p_pageC) {
        // Allocate all pages before getting any address
        p_pageA = p_program.allocPages(p_handleA, p_newAllocA, p_m * p_k);
        p_pageB = p_program.allocPages(p_handleB, p_newAllocB, p_k * p_n);
        p_pageX = p_program.allocPages(p_handleX, p_newAllocX, p_m * p_n);
        p_pageC = p_program.allocPages(p_handleC, p_newAllocC, p_m * p_n);

        // Instruction
        GemmInstrType l_gemmInstr(p_pageA * t_PageSizeBytes, p_pageB * t_PageSizeBytes, p_pageX * t_PageSizeBytes,
                                  p_pageC * t_PageSizeBytes, p_m, p_k, p_n);
        MemInstrType l_memInstr;
        l_gemmInstr.store(l_memInstr);
        l_memInstr.storeMem(p_program.addInstr());

        ResInstrType l_resInstr(0, 0);
        l_resInstr.store(l_memInstr);
        l_memInstr.storeMem(p_program.getBaseResAddr());
    }

    void genDataRnd(ProgramType& p_program,
                    unsigned int p_m,
                    unsigned int p_k,
                    unsigned int p_n,
                    bool p_newAllocA,
                    bool p_newAllocB,
                    bool p_newAllocX,
                    bool p_newAllocC,
                    unsigned int p_pageA,
                    unsigned int p_pageB,
                    unsigned int p_pageX,
                    unsigned int p_pageC,
                    bool p_withGolden) {
        MatType l_matA(p_m, p_k, p_k, p_program.getPageAddr(p_pageA));
        MatType l_matB(p_k, p_n, p_n, p_program.getPageAddr(p_pageB));
        MatType l_matX(p_m, p_n, p_n, p_program.getPageAddr(p_pageX));
        MatType l_matC(p_m, p_n, p_n, p_program.getPageAddr(p_pageC));

        if (p_newAllocA) {
            l_matA.fillMod(67, 1);
        }
        if (p_newAllocB) {
            l_matB.fillMod(129, 65);
        }
        if (p_newAllocX) {
            l_matX.fillMod(1, 0);
        }

        // Calculate reference C = postScale(A * B + X)
        if (p_withGolden) {
            // l_matC.multiplyAddScale(l_matA, l_matB, l_matX, p_postScale);
            gemm_ref<t_DataType>(l_matA, l_matB, l_matX, l_matC);
        }
    }

    void addInstr(ProgramType& p_program,
                  unsigned int p_m,
                  unsigned int p_k,
                  unsigned int p_n,
                  string p_handleA,
                  string p_handleB,
                  string p_handleX,
                  string p_handleC,
                  bool p_withGolden) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
        unsigned int l_pageA;
        unsigned int l_pageB;
        unsigned int l_pageX;
        unsigned int l_pageC;

        genInstr(p_program, p_m, p_k, p_n, p_handleA, p_handleB, p_handleX, p_handleC, l_newAllocA, l_newAllocB,
                 l_newAllocX, l_newAllocC, l_pageA, l_pageB, l_pageX, l_pageC);
        genDataRnd(p_program, p_m, p_k, p_n, l_newAllocA, l_newAllocB, l_newAllocX, l_newAllocC, l_pageA, l_pageB,
                   l_pageX, l_pageC, p_withGolden);
        cout << "Added GEMM " << p_m << "x" << p_k << "x" << p_n << "  ";
    }

    void show(ProgramType& p_program, GemmInstrType p_gemmInstr) {
        unsigned int l_m = p_gemmInstr.m_m, l_k = p_gemmInstr.m_k, l_n = p_gemmInstr.m_n;
        MatType l_matA(l_m, l_k, l_k, p_program.getPageAddr(p_gemmInstr.m_aOffset / t_PageSizeBytes));
        MatType l_matB(l_k, l_n, l_n, p_program.getPageAddr(p_gemmInstr.m_bOffset / t_PageSizeBytes));
        MatType l_matX(l_m, l_n, l_n, p_program.getPageAddr(p_gemmInstr.m_xOffset / t_PageSizeBytes));
        MatType l_matC(l_m, l_n, l_n, p_program.getPageAddr(p_gemmInstr.m_cOffset / t_PageSizeBytes));
        cout << "\n###########  Op Gemm  ###########\n"
             << "  C = A * B + X " << l_m << "x" << l_n << " = " << l_m << "x" << l_k << " * " << l_k << "x" << l_n
             << " + " << l_m << " x " << l_n << "\n"
             << "  A " << l_matA << "\n"
             << "  B " << l_matB << "\n"
             << "  X    " << l_matX << "\n"
             << "  C " << l_matC << "\n";
    }

    bool compare(
        float p_TolRel, float p_TolAbs, ProgramType& p_program0, ProgramType& p_program1, GemmInstrType p_gemmInstr) {
        unsigned int l_m = p_gemmInstr.m_m, l_k = p_gemmInstr.m_k, l_n = p_gemmInstr.m_n;
        MatType l_matC0(l_m, l_n, l_n, p_program0.getPageAddr(p_gemmInstr.m_cOffset / t_PageSizeBytes)),
            l_matC1(l_m, l_n, l_n, p_program1.getPageAddr(p_gemmInstr.m_cOffset / t_PageSizeBytes));
        cout << "\n###########  Op Gemm  ###########\n"
             << "  C = A * B + X " << l_m << "x" << l_n << " = " << l_m << "x" << l_k << " * " << l_k << "x" << l_n
             << " + " << l_m << " x " << l_n << "\n"
             << "  Comparing ...\n";
        bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
        cout << "Gemm C " << (ok ? "Matches" : "Differs") << "\n";
        return (ok);
    }
};

template <typename t_DataType,
          unsigned int t_InstrOffsetBytes,
          unsigned int t_ResOffsetBytes,
          unsigned int t_DataOffsetBytes,
          unsigned int t_MaxNumInstrs,
          unsigned int t_PageSizeBytes,
          unsigned int t_MemWordBytes,
          unsigned int t_InstrBytes,
          unsigned int t_ParEntries,
          unsigned int t_MparWords,
          unsigned int t_KparWords,
          unsigned int t_NparWords>
class GenGemmLdSt {
   public:
    typedef Program<t_DataType,
                    t_InstrOffsetBytes,
                    t_ResOffsetBytes,
                    t_DataOffsetBytes,
                    t_MaxNumInstrs,
                    t_PageSizeBytes,
                    t_MemWordBytes,
                    t_InstrBytes>
        ProgramType;
    typedef MemInstr<t_InstrBytes> MemInstrType;
    typedef ResInstr<t_InstrBytes> ResInstrType;
    typedef GemmLdStInstr<t_InstrBytes> GemmLdStInstrType;
    typedef DenseMat<t_DataType> MatType;

   public:
    bool checkDim(string p_VarName, unsigned int p_Val, unsigned int p_mod, unsigned int p_min) {
        bool l_ok = true;
        if (p_Val % p_mod != 0) {
            cerr << "ERROR: " << p_VarName << " " << p_Val << " must be multiple of " << p_mod << "\n";
            l_ok = false;
        }
        if (p_Val < p_min) {
            cerr << "ERROR: " << p_VarName << " " << p_Val << " must be at least " << p_min << "\n";
            l_ok = false;
        }
        return (l_ok);
    }

    bool check(unsigned int p_m, unsigned int p_k, unsigned int p_n) {
        bool ok = true;
        const unsigned int l_mMin = t_MparWords * t_ParEntries;
        const unsigned int l_kMin =
            t_KparWords * t_ParEntries; // t_KparWords must >= 2, due to kernel Cout control to save area
        const unsigned int l_nMin = t_NparWords * t_ParEntries;

        ok = checkDim("M", p_m, l_mMin, l_mMin) && checkDim("K", p_k, l_kMin, l_kMin) &&
             checkDim("N", p_n, l_nMin, l_nMin);
        return (ok);
    }

    void genInstr(ProgramType& p_program,
                  unsigned int p_m,
                  unsigned int p_k,
                  unsigned int p_n,
                  unsigned int p_bNblocks,
                  unsigned int p_aMblocks,
                  string p_handleA,
                  string p_handleB,
                  string p_handleX,
                  bool& p_newAllocA,
                  bool& p_newAllocB,
                  bool& p_newAllocX,
                  unsigned int& p_pageA,
                  unsigned int& p_pageB,
                  unsigned int& p_pageX,
                  unsigned int& p_pageWrA,
                  unsigned int& p_pageWrB,
                  unsigned int& p_pageWrX) {
        // Allocate all pages before getting any address
        bool l_newAllocWrA, l_newAllocWrB, l_newAllocWrX;
        string l_handleWrA = p_handleA + "Wr";
        string l_handleWrB = p_handleB + "Wr";
        string l_handleWrX = p_handleX + "Wr";

        p_pageA = p_program.allocPages(p_handleA, p_newAllocA, p_m * p_k);
        p_pageB = p_program.allocPages(p_handleB, p_newAllocB, p_k * p_n);
        p_pageX = p_program.allocPages(p_handleX, p_newAllocX, p_m * p_n);

        p_pageWrA = p_program.allocPages(l_handleWrA, l_newAllocWrA, p_m * p_k * p_bNblocks);
        p_pageWrB = p_program.allocPages(l_handleWrB, l_newAllocWrB, p_k * p_n * p_aMblocks);
        p_pageWrX = p_program.allocPages(l_handleWrX, l_newAllocWrX, p_m * p_n);

        // Instruction
        GemmLdStInstrType l_gemmLdStInstr(p_pageA * t_PageSizeBytes, p_pageB * t_PageSizeBytes,
                                          p_pageX * t_PageSizeBytes, p_pageWrA * t_PageSizeBytes,
                                          p_pageWrB * t_PageSizeBytes, p_pageWrX * t_PageSizeBytes, p_m, p_k, p_n);
        MemInstrType l_memInstr;
        l_gemmLdStInstr.store(l_memInstr);
        l_memInstr.storeMem(p_program.addInstr());

        // Result place holder
        ResInstrType l_resInstr(0, 0);
        l_resInstr.store(l_memInstr);
        l_memInstr.storeMem(p_program.getBaseResAddr());
    }

    void genDataRnd(ProgramType& p_program,
                    unsigned int p_m,
                    unsigned int p_k,
                    unsigned int p_n,
                    unsigned int p_bNblocks,
                    unsigned int p_aMblocks,
                    bool p_newAllocA,
                    bool p_newAllocB,
                    bool p_newAllocX,
                    unsigned int p_pageA,
                    unsigned int p_pageB,
                    unsigned int p_pageX,
                    unsigned int p_pageWrA,
                    unsigned int p_pageWrB,
                    unsigned int p_pageWrX,
                    bool p_withGolden) {
        MatType l_matA(p_m, p_k, p_k, p_program.getPageAddr(p_pageA));
        MatType l_matB(p_k, p_n, p_n, p_program.getPageAddr(p_pageB));
        MatType l_matX(p_m, p_n, p_n, p_program.getPageAddr(p_pageX));

        MatType l_matWrA(p_m * p_bNblocks, p_k, p_k, p_program.getPageAddr(p_pageWrA));
        MatType l_matWrB(p_k * p_aMblocks, p_n, p_n, p_program.getPageAddr(p_pageWrB));
        MatType l_matWrX(p_m, p_n, p_n, p_program.getPageAddr(p_pageWrX));

        if (p_newAllocA) {
            l_matA.fillMod(67, 1);
        }
        if (p_newAllocB) {
            l_matB.fillMod(129, 65);
        }
        if (p_newAllocX) {
            l_matX.fillMod(1, 0);
        }

        // Calculate reference C = postScale(A * B + X)
        if (p_withGolden) {
            for (unsigned int i = 0; i < p_bNblocks; ++i) {
                for (unsigned int r = 0; r < p_m; ++r) {
                    for (unsigned int c = 0; c < p_k; ++c) {
                        l_matWrA.getVal(i * p_m + r, c) = l_matA.getVal(r, c);
                    }
                }
            }
            for (unsigned int i = 0; i < p_aMblocks; ++i) {
                for (unsigned int r = 0; r < p_k; ++r) {
                    for (unsigned int c = 0; c < p_n; ++c) {
                        l_matWrB.getVal(i * p_k + r, c) = l_matB.getVal(r, c);
                    }
                }
            }
            for (unsigned int r = 0; r < p_m; ++r) {
                for (unsigned int c = 0; c < p_n; ++c) {
                    l_matWrX.getVal(r, c) = l_matX.getVal(r, c);
                }
            }
        }
    }

    void addInstr(ProgramType& p_program,
                  unsigned int p_m,
                  unsigned int p_k,
                  unsigned int p_n,
                  string p_handleA,
                  string p_handleB,
                  string p_handleX,
                  bool p_withGolden) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocX;
        unsigned int l_pageA, l_pageWrA;
        unsigned int l_pageB, l_pageWrB;
        unsigned int l_pageX, l_pageWrX;

        unsigned int l_bNblocks = p_n / (t_ParEntries * t_NparWords);
        unsigned int l_aMblocks = p_m / (t_ParEntries * t_MparWords);

        genInstr(p_program, p_m, p_k, p_n, l_bNblocks, l_aMblocks, p_handleA, p_handleB, p_handleX, l_newAllocA,
                 l_newAllocB, l_newAllocX, l_pageA, l_pageB, l_pageX, l_pageWrA, l_pageWrB, l_pageWrX);
        genDataRnd(p_program, p_m, p_k, p_n, l_bNblocks, l_aMblocks, l_newAllocA, l_newAllocB, l_newAllocX, l_pageA,
                   l_pageB, l_pageX, l_pageWrA, l_pageWrB, l_pageWrX, p_withGolden);
        cout << "Added GemmLdSt " << p_m << " " << p_k << " " << p_n << "  ";
    }

    void show(ProgramType& p_program, GemmLdStInstrType p_gemmLdStInstr) {
        unsigned int l_m = p_gemmLdStInstr.m_m, l_k = p_gemmLdStInstr.m_k, l_n = p_gemmLdStInstr.m_n;
        unsigned int l_bNblocks = l_n / (t_ParEntries * t_NparWords);
        unsigned int l_aMblocks = l_m / (t_ParEntries * t_MparWords);
        MatType l_matA(l_m, l_k, l_k, p_program.getPageAddr(p_gemmLdStInstr.m_aOffset / t_PageSizeBytes));
        MatType l_matB(l_k, l_n, l_n, p_program.getPageAddr(p_gemmLdStInstr.m_bOffset / t_PageSizeBytes));
        MatType l_matX(l_m, l_n, l_n, p_program.getPageAddr(p_gemmLdStInstr.m_xOffset / t_PageSizeBytes));
        MatType l_matWrA(l_m * l_bNblocks, l_k, l_k,
                         p_program.getPageAddr(p_gemmLdStInstr.m_aWrOffset / t_PageSizeBytes));
        MatType l_matWrB(l_k * l_aMblocks, l_n, l_n,
                         p_program.getPageAddr(p_gemmLdStInstr.m_bWrOffset / t_PageSizeBytes));
        MatType l_matWrX(l_m, l_n, l_n, p_program.getPageAddr(p_gemmLdStInstr.m_xWrOffset / t_PageSizeBytes));
        cout << "\n###########  Op GemmLdSt  ###########\n"
             << "  M, K, N:  " << l_m << "," << l_k << "," << l_n << "\n"
             << "  A " << l_matA << "\n"
             << "  B " << l_matB << "\n"
             << "  X    " << l_matX << "\n"
             << "  A_res " << l_matWrA << "\n"
             << "  B_res " << l_matWrB << "\n"
             << "  X_res    " << l_matWrX << "\n";
    }

    bool compare(float p_TolRel,
                 float p_TolAbs,
                 ProgramType& p_program0,
                 ProgramType& p_program1,
                 GemmLdStInstrType p_gemmLdStInstr) {
        unsigned int l_m = p_gemmLdStInstr.m_m, l_k = p_gemmLdStInstr.m_k, l_n = p_gemmLdStInstr.m_n;
        unsigned int l_bNblocks = l_n / (t_ParEntries * t_NparWords);
        unsigned int l_aMblocks = l_m / (t_ParEntries * t_MparWords);
        cout << "\n###########  Op GemmLdSt  ###########\n"
             << "  M, K, N:  " << l_m << "," << l_k << "," << l_n << "\n";
        MatType l_matA0(l_m * l_bNblocks, l_k, l_k,
                        p_program0.getPageAddr(p_gemmLdStInstr.m_aWrOffset / t_PageSizeBytes));
        MatType l_matA1(l_m * l_bNblocks, l_k, l_k,
                        p_program1.getPageAddr(p_gemmLdStInstr.m_aWrOffset / t_PageSizeBytes));
        cout << "  Comparing A ...\n";
        bool l_res = true;
        bool ok = l_matA1.cmp(p_TolRel, p_TolAbs, l_matA0);
        cout << "GemmLdSt A " << (ok ? "Matches" : "Differs") << "\n";
        l_res = l_res && ok;

        MatType l_matB0(l_k * l_aMblocks, l_n, l_n,
                        p_program0.getPageAddr(p_gemmLdStInstr.m_bWrOffset / t_PageSizeBytes));
        MatType l_matB1(l_k * l_aMblocks, l_n, l_n,
                        p_program1.getPageAddr(p_gemmLdStInstr.m_bWrOffset / t_PageSizeBytes));
        cout << "  Comparing B ...\n";
        ok = l_matB1.cmp(p_TolRel, p_TolAbs, l_matB0);
        cout << "GemmLdSt B " << (ok ? "Matches" : "Differs") << "\n";
        l_res = l_res && ok;

        MatType l_matX0(l_m, l_n, l_n, p_program0.getPageAddr(p_gemmLdStInstr.m_xWrOffset / t_PageSizeBytes));
        MatType l_matX1(l_m, l_n, l_n, p_program1.getPageAddr(p_gemmLdStInstr.m_xWrOffset / t_PageSizeBytes));
        cout << "  Comparing X ...\n";
        ok = l_matX1.cmp(p_TolRel, p_TolAbs, l_matX0);
        cout << "GemmLdSt X " << (ok ? "Matches" : "Differs") << "\n";
        l_res = l_res && ok;
        return (l_res);
    }
};
}
}
#endif
