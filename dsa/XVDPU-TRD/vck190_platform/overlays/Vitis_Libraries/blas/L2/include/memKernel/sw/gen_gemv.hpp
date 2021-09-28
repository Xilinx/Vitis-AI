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
#ifndef XF_BLAS_GEN_GEMV_H
#define XF_BLAS_GEN_GEMV_H

#include "gen_bin.hpp"
#include "matrix.hpp"

typedef GemvType::GemvArgsType GemvArgsType;

template <typename T>
void gemv_ref(DenseMat<T>& p_A, DenseMat<T>& p_B, DenseMat<T>& p_C) {
    T l_val = 0;
    assert(p_A.rows() == p_C.rows());
    assert(p_A.cols() == p_B.rows());
    assert(p_B.cols() == p_C.cols());
    for (unsigned int row = 0; row < p_C.rows(); ++row) {
        for (unsigned int col = 0; col < p_C.cols(); ++col) {
            T l_val = 0;
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
                l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
            }
            // std::cout << "    DEBUG multiply setting row=" << row << " col=" << col << std::endl;
            p_C.getVal(row, col) = l_val;
        }
    }
}

class GenGemv {
   public:
    bool check(unsigned int p_M, unsigned int p_K, unsigned int p_LdA) {
        bool ok = true;

        const unsigned int l_mEdge = BLAS_gemvmGroups * BLAS_ddrWidth;
        const unsigned int l_kEdge = BLAS_transpBlocks * BLAS_ddrWidth;

        if (p_M % l_mEdge != 0) {
            std::cerr << "ERROR: gemv  M dimension " << p_M << " must be multiple of " << l_mEdge << "\n";
            ok = false;
        }
        if (p_K % (l_kEdge) != 0) {
            std::cerr << "ERROR: gemv  K dimension " << p_K << " must be multiple of " << l_kEdge << "\n";
            ok = false;
        }
        return (ok);
    }
    //__attribute__ ((noinline))
    void addInstr(ProgramType& p_Program,
                  unsigned int p_M,
                  unsigned int p_K,
                  unsigned int p_Lda,
                  std::string p_handleA,
                  std::string p_handleB,
                  std::string p_handleC,
                  bool p_WithGolden) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC;
        unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_Lda);
        unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * 1);
        unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * 1);

        // Get addresses where matrices are stored
        MatType l_matA(p_M, p_K, p_Lda, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, 1, 1, p_Program.getPageAddr(l_pageB));
        MatType l_matC(p_M, 1, 1, p_Program.getPageAddr(l_pageC));

        // Instruction
        GemvArgsType l_gemvArgs(l_pageA, l_pageB, l_pageC, p_M, p_K, p_Lda);
        KargsType l_kargs;
        l_kargs.setGemvArgs(l_gemvArgs);
        l_kargs.store(p_Program.addInstr(), 0);

        if (l_newAllocA) {
            l_matA.fillMod(std::numeric_limits<BLAS_dataType>::max());
        }
        if (l_newAllocB) {
            l_matB.fillMod(7);
        }

        // Calculate reference C = A * B
        if (p_WithGolden) {
            gemv_ref<BLAS_dataType>(l_matA, l_matB, l_matC);
        }
        std::cout << "Added GEMV " << p_M << "x" << p_K << "  ";
    }

    void show(ProgramType& p_Program, GemvArgsType p_GemvArgs) {
        unsigned int l_M = p_GemvArgs.m_M, l_K = p_GemvArgs.m_K, l_Lda = p_GemvArgs.m_Lda;
        MatType l_matA(l_M, l_K, l_Lda, p_Program.getPageAddr(p_GemvArgs.m_Aoffset));
        MatType l_matB(l_K, 1, 1, p_Program.getPageAddr(p_GemvArgs.m_Boffset));
        MatType l_matC(l_M, 1, 1, p_Program.getPageAddr(p_GemvArgs.m_Coffset));
        std::cout << "\n###########  Op Gemv  ###########\n"
                  << "  C = A * B  " << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
                  << "  A " << l_matA << "\n"
                  << "  B " << l_matB << "\n"
                  << "  C " << l_matC << "\n";
    }
    bool compare(
        float p_TolRel, float p_TolAbs, ProgramType& p_Program0, ProgramType& p_Program1, GemvArgsType p_GemvArgs) {
        unsigned int l_M = p_GemvArgs.m_M, l_K = p_GemvArgs.m_K;
        MatType l_matC0(l_M, 1, 1, p_Program0.getPageAddr(p_GemvArgs.m_Coffset)),
            l_matC1(l_M, 1, 1, p_Program1.getPageAddr(p_GemvArgs.m_Coffset));
        std::cout << "\n###########  Op Gemv  ###########\n"
                  << "  C = A * B  " << l_M << "x" << 1 << " = " << l_M << "x" << l_K << " * " << l_K << "x" << 1
                  << "  Comparing ...\n";
        bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
        std::cout << "Gemv C " << (ok ? "Matches" : "Differs") << "\n";
        return (ok);
    }
};

#endif