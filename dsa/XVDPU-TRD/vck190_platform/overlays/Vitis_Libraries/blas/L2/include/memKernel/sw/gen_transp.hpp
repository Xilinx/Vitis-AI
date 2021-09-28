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
#ifndef XF_BLAS_GEN_TRANSP_HPP
#define XF_BLAS_GEN_TRANSP_HPP

#include "gen_bin.hpp"
#include "matrix.hpp"

typedef xf::blas::TranspArgs TranspArgsType;

#define BLAS_transpEdgeSize (BLAS_ddrWidth * BLAS_transpBlocks)

class GenTransp {
   public:
    typedef xf::blas::DdrMatrixShape::FormatType MatFormatType;

   public:
    bool check(unsigned int p_M,
               unsigned int p_N,
               unsigned int p_LdIn,
               unsigned int p_LdOut,
               MatFormatType p_FormatA,
               MatFormatType p_FormatB) {
        bool ok = true;

        if (p_FormatB == MatFormatType::GvA) {
            if (p_LdOut != 0) {
                std::cerr << "ERROR: transp  LdOut " << p_LdOut << " is auto computed for GVA matrix, use 0 "
                          << "\n";
                ok = false;
            }
        } else {
            if (p_LdOut < p_M) {
                std::cerr << "ERROR: transp  LdOut " << p_LdOut << " is smaller than p_M " << p_M << "\n";
                ok = false;
            }
        }

        if (p_LdIn < p_N) {
            std::cerr << "ERROR: transp  LdIn " << p_LdIn << " is smaller than p_N " << p_N << "\n";
            ok = false;
        }
        if (p_M % BLAS_transpEdgeSize != 0) {
            std::cerr << "ERROR: transp  p_M " << p_M << " is not divisible by BLAS_transpEdgeSize "
                      << BLAS_transpEdgeSize << "\n";
            ok = false;
        }
        if (p_N % BLAS_transpEdgeSize != 0) {
            std::cerr << "ERROR: transp  p_N " << p_N << " is not divisible by BLAS_transpEdgeSize "
                      << BLAS_transpEdgeSize << "\n";
            ok = false;
        }
        if (p_LdIn % BLAS_ddrWidth != 0) {
            std::cerr << "ERROR: transp  p_LdIn " << p_LdIn << " is not divisible by BLAS_ddrWidth " << BLAS_ddrWidth
                      << "\n";
            ok = false;
        }
        if (p_LdOut % BLAS_ddrWidth != 0) {
            std::cerr << "ERROR: transp  p_LdOut " << p_LdOut << " is not divisible by BLAS_ddrWidth " << BLAS_ddrWidth
                      << "\n";
            ok = false;
        }
        if (p_FormatA == MatFormatType::Unknown) {
            std::cerr << "ERROR: transp  formatA " << p_FormatA << " is not valid\n";
            ok = false;
        }
        if (p_FormatB == MatFormatType::Unknown) {
            std::cerr << "ERROR: transp  formatB " << p_FormatB << " is not valid\n";
            ok = false;
        }

        return (ok);
    }
    void addInstr(ProgramType& p_Program,
                  unsigned int p_M,
                  unsigned int p_N,
                  unsigned int p_LdIn,
                  unsigned int p_LdOut,
                  MatFormatType p_FormatA,
                  MatFormatType p_FormatB,
                  std::string p_handleA,
                  std::string p_handleB,
                  bool p_WithGolden) {
        // Dimensions
        unsigned int l_outRows = p_N;
        unsigned int l_outCols = p_M;
        unsigned int l_outLd = p_LdOut;
        if (p_FormatB == MatFormatType::GvA) {
            unsigned int l_Width = BLAS_ddrWidth;
            l_outRows = p_M / l_Width;
            l_outCols = p_N * l_Width;
            l_outLd = l_outCols;
        }

        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB;
        unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdIn);
        unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, l_outRows * l_outLd);

        // Get addresses where matrices are stored
        MatType l_matA(p_M, p_N, p_LdIn, p_Program.getPageAddr(l_pageA));
        MatType l_matB(l_outRows, l_outCols, l_outLd, p_Program.getPageAddr(l_pageB));

        // Instruction
        DdrMatrixShapeType l_srcShape(l_pageA, p_M, p_N, p_LdIn, 0, p_FormatA),
            l_dstShape(l_pageB, l_outRows, l_outCols, l_outLd, 0, p_FormatB);
        TranspArgsType l_transpArgs(l_srcShape, l_dstShape);
        KargsType l_kargs;
        l_kargs.setTranspArgs(l_transpArgs);
        l_kargs.store(p_Program.addInstr(), 0);

        if (l_newAllocA) {
            l_matA.fillMod(std::numeric_limits<BLAS_dataType>::max());
        }
        if (l_newAllocB) {
            l_matB.fillMod(7);
        }

        // Calculate reference C = A * B
        if (p_WithGolden) {
            if (p_FormatB == MatFormatType::Cm) {
                l_matB.transpose(l_matA);
            } else if (p_FormatB == MatFormatType::GvA) {
                l_matB.transposeGva(l_matA, BLAS_ddrWidth * BLAS_gemvmGroups, BLAS_ddrWidth);
            } else {
                assert(false);
            }
        }
        std::cout << "Added TRANSP " << p_M << "x" << p_N << "  ";
    }
    void show(ProgramType& p_Program, TranspArgsType p_TranspArgs) {
        DdrMatrixShapeType l_src = p_TranspArgs.m_Src, l_dst = p_TranspArgs.m_Dst;
        unsigned int l_pageA = l_src.m_Offset, l_pageB = l_dst.m_Offset;
        MatType l_matA(l_src.m_Rows, l_src.m_Cols, l_src.m_Ld, p_Program.getPageAddr(l_pageA));
        MatType l_matB(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program.getPageAddr(l_pageB));
        std::cout << "\n###########  Op Transp  ###########\n"
                  << "  " << l_src << "  ->  " << l_dst << "\n"
                  << "  A  Page=" << l_pageA << "  " << l_matA << "\n"
                  << "  B  Page=" << l_pageB << "  " << l_matB << "\n";
    }
    bool compare(
        float p_TolRel, float p_TolAbs, ProgramType& p_Program0, ProgramType& p_Program1, TranspArgsType p_TranspArgs) {
        DdrMatrixShapeType l_src = p_TranspArgs.m_Src, l_dst = p_TranspArgs.m_Dst;
        unsigned int l_pageB = l_dst.m_Offset;
        MatType l_matB0(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program0.getPageAddr(l_pageB)),
            l_matB1(l_dst.m_Rows, l_dst.m_Cols, l_dst.m_Ld, p_Program1.getPageAddr(l_pageB));
        std::cout << "\n###########  Op Transp  ###########\n"
                  << "  " << l_src << "  ->  " << l_dst << "\n"
                  << "  Comparing ...\n";
        bool ok = l_matB1.cmp(p_TolRel, p_TolAbs, l_matB0);
        std::cout << "Transp B " << (ok ? "Matches" : "Differs") << "\n";
        return (ok);
    }
};

#endif