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

typedef xf::blas::GemmArgs GemmArgsType;
typedef DenseMat<BLAS_XdataType> XMatType;

template <typename T>
void gemm_ref(
    DenseMat<T>& p_A, DenseMat<T>& p_B, DenseMat<T>& p_C, DenseMat<BLAS_XdataType>& p_X, int32_t p_postScale) {
    assert(p_A.rows() == p_C.rows());
    assert(p_A.cols() == p_B.rows());
    assert(p_B.cols() == p_C.cols());
    assert(p_X.rows() == p_C.rows());
    assert(p_X.cols() == p_C.cols());
    for (unsigned int row = 0; row < p_C.rows(); ++row) {
        for (unsigned int col = 0; col < p_C.cols(); ++col) {
#if BLAS_keepMacBits
            int64_t l_val = 0;
#else
            T l_val = 0;
#endif
            for (unsigned int k = 0; k < p_A.cols(); ++k) {
                l_val += p_A.getVal(row, k) * p_B.getVal(k, col);
            }
            l_val += p_X.getVal(row, col);
#if BLAS_keepMacBits
            unsigned int l_psShift = p_postScale & 0x00ff;
            int64_t l_psVal = p_postScale >> 8;
            l_val = l_val * l_psVal;
            l_val = l_val >> l_psShift;
#endif
            T l_entry = (T)(l_val);
            p_C.getVal(row, col) = l_entry;
        }
    }
}

class GenGemm {
   public:
    bool checkDim(std::string p_VarName, unsigned int p_Val, unsigned int p_Mod, unsigned int p_Min) {
        bool l_ok = true;
        if (p_Val % p_Mod != 0) {
            std::cerr << "ERROR: " << p_VarName << " " << p_Val << " must be multiple of " << p_Mod << "\n";
            l_ok = false;
        }
        if (p_Val < p_Min) {
            std::cerr << "ERROR: " << p_VarName << " " << p_Val << " must be at least " << p_Min << "\n";
            l_ok = false;
        }
        return (l_ok);
    }

    bool check(unsigned int p_M,
               unsigned int p_K,
               unsigned int p_N,
               unsigned int p_LdA,
               unsigned int p_LdB,
               unsigned int p_LdC,
               unsigned int p_LdX) {
        bool ok = true;
        const unsigned int l_Edge = BLAS_ddrWidth;
        const unsigned int l_mMin = BLAS_gemmMBlocks * BLAS_ddrWidth;
        const unsigned int l_kMin =
            BLAS_gemmKBlocks * BLAS_ddrWidth; // BLAS_gemmKBlocks must >= 2, due to kernel Cout control to save area
        const unsigned int l_nMin = BLAS_gemmNBlocks * BLAS_ddrWidth;

        ok = checkDim("M", p_M, l_mMin, l_mMin) && checkDim("K", p_K, l_kMin, l_kMin) &&
             checkDim("N", p_N, l_nMin, l_nMin) && checkDim("LdA", p_LdA, l_Edge, p_K) &&
             checkDim("LdB", p_LdB, l_Edge, p_N) && checkDim("LdC", p_LdC, l_Edge, p_N) &&
             checkDim("LdX", p_LdX, l_Edge, p_N);
        return (ok);
    }

    void addInstr(ProgramType& p_Program,
                  unsigned int p_M,
                  unsigned int p_K,
                  unsigned int p_N,
                  unsigned int p_LdA,
                  unsigned int p_LdB,
                  unsigned int p_LdC,
                  unsigned int p_LdX,
                  int32_t p_postScale,
                  std::string p_handleA,
                  std::string p_handleB,
                  std::string p_handleC,
                  std::string p_handleX,
                  bool p_WithGolden) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
        unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdA);
        unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * p_LdB);
        unsigned int l_pageX = p_Program.allocPages(p_handleX, l_newAllocX,
                                                    p_M * p_LdX * (sizeof(BLAS_XdataType) / sizeof(BLAS_dataType)));
        unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * p_LdC);

        // Get addresses where matrices are stored
        MatType l_matA(p_M, p_K, p_LdA, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, p_N, p_LdB, p_Program.getPageAddr(l_pageB));
        XMatType l_matX(p_M, p_N, p_LdX, (BLAS_XdataType*)p_Program.getPageAddr(l_pageX));
        MatType l_matC(p_M, p_N, p_LdC, p_Program.getPageAddr(l_pageC));

        // Instruction
        GemmArgsType l_gemmArgs(l_pageA, l_pageB, l_pageC, l_pageX, p_M, p_K, p_N, p_LdA, p_LdB, p_LdC, p_LdX,
                                p_postScale);
        KargsType l_kargs;
        l_kargs.setGemmArgs(l_gemmArgs);
        l_kargs.store(p_Program.addInstr(), 0);

        if (l_newAllocA) {
            l_matA.fillMod(67, 1);
        }
        if (l_newAllocB) {
            l_matB.fillMod(129, 65);
        }
        if (l_newAllocX) {
            l_matX.fillMod(1, 0);
        }

        // Calculate reference C = postScale(A * B + X)
        if (p_WithGolden) {
            // l_matC.multiplyAddScale(l_matA, l_matB, l_matX, p_postScale);
            gemm_ref<BLAS_dataType>(l_matA, l_matB, l_matC, l_matX, p_postScale);
        }
        std::cout << "Added GEMM " << p_M << "x" << p_K << "x" << p_N << "  ";
    }

    void addInstrFromBin(ProgramType& p_Program,
                         unsigned int p_M,
                         unsigned int p_K,
                         unsigned int p_N,
                         unsigned int p_LdA,
                         unsigned int p_LdB,
                         unsigned int p_LdC,
                         unsigned int p_LdX,
                         int32_t p_postScale,
                         std::string p_handleA,
                         std::string p_handleB,
                         std::string p_handleC,
                         std::string p_handleX,
                         std::string p_MatAName,
                         std::string p_MatBName,
                         std::string p_MatXName,
                         bool p_WithGolden) {
        // Allocate all pages before getting any address
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;
        unsigned int l_pageA = p_Program.allocPages(p_handleA, l_newAllocA, p_M * p_LdA);
        unsigned int l_pageB = p_Program.allocPages(p_handleB, l_newAllocB, p_K * p_LdB);
        unsigned int l_pageX = p_Program.allocPages(p_handleX, l_newAllocX,
                                                    p_M * p_LdX * (sizeof(BLAS_XdataType) / sizeof(BLAS_dataType)));
        unsigned int l_pageC = p_Program.allocPages(p_handleC, l_newAllocC, p_M * p_LdC);

        // Get addresses where matrices are stored
        MatType l_matA(p_M, p_K, p_LdA, p_Program.getPageAddr(l_pageA));
        MatType l_matB(p_K, p_N, p_LdB, p_Program.getPageAddr(l_pageB));
        XMatType l_matX(p_M, p_N, p_LdX, (BLAS_XdataType*)p_Program.getPageAddr(l_pageX));
        MatType l_matC(p_M, p_N, p_LdC, p_Program.getPageAddr(l_pageC));

        // Instruction
        GemmArgsType l_gemmArgs(l_pageA, l_pageB, l_pageC, l_pageX, p_M, p_K, p_N, p_LdA, p_LdB, p_LdC, p_LdX,
                                p_postScale);
        KargsType l_kargs;
        l_kargs.setGemmArgs(l_gemmArgs);
        l_kargs.store(p_Program.addInstr(), 0);

        if (l_newAllocA) {
            l_matA.fillModFromBin(p_MatAName);
        }
        if (l_newAllocB) {
            l_matB.fillModFromBin(p_MatBName);
        }
        if (l_newAllocX) {
            l_matX.fillModFromBin(p_MatXName);
        }

        // Calculate reference C = postScale(A * B + X)
        if (p_WithGolden) {
            // l_matC.multiplyAddScale(l_matA, l_matB, l_matX, p_postScale);
            gemm_ref<BLAS_dataType>(l_matA, l_matB, l_matC, l_matX, p_postScale);
        }
        std::cout << "Added GEMM " << p_M << "x" << p_K << "x" << p_N << "  ";
    }

    void addInstrFromFiles(unsigned int l_instrIndex,
                           ProgramType& p_Program,
                           std::string p_InsName,
                           std::string p_MatAName,
                           std::string p_MatBName,
                           std::string p_MatXName,
                           bool p_WithGolden) {
        unsigned int l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX;
        unsigned int l_pageA, l_pageB, l_pageC, l_pageX;
        int32_t l_postScale;
        std::string l_handleA, l_handleB, l_handleC, l_handleX;
        bool l_newAllocA, l_newAllocB, l_newAllocC, l_newAllocX;

        std::cout << "INFO: loading input matrices files " << p_InsName << " " << p_MatAName << " " << p_MatBName << " "
                  << p_MatXName << "\n";
        std::ifstream l_fs_ins(p_InsName.c_str(), std::ios_base::in | std::ios_base::binary);
        std::ifstream l_fs_matA(p_MatAName.c_str(), std::ios_base::in | std::ios_base::binary);
        std::ifstream l_fs_matB(p_MatBName.c_str(), std::ios_base::in | std::ios_base::binary);
        std::ifstream l_fs_matX(p_MatXName.c_str(), std::ios_base::in | std::ios_base::binary);

        bool l_good = l_fs_ins.good();

        bool ok = true;
        const unsigned int l_Edge = BLAS_ddrWidth;

        if (l_good) {
            unsigned int l_index;
            while (l_fs_ins.peek() == '#') l_fs_ins.ignore(2048, '\n');
            l_fs_ins >> l_index;
            while (l_index < l_instrIndex) {
                l_fs_ins.ignore(2048, '\n');
                l_fs_ins >> l_index;
            }
            std::cout << "INFO instr number : " << l_index << "\n";
            l_fs_ins >> l_postScale;
            std::cout << "INFO " << l_postScale << "\n";
            // read A dimensions
            l_fs_ins >> l_handleA >> l_M >> l_K >> l_LdA;
            std::cout << "INFO " << l_handleA << " " << l_M << " " << l_K << " " << l_LdA << "\n";
            // check A dimention
            ok = checkDim("M", l_M, l_Edge, 1) && checkDim("K", l_K, l_Edge, 2 * l_Edge) &&
                 checkDim("LdA", l_LdA, l_Edge, l_K);
            if (!ok) {
                return;
            }
            // allocate host memory and initialize it with data from the file
            l_pageA = p_Program.allocPages(l_handleA, l_newAllocA, l_M * l_LdA);
            MatType l_matA(l_M, l_K, l_LdA, p_Program.getPageAddr(l_pageA));
            if (l_fs_matA.good()) {
                while (l_fs_matA.peek() == '#') l_fs_matA.ignore(2048, '\n');
                if (l_newAllocA) {
                    l_matA.fillFromFile(l_fs_matA);
                }
            }
            // read matrix B dimensions
            unsigned int l_bK;
            l_fs_ins >> l_handleB >> l_bK >> l_N >> l_LdB;
            std::cout << "INFO " << l_handleB << " " << l_bK << " " << l_N << " " << l_LdB << "\n";
            assert(l_bK == l_K);
            ok = checkDim("N", l_N, l_Edge, 1) && checkDim("LdB", l_LdB, l_Edge, l_N);
            if (!ok) {
                l_fs_ins.close();
                return;
            }

            l_pageB = p_Program.allocPages(l_handleB, l_newAllocB, l_K * l_LdB);
            MatType l_matB(l_K, l_N, l_LdB, p_Program.getPageAddr(l_pageB));
            if (l_fs_matB.good()) {
                while (l_fs_matB.peek() == '#') l_fs_matB.ignore(2048, '\n');
                if (l_newAllocB) {
                    l_matB.fillFromFile(l_fs_matB);
                }
            }
            // read matrix X dimensions
            unsigned int l_xM, l_xN;
            l_fs_ins >> l_handleX >> l_xM >> l_xN >> l_LdX;
            std::cout << "INFO " << l_handleX << " " << l_xM << " " << l_xN << " " << l_LdX << "\n";
            assert(l_xM == l_M);
            assert(l_xN == l_N);
            ok = checkDim("LdX", l_LdX, l_Edge, l_N);
            if (!ok) {
                l_fs_ins.close();
                return;
            }

            l_pageX = p_Program.allocPages(l_handleX, l_newAllocX,
                                           l_M * l_N * (sizeof(BLAS_XdataType) / sizeof(BLAS_dataType)));
            XMatType l_matX(l_M, l_N, l_LdX, (BLAS_XdataType*)p_Program.getPageAddr(l_pageX));
            if (l_fs_matX.good()) {
                while (l_fs_matX.peek() == '#') l_fs_matX.ignore(2048, '\n');
                if (l_newAllocX) {
                    l_matX.fillFromFile(l_fs_matX);
                }
            }
            // read matrix C dimensions
            unsigned int l_cM, l_cN;
            l_fs_ins >> l_handleC >> l_cM >> l_cN >> l_LdC;
            std::cout << "INFO " << l_handleC << " " << l_cM << " " << l_cN << " " << l_LdC << "\n";
            assert(l_cM == l_M);
            assert(l_cN == l_N);
            ok = checkDim("LdC", l_LdC, l_Edge, l_N);
            if (!ok) {
                l_fs_ins.close();
                return;
            }
            l_pageC = p_Program.allocPages(l_handleC, l_newAllocC, l_M * l_LdC);
            MatType l_matC(l_M, l_N, l_LdC, p_Program.getPageAddr(l_pageC));

            if (p_WithGolden) {
                gemm_ref<BLAS_dataType>(l_matA, l_matB, l_matC, l_matX, l_postScale);
            }

            GemmArgsType l_gemmArgs(l_pageA, l_pageB, l_pageC, l_pageX, l_M, l_K, l_N, l_LdA, l_LdB, l_LdC, l_LdX,
                                    l_postScale);
            KargsType l_kargs;
            l_kargs.setGemmArgs(l_gemmArgs);
            l_kargs.store(p_Program.addInstr(), 0);

            std::cout << "Added GEMM" << l_M << "x" << l_K << "x" << l_N << " postScale: " << l_postScale << "  ";
            l_fs_ins.close();
            l_fs_matA.close();
            l_fs_matB.close();
            l_fs_matX.close();

        } else {
            std::cout << "ERROR: bad filename"
                      << "\n";
        }
    }

    void show(ProgramType& p_Program, GemmArgsType p_GemmArgs) {
        unsigned int l_M = p_GemmArgs.m_M, l_K = p_GemmArgs.m_K, l_N = p_GemmArgs.m_N, l_ldA = p_GemmArgs.m_Lda,
                     l_ldB = p_GemmArgs.m_Ldb, l_ldC = p_GemmArgs.m_Ldc, l_ldX = p_GemmArgs.m_Ldx;
        int32_t l_postScale = p_GemmArgs.m_postScale;
        MatType l_matA(l_M, l_K, l_ldA, p_Program.getPageAddr(p_GemmArgs.m_Aoffset));
        MatType l_matB(l_K, l_N, l_ldB, p_Program.getPageAddr(p_GemmArgs.m_Boffset));
        XMatType l_matX(l_M, l_N, l_ldX, (BLAS_XdataType*)p_Program.getPageAddr(p_GemmArgs.m_Xoffset));
        MatType l_matC(l_M, l_N, l_ldC, p_Program.getPageAddr(p_GemmArgs.m_Coffset));
        std::cout << "\n###########  Op Gemm  ###########\n"
                  << "  C = postScale(A * B + X) " << l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K
                  << "x" << l_N << " + " << l_M << " x " << l_N << "\n"
                  << " postScale " << l_postScale << "\n"
                  << "  A " << l_matA << "\n"
                  << "  B " << l_matB << "\n"
                  << "  X    " << l_matX << "\n"
                  << "  C " << l_matC << "\n";
    }

    bool compare(
        float p_TolRel, float p_TolAbs, ProgramType& p_Program0, ProgramType& p_Program1, GemmArgsType p_GemmArgs) {
        unsigned int l_M = p_GemmArgs.m_M, l_K = p_GemmArgs.m_K, l_N = p_GemmArgs.m_N, l_ldC = p_GemmArgs.m_Ldc;
        MatType l_matC0(l_M, l_N, l_ldC, p_Program0.getPageAddr(p_GemmArgs.m_Coffset)),
            l_matC1(l_M, l_N, l_ldC, p_Program1.getPageAddr(p_GemmArgs.m_Coffset));
        std::cout << "\n###########  Op Gemm  ###########\n"
                  << "  C = postScale(A * B + X) " << l_M << "x" << l_N << " = " << l_M << "x" << l_K << " * " << l_K
                  << "x" << l_N << " + " << l_M << " x " << l_N << "\n"
                  << "  Comparing ...\n";
        bool ok = l_matC1.cmp(p_TolRel, p_TolAbs, l_matC0);
        std::cout << "Gemm C " << (ok ? "Matches" : "Differs") << "\n";
        return (ok);
    }
};

#endif