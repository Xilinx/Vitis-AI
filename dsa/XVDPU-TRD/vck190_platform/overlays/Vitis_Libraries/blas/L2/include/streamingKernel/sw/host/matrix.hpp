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
#ifndef XF_BLAS_MATRIX_HPP
#define XF_BLAS_MATRIX_HPP

#include <iostream>
#include <fstream>

class MtxRow {
   private:
    unsigned int m_Row, m_Col;
    double m_Val;

   public:
    MtxRow() : m_Row(0), m_Col(0), m_Val(0) {}
    MtxRow(double p_Val, unsigned int p_Row, unsigned int p_Col) : m_Row(p_Row), m_Col(p_Col), m_Val(p_Val) {}
    unsigned int getRow() { return m_Row; }
    unsigned int getCol() { return m_Col; }
    double getVal() { return m_Val; }
    void scan(std::istream& p_Is) {
        p_Is >> m_Row >> m_Col >> m_Val;
        if ((m_Row <= 0) || (m_Col <= 0)) {
            std::cerr << "  Error: invalid MTX file line row=" << m_Row << " col=" << m_Col << " val=" << m_Val << "\n";
            assert(0);
        }
        // Indices start from 1 in MTX; 0 locally
        m_Row--;
        m_Col--;
    }
    friend bool operator<(MtxRow& a, MtxRow& b) {
        if (a.getRow() < b.getRow()) {
            return (true);
        } else if (a.getRow() == b.getRow()) {
            if (a.getCol() < b.getCol()) {
                return (true);
            }
        }
        return (false);
    }
    struct compareCol {
        bool operator()(MtxRow& a, MtxRow& b) {
            if (a.getCol() < b.getCol()) {
                return (true);
            } else if (a.getCol() == b.getCol()) {
                if (a.getRow() < b.getRow()) {
                    return (true);
                }
            }
            return (false);
        }
    };

    void print(std::ostream& os) {
        os << std::setw(BLAS_FLOAT_WIDTH) << int(getRow()) << " " << std::setw(BLAS_FLOAT_WIDTH) << int(getCol())
           << "    " << std::setw(BLAS_FLOAT_WIDTH) << getVal();
    }
};
inline std::ostream& operator<<(std::ostream& os, MtxRow& p_Val) {
    p_Val.print(os);
    return (os);
}

// Regula Matrix descriptor with data itself stored in caller's space
template <typename T>
class DenseMat {
   private:
    unsigned int m_Rows, m_Cols, m_Ld;
    T* m_Addr;

   public:
    DenseMat() {}
    DenseMat(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T* p_Addr)
        : m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_Addr(p_Addr) {}
    DenseMat& operator=(const DenseMat& p_Src) {
        assert(p_Src.rows() == rows());
        assert(p_Src.cols() == cols());
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < m_Cols; ++col) {
                m_Addr[row][col] = p_Src.getVal(row, col);
            }
        }
        return *this;
    }
    inline T& getVal(unsigned int p_Row, unsigned int p_Col) { return m_Addr[p_Row * ld() + p_Col]; }
    inline unsigned int rows() { return m_Rows; }
    inline unsigned int cols() { return m_Cols; }
    inline unsigned int ld() { return m_Ld; }
    void init(unsigned int p_Rows, unsigned int p_Cols, unsigned int p_Ld, T* p_Addr) {
        m_Rows = p_Rows;
        m_Cols = p_Cols;
        m_Ld = p_Ld;
        m_Addr = p_Addr;
    }
    void fillMod(T p_Max, T p_First = 0, bool p_small = false) {
        T l_val = p_First;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val;
                l_val++;
                l_val %= p_Max;
            }
        }
    }

    void fillModFromFile(std::string l_fs) {
        std::ifstream inputFile(l_fs);
        T l_val;
        if (inputFile.good()) {
            for (unsigned int row = 0; row < m_Rows; ++row) {
                for (unsigned int col = 0; col < ld(); ++col) {
                    inputFile >> l_val;
                    getVal(row, col) = l_val;
                }
            }
        } else {
            std::cout << "no file no loading!\n";
        }
    }

    void fillModFromBin(std::string l_fs) {
        std::ifstream inFile;
        inFile.open(l_fs);
        if (inFile.is_open()) {
            inFile.read((char*)m_Addr, sizeof(T) * m_Rows * ld());
            inFile.close();
        } else {
            std::cout << "no file no loading!\n";
        }
    }

    void fillFromFile(std::istream& p_Is) {
        T l_val;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                p_Is >> l_val;
                getVal(row, col) = l_val;
            }
        }
    }
#if BLAS_runTransp == 1
    // Golden comparsion functions for transp engine
    void transpose(DenseMat& p_A) {
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                getVal(row, col) = p_A.getVal(col, row);
            }
        }
        std::swap(m_Rows, m_Cols);
    }
    void transposeGva(DenseMat& p_A, unsigned int p_rowEdgeWidth, unsigned int p_colEdgeWidth) {
        unsigned int l_pos = 0;
        for (unsigned int rowBlock = 0; rowBlock < p_A.rows() / p_rowEdgeWidth; ++rowBlock) {
            for (unsigned int colBlock = 0; colBlock < p_A.cols() / p_colEdgeWidth; ++colBlock) {
                for (unsigned int col = 0; col < p_colEdgeWidth; ++col) {
                    for (unsigned int row = 0; row < p_rowEdgeWidth; ++row) {
                        getVal(l_pos / cols(), l_pos % cols()) =
                            p_A.getVal(row + rowBlock * p_rowEdgeWidth, col + colBlock * p_colEdgeWidth);
                        l_pos++;
                    }
                    // std::cout << "DEBUG transposeGva step " << *this << "\n";
                }
            }
        }
        std::swap(m_Rows, m_Cols);
    }
#endif
    void print(std::ostream& os) {
        os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
        unsigned int l_cols = cols(); // normal matrix
                                      // ld();; // parent matrix (within Ld
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < l_cols; ++col) {
                os << std::setw(BLAS_FLOAT_WIDTH) << int(getVal(row, col)) << " ";
            }
            os << "\n";
        }
    }
    bool cmp(float p_TolRel, float p_TolAbs, DenseMat& p_Ref) {
        bool ok = true;
        unsigned int l_numExactMatches = 0, l_numMismatches = 0;
        for (unsigned int row = 0; row < rows(); ++row) {
            for (unsigned int col = 0; col < cols(); ++col) {
                std::string l_Prefix = "      row " + std::to_string(row) + " col " + std::to_string(col);
                T v = getVal(row, col);
                T vRef = p_Ref.getVal(row, col);
                bool l_exactMatch = false;
                bool l_ok = xf::blas::cmpVal<T>(p_TolRel, p_TolAbs, vRef, v, l_Prefix, l_exactMatch, 1);
                ok = ok && l_ok;
                if (l_exactMatch) {
                    l_numExactMatches++;
                }
                if (!l_ok) {
                    l_numMismatches++;
                }
            }
        }
        unsigned int l_total = rows() * cols();
        unsigned int l_withinTolerance = l_total - l_numExactMatches - l_numMismatches;
        std::cout << "  Compared " << l_total << " values:"
                  << "  exact match " << l_numExactMatches << "  within tolerance " << l_withinTolerance
                  << "  mismatch " << l_numMismatches << "\n";
        return (ok);
    }
};
template <typename T1>
std::ostream& operator<<(std::ostream& os, DenseMat<T1>& p_Val) {
    p_Val.print(os);
    return (os);
}

/*Type define and float specialization*/

typedef DenseMat<float> MatType_ForFloat;

template <>
void MatType_ForFloat::fillMod(float p_Max, float p_First, bool p_small) {
    if (p_small) {
        // for easier debug of matrices
        float l_val = p_First;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val;
                l_val += 0.01;
                if (l_val > p_Max) {
                    l_val -= p_Max;
                }
            }
        }
    } else {
        // for better float robustness of large matrices
        float l_val = 1.0;
        float l_step = 0.3;
        float l_drift = 0.00001;
        float l_sign = 1;
        for (unsigned int row = 0; row < m_Rows; ++row) {
            for (unsigned int col = 0; col < ld(); ++col) {
                getVal(row, col) = l_val;
                l_val += l_sign * l_step;
                l_step += l_drift;
                l_sign = -l_sign;
                if (l_val > p_Max) {
                    l_val -= p_Max;
                }
            }
        }
    }
}
template <>
void MatType_ForFloat::print(std::ostream& os) {
    os << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << "\n";
    unsigned int l_cols = cols(); // normal matrix
                                  // ld();; // parent matrix (within Ld
    for (unsigned int row = 0; row < rows(); ++row) {
        for (unsigned int col = 0; col < l_cols; ++col) {
            os << std::setw(BLAS_FLOAT_WIDTH) << std::fixed << std::setprecision(3) << getVal(row, col) << " ";
        }
        os << "\n";
    }
}

#endif
