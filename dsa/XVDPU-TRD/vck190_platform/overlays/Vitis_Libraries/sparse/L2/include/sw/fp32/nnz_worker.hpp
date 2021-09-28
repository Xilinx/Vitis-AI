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
 * @file nnz_worker.hpp
 * @brief header file for data types used in L2/L3 host code.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_NNZ_WORKER_HPP
#define XF_SPARSE_NNZ_WORKER_HPP

#include <algorithm>
#include <cassert>
#include <vector>
#include "L1_utils.hpp"

using namespace std;

namespace xf {
namespace sparse {

template <typename t_DataType, typename t_IndexType>
class NnzUnit {
   public:
    NnzUnit() {}
    NnzUnit(t_IndexType p_row, t_IndexType p_col, t_DataType p_val) : m_row(p_row), m_col(p_col), m_val(p_val) {}
    inline t_IndexType& getRow() { return m_row; }
    inline t_IndexType& getCol() { return m_col; }
    inline t_DataType& getVal() { return m_val; }
    void scan(istream& p_is) {
        double l_val;
        p_is >> m_row >> m_col >> l_val;
        m_val = (t_DataType)(l_val);
        if ((m_row <= 0) || (m_col <= 0)) {
            cerr << "Error: invalid MTX file line row=" << m_row << "  col=" << m_col << "  val=" << m_val << endl;
            return;
        }
        // indices start from 1 in .mtx file, 0 locally
        m_row--;
        m_col--;
        return;
    }
    bool isSimilar(NnzUnit<t_DataType, t_IndexType>& p_x) {
        if (m_row != p_x.getRow()) {
            return false;
        } else if (m_col != p_x.getCol()) {
            return false;
        } else if (!compare<t_DataType>(m_val, p_x.getVal())) {
            return false;
        }
        return true;
    }
    friend bool operator==(NnzUnit<t_DataType, t_IndexType>& a, NnzUnit<t_DataType, t_IndexType>& b) {
        if (a.getRow() != b.getRow()) {
            return false;
        } else if (a.getCol() != b.getCol()) {
            return false;
        } else if (a.getVal() != b.getVal()) {
            return false;
        }
        return true;
    }
    friend bool operator<(NnzUnit<t_DataType, t_IndexType>& a, NnzUnit<t_DataType, t_IndexType>& b) {
        if (a.getCol() < b.getCol()) {
            return true;
        } else if (a.getCol() == b.getCol()) {
            if (a.getRow() <= b.getRow()) {
                return true;
            }
        }
        return false;
    }

    void writeBinFile(ofstream& p_of) {
        p_of.write(reinterpret_cast<char*>(&m_row), sizeof(t_IndexType));
        p_of.write(reinterpret_cast<char*>(&m_col), sizeof(t_IndexType));
        p_of.write(reinterpret_cast<char*>(&m_val), sizeof(t_DataType));
    }
    void readBinFile(ifstream& p_if) {
        p_if.read(reinterpret_cast<char*>(&m_row), sizeof(t_IndexType));
        p_if.read(reinterpret_cast<char*>(&m_col), sizeof(t_IndexType));
        p_if.read(reinterpret_cast<char*>(&m_val), sizeof(t_DataType));
    }
    void print(ostream& p_os) {
        p_os << setw(SPARSE_printWidth) << m_row << " " << setw(SPARSE_printWidth) << m_col << "  "
             << setw(SPARSE_printWidth) << m_val;
    }
    friend ostream& operator<<(ostream& p_os, NnzUnit& p_nnzUnit) {
        p_nnzUnit.print(p_os);
        return (p_os);
    }

   private:
    t_IndexType m_row, m_col;
    t_DataType m_val;
};

template <typename t_DataType, typename t_IndexType>
struct compareRow {
    bool operator()(NnzUnit<t_DataType, t_IndexType>& a, NnzUnit<t_DataType, t_IndexType>& b) {
        if (a.getRow() < b.getRow()) {
            return (true);
        } else if (a.getRow() == b.getRow()) {
            if (a.getCol() < b.getCol()) {
                return (true);
            }
        }
        return (false);
    }
};

template <typename t_DataType, typename t_IndexType>
class NnzWorker {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;

   public:
    NnzWorker() {}

   public:
    void genNnzUnits(vector<t_DataType>& p_val,
                     vector<t_IndexType>& p_rowIdx,
                     vector<t_IndexType>& p_colPtr,
                     unsigned int p_minRowId,
                     unsigned int p_minColId,
                     vector<t_NnzUnitType>& p_nnzUnits) {
        unsigned int l_cols = p_colPtr.size();
        unsigned int l_pre = 0;
        for (unsigned int i = 0; i < l_cols; ++i) {
            unsigned int l_colId = p_minColId + i;
            for (unsigned int j = l_pre; j < p_colPtr[i]; ++j) {
                unsigned int l_rowId = p_rowIdx[j] + p_minRowId;
                t_DataType l_val = p_val[j];
                t_NnzUnitType l_nnzUnit(l_rowId, l_colId, l_val);
                p_nnzUnits.push_back(l_nnzUnit);
            }
            l_pre = p_colPtr[i];
        }
    }

    void spmv(vector<t_NnzUnitType>& p_mat,
              vector<t_DataType>& p_inVec,
              unsigned int p_rows,
              vector<t_DataType>& p_res) {
        p_res.resize(p_rows, 0);
        unsigned int l_nnzs = p_mat.size();
        // t_DataType l_res0=0;
        // t_DataType l_res1=0;
        // t_DataType l_res = 0;
        for (unsigned int i = 0; i < l_nnzs; ++i) {
            t_NnzUnitType l_nnzUnit = p_mat[i];
            t_IndexType l_rowId = l_nnzUnit.getRow();
            t_IndexType l_colId = l_nnzUnit.getCol();
            t_DataType l_val = l_nnzUnit.getVal();
            t_DataType l_mulVal = l_val * p_inVec[l_colId];
            p_res[l_rowId] += l_mulVal;
            /*if (l_rowId==921) {
                cout << l_nnzUnit << endl;
                l_res += l_mulVal;
            }*/
        }
        // cout << l_res0 << endl;
        // cout << l_res1 << endl;
        // cout << l_res0 + l_res1 << endl;
        // cout << l_res << endl;
    }
    bool compareNnzs(vector<t_NnzUnitType>& p_nnzs1, vector<t_NnzUnitType>& p_nnzs2) {
        bool l_res = true;
        unsigned int l_errs = 0;
        unsigned int l_size = p_nnzs1.size();
        if (l_size != p_nnzs2.size()) {
            cout << "ERROR: in compareNnzs, NnzUnit Arrays have different sizes, one is " << l_size << " the other is "
                 << p_nnzs2.size() << endl;
            return false;
        }
        for (unsigned int i = 0; i < l_size; ++i) {
            t_NnzUnitType l_val1 = p_nnzs1[i];
            t_NnzUnitType l_val2 = p_nnzs2[i];
            if (!(l_val1 == l_val2)) {
                cout << "ERROR: in compareNnzs, NnzUnits are not the same" << endl;
                cout << "First NnzUnit: " << l_val1 << endl;
                cout << "Second NnzUnit: " << l_val2 << endl;
                l_errs++;
            }
        }
        l_res = (l_errs == 0);
        if (!l_res) {
            cout << "Total NnzUnit errors: " << l_errs << endl;
        }
        return l_res;
    }
    bool similarNnzs(vector<t_NnzUnitType>& p_nnzs1, vector<t_NnzUnitType>& p_nnzs2) {
        bool l_res = true;
        unsigned int l_errs = 0;
        unsigned int l_size = p_nnzs1.size();
        if (l_size != p_nnzs2.size()) {
            cout << "ERROR: in similarNnzs, NnzUnit Arrays have different sizes, one is " << l_size << " the other is "
                 << p_nnzs2.size() << endl;
            return false;
        }
        for (unsigned int i = 0; i < l_size; ++i) {
            t_NnzUnitType l_val1 = p_nnzs1[i];
            t_NnzUnitType l_val2 = p_nnzs2[i];
            if (!(l_val1.isSimilar(l_val2))) {
                cout << "ERROR: NnzUnits are not close enough" << endl;
                cout << "First NnzUnit: " << l_val1 << endl;
                cout << "Second NnzUnit: " << l_val2 << endl;
                l_errs++;
            }
        }
        l_res = (l_errs == 0);
        if (!l_res) {
            cout << "Total NnzUnit errors: " << l_errs << endl;
        }
        return l_res;
    }
};

} // end namespace sparse
} // end namespace xf
#endif
