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
 * @file mtxFile.hpp
 * @brief header file for processing .mtx file.
 *
 * This file is part of Vitis SPARSE Library.
 */
#ifndef XF_SPARSE_MTXFILE_HPP
#define XF_SPARSE_MTXFILE_HPP

#include <iostream>
#include <fstream>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include "L2_types.hpp"

using namespace std;

namespace xf {
namespace sparse {

inline void align(unsigned int& dst, unsigned int width) {
    dst = width * ((dst + width - 1) / width);
}

template <typename t_DataType, typename t_IndexType>
class MtxFile {
   public:
    typedef NnzUnit<t_DataType, t_IndexType> t_NnzUnitType;

   public:
    MtxFile() {}
    bool good() { return (m_good); }
    unsigned int rows() { return m_rows; }
    unsigned int cols() { return m_cols; }
    unsigned int nnzs() { return m_nnzs; }
    vector<t_NnzUnitType>& getNnzUnits() { return m_nnzUnits; }
    string fileName() { return m_fileName; }
    void loadFile(string p_FileName) {
        m_good = false;
        m_rows = 0;
        m_cols = 0;
        m_nnzs = 0;
        m_fileName = p_FileName;
        if (m_fileName != "none") {
            cout << "INFO: loading Mtx file  " << m_fileName << "\n";
            ifstream l_fs(m_fileName.c_str(), ios_base::in | ios_base::binary);

            boost::iostreams::filtering_istream l_bs;
            string l_ext = m_fileName.substr(m_fileName.find_last_of(".") + 1);
            transform(l_ext.begin(), l_ext.end(), l_ext.begin(), ::tolower);

            if (l_ext == "gz") {
                l_bs.push(boost::iostreams::gzip_decompressor());
            } else if (l_ext == "mtx") {
                // noop
            } else {
                cerr << "ERROR: MtxFile failed due to unknown extension \"" << l_ext << "\", file  " << m_fileName
                     << endl;
                assert(0);
            }
            l_bs.push(l_fs);
            m_good = l_bs.good();
            if (m_good) {
                bool l_isSym = false;
                while (l_bs.peek() == '%') {
                    string l_line;
                    getline(l_bs, l_line);
                    if ((l_line.find("SYMMETRIC") != string::npos) || (l_line.find("symmetric") != string::npos)) {
                        l_isSym = true;
                    }
                    // l_bs.ignore(2048, '\n');
                }
                l_bs >> m_rows >> m_cols >> m_nnzs;
                unsigned l_symNnzs = 0;
                for (unsigned int i = 0; i < m_nnzs; ++i) {
                    t_NnzUnitType l_nnzUnit;
                    l_nnzUnit.scan(l_bs);
                    m_nnzUnits.push_back(l_nnzUnit);
                    if (l_isSym && (l_nnzUnit.getRow() != l_nnzUnit.getCol())) {
                        t_NnzUnitType l_dupNnz(l_nnzUnit.getCol(), l_nnzUnit.getRow(), l_nnzUnit.getVal());
                        m_nnzUnits.push_back(l_dupNnz);
                        l_symNnzs++;
                    }
                }
                m_nnzs += l_symNnzs;
                boost::iostreams::close(l_bs);
                assert(m_nnzs == m_nnzUnits.size());
                cout << "INFO: loaded mtx file"
                     << "  rows " << rows() << "  cols " << cols() << "  Nnzs " << nnzs() << endl;
            }
        }
    }

   private:
    string m_fileName;
    bool m_good;
    unsigned int m_rows, m_cols, m_nnzs;
    vector<t_NnzUnitType> m_nnzUnits;
};

} // end namespace sparse
} // end namespace xf

#endif
