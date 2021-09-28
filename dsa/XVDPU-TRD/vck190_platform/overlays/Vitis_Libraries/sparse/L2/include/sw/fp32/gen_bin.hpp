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
 * @file gen_bin.hpp
 * @brief functions used in gen_bin.cpp host coce.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_GEN_BIN_HPP
#define XF_SPARSE_GEN_BIN_HPP

#include "gen_cscmv.hpp"

using namespace std;

namespace xf {
namespace sparse {

template <typename t_DataType, typename t_IndexType, unsigned int t_PageSize>
bool writeCsc(string p_mtxFileName, string p_valFileName, string p_rowFileName, string p_colPtrFileName) {
    MatCsc<t_DataType, t_IndexType> l_mat;
    Program<t_PageSize> l_program;
    GenMatCsc<t_DataType, t_IndexType, t_PageSize> l_gen;

    if (!l_gen.genMatFromMtxFile(p_mtxFileName, l_program, l_mat)) {
        return false;
    }
    if (!l_mat.write2BinFile(p_valFileName, p_rowFileName, p_colPtrFileName)) {
        return false;
    }
    return true;
}

template <typename t_DataType, typename t_IndexType, unsigned int t_PageSize>
bool readCsc(string p_valFileName, string p_rowFileName, string p_colPtrFileName) {
    MatCsc<t_DataType, t_IndexType> l_mat;
    Program<t_PageSize> l_program;
    GenMatCsc<t_DataType, t_IndexType, t_PageSize> l_gen;

    if (!l_gen.genMatFromCscFiles(p_valFileName, p_rowFileName, p_colPtrFileName, l_program, l_mat)) {
        return false;
    }

    cout << l_mat;
    return true;
}

template <typename t_DataType, typename t_IndexType, unsigned int t_PageSize>
bool readCheckCsc(string p_mtxFileName, string p_valFileName, string p_rowFileName, string p_colPtrFileName) {
    MatCsc<t_DataType, t_IndexType> l_mat1;
    MatCsc<t_DataType, t_IndexType> l_mat2;
    Program<t_PageSize> l_program;
    GenMatCsc<t_DataType, t_IndexType, t_PageSize> l_gen;

    if (!l_gen.genMatFromMtxFile(p_mtxFileName, l_program, l_mat1)) {
        return false;
    }
    if (!l_gen.genMatFromCscFiles(p_valFileName, p_rowFileName, p_colPtrFileName, l_program, l_mat2)) {
        return false;
    }

    bool l_res = l_mat1.isEqual(l_mat2);
    return l_res;
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsInPar,
          unsigned int t_MaxColsInPar,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_memBits,
          unsigned int t_HbmChannels>
bool writePar(string p_parFileName, string p_mtxFileName) {
    MtxFile<t_DataType, t_IndexType> l_mtxFile;
    MatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_HbmChannels> l_matPar;
    GenMatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_ParEntries, t_ParGroups, t_memBits,
              t_HbmChannels>
        l_genMatPar;

    l_mtxFile.loadFile(p_mtxFileName);
    vector<NnzUnit<t_DataType, t_IndexType> > l_nnzUnits = l_mtxFile.getNnzUnits();
    unsigned int l_rows = l_mtxFile.rows();
    unsigned int l_cols = l_mtxFile.cols();
    l_genMatPar.genMatPar(l_nnzUnits, l_rows, l_cols, l_matPar);

    ofstream l_of(p_parFileName.c_str(), ios::binary);
    if (!l_of.is_open()) {
        cout << "ERROR: open file " << p_parFileName << endl;
        return false;
    }
    l_matPar.writeBinFile(l_of);
    l_of.close();
    return true;
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsInPar,
          unsigned int t_MaxColsInPar,
          unsigned int t_HbmChannels>
bool readPar(string p_parFileName) {
    MatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_HbmChannels> l_matPar;
    ifstream l_if(p_parFileName.c_str(), ios::binary);
    if (!l_if.is_open()) {
        cout << "ERROR: open file " << p_parFileName << endl;
        return false;
    }
    l_matPar.readBinFile(l_if);
    l_if.close();
    cout << l_matPar;
    return true;
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsInPar,
          unsigned int t_MaxColsInPar,
          unsigned int t_HbmChannels>
bool readCheckPar(string p_parFileName, string p_mtxFileName) {
    MtxFile<t_DataType, t_IndexType> l_mtxFile;
    l_mtxFile.loadFile(p_mtxFileName);
    vector<NnzUnit<t_DataType, t_IndexType> > l_nnzUnits = l_mtxFile.getNnzUnits();
    unsigned int l_rows = l_mtxFile.rows();
    unsigned int l_cols = l_mtxFile.cols();

    MatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_HbmChannels> l_matPar;
    ifstream l_if(p_parFileName.c_str(), ios::binary);
    if (!l_if.is_open()) {
        cout << "ERROR: open file " << p_parFileName << endl;
        return false;
    }
    l_matPar.readBinFile(l_if);
    l_if.close();

    bool l_res = l_matPar.check(l_rows, l_cols, l_nnzUnits);
    return l_res;
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsInPar,
          unsigned int t_MaxColsInPar,
          unsigned int t_MaxParamDdrBlocks,
          unsigned int t_MaxParamHbmBlocks,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmMemBits,
          unsigned int t_HbmChannels,
          unsigned int t_ParamOffset = 1024,
          unsigned int t_PageSize = 4096>
bool writeConfig(string p_configFileName, string p_inVecFileName, string p_outVecFileName, string p_mtxFileName) {
    RunConfig<t_DataType, t_IndexType, t_ParEntries, t_ParGroups, t_DdrMemBits, t_HbmMemBits, t_HbmChannels,
              t_ParamOffset, t_PageSize>
        l_runConf;
    MtxFile<t_DataType, t_IndexType> l_mtxFile;
    Program<t_PageSize> l_prog;
    GenRunConfig<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_MaxParamDdrBlocks, t_MaxParamHbmBlocks,
                 t_ParEntries, t_ParGroups, t_DdrMemBits, t_HbmMemBits, t_HbmChannels, t_ParamOffset, t_PageSize>
        l_genRunConf;

    l_mtxFile.loadFile(p_mtxFileName);
    vector<NnzUnit<t_DataType, t_IndexType> > l_nnzUnits = l_mtxFile.getNnzUnits();
    unsigned int l_rows = l_mtxFile.rows();
    unsigned int l_cols = l_mtxFile.cols();
    vector<t_DataType> l_colVec(l_cols);
    for (unsigned int i = 0; i < l_cols; ++i) {
        l_colVec[i] = i / 10.0;
    }
    vector<t_DataType> l_resVec;
    NnzWorker<t_DataType, t_IndexType> l_nnzWorker;
    l_nnzWorker.spmv(l_nnzUnits, l_colVec, l_rows, l_resVec);

    l_genRunConf.genRunConfig(l_prog, l_nnzUnits, l_colVec, l_rows, l_cols, l_runConf);

    ofstream l_of(p_configFileName.c_str(), ios::binary);
    if (!l_of.is_open()) {
        cout << "ERROR: open file " << p_configFileName << endl;
        return false;
    }
    l_runConf.writeBinFile(l_of, l_prog);
    l_of.close();

    ofstream l_ofInVec(p_inVecFileName.c_str(), ios::binary);
    if (!l_ofInVec.is_open()) {
        cout << "ERROR: open file " << p_inVecFileName << endl;
        return false;
    }
    unsigned long long l_inVecSz = l_colVec.size() * sizeof(t_DataType);
    l_ofInVec.write(reinterpret_cast<char*>(l_colVec.data()), l_inVecSz);
    l_ofInVec.close();

    ofstream l_ofOutVec(p_outVecFileName.c_str(), ios::binary);
    if (!l_ofOutVec.is_open()) {
        cout << "ERROR: open file " << p_outVecFileName << endl;
        return false;
    }
    l_ofOutVec.write(reinterpret_cast<char*>(l_resVec.data()), l_inVecSz);
    l_ofOutVec.close();
    return true;
}
template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmMemBits,
          unsigned int t_HbmChannels,
          unsigned int t_ParamOffset,
          unsigned int t_PageSize = 4096>
bool readConfig(string p_configFileName) {
    RunConfig<t_DataType, t_IndexType, t_ParEntries, t_ParGroups, t_DdrMemBits, t_HbmMemBits, t_HbmChannels,
              t_ParamOffset, t_PageSize>
        l_runConf;
    Program<t_PageSize> l_prog;

    ifstream l_if(p_configFileName.c_str(), ios::binary);
    if (!l_if.is_open()) {
        cout << "ERROR: open file " << p_configFileName << endl;
        return false;
    }
    l_runConf.readBinFile(l_if, l_prog);
    l_if.close();
    cout << l_runConf;
    return true;
}

template <typename t_DataType,
          typename t_IndexType,
          unsigned int t_MaxRowsInPar,
          unsigned int t_MaxColsInPar,
          unsigned int t_ParEntries,
          unsigned int t_ParGroups,
          unsigned int t_DdrMemBits,
          unsigned int t_HbmMemBits,
          unsigned int t_HbmChannels,
          unsigned int t_ParamOffset = 1024,
          unsigned int t_PageSize = 4096>
bool readCheckConfig(string p_configFileName, string p_inVecFileName, string p_outVecFileName, string p_mtxFileName) {
    MtxFile<t_DataType, t_IndexType> l_mtxFile;
    MatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_HbmChannels> l_matPar;
    GenMatPar<t_DataType, t_IndexType, t_MaxRowsInPar, t_MaxColsInPar, t_ParEntries, t_ParGroups, t_DdrMemBits,
              t_HbmChannels>
        l_genMatPar;

    l_mtxFile.loadFile(p_mtxFileName);
    vector<NnzUnit<t_DataType, t_IndexType> > l_nnzUnits = l_mtxFile.getNnzUnits();
    unsigned int l_rows = l_mtxFile.rows();
    unsigned int l_cols = l_mtxFile.cols();
    l_genMatPar.genMatPar(l_nnzUnits, l_rows, l_cols, l_matPar);

    RunConfig<t_DataType, t_IndexType, t_ParEntries, t_ParGroups, t_DdrMemBits, t_HbmMemBits, t_HbmChannels,
              t_ParamOffset, t_PageSize>
        l_runConf;
    Program<t_PageSize> l_prog;

    ifstream l_if(p_configFileName.c_str(), ios::binary);
    if (!l_if.is_open()) {
        cout << "ERROR: open file " << p_configFileName << endl;
        return false;
    }
    l_runConf.readBinFile(l_if, l_prog);
    l_if.close();

    vector<t_DataType> l_colVec(l_cols);
    ifstream l_ifInVec(p_inVecFileName.c_str(), ios::binary);
    if (!l_ifInVec.is_open()) {
        cout << "ERROR: open file " << p_inVecFileName << endl;
        return false;
    }
    unsigned long long l_inVecSz = l_cols * sizeof(t_DataType);
    l_ifInVec.read(reinterpret_cast<char*>(l_colVec.data()), l_inVecSz);
    l_ifInVec.close();

    vector<t_DataType> l_resVec(l_rows);
    ifstream l_ifOutVec(p_outVecFileName.c_str(), ios::binary);
    if (!l_ifOutVec.is_open()) {
        cout << "ERROR: open file " << p_outVecFileName << endl;
        return false;
    }
    unsigned long long l_outVecSz = l_rows * sizeof(t_DataType);
    l_ifOutVec.read(reinterpret_cast<char*>(l_resVec.data()), l_outVecSz);
    l_ifOutVec.close();

    bool l_res = true;
    if (l_runConf.validPars() != l_matPar.validPars()) {
        cout << "ERROR: different validPars" << endl;
        return false;
    }
    if (l_runConf.rows() != l_matPar.roCooPar().rows()) {
        cout << "ERROR: different rows" << endl;
        return false;
    }
    if (l_runConf.cols() != l_matPar.roCooPar().cols()) {
        cout << "ERROR: different cols" << endl;
        return false;
    }
    if (l_runConf.nnzs() != l_matPar.roCooPar().nnzs()) {
        cout << "ERROR: different nnzs" << endl;
        return false;
    }
    if (l_runConf.rowPars() != l_matPar.roCooPar().rowPars()) {
        cout << "ERROR: different rowPars" << endl;
        return false;
    }
    if (l_runConf.colPars() != l_matPar.roCooPar().colPars()) {
        cout << "ERROR: different colPars" << endl;
        return false;
    }
    if (l_runConf.totalPars() != l_matPar.roCooPar().totalPars()) {
        cout << "ERROR: different totalPars" << endl;
        return false;
    }
    for (unsigned int ch = 0; ch < t_HbmChannels; ++ch) {
        if (l_runConf.validRowPars()[ch] != l_matPar.validRowPars()[ch]) {
            cout << "ERROR: different validRowPars" << endl;
            return false;
        }
        if (l_runConf.nnzBlocks()[ch] != l_matPar.nnzBlocks()[ch]) {
            cout << "ERROR: different nnzBlocks" << endl;
            return false;
        }
        if (l_runConf.rowResBlocks()[ch] != l_matPar.rowResBlocks()[ch]) {
            cout << "ERROR: different rowResBlocks" << endl;
            return false;
        }
    }
    l_res = l_res && l_runConf.check(l_matPar.roCooPar().matPars(), l_colVec, l_resVec);
    return l_res;
}

} // end namespace sparse
} // end namespace xf
#endif
