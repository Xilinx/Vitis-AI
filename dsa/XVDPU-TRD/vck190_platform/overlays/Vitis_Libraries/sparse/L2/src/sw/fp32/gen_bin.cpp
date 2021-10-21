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
 * @file gen_bin.cpp
 * @brief main function for generating memory image for parallelCscMV kernel
 *
 * This file is part of Vitis SPARSE Library.
 */
#include "L2_definitions.hpp"
#include "gen_bin.hpp"

using namespace std;
using namespace xf::sparse;

void writeNumPars2BinFile(ofstream& p_of, uint32_t p_numPars) {
    p_of.seekp(0);
    p_of.write(reinterpret_cast<char*>(&p_numPars), sizeof(p_numPars));
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "ERROR: passed " << argc << " arguments, expected at least 3 arguments." << endl;
        cout << "  Usage:\n    gen_bin.exe  <-write | -read | -config-write | -config-read> app.bin [mtxFile] "
             << "    Examples:\n"
             << "      gen_bin.exe -write app.bin bcsstm01.mtx     generate partitions from .mtx file\n"
             << "      gen_bin.exe -write app_val.bin app_row.bin app_colPtr.bin bccstm01.mtx    generate CSC Format "
                "files from .mtx file \n"
             << "      gen_bin.exe -read app.bin    interpret a partition file\n"
             << "      gen_bin.exe -read app.bin bccstm01.mtx    interpret and verify a partition file against .mtx "
                "file\n"
             << "      gen_bin.exe -read app_val.bin app_row.bin app_colPtr.bin   interpreset CSC format files\n"
             << "      gen_bin.exe -read app_val.bin app_row.bin app_colPtr.bin bccstm01.mtx   interpret and verify  "
                "CSC format files against .mtx file\n"
             << "      gen_bin.exe -read app.bin    interpret a kernel config file\n"
             << "      gen_bin.exe -read app.bin bccstm01.mtx    interpret and verify a kernel config file against "
                ".mtx file\n"
             << "      gen_bin.exe -config-write app.bin bcsstm01.mtx     generate kernel config from .mtx file\n"
             << "      gen_bin.exe -config-read app.bin interprete a configuration file\n"
             << "      gen_bin.exe -inter-cscRow app2CscRow.bin appFromCscRow.bin  interpreter  .bin files for "
                "debugging cscRow\n"
             << "\n";
        return EXIT_FAILURE;
    }

    string l_mode(argv[1]);
    bool l_write = (l_mode == "-write");
    bool l_read = (l_mode == "-read");
    bool l_writeConf = (l_mode == "-config-write");
    bool l_readConf = (l_mode == "-config-read");
    bool l_interCscRow = (l_mode == "-inter-cscRow");

    string l_binFileName[3];
    string l_mtxFileName;

    bool l_writePar = l_write && (argc == 4);
    bool l_writeCsc = l_write && (argc == 6);
    bool l_readPar = l_read && (argc == 3);
    bool l_readCheckPar = l_read && (argc == 4);
    bool l_readCsc = l_read && (argc == 5);
    bool l_readCheckCsc = l_read && (argc == 6);
    l_writeConf = l_writeConf && (argc == 6);
    bool l_readCheckConf = l_readConf && (argc == 6);
    l_readConf = l_readConf && (argc == 3);

    l_mtxFileName = (l_writePar || l_readCheckPar)
                        ? argv[3]
                        : (l_writeCsc || l_writeConf || l_readCheckConf || l_readCheckCsc) ? argv[5] : "";
    l_binFileName[0] = argv[2]; // valFileName
    l_binFileName[1] = (l_writeCsc || l_writeConf || l_readCheckConf || l_readCsc || l_readCheckCsc || l_interCscRow)
                           ? argv[3]
                           : ""; // rowFileName
    l_binFileName[2] =
        (l_writeCsc || l_writeConf || l_readCheckConf || l_readCsc || l_readCheckCsc) ? argv[4] : ""; // colPtrFileName

    bool l_res = true;
    if (l_writeCsc) {
        l_res = writeCsc<SPARSE_dataType, SPARSE_indexType, SPARSE_pageSize>(l_mtxFileName, l_binFileName[0],
                                                                             l_binFileName[1], l_binFileName[2]);
    }
    if (l_readCsc) {
        l_res = readCsc<SPARSE_dataType, SPARSE_indexType, SPARSE_pageSize>(l_binFileName[0], l_binFileName[1],
                                                                            l_binFileName[2]);
    }
    if (l_readCheckCsc) {
        l_res = readCheckCsc<SPARSE_dataType, SPARSE_indexType, SPARSE_pageSize>(l_mtxFileName, l_binFileName[0],
                                                                                 l_binFileName[1], l_binFileName[2]);
        if (l_res) {
            cout << "Pass CSC format check for mtx file " << l_mtxFileName << endl;
        } else {
            cout << "ERROR: failed CSC format check for mtx file " << l_mtxFileName << endl;
        }
    }
    if (l_writePar) {
        l_res = writePar<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_maxCols, SPARSE_parEntries,
                         SPARSE_parGroups, SPARSE_ddrMemBits, SPARSE_hbmChannels>(l_binFileName[0], l_mtxFileName);
    }
    if (l_readPar) {
        l_res = readPar<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_maxCols, SPARSE_hbmChannels>(
            l_binFileName[0]);
    }
    if (l_readCheckPar) {
        l_res = readCheckPar<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_maxCols, SPARSE_hbmChannels>(
            l_binFileName[0], l_mtxFileName);
    }
    if (l_writeConf) {
        l_res = writeConfig<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_maxCols, SPARSE_maxParamDdrBlocks,
                            SPARSE_maxParamHbmBlocks, SPARSE_parEntries, SPARSE_parGroups, SPARSE_ddrMemBits,
                            SPARSE_hbmMemBits, SPARSE_hbmChannels, SPARSE_paramOffset, SPARSE_pageSize>(
            l_binFileName[0], l_binFileName[1], l_binFileName[2], l_mtxFileName);
    }
    if (l_readConf) {
        l_res =
            readConfig<SPARSE_dataType, SPARSE_indexType, SPARSE_parEntries, SPARSE_parGroups, SPARSE_ddrMemBits,
                       SPARSE_hbmMemBits, SPARSE_hbmChannels, SPARSE_paramOffset, SPARSE_pageSize>(l_binFileName[0]);
    }
    if (l_readCheckConf) {
        l_res = readCheckConfig<SPARSE_dataType, SPARSE_indexType, SPARSE_maxRows, SPARSE_maxCols, SPARSE_parEntries,
                                SPARSE_parGroups, SPARSE_ddrMemBits, SPARSE_hbmMemBits, SPARSE_hbmChannels,
                                SPARSE_paramOffset, SPARSE_pageSize>(l_binFileName[0], l_binFileName[1],
                                                                     l_binFileName[2], l_mtxFileName);
        if (l_res) {
            cout << "Config Test Pass!" << endl;
        } else {
            cout << "Config Test Failed!" << endl;
        }
    }
    if (l_interCscRow) {
        CscRowIntType l_cscRowInt;
        l_res = l_res && l_cscRowInt.interToCscRowFile(l_binFileName[0]);
        if (l_res) {
            cout << l_cscRowInt << endl;
        }
        l_res = l_res && l_cscRowInt.interFromCscRowFile(l_binFileName[1]);
        if (l_res) {
            cout << l_cscRowInt << endl;
        }
    }

    if (l_res) {
        return EXIT_SUCCESS;
    } else {
        return EXIT_FAILURE;
    }
}
