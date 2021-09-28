/*
 * Copyright 2020 Xilinx, Inc.
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
#include "xf_database/gqe_join_config.hpp"
#include "xf_database/kernel_command.hpp"
#define USER_DEBUG 1
namespace xf {
namespace database {
namespace gqe {

// init to setup the join config, which includes:
//- sw_cfg: sw_shuffle cfg for scan and wr
//- krn_cfg: gqe kernel used config
JoinConfig::JoinConfig(Table a,
                       std::string filter_a,
                       Table b,
                       std::string filter_b,
                       std::string join_str, // comma separated
                       Table c,
                       std::string output_str,
                       int join_type) {
    // 0) process gen-rowid_en  and val_en
    bool gen_rowID_en[2];
    bool valid_en[2];

    gen_rowID_en[0] = a.getRowIDEnableFlag();
    gen_rowID_en[1] = b.getRowIDEnableFlag();
    valid_en[0] = a.getValidEnableFlag();
    valid_en[1] = b.getValidEnableFlag();

    // 1) read in col names of table a and b
    std::vector<std::string> a_col_names = a.getColNames();
    std::vector<std::string> b_col_names = b.getColNames();
    int table_b_valid_col_num = b_col_names.size();
    // verify the column num in table A and B is no more than 3
    CHECK_0(a_col_names, 3, "A");
    CHECK_0(b_col_names, 3, "B");
    // remove the space in str
    xf::database::internals::filter_config::trim(join_str);
    xf::database::internals::filter_config::trim(output_str);

#ifdef USER_DEBUG
    std::cout << "1. Get cols from table" << std::endl;
    for (size_t i = 0; i < a_col_names.size(); i++) {
        std::cout << a_col_names[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < b_col_names.size(); i++) {
        std::cout << b_col_names[i] << " ";
    }
    std::cout << std::endl << "------------------" << std::endl;
#endif

    // 2) extract join key col and write out col from input info
    //  join keys extracted from two input tables
    std::vector<std::vector<std::string> > join_keys;
    join_keys = extractKeys(join_str);

    // write out cols name extracted from table out
    std::vector<std::string> write_out_cols;
    write_out_cols = extractWcols(join_keys, output_str);

    // 3) calc the sw_shuffle_scan cfg
    sw_shuffle_scan_hj.resize(2);
    // sw scan-shuffle, puts key cols to 1st and 2nd col
    ShuffleScan(filter_a, join_keys[0], write_out_cols, a_col_names, sw_shuffle_scan_hj[0]);
    ShuffleScan(filter_b, join_keys[1], write_out_cols, b_col_names, sw_shuffle_scan_hj[1]);

#ifdef USER_DEBUG
    std::cout << "3.1. sw_shuffle_scan_hj_a: ";
    for (size_t i = 0; i < sw_shuffle_scan_hj[0].size(); i++) {
        std::cout << (int)sw_shuffle_scan_hj[0][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "after sw_shuffle scan column names: " << std::endl;
    for (size_t i = 0; i < a_col_names.size(); i++) {
        std::cout << a_col_names[i] << " ";
    }
    std::cout << std::endl;

    std::cout << "3.2. sw_shuffle_scan_hj_b: ";
    for (size_t i = 0; i < sw_shuffle_scan_hj[1].size(); i++) {
        std::cout << (int)sw_shuffle_scan_hj[1][i] << " ";
    }
    std::cout << std::endl;
    std::cout << "after sw_shuffle scan column names: " << std::endl;
    for (size_t i = 0; i < b_col_names.size(); i++) {
        std::cout << b_col_names[i] << " ";
    }
    std::cout << std::endl;

#endif
    // updating row-id col name
    UpdateRowIDCol(filter_a, a_col_names, a.getRowIDColName());
    UpdateRowIDCol(filter_b, b_col_names, b.getRowIDColName());

#ifdef USER_DEBUG
    std::cout << "3.3 after add row-id, column names: " << std::endl;
    // tab a
    for (size_t i = 0; i < a_col_names.size(); i++) {
        std::cout << a_col_names[i] << " ";
    }
    // tab b
    for (size_t i = 0; i < b_col_names.size(); i++) {
        std::cout << b_col_names[i] << " ";
    }

    std::cout << std::endl << "------------------" << std::endl;
#endif

    // 4) calc the shuffle wr cfg by "mimic kernel out col ==> compare with write out col"
    // define the joined cols that mimic the hardware shuffle Join Keys
    int join_keys_num = 1;
    std::vector<std::string> col_out_names;
    if (join_str != "") {
        join_keys_num = join_keys[0].size();

        // l table rowid col name write to col_out_names
        col_out_names.push_back(b.getRowIDColName());
        col_out_names.push_back(a.getRowIDColName());
        // insert joined keys
        col_out_names.insert(col_out_names.end(), a_col_names.begin(), a_col_names.begin() + join_keys_num);
        if (join_keys_num == 1) {
            col_out_names.push_back("unused");
        }
    } else {
        if (table_b_valid_col_num != 0) {
            std::cout << "WARNING: Bypass data and ignore table b." << std::endl;
        }
        std::copy(a_col_names.begin(), a_col_names.end(), std::back_inserter(col_out_names));
    }
#ifdef USER_DEBUG
    std::cout << "4.1. After join, column names: " << std::endl;
    for (size_t i = 0; i < col_out_names.size(); i++) {
        std::cout << col_out_names[i] << " ";
    }
    std::cout << std::endl;
#endif

    // get the shuffle_wr config
    // the cols before shuffle_wr: col_out_names
    // the cols after shuffle_wr: extractWCols result: write_out_cols
    sw_shuffle_wr_hj = ShuffleWrite(col_out_names, write_out_cols);

#ifdef USER_DEBUG
    for (size_t i = 0; i < sw_shuffle_wr_hj.size(); i++) {
        std::cout << "sw_shuffle_wr_hj: " << (int)sw_shuffle_wr_hj[i] << std::endl;
    }
    for (size_t i = 0; i < write_out_cols.size(); i++) {
        std::cout << "write_out_cols: " << write_out_cols[i] << std::endl;
    }
    std::cout << "----------cfg setup done---------" << std::endl;
#endif

    // 5) setup kernel cfg, which includes join setup, input_col_en, wr_col_en
    // input_col_en and wr_col_en are taken from sw_shuffle_scan/wr cfg
    SetupKernelConfig(join_type, filter_a, filter_b, gen_rowID_en, valid_en, join_keys, sw_shuffle_scan_hj,
                      sw_shuffle_wr_hj);
}

/**
 * @brief setup gqe kernel used config
 *
 * kernel config is consist of 3 parts:
 * - krn join setup
 * - scan input col enable, get from sw_shuffle_scan
 * - write out col enable, get from sw_shuffle_wr
**/
void JoinConfig::SetupKernelConfig(int join_type,
                                   std::string filter_a,
                                   std::string filter_b,
                                   bool gen_rowID_en[2],
                                   bool valid_en[2],
                                   std::vector<std::vector<std::string> > join_keys,
                                   std::vector<std::vector<int8_t> > sw_shuffle_scan_hj,
                                   std::vector<int8_t> sw_shuffle_wr_hj) {
    using krncmdclass = xf::database::gqe::KernelCommand;
    krncmdclass krncmd = krncmdclass();
    if (join_keys[0].size() == 2) {
        krncmd.setDualKeyOn();
    }
    krncmd.setJoinType(join_type);

    // setting gen_rowID_en and valid_en for tab a
    krncmd.setRowIDValidEnable(0, 0, gen_rowID_en[0], valid_en[0]);
    // setting gen_rowID_en and valid_en for tab b
    krncmd.setRowIDValidEnable(0, 1, gen_rowID_en[1], valid_en[1]);

    // using join solution, se filter in join kernel
    krncmd.setScanColEnable(0, 0, sw_shuffle_scan_hj[0]);
    if (filter_a != "") krncmd.setFilter(0, filter_a);

    // using join solution, se filter in join kernel
    krncmd.setScanColEnable(0, 1, sw_shuffle_scan_hj[1]);
    if (filter_b != "") krncmd.setFilter(1, filter_b);

    // setup the write col en
    krncmd.setJoinWriteColEnable(0, 0, sw_shuffle_wr_hj);

    ap_uint<512>* config_bits = krncmd.getConfigBits();
    table_join_cfg = mm.aligned_alloc<ap_uint<512> >(14);
    memcpy(table_join_cfg, config_bits, sizeof(ap_uint<512>) * 14);
}

ap_uint<512>* JoinConfig::getJoinConfigBits() const {
    return table_join_cfg;
}

// get the sw scan-shuffle config which shuffles key col to the front
std::vector<std::vector<int8_t> > JoinConfig::getShuffleScan() const {
    return sw_shuffle_scan_hj;
}

std::vector<int8_t> JoinConfig::getShuffleWrite() const {
    return sw_shuffle_wr_hj;
}

} // database
} // gqe
} // xf
