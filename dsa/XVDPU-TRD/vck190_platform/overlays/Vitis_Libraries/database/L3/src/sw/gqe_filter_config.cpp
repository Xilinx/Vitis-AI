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
#include "xf_database/gqe_filter_config.hpp"
#include "xf_database/kernel_command.hpp"
#define USER_DEBUG 1
namespace xf {
namespace database {
namespace gqe {

// to setup the filter config, which includes:
//- sw_cfg: sw_shuffle cfg for scan and write
//- krn_cfg: gqe kernel used config
BloomFilterConfig::BloomFilterConfig(Table tab_in,
                                     std::string filter_condition,
                                     std::string input_str, // comma separated
                                     uint64_t bf_size,
                                     Table tab_out,
                                     std::string output_str) { // comma separated
    // 1) read in col names of input table
    std::vector<std::string> in_col_names = tab_in.getColNames();
    // verify the column num in input table is no more than 3
    CHECK_0(in_col_names, 3, "tab_in");
    // remove the space in str
    xf::database::internals::filter_config::trim(input_str);
    xf::database::internals::filter_config::trim(output_str);

#ifdef USER_DEBUG
    std::cout << "1. Get cols from table" << std::endl;
    std::cout << std::endl;
    for (size_t i = 0; i < in_col_names.size(); i++) {
        std::cout << in_col_names[i] << " ";
    }
    std::cout << std::endl << "------------------" << std::endl;
#endif

    // 2) extract filter key col and write out col from input info
    //  filter key(s) extracted from input table
    std::vector<std::string> filter_keys;
    filter_keys = extractKey(input_str);

    // write out column name(s) extracted from table out
    std::vector<std::string> write_out_cols;
    write_out_cols = extractWcol(output_str);
    if (write_out_cols.size() > 3) {
        std::cout << "No more than 3 columns can be output since we only have 3 input columns at most\n";
        exit(1);
    }

    // 3) calc the sw_shuffle_scan cfg
    sw_shuffle_scan.resize(2);
    // sw scan-shuffle, puts key cols to 1st and 2nd col
    ShuffleScan(filter_condition, filter_keys, write_out_cols, in_col_names, sw_shuffle_scan);

#ifdef USER_DEBUG
    std::cout << "3.1. sw_shuffle_scan: ";
    for (size_t i = 0; i < sw_shuffle_scan.size(); i++) {
        std::cout << (int)sw_shuffle_scan[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "after sw_shuffle scan column names: " << std::endl;
    for (size_t i = 0; i < in_col_names.size(); i++) {
        std::cout << in_col_names[i] << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl << "------------------" << std::endl;
#endif

    // 4) calc the shuffle write cfg by "mimic kernel out col ==> compare with write out col"
    std::vector<std::string> col_out_names;
    int filter_keys_num = filter_keys.size();
    std::cout << "filter_keys_num: " << filter_keys_num << std::endl;
    std::cout << "in_col_names.size(): " << in_col_names.size() << std::endl;
    // if input table have payload
    if (write_out_cols.size() > filter_keys_num) {
        col_out_names.push_back(in_col_names[2]);
    } else {
        col_out_names.push_back("unused");
    }
    // kernel output column 1 is unused in gqeFilter
    col_out_names.push_back("unused");
    // key0 + key1 (if existed) to kernel output column 2 + 3
    std::copy(in_col_names.begin(), in_col_names.begin() + filter_keys_num, std::back_inserter(col_out_names));
    if (filter_keys_num == 1) {
        col_out_names.push_back("unused");
    }
#ifdef USER_DEBUG
    std::cout << "4.1. After bloomfilter, column names: " << std::endl;
    for (size_t i = 0; i < col_out_names.size(); i++) {
        std::cout << col_out_names[i] << " ";
    }
    std::cout << std::endl;
#endif

    // get the shuffle_write config
    // the cols before shuffle_write: col_out_names
    // the cols after shuffle_write: extractWCols result: write_out_cols
    sw_shuffle_write = ShuffleWrite(col_out_names, write_out_cols);

#ifdef USER_DEBUG
    for (size_t i = 0; i < sw_shuffle_write.size(); i++) {
        std::cout << "sw_shuffle_write: " << (int)sw_shuffle_write[i] << std::endl;
    }
    for (size_t i = 0; i < write_out_cols.size(); i++) {
        std::cout << "write_out_cols: " << write_out_cols[i] << std::endl;
    }
    std::cout << "----------cfg setup done---------" << std::endl;
#endif

    // 5) setup kernel cfg, which includes filter setup, input_col_en, wr_col_en
    // input_col_en and wr_col_en are taken from sw_shuffle_scan/write cfg
    SetupKernelConfig(bf_size, filter_condition, filter_keys, sw_shuffle_scan, sw_shuffle_write);
}

/**
 * setup gqeFilter kernel config
 *
 * kernel config is consist of 3 parts:
 * - krn filter setup
 * - scan input col enable, get from sw_shuffle_scan
 * - write out col enable, get from sw_shuffle_write
 */
void BloomFilterConfig::SetupKernelConfig(uint64_t bf_size,
                                          std::string filter_condition,
                                          std::vector<std::string> filter_keys,
                                          std::vector<int8_t> sw_shuffle_scan,
                                          std::vector<int8_t> sw_shuffle_write) {
    using krncmdclass = xf::database::gqe::KernelCommand;
    krncmdclass krncmd = krncmdclass();

    // setup dual key switcher
    if (filter_keys.size() == 2) {
        krncmd.setDualKeyOn();
    }

    // setup scan column enable
    krncmd.setScanColEnable(2, 1, sw_shuffle_scan);

    // setup bloom-filter switcher & size
    if (bf_size > (uint64_t)16 * 1024 * 1024 * 1024) {
        std::cout << "Maximum size of bloom-filter is limited to 16 Gbits\n";
        exit(1);
    }
    krncmd.setBloomfilterOn(bf_size);

    // setup dynamic filter
    if (filter_condition != "") krncmd.setFilter(1, filter_condition);

    // setup the write col en
    krncmd.setJoinWriteColEnable(2, 1, sw_shuffle_write);

    // setup kernel config
    ap_uint<512>* config_bits = krncmd.getConfigBits();
    table_filter_cfg = mm.aligned_alloc<ap_uint<512> >(14);
    memcpy(table_filter_cfg, config_bits, sizeof(ap_uint<512>) * 14);
}

// gets kernel config
ap_uint<512>* BloomFilterConfig::getFilterConfigBits() const {
    return table_filter_cfg;
}

// gets the sw scan-shuffle config which shuffles key col to the front
std::vector<int8_t> BloomFilterConfig::getShuffleScan() const {
    return sw_shuffle_scan;
}
// gets the sw write-shuffle config which shuffles input table columns to the output table columns
std::vector<int8_t> BloomFilterConfig::getShuffleWrite() const {
    return sw_shuffle_write;
}

} // namespace gqe
} // namespace database
} // namespace xf
