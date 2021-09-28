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

#ifndef _GQE_FILTER_CONFIG_L3_
#define _GQE_FILTER_CONFIG_L3_
// commmon
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <iomanip>
// HLS
#include <ap_int.h>
// L2
#include "xf_database/gqe_utils.hpp"
// L3
#include "xf_database/gqe_table.hpp"
#include "xf_database/gqe_base_config.hpp"

namespace xf {
namespace database {
namespace gqe {

class BloomFilterConfig : protected BaseConfig {
   private:
    // sw-shuffle for scan, determins the column order while scanning data in
    std::vector<int8_t> sw_shuffle_scan;
    // sw-shuffle for write out, determins the column order while writing data out
    std::vector<int8_t> sw_shuffle_write;

    // the kernel config passed to gqeFilter kernel
    ap_uint<512>* table_filter_cfg;

    // setup kernel config (table_filter_cfg) for gqeFilter
    void SetupKernelConfig(uint64_t bf_size,
                           std::string filter_condition,
                           std::vector<std::string> filter_keys,
                           std::vector<int8_t> sw_shuffle_scan,
                           std::vector<int8_t> sw_shuffle_write);

   public:
    /**
     * @brief constructor of BloomFilterConfig.
     *
     * This class generates filter configuration bits by paring the .run() arguments
     *
     * @param tab_in input table
     * @param filter_condition filter condition for input table
     * @param input_str column name(s) of input table to be filtered
     * @param bf_size bloom-filter size in bits
     * @param tab_out result table
     * @param output_str output column mapping
     *
     */
    BloomFilterConfig(Table tab_in,
                      std::string filter_condition,
                      std::string input_str, // comma separated
                      uint64_t bf_size,
                      Table tab_out,
                      std::string output_str); // comma separated

    /**
     * @brief get the gqeFilter kernel config
     *
     * @return gqeFilter config bits (14 * ap_uint<512>)
     */
    ap_uint<512>* getFilterConfigBits() const;

    /**
     * @brief get the sw-shuffle config for scan
     *
     * @return the scan sw_shuffle cfg
    */
    std::vector<int8_t> getShuffleScan() const;

    /**
     * @brief get the sw-shuffle config for write out
     *
     * @return the write out sw_shuffle cfg
    */
    std::vector<int8_t> getShuffleWrite() const;

}; // end class BloomFilterConfig

} // namespace gqe
} // namespace database
} // namespace xf
#endif
