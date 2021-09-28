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

#ifndef _GQE_BLOOMFILTER_L3_
#define _GQE_BLOOMFILTER_L3_
// commmon
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <iomanip>
#include <math.h>
// HLS
#include <ap_int.h>
// L1, just for `trim`
#include "xf_database/filter_config.hpp"
// L2
#include "xf_database/gqe_utils.hpp"
// L3
#include "xf_database/gqe_table.hpp"

namespace xf {
namespace database {
namespace gqe {

class BloomFilter {
   private:
    // memory allocator
    gqe::utils::MM mm;

    // false positive probability (FPP)
    const float FPP;

    // size of the bloom-filter in bits
    uint64_t bloom_filter_bits;
    // log2 of the number of bits needed for hash-table
    uint64_t bloom_filter_addr_bits;

    // hash-table of the bloom-filter
    ap_uint<256>** hash_table;

    // calculates the log2 of the number of bits needed for hash-table
    uint64_t calcLogSize(uint64_t num_keys);

    // calculates hash value with the input key(s)
    uint64_t hashLookup3(uint64_t key0, uint64_t key1);

    // gets sub-string index
    int findStrInd(std::vector<std::string> whole_str, std::string sub_str);

    // extracts key names from input string
    std::vector<std::string> extractKeyNames(std::string input_str);

    // get mask with `n` ones on LSB.
    inline uint64_t mask(int n) const { return (~0ULL) << (64 - n) >> (64 - n); }

    // print debug info
    const bool dbg = false;

   public:
    /**
     * @brief constructor of BloomFilter
     *
     * Calculates the size of the bloom-filter based on the number of unique keys and the equation provided in:
     * https://en.wikipedia.org/wiki/Bloom_filter, as well as allocates buffer for the internal hash-table
     *
     * @param num_keys number of unique keys to be built into the hash-table of the bloom-filter
     * @param fpp false positive probability (5% by default)
     *
     */
    BloomFilter(uint64_t num_keys, float fpp = 0.05f);

    /**
     * @brief build the hash-table with the given key column from input table,
     *
     * key_names_str should be comma separated, e.g. "key0, key1"
     *
     * @param tab_in input table
     * @param key_names_str key column names (comma separated) of the input table to be built into hash-table
     *
     */
    void build(Table tab_in, std::string col_names);

    /**
     * @brief merge the input bloom-filter into the current one
     *
     * @param bf_in input bloom-filter
     *
     */
    void merge(BloomFilter& bf_in);

    /**
     * @brief get the bloom-filter hash-table
     *
     * @return hash-table of the bloom-filter
     */
    ap_uint<256>** getHashTable() const;

    /**
     * @brief get the bloom-filter size
     *
     * @return size of the bloom-filter
    */
    uint64_t getBloomFilterSize() const;

}; // end class BloomFilter

} // namespace gqe
} // namespace database
} // namespace xf
#endif
