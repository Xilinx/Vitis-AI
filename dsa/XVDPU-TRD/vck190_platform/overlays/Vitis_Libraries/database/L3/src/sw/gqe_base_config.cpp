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
#include <map>

#include "xf_database/gqe_base_config.hpp"

#define USER_DEBUG 1

namespace xf {
namespace database {
namespace gqe {

//-----------------------------------------------------------------------------------------------
// detailed implementations

// check the input table columns is no more than supported maximum
void BaseConfig::CHECK_0(std::vector<std::string> str, size_t len, std::string sinfo) {
    if (str.size() > len) {
        std::cout << "Maximum Number of Columns Supported by Table " << sinfo << " is " << len << std::endl;
        exit(1);
    }
}

// align the joined key in two table to the same col name
void BaseConfig::AlignColName(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

// get the ss position in join_str
int BaseConfig::findStrInd(std::vector<std::string> join_str, std::string ss) {
    auto pos_iter = std::find(join_str.begin(), join_str.end(), ss);
    if (pos_iter != join_str.end()) return (pos_iter - join_str.begin());
    return -1;
}

// extract joined key from join_str, the key is stroeed to join_keys[2][N],
// where the first 2 is left/right table. the N is joined key number for left/right table.
// for single key, N = 1; for dual key, N = 2
std::vector<std::vector<std::string> > BaseConfig::extractKeys(std::string join_str) {
    // join keys extracted from two input tables
    std::vector<std::vector<std::string> > join_keys;

    std::vector<std::string> values_0;
    std::vector<std::string> values_1;
    if (join_str == "") {
        std::cout << "WARNING: the join key condition is empty!" << std::endl;
        return join_keys;
    }
    std::vector<std::string> maps;
    std::istringstream f(join_str);
    std::string s;
    while (getline(f, s, ',')) {
        maps.push_back(s);
    }
    // int i = 0;
    for (auto eval_0 : maps) {
        int notation_eq = eval_0.find_first_of("=");
        std::string key_0 = eval_0.substr(0, notation_eq);
        std::string key_1 = eval_0.substr(notation_eq + 1);
        values_0.push_back(key_0);
        values_1.push_back(key_1);
    }
    join_keys.push_back(values_0);
    join_keys.push_back(values_1);
    if (join_keys[0].size() != join_keys[1].size()) {
        std::cout << "Join Input Error!" << std::endl;
        exit(1);
    }
#ifdef USER_DEBUG
    std::cout << "2.1. ExtractKeys " << std::endl;
    for (size_t i = 0; i < join_keys.size(); i++) {
        std::cout << "Join keys in table " << i << std::endl;
        for (size_t j = 0; j < join_keys[i].size(); j++) {
            std::cout << join_keys[i][j] << " ";
        }
        std::cout << std::endl;
    }
    std::cout << "------------------" << std::endl;
#endif
    return join_keys;
}

// extract filter key from input_str, the key is stored to filter_keys[N],
// for single key, N = 1; for dual key, N = 2
std::vector<std::string> BaseConfig::extractKey(std::string input_str) {
    // single key or dual keys
    std::vector<std::string> filter_keys;

    if (input_str == "") {
        std::cout << "WARNING: the input key selection is empty!" << std::endl;
        return filter_keys;
    }
    std::istringstream f(input_str);
    std::string s;
    while (getline(f, s, ',')) {
        filter_keys.push_back(s);
    }
#ifdef USER_DEBUG
    std::cout << "2.1. ExtractKeys " << std::endl;
    for (size_t j = 0; j < filter_keys.size(); j++) {
        std::cout << filter_keys[j] << " ";
    }
    std::cout << "\n------------------" << std::endl;
#endif
    if (filter_keys.size() > 2) {
        std::cout << "ERROR: the input key selection has to be no more than 2\n";
        exit(1);
    }
    return filter_keys;
}

// extract the write out cols from table c string
std::vector<std::string> BaseConfig::extractWcols(std::vector<std::vector<std::string> > join_keys,
                                                  std::string outputs) {
    // write out cols idx extracted from table out, unuseful
    std::vector<std::string> write_out_col_idx;
    // write out cols name extracted from table out
    std::vector<std::string> write_out_cols;

    std::vector<std::string> maps;
    std::istringstream f(outputs);
    std::string s;
    while (getline(f, s, ',')) {
        maps.push_back(s);
    }
    for (auto ss : maps) {
        int notation_eq = ss.find_first_of("=");
        std::string idx_ = ss.substr(0, notation_eq);
        std::string key_ = ss.substr(notation_eq + 1);
        // if include L join keys rep with O join keys
        for (size_t j = 0; j < join_keys[0].size(); j++) {
            size_t found = key_.find(join_keys[1][j]);
            if (found != std::string::npos) {
                AlignColName(key_, join_keys[1][j], join_keys[0][j]);
            }
        }
        write_out_col_idx.push_back(idx_);
        write_out_cols.push_back(key_);
    }
#ifdef USER_DEBUG
    std::cout << "2.2. extractWcols" << std::endl;
    for (size_t i = 0; i < write_out_cols.size(); i++) {
        std::cout << write_out_col_idx[i] << "<--" << write_out_cols[i] << " ";
    }
    std::cout << std::endl << "------------------" << std::endl;
#endif
    return write_out_cols;
}

// extract the write out cols from table c string (for gqeFilter only)
std::vector<std::string> BaseConfig::extractWcol(std::string outputs) {
    // write out cols idx extracted from table out, unuseful
    std::vector<std::string> write_out_col_idx;
    // write out cols name extracted from table out
    std::vector<std::string> write_out_cols;

    std::vector<std::string> maps;
    std::istringstream f(outputs);
    std::string s;
    while (getline(f, s, ',')) {
        maps.push_back(s);
    }
    for (auto ss : maps) {
        int notation_eq = ss.find_first_of("=");
        std::string idx_ = ss.substr(0, notation_eq);
        // which column in input table to be written out to output table
        std::string key_ = ss.substr(notation_eq + 1);
        write_out_col_idx.push_back(idx_);
        write_out_cols.push_back(key_);
    }
#ifdef USER_DEBUG
    std::cout << "2.2. extractWcols" << std::endl;
    for (size_t i = 0; i < write_out_cols.size(); i++) {
        std::cout << write_out_col_idx[i] << "<--" << write_out_cols[i] << " ";
    }
    std::cout << std::endl << "\n------------------" << std::endl;
#endif
    return write_out_cols;
}

// sw scan-shuffle
// - shuffle the key cols to col0/1. the pld col to col2
void BaseConfig::ShuffleScan(std::string& filter_str_,
                             std::vector<std::string> join_keys_,
                             std::vector<std::string> write_out_cols_,
                             std::vector<std::string>& col_names_,
                             std::vector<int8_t>& sw_shuffle_scan_) {
    // re-order the input col with the following orders:
    // join-key, pld
    std::vector<int> idx_keys_; // join key
    std::vector<int> idx_plds_; // pld

    for (size_t i = 0; i < col_names_.size(); ++i) {
        // check if col is key, get the idx
        int found_key_idx = findStrInd(join_keys_, col_names_[i]);
        // save the key idx
        if (found_key_idx != -1) {
            idx_keys_.push_back(i);
        } else {
            idx_plds_.push_back(i);
        }
    }
    assert(idx_keys_.size() < 3);
    assert(idx_plds_.size() < 2);

    // push the cols into cols_tab_ with the order: join_key, plds
    std::vector<int8_t> cols_tab_;
    for (size_t i = 0; i < idx_keys_.size(); ++i) {
        cols_tab_.push_back(idx_keys_[i]);
    }
    if (idx_keys_.size() == 1) {
        cols_tab_.push_back(-1);
    }
    for (size_t i = 0; i < idx_plds_.size(); ++i) {
        cols_tab_.push_back(idx_plds_[i]);
    }
    if (idx_plds_.size() == 0) {
        cols_tab_.push_back(-1);
    }
    assert(cols_tab_.size() == 3);
    //#ifdef USER_DEBUG
    //    for (int i = 0; i < cols_tab_.size(); ++i) {
    //        std::cout << "cols_tab_, col[" << i << "]: " << (int)cols_tab_[i] << std::endl;
    //    }
    //#endif

    // the sw shuffle_scan config is ready, aka, cols_tab_
    sw_shuffle_scan_ = cols_tab_;
    // generating corresponding col_names after shuffle_scan
    std::vector<std::string> col_names_helper;
    std::copy(col_names_.begin(), col_names_.end(), std::back_inserter(col_names_helper));

    col_names_.resize(3);
    for (int i = 0; i < 3; i++) {
        if (cols_tab_[i] == -1)
            col_names_[i] = "unused";
        else
            col_names_[i] = col_names_helper[cols_tab_[i]];
    }
    //#ifdef USER_DEBUG
    //    std::cout << "After (SW)Scan shuffle, column names: ";
    //    for (size_t i = 0; i < 3; i++) {
    //        std::cout << col_names_[i] << " ";
    //    }
    //    std::cout << std::endl;
    //#endif
}

// the shuffle used for write out cols, saves the shuffle targeting col
std::vector<int8_t> BaseConfig::ShuffleWrite(std::vector<std::string> col_in_names,
                                             std::vector<std::string> col_wr_names) {
    std::vector<int8_t> shuffle_wr_order;
    for (size_t i = 0; i < col_in_names.size(); i++) {
#ifdef USER_DEBUG
        std::cout << "col_in_names[" << i << "]: " << col_in_names[i] << std::endl;
#endif
        int found = findStrInd(col_wr_names, col_in_names[i]);
        if (found != -1) {
#ifdef USER_DEBUG
            std::cout << "to wr: " << found << ", colname: " << col_wr_names[found] << std::endl;
#endif
            shuffle_wr_order.push_back(found);
        } else {
            shuffle_wr_order.push_back(-1);
        }
    };
    return shuffle_wr_order;
}

void BaseConfig::UpdateRowIDCol(std::string& filter_str_,
                                std::vector<std::string>& col_names_,
                                std::string rowID_str_,
                                std::vector<std::string> strs_) {
    // 1)add rowID col name to col_names
    col_names_[2] = rowID_str_;
    // 2)replace the filter_str with col name
    for (int i = 0; i < 3; i++) {
        AlignColName(filter_str_, col_names_[i], strs_[i]);
    }
}
} // database
} // gqe
} // xf
