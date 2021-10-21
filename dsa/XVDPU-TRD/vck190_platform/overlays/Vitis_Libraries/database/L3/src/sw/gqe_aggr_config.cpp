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

#include "xf_database/gqe_aggr_config.hpp"
#include "xf_database/aggr_command.hpp"

namespace xf {
namespace database {
namespace gqe {

void AggrConfig::CHECK_0(std::vector<std::string> str, size_t len, std::string sinfo) {
    if (str.size() > len) {
        std::cout << "Most Support " << len << " columns in " << sinfo << std::endl;
        exit(1);
    }
}

// check output non-aggr column if group keys
void AggrConfig::CHECK_1(std::string key) {
    if (std::find(group_keys.begin(), group_keys.end(), key) == group_keys.end()) {
        std::cout << "Output columns must in group keys!" << std::endl;
        exit(1);
    }
}

// check input contain all group keys
void AggrConfig::CHECK_2(std::vector<std::string> col_names) {
    for (std::string gkey : group_keys) {
        if (std::find(col_names.begin(), col_names.end(), gkey) == col_names.end()) {
            std::cout << "Input columns must contain all group keys!" << std::endl;
            exit(1);
        }
    }
}

std::string AggrConfig::getAggrInfo(int key_info) {
    switch (key_info) {
        case xf::database::enums::AOP_MIN:
            return "AOP_MIN";
            break;
        case xf::database::enums::AOP_MAX:
            return "AOP_MAX";
            break;
        case xf::database::enums::AOP_SUM:
            return "AOP_SUM";
            break;
        case xf::database::enums::AOP_COUNT:
            return "AOP_COUNT";
            break;
        case xf::database::enums::AOP_COUNTNONZEROS:
            return "AOP_COUNTNONZEROS";
            break;
        case xf::database::enums::AOP_MEAN:
            return "AOP_MEAN";
            break;
        case xf::database::enums::AOP_VARIANCE:
            return "AOP_VARIANCE";
            break;
        case xf::database::enums::AOP_NORML1:
            return "AOP_NORML1";
            break;
        case xf::database::enums::AOP_NORML2:
            return "AOP_NORML2";
    }
    return "INVALID_STR";
}

void AggrConfig::ReplaceAll(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = 0;
    while ((start_pos = str.find(from)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // Handles case where 'to' is a substring of 'from'
    }
}

void AggrConfig::extractGroupKeys(std::string group_keys_str) {
    std::istringstream f(group_keys_str);
    std::string s;
#ifdef USER_DEBUG
    std::cout << "GROUP KEYS:";
#endif
    while (getline(f, s, ',')) {
        group_keys.push_back(s);
#ifdef USER_DEBUG
        std::cout << s << " ";
#endif
    }
    std::cout << std::endl;
}

void AggrConfig::compressCol(std::vector<int8_t>& shuffle,
                             std::vector<std::string>& col_names,
                             std::vector<std::string> ref_cols) {
    for (size_t i = 0; i < shuffle.size(); i++) {
        auto pos_iter = std::find(ref_cols.begin(), ref_cols.end(), col_names[i]);
        if (pos_iter == ref_cols.end()) {
            col_names.erase(col_names.begin() + i);
            shuffle.erase(shuffle.begin() + i);
            i--;
        }
    }
}

void AggrConfig::extractCompressRefKeys(std::string input_str,
                                        std::vector<std::string> col_names,
                                        std::vector<std::string>& ref_col) {
    std::vector<std::string> keys;
    if (input_str == "") return;
    for (size_t i = 0; i < col_names.size(); i++) {
        size_t found = input_str.find(col_names[i]);
        if (found != std::string::npos) {
            keys.push_back(col_names[i]);
        }
    }
    for (auto k : keys) {
        ref_col.push_back(k);
    }
#ifdef USER_DEBUG
    std::cout << "extractKeys From " << input_str << std::endl;
    for (size_t j = 0; j < keys.size(); j++) {
        std::cout << keys[j] << " ";
    }
    std::cout << std::endl;
#endif
}

ap_uint<4> AggrConfig::checkIsAggr(std::string& str) {
    int size = str.length();
    if (size > 4 && str.substr(0, 4) == "sum(") {
        str = str.substr(4, size - 5);
        return xf::database::enums::AOP_SUM;
    }
    if (size > 4 && str.substr(0, 4) == "avg(") {
        str = str.substr(4, size - 5);
        return xf::database::enums::AOP_MEAN;
    }
    if (size > 4 && str.substr(0, 6) == "count(") {
        str = str.substr(6, size - 7);
        return xf::database::enums::AOP_COUNT;
    }
    if (size > 4 && str.substr(0, 4) == "min(") {
        str = str.substr(4, size - 5);
        return xf::database::enums::AOP_MIN;
    }
    if (size > 4 && str.substr(0, 4) == "max(") {
        str = str.substr(4, size - 5);
        return xf::database::enums::AOP_MAX;
    }
    return 0xf;
}

void AggrConfig::extractWcols(std::string outputs, std::vector<std::string>& col_names, bool avg_to_sum) {
    std::vector<std::string> maps;
    std::istringstream f(outputs);
    std::string s;
    while (getline(f, s, ',')) {
        maps.push_back(s);
    }
    output_col_num = maps.size();
#ifdef USER_DEBUG
    std::cout << "-----------------------PRE MERGE1-----------------------" << std::endl;
#endif
    for (unsigned i = 0; i < maps.size(); i++) {
        std::string ss = maps[i];
        unsigned notation_eq = ss.find("=");
        std::string key_1_ = ss;
        if (notation_eq != std::string::npos) {
            key_1_ = ss.substr(notation_eq + 1);
        }
        ap_uint<4> chk_aggr = checkIsAggr(key_1_);
        int col_ind;
        int key_info = 1;

        // if pld aggr, then disable key_info
        if (chk_aggr != 0x0f) {
            key_info = -1;
        } else {
            CHECK_1(key_1_);
        }
        // for count(*)
        if (key_1_ == "*") {
            if (chk_aggr != xf::database::enums::AOP_COUNT) {
                std::cout << "please input col_name instead of *" << std::endl;
                exit(1);
            }
            key_1_ = group_keys[0];
            col_ind = std::find(col_names.begin(), col_names.end(), key_1_) - col_names.begin();
            key_1_ = group_keys[0] + getAggrInfo(chk_aggr); //"_xilix_dup";
            col_names.push_back(key_1_);
            // other operations for group keys
        } else if (std::find(group_keys.begin(), group_keys.end(), key_1_) != group_keys.end() && key_info != 1) {
            col_ind = std::find(col_names.begin(), col_names.end(), key_1_) - col_names.begin();
            key_1_ = key_1_ + getAggrInfo(chk_aggr); //"_xilix_dup";
            col_names.push_back(key_1_);

        } else if (hashGrpInfoMap.find(key_1_) != hashGrpInfoMap.end()) {
            col_ind = std::find(col_names.begin(), col_names.end(), key_1_) - col_names.begin();
            col_ind = hashGrpInfoMap[key_1_].merge_ind;
            key_1_ = key_1_ + getAggrInfo(chk_aggr); //"_xilix_dup";
            col_names.push_back(key_1_);
        } else {
            col_ind = std::find(col_names.begin(), col_names.end(), key_1_) - col_names.begin();
        }
        // find count key easily
        if (avg_to_sum) {
            if (chk_aggr == xf::database::enums::AOP_COUNT) {
                count_key = key_1_;
            }
        }

        struct hashGrpInfo hginfo = {key_1_, (int)i, col_ind, key_info, chk_aggr, -1};
        hashGrpInfoMap[key_1_] = hginfo;
        read_sort_keys.push_back(key_1_);
#ifdef USER_DEBUG
        std::cout << "col_name:" << std::setw(30) << key_1_ << " windex:" << std::setw(2) << i
                  << " mergeid:" << std::setw(2) << col_ind << " keyinfo:" << std::setw(4) << (int)key_info
                  << " aggrinfo:" << std::setw(6) << (int)chk_aggr << std::endl;
#endif
    }
#ifdef USER_DEBUG
    std::cout << "After extractWcols" << std::endl;
    for (size_t i = 0; i < col_names.size(); i++) {
        std::cout << col_names[i] << " ";
    }
    std::cout << std::endl;
#endif
}

std::vector<int8_t> AggrConfig::shuffle(size_t shuf_id,
                                        std::string& eval_str, // or filter_str
                                        std::vector<std::string>& col_names,
                                        std::vector<std::string> strs,
                                        bool replace) {
    // kk.key - the index by findin
    std::vector<int8_t> shuffle0_last;
    std::map<int, int> kk;
    for (size_t i = 0; i < col_names.size(); i++) {
        if (col_names[i] == "undefined") continue;
        if (eval_str == "") {
            shuffle0_last.push_back(i);
        } else {
            size_t found = eval_str.find(col_names[i]);
            if (found != std::string::npos) {
                kk.insert(std::make_pair(found, i));
            } else {
                shuffle0_last.push_back(i);
            }
        }
    }
    std::vector<int8_t> shuffle0;
    std::vector<std::string> col_names_helper;
    std::copy(col_names.begin(), col_names.end(), std::back_inserter(col_names_helper));
    int i = 0;
    for (auto it = kk.begin(); it != kk.end(); ++it) {
        int ind = it->second;
        shuffle0.push_back(ind);
        if (replace) {
            ReplaceAll(eval_str, col_names[ind], strs[i]);
            i++;
        }
    }
    shuffle0.insert(shuffle0.end(), shuffle0_last.begin(), shuffle0_last.end());
    col_names.resize(shuffle0.size());
    for (size_t i = 0; i < shuffle0.size(); i++) {
        col_names[i] = col_names_helper[shuffle0[i]];
    }
    std::vector<std::string> ref_cols;
    ref_cols.insert(ref_cols.end(), ref_col_writeout.begin(), ref_col_writeout.end());
    ref_cols.insert(ref_cols.end(), ref_col_filter.begin(), ref_col_filter.end());
    if (shuf_id < 2) {
        ref_cols.insert(ref_cols.end(), ref_col_evals[1].begin(), ref_col_evals[1].end());
    }
    if (shuf_id == 0) {
        ref_cols.insert(ref_cols.end(), ref_col_evals[0].begin(), ref_col_evals[0].end());
    }
    compressCol(shuffle0, col_names, ref_cols);
    CHECK_0(col_names, 9, "After shuf" + shuf_id);
    return shuffle0;
}

void AggrConfig::setPartList(std::vector<std::string> init_col_names) {
    for (unsigned i = 0; i < group_keys.size(); i++) {
        std::string grp_key = group_keys[i];
        auto it = std::find(init_col_names.begin(), init_col_names.end(), grp_key);
        if (it != init_col_names.end()) {
            part_list.push_back(it - init_col_names.begin());
        } else {
            std::cout << "grp keys must in input cols!" << std::endl;
        }
    }
    for (unsigned i = 0; i < init_col_names.size(); i++) {
        auto it = std::find(group_keys.begin(), group_keys.end(), init_col_names[i]);
        if (it == group_keys.end()) {
            part_list.push_back(i);
        }
    }
#ifdef USER_DEBUG
    std::cout << "-----------------------Part Input List-----------------------" << std::endl;
    for (auto k : init_col_names) {
        std::cout << k << " ";
    }
    for (auto k : part_list) {
        std::cout << (int)k << " ";
    }
    std::cout << std::endl << std::endl;

#endif
}

AggrConfig::AggrConfig(Table tab_a,
                       std::vector<EvaluationInfo> evals_info,
                       std::string filter_str,
                       std::string group_keys_str,
                       std::string output_str,
                       bool avg_to_sum_) {
    avg_to_sum = avg_to_sum_;
    count_key = "";
    std::vector<std::vector<int> > evals_conts;
    std::vector<int> evals_div;
    std::vector<std::string> evals;
    for (auto eval_info : evals_info) {
        int div_scale = 0;
        std::string eval_str = eval_info.eval_str;

        std::string eval_str_all = eval_info.eval_str;
        xf::database::internals::filter_config::trim(eval_str_all);
        size_t find_div = eval_str_all.find("/");
        if (find_div != std::string::npos) {
            eval_str = eval_str_all.substr(0, find_div);
            div_scale = std::stoi(eval_str_all.substr(find_div + 1));
            if (div_scale == 10) {
                div_scale = 4;
            } else if (div_scale == 100) {
                div_scale = 5;
            } else if (div_scale == 1000) {
                div_scale = 6;
            } else if (div_scale == 10000) {
                div_scale = 7;
            } else {
                std::cout << "Invalid DIV!" << std::endl;
                exit(1);
            }
        }
        evals.push_back(eval_str);
        evals_div.push_back(div_scale);
        evals_conts.push_back(eval_info.eval_const);
    }
    xf::database::internals::filter_config::trim(filter_str);
    xf::database::internals::filter_config::trim(output_str);
    xf::database::internals::filter_config::trim(group_keys_str);
    std::vector<std::string> col_names = tab_a.getColNames();
    extractGroupKeys(group_keys_str);
    setPartList(col_names);

    // check input contain all group keys
    CHECK_2(col_names);
    // init write_cols
    for (int i = 0; i < 16; i++) {
        write_flag.push_back(false);
    }
    extractCompressRefKeys(filter_str, col_names, ref_col_filter);
    std::istringstream f(output_str);
    std::string ss;
    while (getline(f, ss, ',')) {
        unsigned notation_eq = ss.find("=");
        std::string key_1_ = ss;
        if (notation_eq != std::string::npos) {
            key_1_ = ss.substr(notation_eq + 1);
        }
        if (key_1_.find("eval0") != std::string::npos) {
            ref_col_writeout.push_back("eval0");
        }
        if (key_1_.find("eval1") != std::string::npos) {
            ref_col_writeout.push_back("eval1");
        }
        extractCompressRefKeys(key_1_, col_names, ref_col_writeout);
    }
    ref_col_evals.resize(2);
    for (unsigned i = 0; i < evals.size(); i++) {
        extractCompressRefKeys(evals[i], col_names, ref_col_evals[i]);
    }

#ifdef USER_DEBUG
    std::cout << "-----------------------INIT COMPRESS-----------------------" << std::endl;
    for (auto k : ref_col_writeout) {
        std::cout << k << " ";
    }
    std::cout << std::endl << std::endl;
    std::cout << "-----------------------ORIGIN COLS-----------------------" << std::endl;
    for (size_t i = 0; i < col_names.size(); i++) {
        std::cout << col_names[i] << " ";
    }
    std::cout << std::endl << std::endl;
#endif
    CHECK_0(col_names, 8, "scan");
    using acmdclass = xf::database::gqe::AggrCommand;
    acmdclass acmd = acmdclass();

    // CHECK EVAL COUNT
    if (evals.size() > 2) {
        std::cout << "Most Support 2 Evaluations" << std::endl;
        exit(1);
    }
    for (size_t i = 0; i < 2; i++) {
        std::string eval_str = "";
        if (i < evals.size()) eval_str = evals[i];
        std::vector<int8_t> eval_list = shuffle(i, eval_str, col_names);
        if (i == 0) {
            // so that suppose scan have shuffle function
            acmd.Scan(eval_list);
            scan_list = eval_list;
        } else if (i == 1) {
            acmd.setShuffle0(eval_list);
        } else {
            std::cout << "most support two evals" << std::endl;
            exit(1);
        }
        for (size_t j = col_names.size(); j < 8; j++) {
            col_names.push_back("undefined");
        }
        col_names.push_back("eval" + std::to_string(i));
        if (i < evals.size()) {
            int div_scal = evals_div[i];
            std::vector<int> eval_const = evals_conts[i];
            acmd.setEvaluation(i, eval_str, eval_const, div_scal);
        }
#ifdef USER_DEBUG
        if (i == 0) std::cout << "-----------------------SCAN------------------------" << std::endl;
        if (i == 1) std::cout << "-----------------------SHUFFLE0------------------------" << std::endl;
        std::cout << evals[i] << "--->" << eval_str << std::endl;
        for (size_t j = 0; j < eval_list.size(); j++) {
            std::cout << (int)eval_list[j] << " ";
        }
        std::cout << std::endl;
        for (size_t j = 0; j < col_names.size(); j++) {
            std::cout << col_names[j] << " ";
        }
        std::cout << std::endl;
#endif
    }

    // acmd.setShuffle1({0, 1, 8, 3, 4, 5, 6, 7});
    // acmd.setFilter("d<=19980902");
    std::vector<int8_t> shuf1_list = shuffle(2, filter_str, col_names, {"a", "b", "c", "d"});
    acmd.setShuffle1(shuf1_list);
    acmd.setFilter(filter_str);
#ifdef USER_DEBUG
    std::cout << "-----------------------SHUFFLE1-----------------------" << std::endl;
    for (size_t i = 0; i < shuf1_list.size(); i++) {
        std::cout << (int)shuf1_list[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < col_names.size(); i++) {
        std::cout << col_names[i] << " ";
    }
    std::cout << std::endl;
    std::cout << filter_str << std::endl;
#endif

    extractWcols(output_str, col_names, avg_to_sum);
    if (avg_to_sum) {
        // for avg_to_sum
        for (auto it = hashGrpInfoMap.begin(); it != hashGrpInfoMap.end(); it++) {
            std::string key = it->first;
            struct hashGrpInfo hginfo = hashGrpInfoMap[key];
            if (hginfo.aggr_info == xf::database::enums::AOP_MEAN) {
                if (key.find("AOP_MEAN") == std::string::npos) {
                    if (hashGrpInfoMap.find(key + "AOP_SUM") != hashGrpInfoMap.end()) {
                        avg_to_sum_map[key] = key + "AOP_SUM";
                    }
                } else {
                    std::string origin_key = key.substr(0, key.length() - 8);
                    // std::cout << key << "->" << origin_key << std::endl;
                    if (hashGrpInfoMap.find(origin_key + "AOP_SUM") != hashGrpInfoMap.end()) {
                        avg_to_sum_map[key] = origin_key + "AOP_SUM";
                    } else if (hashGrpInfoMap[origin_key].aggr_info == xf::database::enums::AOP_SUM) {
                        avg_to_sum_map[key] = origin_key;
                    }
                }
            }
        }
        for (auto it = hashGrpInfoMap.begin(); it != hashGrpInfoMap.end(); it++) {
            std::string key = it->first;
            struct hashGrpInfo hginfo = hashGrpInfoMap[key];
            if (hginfo.aggr_info == xf::database::enums::AOP_MEAN) {
                hashGrpInfoMap[key].route = 1;
                if (avg_to_sum_map.find(key) == avg_to_sum_map.end()) {
                    hashGrpInfoMap[key].aggr_info = xf::database::enums::AOP_SUM;
                    if (count_key == "") {
                        int col_ind = std::find(col_names.begin(), col_names.end(), group_keys[0]) - col_names.begin();
                        count_key = group_keys[0] + "AOP_COUNT";
                        hashGrpInfoMap[count_key] = {count_key, (int)read_sort_keys.size(),     col_ind,
                                                     -1,        xf::database::enums::AOP_COUNT, -1};
                        read_sort_keys.push_back(count_key);
                    }
                } else if (count_key == "") {
                    hashGrpInfoMap[key].aggr_info = xf::database::enums::AOP_COUNT;
                }
            }
        }
#ifdef USER_DEBUG
        std::cout << std::endl;
        std::cout << "Print var_to_sum_map" << std::endl;
        for (auto it = avg_to_sum_map.begin(); it != avg_to_sum_map.end(); it++) {
            std::cout << it->first << "--" << it->second << std::endl;
        }
        std::cout << std::endl;
        std::cout << "count_key: " << count_key << std::endl;
#endif
    }
    std::vector<int8_t> shuf2_list;
    std::vector<int8_t> shuf3_helper_list;
    std::vector<int8_t> shuf3_list;
    std::vector<uint8_t> merge2;
    std::vector<uint8_t> merge3;
    std::vector<int8_t> write_list;
    std::vector<std::string> pld_helper_keys;
    for (unsigned i = 0; i < col_names.size(); i++) {
        std::string key = col_names[i];
        if (hashGrpInfoMap.find(key) != hashGrpInfoMap.end()) {
            struct hashGrpInfo hginfo = hashGrpInfoMap[key];
            // key
            if (hginfo.key_info != -1) {
                shuf2_list.push_back(hginfo.merge_ind); // is i
                key_keys.push_back(col_names[i]);
                // merge id is the index in shuf2_list
                int shuf2_list_counter = shuf2_list.size();
                merge2.push_back(shuf2_list_counter - 1);
                // write_list.push_back(shuf2_list_counter + 7);
            } else { // pld
                ap_uint<4> aggr_info = hginfo.aggr_info;
                if (aggr_info == xf::database::enums::AOP_SUM || key == "eval0" || key == "eval1") {
                    shuf3_helper_list.push_back(hginfo.merge_ind);
                    pld_helper_keys.push_back(col_names[i]);
                } else if (!avg_to_sum || aggr_info != xf::database::enums::AOP_MEAN) {
                    shuf3_list.push_back(hginfo.merge_ind); //
                    pld_keys.push_back(col_names[i]);
                }
                if (aggr_info == xf::database::enums::AOP_COUNT) {
                    merge3.push_back(shuf3_list.size() - 1);
                }
            }
        }
        // else discard unnecessary cols
    }
    int offset = shuf3_list.size() >= shuf2_list.size() ? 0 : (shuf2_list.size() - shuf3_list.size());
    for (int i = 0; i < offset + (int)shuf3_helper_list.size(); i++) {
        if (i < offset) {
            shuf3_list.push_back(-1);
            pld_keys.push_back("undefined");
        } else {
            int ind = i - offset;
            shuf3_list.push_back(shuf3_helper_list[ind]);
            pld_keys.push_back(pld_helper_keys[ind]);
        }
    }

    for (size_t i = 0; i < pld_keys.size(); i++) {
        std::string key = pld_keys[i];
        if (key != "undefined") {
            struct hashGrpInfo hginfo = hashGrpInfoMap[key];
            acmd.setGroupAggr(i, hginfo.aggr_info);

            write_list.push_back(i);
            write_list.push_back(i + 8);
        } else if (i < key_keys.size()) {
            write_list.push_back(i + 8);
        }
    }
    // if key count > pld count
    for (size_t i = pld_keys.size(); i < key_keys.size(); i++) {
        write_list.push_back(i + 8);
    }

    for (auto ind : write_list) {
        if (ind >= 0 && ind < 16) {
            write_flag[ind] = true;
        } else {
            std::cout << "Invalid WriteOut index." << std::endl;
        }
    }

    acmd.setShuffle2(shuf2_list);
    acmd.setShuffle3(shuf3_list);
    acmd.setMerge(2, merge2);
    acmd.setMerge(3, merge3);

#ifdef USER_DEBUG
    std::cout << "-----------------------SHUFFLE2-----------------------" << std::endl;
    for (size_t i = 0; i < shuf2_list.size(); i++) {
        std::cout << (int)shuf2_list[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < key_keys.size(); i++) {
        std::cout << key_keys[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------SHUFFLE3-----------------------" << std::endl;
    for (size_t i = 0; i < shuf3_list.size(); i++) {
        std::cout << (int)shuf3_list[i] << " ";
    }
    std::cout << std::endl;
    for (size_t i = 0; i < pld_keys.size(); i++) {
        std::cout << pld_keys[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------MERGE-----------------------" << std::endl;
    std::cout << "merge key into high bit cols of pld (final 8~15)" << std::endl;
    for (size_t i = 0; i < merge2.size(); i++) {
        std::cout << (int)merge2[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "merge count into low bit cols of pld (final 0~7)" << std::endl;
    for (size_t i = 0; i < merge3.size(); i++) {
        std::cout << (int)merge3[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------SETTING GRP AGGR-----------------------" << std::endl;
    for (size_t i = 0; i < pld_keys.size(); i++) {
        std::string key = pld_keys[i];
        if (key != "undefined") {
            struct hashGrpInfo hginfo = hashGrpInfoMap[key];
            std::cout << std::setw(30) << key << " : " << std::setw(10) << getAggrInfo(hginfo.aggr_info) << std::endl;
        } else {
            std::cout << std::setw(30) << "undefined"
                      << " : " << std::setw(10) << "undefined" << std::endl;
        }
    }
    std::cout << std::endl;
    std::cout << "-----------------------WRITEOUT-----------------------" << std::endl;
    for (size_t i = 0; i < write_list.size(); i++) {
        std::cout << (int)write_list[i] << " ";
    }
    std::cout << std::endl;
#endif
#ifdef USER_DEBUG
    std::cout << "-----------------------SW_SCAN-----------------------" << std::endl;
    for (size_t i = 0; i < scan_list.size(); i++) {
        std::cout << (int)scan_list[i] << " ";
    }
    std::cout << std::endl;
    std::cout << "-----------------------SW_SCAN_SIZE:" << scan_list.size() << "-----------------------" << std::endl;
#endif
    acmd.setWriteCol(write_list);

    table_cfg = acmd.getConfigBits();
    table_out_cfg = acmd.getConfigOutBits();

    table_cfg_part = mm.aligned_alloc<ap_uint<512> >(9);
    memset(table_cfg_part, 0, sizeof(ap_uint<512>) * 9);
    table_cfg_part[0] = 1;
    table_cfg_part[5].range(415, 415) = 1;
    table_cfg_part[8].range(415, 415) = 1;
    if (key_keys.size() == 2) {
        table_cfg_part[0].set_bit(2, 1);
    }
    for (size_t i = 0; i < scan_list.size(); i++) {
        table_cfg_part[0].range(56 + 8 * i + 7, 56 + 8 * i) = i;
    }

    for (size_t i = scan_list.size(); i < 8; i++) {
        table_cfg_part[0].range(56 + 8 * i + 7, 56 + 8 * i) = -1;
    }
}

int AggrConfig::getOutputColNum() const {
    return output_col_num;
}

int AggrConfig::getGrpKeyNum() const {
    return key_keys.size();
}

std::vector<int8_t> AggrConfig::getScanList() const {
    return scan_list;
}

std::vector<int8_t> AggrConfig::getPartList() const {
    return part_list;
}

std::vector<bool> AggrConfig::getWriteFlag() const {
    return write_flag;
}

ap_uint<512>* AggrConfig::getPartConfigBits() const {
    return table_cfg_part;
}

ap_uint<32>* AggrConfig::getAggrConfigBits() const {
    return table_cfg;
}

ap_uint<32>* AggrConfig::getAggrConfigOutBits() const {
    return table_out_cfg;
}

std::vector<int> AggrConfig::getResults(int i) {
    if (i >= output_col_num) {
        std::cout << "OutPut col id must less than " << output_col_num << std::endl;
        exit(1);
    }
    std::vector<int> final_loc;
    std::string key = read_sort_keys[i];
    struct hashGrpInfo hginfo = hashGrpInfoMap[key];
    int final_location = 0;
    // is key
    if (hginfo.key_info != -1) {
        final_location = std::find(key_keys.begin(), key_keys.end(), key) - key_keys.begin();
#ifdef USER_DEBUG
        std::cout << std::setw(30) << key << " : " << final_location + 8 << std::endl;
#endif
        return {final_location + 8};
    } else {
        final_location = std::find(pld_keys.begin(), pld_keys.end(), key) - pld_keys.begin();
        if (avg_to_sum && (hginfo.route == 1)) {
            // must be avg
            if (hginfo.aggr_info == xf::database::enums::AOP_MEAN) {
                // get sum and cout
                std::string sum_key = avg_to_sum_map[key];
                int sum_location = std::find(pld_keys.begin(), pld_keys.end(), sum_key) - pld_keys.begin();
                int count_location = std::find(pld_keys.begin(), pld_keys.end(), count_key) - pld_keys.begin();
#ifdef USER_DEBUG
                std::cout << std::setw(30) << key << " : " << sum_location << "," << (sum_location + 8) << ","
                          << count_location << std::endl;
#endif
                return {sum_location, sum_location + 8, count_location};
            } else if (hginfo.aggr_info == xf::database::enums::AOP_SUM) {
                int count_location = std::find(pld_keys.begin(), pld_keys.end(), count_key) - pld_keys.begin();
#ifdef USER_DEBUG
                std::cout << std::setw(30) << key << " : " << final_location << "," << (final_location + 8) << ","
                          << count_location << std::endl;
#endif
                return {final_location, final_location + 8, count_location};
            } else if (hginfo.aggr_info == xf::database::enums::AOP_COUNT) {
                std::string sum_key = avg_to_sum_map[key];
                int sum_location = std::find(pld_keys.begin(), pld_keys.end(), sum_key) - pld_keys.begin();
#ifdef USER_DEBUG
                std::cout << std::setw(30) << key << " : " << sum_location << "," << (sum_location + 8) << ","
                          << final_location << std::endl;
#endif
                return {sum_location, sum_location + 8, final_location};
            } else {
                std::cout << "Invalid OP" << std::endl;
                exit(1);
            }
        } else {
            if (hginfo.aggr_info == xf::database::enums::AOP_SUM || key == "eval0" || key == "eval1") {
#ifdef USER_DEBUG
                std::cout << std::setw(30) << key << " : " << final_location << "," << (final_location + 8)
                          << std::endl;
#endif
                return {final_location, final_location + 8};
            } else {
#ifdef USER_DEBUG
                std::cout << std::setw(30) << key << " : " << final_location << std::endl;
#endif
                return {final_location};
            }
        }
    }
}

} // gqe
} // database
} // xf
