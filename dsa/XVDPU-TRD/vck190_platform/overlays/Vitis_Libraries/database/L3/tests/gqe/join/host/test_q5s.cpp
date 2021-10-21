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
#include "xf_database/gqe_join.hpp"
#include "prepare.hpp"
#include "x_utils.hpp"
#include <unordered_map>

#include "xf_utils_sw/logger.hpp"

#define VEC_LEN 8
#define USER_DEBUG 1

// load one col data into 1 buffer
template <typename T>
int load_dat(void* data, const std::string& name, const std::string& dir, const int sf, const size_t n) {
    if (!data) {
        return -1;
    }
    std::string fn = dir + "/dat" + std::to_string(sf) + "/" + name + ".dat";
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    size_t cnt = fread(data, sizeof(T), n, f);
    fclose(f);
    if (cnt != n) {
        std::cerr << "ERROR: " << cnt << " entries read from " << fn << ", " << n << " entries required." << std::endl;
        return -1;
    }
    return 0;
}

struct golden_data {
    int64_t nrow;
    int64_t sum;
};
golden_data get_golden_sum(int o_row,
                           int64_t* col_o_orderkey,
                           int64_t* col_o_rowid,
                           bool valid_o,
                           char* tab_o_valid,
                           int l_row,
                           int64_t* col_l_orderkey,
                           int64_t* col_l_rowid,
                           bool valid_l,
                           char* tab_l_valid) {
    int64_t sum = 0;
    int64_t cnt = 0;

    std::unordered_multimap<uint64_t, uint64_t> ht1;

    for (int i = 0; i < o_row; ++i) {
        char val_8 = tab_o_valid[i / 8];
        bool valid = (val_8 >> (i % 8)) & 0x1;
        // valid_o ==0: Table O not using validation buffer
        if (valid_o == 0) valid = 1;
        if (valid) {
            int64_t k = col_o_orderkey[i];
            ht1.insert(std::make_pair(k, i + 1));
        }
    }

    // read t once
    for (int i = 0; i < l_row; ++i) {
        int64_t k = col_l_orderkey[i];
        // check hash table
        auto its = ht1.equal_range(k);
        for (auto it = its.first; it != its.second; ++it) {
#ifdef USER_DEBUG
            if (cnt < 16) {
                std::cout << "key: " << k << ", o_rowid: " << it->second << ", l_rowid: " << i + 1 << std::endl;
            }
#endif
            int64_t sum_i = (k * k) % 10000;
            sum += sum_i;
            cnt++;
        }
    }

    golden_data result;

    result.sum = sum;
    result.nrow = cnt;
#ifdef USER_DEBUG
    std::cout << "INFO: CPU ref matched " << cnt << " rows, sum = " << sum << std::endl;
#endif

    return result;
}

int main(int argc, const char* argv[]) {
    std::cout << "--------------- Query 5 simplified, join --------------- " << std::endl;

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    x_utils::ArgParser parser(argc, argv);

    std::string xclbin_path;
#ifndef HLS_TEST
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
#endif
    std::string in_dir;
    if (!parser.getCmdOption("-in", in_dir)) {
        in_dir = "db_data/";
    }

    std::string scale;
    int factor_o = 1;
    int factor_l = 1;
    if (parser.getCmdOption("-O", scale)) {
        try {
            factor_o = std::stoi(scale);
        } catch (...) {
            factor_o = 1;
        }
    }
    if (parser.getCmdOption("-L", scale)) {
        try {
            factor_l = std::stoi(scale);
        } catch (...) {
            factor_l = 1;
        }
    }
    int sim_scale = 1;
    if (parser.getCmdOption("-scale", scale)) {
        try {
            sim_scale = std::stoi(scale);
        } catch (...) {
            sim_scale = 10000;
        }
    }
    std::string validate = "on";
    parser.getCmdOption("-validate", validate);
    if (validate != "on" && validate != "off") validate = "on";

    std::vector<std::string> cols_lt{"o_orderkey", "o_orderdate"};
    std::vector<std::string> cols_rt{"l_orderkey"};

    std::string in_dir_datl = prepare(in_dir, factor_o, cols_lt);
    std::string in_dir_datr = prepare(in_dir, factor_l, cols_rt);
    std::cout << "Read left table form " << in_dir_datl << std::endl;
    std::cout << "Read right table form " << in_dir_datr << std::endl;

    // XXX dat files are still 32bit.
    generate_valid(in_dir, factor_o, "o_valid", "o_orderdate", 32, 19940101UL, 19950101UL);

    size_t solution = 0;
    size_t sec_o = 1; // set 0, use user's input section
    size_t sec_l = 1; // set 0, use user's input section
    size_t slice_num = 1;
    size_t log_part = 2;

    // the coefficiency of partO output buffer expansion
    float coef_exp_partO = 2;
    // the coefficiency of partL output buffer expansion
    float coef_exp_partL = 2;
    // the coefficiency of join output buffer expansion
    float coef_exp_join = 2;

    std::string mode = "manual";
    parser.getCmdOption("-mode", mode);

    if (mode == "manual") {
        std::cout << "Using StrategyManualSet" << std::endl;

        if (parser.getCmdOption("-solution", scale)) {
            try {
                solution = std::stoi(scale);
            } catch (...) {
                solution = 2;
            }
        }
        std::cout << "Select solution:" << solution << std::endl;

        if (solution > 2) {
            std::cout << "No supported Strategy" << std::endl;
            return 1;
        }

        if (parser.getCmdOption("-sec_o", scale)) {
            try {
                sec_o = std::stoi(scale);
            } catch (...) {
                sec_o = 1;
            }
        }
        if (parser.getCmdOption("-sec_l", scale)) {
            try {
                sec_l = std::stoi(scale);
            } catch (...) {
                sec_l = 1;
            }
        }
        if (parser.getCmdOption("-log_part", scale)) {
            try {
                log_part = std::stoi(scale);
            } catch (...) {
                log_part = 2;
            }
        }
        if (solution == 2 && log_part < 2) {
            std::cout << "ERROR: partition number only supports >= 4 for now!!" << std::endl;
            return -1;
        }
        if (parser.getCmdOption("-slice_num", scale)) {
            try {
                slice_num = std::stoi(scale);
            } catch (...) {
                slice_num = 1;
            }
        }
        if (parser.getCmdOption("-coef_exp_partO", scale)) {
            try {
                coef_exp_partO = std::stof(scale);
            } catch (...) {
                coef_exp_partO = 2;
            }
        }
        if (parser.getCmdOption("-coef_exp_partL", scale)) {
            try {
                coef_exp_partL = std::stof(scale);
            } catch (...) {
                coef_exp_partL = 2;
            }
        }
        if (parser.getCmdOption("-coef_exp_join", scale)) {
            try {
                coef_exp_join = std::stof(scale);
            } catch (...) {
                coef_exp_join = 1;
            }
        }
    } else if (mode == "v1") {
        std::cout << "Using StrategyV1" << std::endl;
    } else {
        std::cout << "No supported Strategy" << std::endl;
        return 1;
    }
    int64_t table_o_nrow = 1500000 * factor_o;
    int64_t table_l_nrow = 6001215;
    switch (factor_l) {
        case 1:
            table_l_nrow = 6001215;
            break;
        case 2:
            table_l_nrow = 11997996;
            break;
        case 4:
            table_l_nrow = 23996604;
            break;
        case 8:
            table_l_nrow = 47989007;
            break;
        case 10:
            table_l_nrow = 59986052;
            break;
        case 12:
            table_l_nrow = 71985077;
            break;
        case 20:
            table_l_nrow = 119994608;
            break;
        case 30:
            table_l_nrow = 179998372;
            break;
        case 32:
            table_l_nrow = 192000000;
            break;
        case 33:
            table_l_nrow = 198000000;
            break;
        case 34:
            table_l_nrow = 204000000;
            break;
        case 35:
            table_l_nrow = 210000000;
            break;
        case 36:
            table_l_nrow = 216000000;
            break;
        case 37:
            table_l_nrow = 222000000;
            break;
        case 38:
            table_l_nrow = 228000000;
            break;
        case 39:
            table_l_nrow = 234000000;
            break;
        case 40:
            table_l_nrow = 240012290;
            break;
        case 60:
            table_l_nrow = 360011594;
            break;
        case 80:
            table_l_nrow = 480025129;
            break;
        case 100:
            table_l_nrow = 600037902;
            break;
        case 150:
            table_l_nrow = 900035147;
            break;
        case 200:
            table_l_nrow = 1200018434;
            break;
        case 250:
            table_l_nrow = 1500000714;
            break;
        default:
            table_l_nrow = 6001215;
            std::cerr << "L factor not supported, using SF1" << std::endl;
    }
    if (factor_l > 30 && factor_l < 40) {
        factor_l = 40;
    }
    table_o_nrow /= sim_scale;
    table_l_nrow /= sim_scale;

    std::cout << "Orders SF(" << factor_o << ")\t" << table_o_nrow << " rows\n"
              << "Lineitem SF(" << factor_l << ")\t" << table_l_nrow << " rows\n";

    using namespace xf::database;
    gqe::utils::MM mm;

    // 32-bit data load from tpch data
    int32_t* table_o_in_0 = mm.aligned_alloc<int32_t>(table_o_nrow);
    int32_t* table_o_in_1 = mm.aligned_alloc<int32_t>(table_o_nrow);

    // 64-bit data actually used in gqe-int64 kernel
    int64_t* tab_o_col0 = mm.aligned_alloc<int64_t>(table_o_nrow);
    int64_t* tab_o_col1 = mm.aligned_alloc<int64_t>(table_o_nrow);

    // 32-bit data load from tpch data

    int32_t* table_l_in_0 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* table_l_in_1 = mm.aligned_alloc<int32_t>(table_l_nrow);

    // 64-bit data actually used in gqe-int64 kernel
    int64_t* tab_l_col0 = mm.aligned_alloc<int64_t>(table_l_nrow);
    int64_t* tab_l_col1 = mm.aligned_alloc<int64_t>(table_l_nrow);

    // define the validation buffer
    int64_t tab_o_val_len = (table_o_nrow + 7) / 8;
    char* tab_o_valid = mm.aligned_alloc<char>(tab_o_val_len);

    int64_t tab_l_val_len = (table_l_nrow + 7) / 8;
    char* tab_l_valid = mm.aligned_alloc<char>(tab_l_val_len);
    for (int i = 0; i < tab_l_val_len; i++) {
        tab_l_valid[i] = 0xff;
    }

    int64_t table_c_nrow = coef_exp_join * (float)table_l_nrow;
    int64_t table_c_nrow_depth = (table_c_nrow + VEC_LEN - 1) / VEC_LEN;

    ap_uint<512>* tab_c_col0 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);
    ap_uint<512>* tab_c_col1 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);
    ap_uint<512>* tab_c_col2 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);
    // ap_uint<512>* tab_c_col3 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);

    int err = 0;

    err += load_dat<int32_t>(table_o_in_0, "o_orderkey", in_dir, factor_o, table_o_nrow);
    err += load_dat<char>(tab_o_valid, "o_valid", in_dir, factor_o, tab_o_val_len);
    if (err) return err;
    std::cout << "Orders table has been read from disk" << std::endl;

    // convert data from 32-bit to 64-bit, for testing only
    for (int i = 0; i < table_o_nrow; ++i) {
        tab_o_col0[i] = table_o_in_0[i];
        tab_o_col1[i] = table_o_in_1[i];
        if (i < 64) std::cout << "Key: " << tab_o_col0[i] << ", o_rowid: " << i + 1 << std::endl;
    }

    err += load_dat<int32_t>(table_l_in_0, "l_orderkey", in_dir, factor_l, table_l_nrow);
    // err += load_dat<int32_t>(table_l_in_1, "l_orderkey", in_dir, factor_l, table_l_nrow);
    if (err) return err;
    std::cout << "LineItem table has been read from disk" << std::endl;

    // convert data from 32-bit to 64-bit, for testing only
    for (int i = 0; i < table_l_nrow; ++i) {
        tab_l_col0[i] = table_l_in_0[i];
        tab_l_col1[i] = table_l_in_1[i];
        if (i < 64) std::cout << "Key: " << tab_l_col0[i] << ", l_rowid: " << i + 1 << std::endl;
    }

    bool valid_o = 1;
    bool valid_l = 0;
    gqe::Table tab_o("Table O");
    tab_o.addCol("o_orderkey", gqe::TypeEnum::TypeInt64, tab_o_col0, table_o_nrow);
    // tab_o.addCol("o_orderkey1", gqe::TypeEnum::TypeInt64, tab_o_col1, table_o_nrow);
    tab_o.genRowIDWithValidation("o_rowid", "o_valid", 1, valid_o, tab_o_valid, table_o_nrow); // gen row-id, use valid

    gqe::Table tab_l("Table L");
    tab_l.addCol("l_orderkey", gqe::TypeEnum::TypeInt64, tab_l_col0, table_l_nrow);
    // tab_l.addCol("l_orderkey1", gqe::TypeEnum::TypeInt64, tab_l_col1, table_l_nrow);
    tab_l.genRowIDWithValidation("l_rowid", "l_valid", 1, valid_l, tab_l_valid, table_l_nrow); // gen row-id, use valid

    gqe::Table tab_c("Table C");
    tab_c.addCol("c1", gqe::TypeEnum::TypeInt64, tab_c_col0, table_c_nrow);
    tab_c.addCol("c2", gqe::TypeEnum::TypeInt64, tab_c_col1, table_c_nrow);
    tab_c.addCol("c3", gqe::TypeEnum::TypeInt64, tab_c_col2, table_c_nrow);
    // tab_c.addCol("c4", gqe::TypeEnum::TypeInt64, tab_c_col3, table_c_nrow);

    tab_o.info();
    tab_l.info();

    gqe::FpgaInit init_ocl(xclbin_path);

    init_ocl.createHostBufs();
    init_ocl.createDevBufs();

    // constructor
    gqe::Joiner bigjoin(init_ocl);
    bigjoin.SetBufAllocMaxNum(100);

    gqe::ErrCode err_code;
    if (mode == "v1") {
        // use JoinStrategyV1
        auto sv1 = new gqe::JoinStrategyV1();
        err_code = bigjoin.run(tab_o, "o_rowid>0", tab_l, "", "o_orderkey = l_orderkey", tab_c,
                               "c1=l_orderkey, c2=o_rowid, c3=l_rowid", gqe::INNER_JOIN, sv1);
        if (!err_code) {
            delete sv1;
            std::cout << "Join done" << std::endl;
        } else {
            return err_code;
        }
    } else if (mode == "manual") {
        // use JoinStrategyManualSet
        auto smanual = new gqe::JoinStrategyManualSet(solution, sec_o, sec_l, slice_num, log_part, coef_exp_partO,
                                                      coef_exp_partL, coef_exp_join);

        err_code = bigjoin.run(tab_o, "o_rowid>0", tab_l, "", "o_orderkey = l_orderkey", tab_c,
                               "c1=l_orderkey,c2=o_rowid, c3=l_rowid", gqe::INNER_JOIN, smanual);

        if (!err_code) {
            delete smanual;
        } else {
            return err_code;
        }
    }

    std::cout << "Aggregate in CPU" << std::endl;
    // calculate the aggr results: sum(l_extendedprice + orderkey)
    // col0: l_extendedprice; col1: l_orderkey
    int64_t sum = 0;
    int p_nrow = tab_c.getRowNum();
#ifdef USER_DEBUG
    int cnt = 0;
#endif
    for (int n = 0; n < p_nrow / 8; n++) {
        for (int i = 0; i < 8; i++) {
            int64_t key = tab_c_col0[n](63 + 64 * i, 64 * i);
            int64_t o_rowid = tab_c_col1[n](63 + 64 * i, 64 * i);
            int64_t l_rowid = tab_c_col2[n](63 + 64 * i, 64 * i);

            int64_t sum_i = (key * key) % 10000;
            sum += sum_i;

#ifdef USER_DEBUG
            if (cnt < 16) {
                std::cout << "key: " << key << ", ";
                std::cout << "o_rowid: (" << (o_rowid >> 32) << ", " << (o_rowid & 0xffffffff) << "), ";
                std::cout << "l_rowid: (" << (l_rowid >> 32) << ", " << (l_rowid & 0xffffffff) << ") ";
                std::cout << std::endl;
            }
            cnt++;
#endif
        }
    }
    for (int n = 0; n < p_nrow % 8; n++) {
        int64_t key = tab_c_col0[p_nrow / 8](63 + 64 * n, 64 * n);
        int64_t o_rowid = tab_c_col1[p_nrow / 8](63 + 64 * n, 64 * n);
        int64_t l_rowid = tab_c_col2[p_nrow / 8](63 + 64 * n, 64 * n);
        int64_t sum_i = (key * key) % 10000;
        sum += sum_i;
#ifdef USER_DEBUG
        if (cnt < 16) {
            std::cout << "key: " << key << ", ";
            std::cout << "o_rowid: (" << (o_rowid >> 32) << ", " << (o_rowid & 0xffffffff) << "), ";
            std::cout << "l_rowid: (" << (l_rowid >> 32) << ", " << (l_rowid & 0xffffffff) << ") ";
            std::cout << std::endl;
        }
        cnt++;
#endif
    }
    std::cout << "Q5S Join done, rows: " << p_nrow << ", result: " << sum << std::endl;

    int ret = 0;
    if (validate == "on") {
        std::cout << "---------------------------------Checking result---------------------------------" << std::endl;
        golden_data golden;
        golden = get_golden_sum(table_o_nrow, tab_o_col0, tab_o_col1, valid_o, tab_o_valid, table_l_nrow, tab_l_col0,
                                tab_l_col1, valid_l, tab_l_valid);
        std::cout << "Golen: rows: " << golden.nrow << ", value: " << golden.sum << std::endl;

        ret = (golden.sum == sum) ? 0 : 1;
        ret ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);
    }

    return ret;
}
