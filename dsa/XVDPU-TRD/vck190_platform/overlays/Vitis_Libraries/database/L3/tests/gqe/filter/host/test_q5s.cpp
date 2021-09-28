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
#include "xf_database/gqe_filter.hpp"
#include "xf_database/gqe_bloomfilter.hpp"
#include "prepare.hpp"
#include "x_utils.hpp"
#include <unordered_map>

#include "xf_utils_sw/logger.hpp"

#define VEC_LEN 8
// 1 / BUILD_FACTOR of L table rows will be built into bloom-filter
#define BUILD_FACTOR 10

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

int main(int argc, const char* argv[]) {
    std::cout << "--------------- Query 5 simplified, filter --------------- " << std::endl;

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    x_utils::ArgParser parser(argc, argv);

    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string in_dir;
    if (!parser.getCmdOption("-in", in_dir)) {
        std::cout << "Please provide the path to the input tables\n";
        return 1;
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

    std::string section;
    int sec_l = 1;
    if (parser.getCmdOption("-sec", section)) {
        try {
            sec_l = std::stoi(section);
        } catch (...) {
            sec_l = 1;
        }
    }

    std::vector<std::string> cols_rt;
    cols_rt.push_back("l_orderkey");
    cols_rt.push_back("l_extendedprice");
    cols_rt.push_back("l_discount");
    std::string in_dir_datr = prepare(in_dir, factor_l, cols_rt);
    std::cout << "Read right table form " << in_dir_datr << std::endl;

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

    int sim_scale = 10000;
    if (parser.getCmdOption("-scale", scale)) {
        try {
            sim_scale = std::stoi(scale);
        } catch (...) {
            sim_scale = 10000;
        }
    }

    table_o_nrow /= sim_scale;
    table_l_nrow /= sim_scale;

    std::cout << "Orders SF(" << factor_o << ")\t" << table_o_nrow << " rows\n"
              << "Lineitem SF(" << factor_l << ")\t" << table_l_nrow << " rows\n";

    using namespace xf::database;
    gqe::utils::MM mm;

    // 32-bit data load from tpch data
    int32_t* table_l_in_0 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* table_l_in_1 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* table_l_in_2 = mm.aligned_alloc<int32_t>(table_l_nrow);

    // 64-bit data actually used in gqe-int64 kernel
    int64_t* tab_l_col0 = mm.aligned_alloc<int64_t>(table_l_nrow);
    int64_t* tab_l_col1 = mm.aligned_alloc<int64_t>(table_l_nrow);
    int64_t* tab_l_col2 = mm.aligned_alloc<int64_t>(table_l_nrow);

    int64_t* tab_o1_col0 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);
    int64_t* tab_o1_col1 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);
    int64_t* tab_o1_col2 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);

    int64_t* tab_o2_col0 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);
    int64_t* tab_o2_col1 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);
    int64_t* tab_o2_col2 = mm.aligned_alloc<int64_t>(table_l_nrow / BUILD_FACTOR);

    int64_t table_c_nrow = table_l_nrow;
    int64_t table_c_nrow_depth = (table_c_nrow + VEC_LEN - 1) / VEC_LEN;

    ap_uint<512>* tab_c_col0 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);
    memset(tab_c_col0, 0, table_c_nrow_depth * sizeof(ap_uint<512>));
    ap_uint<512>* tab_c_col1 = mm.aligned_alloc<ap_uint<512> >(table_c_nrow_depth);
    memset(tab_c_col1, 0, table_c_nrow_depth * sizeof(ap_uint<512>));

    int err = 0;
    err += load_dat<int32_t>(table_l_in_0, "l_orderkey", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(table_l_in_1, "l_extendedprice", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(table_l_in_2, "l_discount", in_dir, factor_l, table_l_nrow);
    if (err) return err;
    std::cout << "LineItem table has been read from disk" << std::endl;

    // convert data from 32-bit to 64-bit, for testing only
    for (int i = 0; i < table_l_nrow; ++i) {
        tab_l_col0[i] = table_l_in_0[i];
        tab_l_col1[i] = table_l_in_1[i];
        tab_l_col2[i] = table_l_in_2[i];
    }
    // build 0 - 1/BUILD_FACTOR of table L into bf1
    for (int i = 0; i < table_l_nrow / BUILD_FACTOR; i++) {
        tab_o1_col0[i] = table_l_in_0[i];
        tab_o1_col1[i] = table_l_in_1[i];
        tab_o1_col2[i] = table_l_in_2[i];
    }
    // build 1/BUILD_FACTOR - 2/BUILD_FACTOR of table L into bf2
    for (int i = table_l_nrow / BUILD_FACTOR; i < (table_l_nrow / BUILD_FACTOR) * 2; i++) {
        tab_o2_col0[i - table_l_nrow / BUILD_FACTOR] = table_l_in_0[i];
        tab_o2_col1[i - table_l_nrow / BUILD_FACTOR] = table_l_in_1[i];
        tab_o2_col2[i - table_l_nrow / BUILD_FACTOR] = table_l_in_2[i];
    }

    gqe::Table tab_l("Table L");
    tab_l.addCol("l_orderkey", gqe::TypeEnum::TypeInt64, tab_l_col0, table_l_nrow);
    tab_l.addCol("l_extendedprice", gqe::TypeEnum::TypeInt64, tab_l_col1, table_l_nrow);

    gqe::Table tab_o1("Table O1");
    tab_o1.addCol("l_orderkey", gqe::TypeEnum::TypeInt64, tab_o1_col0, table_l_nrow / BUILD_FACTOR);
    tab_o1.addCol("l_extendedprice", gqe::TypeEnum::TypeInt64, tab_o1_col1, table_l_nrow / BUILD_FACTOR);

    gqe::BloomFilter bf1((uint64_t)table_l_nrow / BUILD_FACTOR * 2);
    bf1.build(tab_o1, "l_orderkey");

    gqe::Table tab_o2("Table O2");
    tab_o2.addCol("l_orderkey", gqe::TypeEnum::TypeInt64, tab_o2_col0, table_l_nrow / BUILD_FACTOR);
    tab_o2.addCol("l_extendedprice", gqe::TypeEnum::TypeInt64, tab_o2_col1, table_l_nrow / BUILD_FACTOR);

    gqe::BloomFilter bf2((uint64_t)table_l_nrow / BUILD_FACTOR * 2);
    bf2.build(tab_o2, "l_orderkey");

    bf1.merge(bf2);

    gqe::Table tab_c("Table C");
    tab_c.addCol("c1", gqe::TypeEnum::TypeInt64, tab_c_col0, table_c_nrow);
    tab_c.addCol("c2", gqe::TypeEnum::TypeInt64, tab_c_col1, table_c_nrow);

    tab_l.info();
    tab_o1.info();
    tab_o2.info();

    gqe::FpgaInit init_ocl(xclbin_path);

    init_ocl.createHostBufs();
    init_ocl.createDevBufs();
    // constructor
    gqe::Filter bigfilter(init_ocl);
    bigfilter.SetBufAllocMaxNum(100);

    gqe::StrategySet params;
    params.sec_l = sec_l;
    gqe::ErrCode err_code;
    err_code = bigfilter.run(tab_l, "l_orderkey", bf1, "", tab_c, "c1=l_extendedprice, c2=l_orderkey", params);

    if (err_code) {
        return err_code;
    }

    std::cout << "Check results on CPU" << std::endl;
    // col0: l_extendedprice; col1: l_orderkey
    int p_nrow = tab_c.getRowNum();
    std::cout << "Output buffer has " << p_nrow << " rows." << std::endl;

    std::cout << "------------Checking result-------------" << std::endl;
    // save filtered key/payload to std::unordered_map for checking
    std::unordered_map<int64_t, int64_t> filtered_pairs;
    for (int n = 0; n < p_nrow / 8; n++) {
        for (int i = 0; i < 8; i++) {
            // l_orderkey is stored to tab_c::col1, see command line bigfilter.run()
            filtered_pairs.insert(std::make_pair<int64_t, int64_t>(tab_c_col1[n](63 + 64 * i, 64 * i),
                                                                   tab_c_col0[n](63 + 64 * i, 64 * i)));
        }
    }
    for (int i = 0; i < p_nrow % 8; i++) {
        // l_orderkey is stored to tab_c::col1, see command line bigfilter.run()
        filtered_pairs.insert(std::make_pair<int64_t, int64_t>(tab_c_col1[p_nrow / 8](63 + 64 * i, 64 * i),
                                                               tab_c_col0[p_nrow / 8](63 + 64 * i, 64 * i)));
    }
    // test each added key in the filtered key list
    int nerror = 0;
    for (int i = 0; i < (table_l_nrow / BUILD_FACTOR) * 2; i++) {
        std::unordered_map<int64_t, int64_t>::const_iterator got = filtered_pairs.find((int64_t)tab_l_col0[i]);
        if (got == filtered_pairs.end()) {
            nerror++;
            std::cout << "Missing key = " << tab_l_col0[i] << " in bloom-filter." << std::endl;
        }
    }
    if (nerror) std::cout << nerror << " errors found in " << table_l_nrow << " inputs.\n";

    nerror ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    return nerror;
}
