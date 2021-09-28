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
#include "xf_database/gqe_aggr.hpp"
#include "prepare.hpp"
#include "x_utils.hpp"
#include "q1.hpp"

#include "xf_utils_sw/logger.hpp"

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
    std::cout << "--------------- Query 1, aggregation --------------- " << std::endl;

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
        in_dir = "db_data/";
    }
    std::vector<std::string> col_names;
    col_names.push_back("l_returnflag");
    col_names.push_back("l_linestatus");
    col_names.push_back("l_extendedprice");
    col_names.push_back("l_quantity");
    col_names.push_back("l_discount");
    col_names.push_back("l_tax");
    col_names.push_back("l_shipdate");

    std::string scale;
    int factor_l = 1;
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

    std::string in_dir_dat = prepare(in_dir, factor_l, col_names);
    std::cout << "Read Input table form " << in_dir_dat << std::endl;

    std::string mode = "manual";
    size_t solution = 2;
    size_t sec_l = 2;
    size_t slice_num = 4;
    size_t log_part = 2;
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
    } else {
        std::cout << "No supported Strategy" << std::endl;
        return 1;
    }

    int32_t table_l_nrow = 6001215;
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

    table_l_nrow /= sim_scale;
    std::cout << "Lineitem SF(" << factor_l << ")\t" << table_l_nrow << " rows\n";
    int table_c_nrow = 1000;
    std::cout << "Result table\t" << table_c_nrow << " rows\n";

    using namespace xf::database;

    gqe::utils::MM mm;
    int32_t* tab_l_col0 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col1 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col2 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col3 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col4 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col5 = mm.aligned_alloc<int32_t>(table_l_nrow);
    int32_t* tab_l_col6 = mm.aligned_alloc<int32_t>(table_l_nrow);

    int64_t* tab_c_col0 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col1 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col2 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col3 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col4 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col5 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col6 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col7 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col8 = mm.aligned_alloc<int64_t>(table_c_nrow);
    int64_t* tab_c_col9 = mm.aligned_alloc<int64_t>(table_c_nrow);

    int err = 0;
    err += load_dat<int32_t>(tab_l_col0, "l_returnflag", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col1, "l_linestatus", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col2, "l_quantity", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col3, "l_extendedprice", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col4, "l_discount", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col5, "l_tax", in_dir, factor_l, table_l_nrow);
    err += load_dat<int32_t>(tab_l_col6, "l_shipdate", in_dir, factor_l, table_l_nrow);
    if (err) return err;
    std::cout << "LineItem table has been read from disk" << std::endl;

    gqe::Table tab_l("Table L");
    tab_l.addCol("l_returnflag", gqe::TypeEnum::TypeInt32, tab_l_col0, table_l_nrow);
    tab_l.addCol("l_linestatus", gqe::TypeEnum::TypeInt32, tab_l_col1, table_l_nrow);
    tab_l.addCol("l_quantity", gqe::TypeEnum::TypeInt32, tab_l_col2, table_l_nrow);
    tab_l.addCol("l_extendedprice", gqe::TypeEnum::TypeInt32, tab_l_col3, table_l_nrow);
    tab_l.addCol("l_discount", gqe::TypeEnum::TypeInt32, tab_l_col4, table_l_nrow);
    tab_l.addCol("l_tax", gqe::TypeEnum::TypeInt32, tab_l_col5, table_l_nrow);
    tab_l.addCol("l_shipdate", gqe::TypeEnum::TypeInt32, tab_l_col6, table_l_nrow);

    gqe::Table tab_c("Table C");
    tab_c.addCol("c0", gqe::TypeEnum::TypeInt64, tab_c_col0, table_c_nrow);
    tab_c.addCol("c1", gqe::TypeEnum::TypeInt64, tab_c_col1, table_c_nrow);
    tab_c.addCol("c2", gqe::TypeEnum::TypeInt64, tab_c_col2, table_c_nrow);
    tab_c.addCol("c3", gqe::TypeEnum::TypeInt64, tab_c_col3, table_c_nrow);
    tab_c.addCol("c4", gqe::TypeEnum::TypeInt64, tab_c_col4, table_c_nrow);
    tab_c.addCol("c5", gqe::TypeEnum::TypeInt64, tab_c_col5, table_c_nrow);
    tab_c.addCol("c6", gqe::TypeEnum::TypeInt64, tab_c_col6, table_c_nrow);
    tab_c.addCol("c7", gqe::TypeEnum::TypeInt64, tab_c_col7, table_c_nrow);
    tab_c.addCol("c8", gqe::TypeEnum::TypeInt64, tab_c_col8, table_c_nrow);
    tab_c.addCol("c9", gqe::TypeEnum::TypeInt64, tab_c_col9, table_c_nrow);

    // constructor
    gqe::Aggregator bigaggr(xclbin_path);

    auto smanual = new gqe::AggrStrategyManualSet(solution, sec_l, slice_num, log_part);
    gqe::ErrCode err_code;
    // input string
    std::string filter_str = "l_shipdate<=19980902";
    std::string group_keys_str = "l_returnflag,l_linestatus";
    std::string ouput_str =
        "c0=l_returnflag, c1=l_linestatus, c2=sum(l_quantity), c3=sum(l_extendedprice),c4=sum(eval0),  "
        "c5=sum(eval1),c6=avg(l_quantity),c7=avg(l_extendedprice),c8=avg(l_discount),c9=count(*)";

    // do aggregation
    err_code = bigaggr.aggregate(tab_l, {{"l_extendedprice * (-l_discount+c2) / 100", {0, 100}},
                                         {"l_extendedprice * (-l_discount+c2) * (l_tax+c3) / 10000", {0, 100, 100}}},
                                 filter_str, group_keys_str, ouput_str, tab_c, smanual);

    std::cout << "Result after OrderBy in CPU:" << std::endl;
    // sort table
    if (!err_code) {
        delete smanual;
    } else {
        return err_code;
    }
    std::cout << "Q1 Demo Finished!" << std::endl;
    int nerror = 0;
    if (validate == "on") {
        std::cout << "---------------Validate results---------------" << std::endl;
        std::cout << "Run Q1 in CPU as Golen" << std::endl;
        int64_t* golden[10];
        int32_t golden_n = 0;
        for (int i = 0; i < 10; i++) {
            golden[i] = mm.aligned_alloc<int64_t>(100);
        }
        cpuQ1(tab_l_col0, tab_l_col1, tab_l_col2, tab_l_col3, tab_l_col4, tab_l_col5, tab_l_col6, table_l_nrow, golden,
              golden_n);
        std::cout << "Q1 CPU run done,Start Validate..." << std::endl;
        nerror = check_result(tab_c, golden, golden_n);
    }
    nerror ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);
    return nerror;
}
