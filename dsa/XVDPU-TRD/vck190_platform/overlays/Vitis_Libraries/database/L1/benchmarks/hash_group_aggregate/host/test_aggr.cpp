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

#include "hash_aggr_kernel.hpp"
#include "xf_database/enums.hpp"
#include "table_dt.hpp"
#include "utils.hpp"

#include <sys/time.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unordered_map>

#include "xf_utils_sw/logger.hpp"

using namespace std;

#ifndef HLS_TEST

#include <xcl2.hpp>
#define XCL_BANK(n) (XCL_MEM_TOPOLOGY | unsigned(n))

int check_result(ap_uint<1024>* data,
                 int num,
                 ap_uint<4> op,
                 ap_uint<32> key_col,
                 ap_uint<32> pld_col,
                 std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    int nerror = 0;
    int ncorrect = 0;
    ap_uint<8 * KEY_SZ * KEY_COL + 3 * 8 * MONEY_SZ * PLD_COL> result;

    for (int i = 0; i < num; i++) {
        result = data[i](8 * KEY_SZ * KEY_COL + 3 * 8 * MONEY_SZ * PLD_COL - 1, 0);
        ap_uint<3 * 8 * MONEY_SZ> p = result(3 * 8 * MONEY_SZ - 1, 0);

        TPCH_INT key = result(8 * KEY_SZ + 3 * 8 * MONEY_SZ * PLD_COL - 1, 3 * 8 * MONEY_SZ * PLD_COL);
        TPCH_INT pld;

        if (op == xf::database::enums::AOP_MIN || op == xf::database::enums::AOP_COUNT) {
            pld = p(8 * MONEY_SZ - 1, 0);
        } else if (op == xf::database::enums::AOP_COUNTNONZEROS) {
            pld = p(3 * 8 * MONEY_SZ - 1, 2 * 8 * MONEY_SZ);
        } else if (op == xf::database::enums::AOP_SUM || op == xf::database::enums::AOP_MEAN) {
            pld = p(3 * 8 * MONEY_SZ - 1, 8 * MONEY_SZ);
        } else {
            // not supported yet
            pld = 0;
        }

        std::cout << std::hex << "Checking: idx=" << i << " key:" << key << " pld:" << pld << std::endl;

        auto it = ref_map.find(key);
        if (it != ref_map.end()) {
            TPCH_INT golden_pld = it->second;
            if (pld != golden_pld) {
                std::cout << "ERROR! key:" << key << ", pld:" << pld << ", refpld:" << golden_pld << std::endl;
                ++nerror;
            } else {
                ++ncorrect;
            }
        } else {
            std::cout << "ERROR! k:" << key << " does not exist in ref" << std::endl;
        }
    }

    if (nerror == 0) {
        std::cout << "No error found!" << std::endl;
    } else {
        std::cout << "Found " << nerror << " errors!" << std::endl;
    }
    return nerror;
}

typedef struct print_buf_result_data_ {
    int i;
    ap_uint<1024>* aggr_result_buf;
    ap_uint<32>* pu_end_status;
    ap_uint<4> op;
    ap_uint<32> key_column;
    ap_uint<32> pld_column;
    std::unordered_map<TPCH_INT, TPCH_INT>* map0;
    int* r;
} print_buf_result_data_t;

void CL_CALLBACK aggr_kernel_finish(cl_event event, cl_int cmd_exec_status, void* ptr) {
    print_buf_result_data_t* d = (print_buf_result_data_t*)ptr;
    (*(d->r)) += check_result((d->aggr_result_buf), (d->pu_end_status)[3], (d->op), (d->key_column), (d->pld_column),
                              (*(d->map0)));
    std::cout << "kernel done!" << std::endl;
    std::cout << "kernel_result_num=" << (d->pu_end_status)[3] << std::endl;
}

#endif // HLS_TEST

std::unordered_map<TPCH_INT, TPCH_INT> map0, map1, map2;

TPCH_INT group_sum(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second + p; // calculate
#if 0
            if (i < 1024) {
                std::cout << std::hex << "update: idx=" << i << " key=" << k << " i_pld=" << p
                          << " old_pld=" << it->second << " new_pld=" << s << std::endl;
            }
#endif
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));

            if (i < 1024) {
                std::cout << std::hex << "insert: idx=" << i << " key=" << k << " i_pld=" << p << std::endl;
            }
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_cnt(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second + 1;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, 1));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_mean(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_sum,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_cnt,
    std::unordered_map<TPCH_INT, TPCH_INT>& map_mean) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];

        auto sum = map_sum.find(k);
        auto cnt = map_cnt.find(k);
        if (sum != map_cnt.end() && cnt != map_cnt.end()) {
            TPCH_INT s = sum->second + p;
            TPCH_INT c = cnt->second + 1;
            TPCH_INT m = s / c;

            map_sum[k] = s; // update
            map_cnt[k] = c;
            map_mean[k] = m;
        } else {
            // not in hash table, create
            map_sum.insert(std::make_pair(k, p));
            map_cnt.insert(std::make_pair(k, 1));
            map_mean.insert(std::make_pair(k, p));
        }
    }

    return (TPCH_INT)map_sum.size();
}

TPCH_INT group_max(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second > p ? it->second : p;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_min(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = it->second > p ? p : it->second;
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p));
        }
    }

    return (TPCH_INT)ref_map.size();
}

TPCH_INT group_cnt_nz(
    // input
    TPCH_INT* key,
    TPCH_INT* pay,
    int num,
    std::unordered_map<TPCH_INT, TPCH_INT>& ref_map) {
    for (int i = 0; i < num; ++i) {
        TPCH_INT k = key[i];
        TPCH_INT p = pay[i];
        auto it = ref_map.find(k);
        if (it != ref_map.end()) {
            TPCH_INT s = p == 0 ? it->second : (it->second + 1);
            ref_map[k] = s; // update
        } else {
            ref_map.insert(std::make_pair(k, p == 0 ? 0 : 1));
        }
    }

    return (TPCH_INT)ref_map.size();
}

template <typename T>
int generate_data(T* data, int range, size_t n) {
    if (!data) {
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        data[i] = (T)(rand() % range + 1);
    }

    return 0;
}

int main(int argc, const char* argv[]) {
    std::cout << "\n---------- Query with TPC-H 1G Data ----------\n\n";
    std::cout << " select max(l_extendedprice), min(l_extendedprice),\n"
                 "        sum(l_extendedprice), count(l_extendedprice)\n"
                 " from lineitem\n"
                 " group by l_orderkey\n ";
    std::cout << "---------------------------------------------\n";

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    ArgParser parser(argc, argv);

    std::string xclbin_path; // eg. kernel.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string num_str;

    int num_rep = NUM_REP_HOST;
    if (parser.getCmdOption("-rep", num_str)) {
        try {
            num_rep = std::stoi(num_str);
        } catch (...) {
            num_rep = 1;
        }
        if (num_rep > 20) {
            num_rep = 20;
            std::cout << "WARNING: limited repeat to " << num_rep << " times\n.";
        }
    }

    ap_uint<32> key_column = 8;
    if (parser.getCmdOption("-key_column", num_str)) {
        try {
            key_column = std::stoi(num_str);
        } catch (...) {
            key_column = 1;
        }
    }

    ap_uint<32> pld_column = 8;
    if (parser.getCmdOption("-pld_column", num_str)) {
        try {
            pld_column = std::stoi(num_str);
        } catch (...) {
            pld_column = 1;
        }
    }

    int sim_scale = 1;
    if (parser.getCmdOption("-scale", num_str)) {
        try {
            sim_scale = std::stoi(num_str);
        } catch (...) {
            sim_scale = 1;
        }
    }

    ap_uint<4> op = xf::database::enums::AOP_SUM;
    ap_uint<32> opt_type = (op, op, op, op, op, op, op, op);

    const size_t hbm_size = 32 << 20; // 256MB
    const size_t hbm_depth = 4 << 20; // 4M * 512b
    (void)hbm_depth;
    const size_t l_depth = L_MAX_ROW;
    KEY_T* col_l_orderkey = aligned_alloc<KEY_T>(l_depth);
    MONEY_T* col_l_extendedprice = aligned_alloc<MONEY_T>(l_depth);

    std::cout << "Host map Buffer has been allocated.\n";

    int l_nrow = L_MAX_ROW / sim_scale;
    std::cout << "Lineitem " << l_nrow << " rows\n";

    int err;
    err = generate_data<TPCH_INT>(col_l_orderkey, 1000, l_nrow);
    if (err) return err;

    err = generate_data<TPCH_INT>(col_l_extendedprice, 10000000, l_nrow);
    if (err) return err;

    std::cout << "Lineitem table has been read from disk\n";

    // golden reference
    TPCH_INT result_cnt;
    if (op = xf::database::enums::AOP_SUM)
        result_cnt = group_sum(col_l_orderkey, col_l_extendedprice, l_nrow, map0);
    else if (op == xf::database::enums::AOP_MAX)
        result_cnt = group_max(col_l_orderkey, col_l_extendedprice, l_nrow, map0);
    else if (op == xf::database::enums::AOP_MIN)
        result_cnt = group_min(col_l_orderkey, col_l_extendedprice, l_nrow, map0);
    else if (op == xf::database::enums::AOP_COUNT)
        result_cnt = group_cnt(col_l_orderkey, col_l_extendedprice, l_nrow, map0);
    else if (op == xf::database::enums::AOP_COUNTNONZEROS)
        result_cnt = group_cnt_nz(col_l_orderkey, col_l_extendedprice, l_nrow, map0);
    else if (op == xf::database::enums::AOP_MEAN)
        result_cnt = group_mean(col_l_orderkey, col_l_extendedprice, l_nrow, map1, map2, map0);
    else
        result_cnt = 0;

    size_t r_depth = L_MAX_ROW / (1024 / 32) / sim_scale;
    if (r_depth < 1000) r_depth = l_nrow;

    ap_uint<1024>* aggr_result_buf_a; // result
    aggr_result_buf_a = aligned_alloc<ap_uint<1024> >(r_depth);
    ap_uint<1024>* aggr_result_buf_b; // result
    aggr_result_buf_b = aligned_alloc<ap_uint<1024> >(r_depth);

    ap_uint<32>* pu_begin_status_a;
    ap_uint<32>* pu_begin_status_b;
    ap_uint<32>* pu_end_status_a;
    ap_uint<32>* pu_end_status_b;
    pu_begin_status_a = aligned_alloc<ap_uint<32> >(PU_STATUS_DEPTH);
    pu_begin_status_b = aligned_alloc<ap_uint<32> >(PU_STATUS_DEPTH);
    pu_end_status_a = aligned_alloc<ap_uint<32> >(PU_STATUS_DEPTH);
    pu_end_status_b = aligned_alloc<ap_uint<32> >(PU_STATUS_DEPTH);

    pu_begin_status_a[0] = opt_type;
    pu_begin_status_b[0] = opt_type;
    pu_begin_status_a[1] = key_column;
    pu_begin_status_b[1] = key_column;
    pu_begin_status_a[2] = pld_column;
    pu_begin_status_b[2] = pld_column;
    pu_begin_status_a[3] = 0;
    pu_begin_status_b[3] = 0;

    for (int i = 0; i < PU_STATUS_DEPTH; i++) {
        std::cout << std::hex << "read_config: pu_begin_status_a[" << i << "]=" << pu_begin_status_a[i] << std::endl;
        std::cout << std::hex << "read_config: pu_begin_status_b[" << i << "]=" << pu_begin_status_b[i] << std::endl;

        pu_end_status_a[i] = 0;
        pu_end_status_b[i] = 0;
    }

#ifdef HLS_TEST

    ap_uint<512>* ping_buf0; // ping buffer
    ap_uint<512>* ping_buf1; // ping buffer
    ap_uint<512>* ping_buf2; // ping buffer
    ap_uint<512>* ping_buf3; // ping buffer
    ping_buf0 = aligned_alloc<ap_uint<512> >(hbm_depth);
    ping_buf1 = aligned_alloc<ap_uint<512> >(hbm_depth);
    ping_buf2 = aligned_alloc<ap_uint<512> >(hbm_depth);
    ping_buf3 = aligned_alloc<ap_uint<512> >(hbm_depth);

    ap_uint<512>* pong_buf0; // pong buffer
    ap_uint<512>* pong_buf1; // pong buffer
    ap_uint<512>* pong_buf2; // pong buffer
    ap_uint<512>* pong_buf3; // pong buffer
    pong_buf0 = aligned_alloc<ap_uint<512> >(hbm_depth);
    pong_buf1 = aligned_alloc<ap_uint<512> >(hbm_depth);
    pong_buf2 = aligned_alloc<ap_uint<512> >(hbm_depth);
    pong_buf3 = aligned_alloc<ap_uint<512> >(hbm_depth);

    hash_aggr_kernel((ap_uint<8 * KEY_SZ * VEC_LEN>*)col_l_orderkey,
                     (ap_uint<8 * MONEY_SZ * VEC_LEN>*)col_l_extendedprice, l_nrow,

                     pu_begin_status_a, pu_end_status_a,

                     ping_buf0, ping_buf1, ping_buf2, ping_buf3, pong_buf0, pong_buf1, pong_buf2, pong_buf3,

                     aggr_result_buf_a);

    int agg_result_num = pu_end_status_a[3];
    int nerror = 0; // result_cnt!=agg_result_num;

    check_result(aggr_result_buf_a, agg_result_num, op, key_column, pld_column, map0);

    std::cout << "ref_result_num=" << result_cnt << std::endl;
    std::cout << "kernel_result_num=" << agg_result_num << std::endl;

    std::cout << "---------------------------------------------\n" << std::endl;

    free(ping_buf0);
    free(ping_buf1);
    free(ping_buf2);
    free(ping_buf3);
    free(pong_buf0);
    free(pong_buf1);
    free(pong_buf2);
    free(pong_buf3);

    return nerror;

#else

    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);

    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);

    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel kernel0(program, "hash_aggr_kernel", &err); // XXX must match
    logger.logCreateKernel(err);
    std::cout << "Kernel has been created\n";

    cl_mem_ext_ptr_t mext_l_orderkey = {0, col_l_orderkey, kernel0()};
    cl_mem_ext_ptr_t mext_l_extendedprice = {1, col_l_extendedprice, kernel0()};
    cl_mem_ext_ptr_t mext_result_a = {13, aggr_result_buf_a, kernel0()};
    cl_mem_ext_ptr_t mext_result_b = {13, aggr_result_buf_b, kernel0()};
    cl_mem_ext_ptr_t mext_begin_status_a = {3, pu_begin_status_a, kernel0()};
    cl_mem_ext_ptr_t mext_begin_status_b = {3, pu_begin_status_b, kernel0()};
    cl_mem_ext_ptr_t mext_end_status_a = {4, pu_end_status_a, kernel0()};
    cl_mem_ext_ptr_t mext_end_status_b = {4, pu_end_status_b, kernel0()};

    // Map buffers
    // a
    cl::Buffer buf_l_orderkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(KEY_SZ * l_depth), &mext_l_orderkey);

    cl::Buffer buf_l_extendedprice_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(MONEY_SZ * l_depth), &mext_l_extendedprice);

    cl::Buffer buf_result_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                            (size_t)(1024 / 8 * r_depth), &mext_result_a);

    cl::Buffer buf_begin_status_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  (size_t)(32 / 8 * PU_STATUS_DEPTH), &mext_begin_status_a);

    cl::Buffer buf_end_status_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                (size_t)(32 / 8 * PU_STATUS_DEPTH), &mext_end_status_a);

    // b (need to copy input)
    cl::Buffer buf_l_orderkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(KEY_SZ * l_depth), &mext_l_orderkey);

    cl::Buffer buf_l_extendedprice_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_COPY_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(MONEY_SZ * l_depth), &mext_l_extendedprice);

    cl::Buffer buf_result_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                            (size_t)(1024 / 8 * r_depth), &mext_result_b);

    cl::Buffer buf_begin_status_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                  (size_t)(32 / 8 * PU_STATUS_DEPTH), &mext_begin_status_b);

    cl::Buffer buf_end_status_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                (size_t)(32 / 8 * PU_STATUS_DEPTH), &mext_end_status_b);

    const int PU_NM = 4;
    cl::Buffer buf_ping[PU_NM];
    cl::Buffer buf_pong[PU_NM];
    std::vector<cl::Memory> tb;
    for (int i = 0; i < PU_NM; i++) {
        // ping
        cl_mem_ext_ptr_t me_ping = {0};
        me_ping.banks = XCL_BANK(i * 4);
        buf_ping[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, (size_t)hbm_size, &me_ping);
        tb.push_back(buf_ping[i]);
        // pong
        cl_mem_ext_ptr_t me_pong = {0};
        me_pong.banks = XCL_BANK(i * 4 + 2);
        buf_pong[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX, (size_t)hbm_size, &me_pong);
        tb.push_back(buf_pong[i]);
    }
    q.enqueueMigrateMemObjects(tb, CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, nullptr, nullptr);

    q.finish();
    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";

    std::cout << "Numer of repeat is " << num_rep << "\n";

    struct timeval tv0;
    int exec_us;
    gettimeofday(&tv0, 0);

    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        write_events[i].resize(1);
        kernel_events[i].resize(1);
        read_events[i].resize(1);
    }

    /*
     * W0-. W1----.     W2-.     W3-.
     *    '-K0--. '-K1-/-. '-K2-/-. '-K3---.
     *          '---R0-  '---R1-  '---R2   '--R3
     */

    std::vector<print_buf_result_data_t> cbd(num_rep);
    std::vector<print_buf_result_data_t>::iterator it = cbd.begin();
    print_buf_result_data_t* cbd_ptr = &(*it);

    int ret = 0;
    for (int i = 0; i < num_rep; ++i) {
        int use_a = i & 1;

        // write data to DDR
        std::vector<cl::Memory> ib;
        if (use_a) {
            ib.push_back(buf_l_orderkey_a);
            ib.push_back(buf_l_extendedprice_a);
            ib.push_back(buf_begin_status_a);
        } else {
            ib.push_back(buf_l_orderkey_b);
            ib.push_back(buf_l_extendedprice_b);
            ib.push_back(buf_begin_status_b);
        }
        if (i > 1) {
            q.enqueueMigrateMemObjects(ib, 0, &read_events[i - 2], &write_events[i][0]);
        } else {
            q.enqueueMigrateMemObjects(ib, 0, nullptr, &write_events[i][0]);
        }

        // set args and enqueue kernel
        if (use_a) {
            int j = 0;
            kernel0.setArg(j++, buf_l_orderkey_a);
            kernel0.setArg(j++, buf_l_extendedprice_a);
            kernel0.setArg(j++, l_nrow);
            kernel0.setArg(j++, buf_begin_status_a);
            kernel0.setArg(j++, buf_end_status_a);
            kernel0.setArg(j++, buf_ping[0]);
            kernel0.setArg(j++, buf_ping[1]);
            kernel0.setArg(j++, buf_ping[2]);
            kernel0.setArg(j++, buf_ping[3]);
            kernel0.setArg(j++, buf_pong[0]);
            kernel0.setArg(j++, buf_pong[1]);
            kernel0.setArg(j++, buf_pong[2]);
            kernel0.setArg(j++, buf_pong[3]);
            kernel0.setArg(j++, buf_result_a);
        } else {
            int j = 0;
            kernel0.setArg(j++, buf_l_orderkey_b);
            kernel0.setArg(j++, buf_l_extendedprice_b);
            kernel0.setArg(j++, l_nrow);
            kernel0.setArg(j++, buf_begin_status_b);
            kernel0.setArg(j++, buf_end_status_b);
            kernel0.setArg(j++, buf_ping[0]);
            kernel0.setArg(j++, buf_ping[1]);
            kernel0.setArg(j++, buf_ping[2]);
            kernel0.setArg(j++, buf_ping[3]);
            kernel0.setArg(j++, buf_pong[0]);
            kernel0.setArg(j++, buf_pong[1]);
            kernel0.setArg(j++, buf_pong[2]);
            kernel0.setArg(j++, buf_pong[3]);
            kernel0.setArg(j++, buf_result_b);
        }
        q.enqueueTask(kernel0, &write_events[i], &kernel_events[i][0]);

        // read data from DDR
        std::vector<cl::Memory> ob;
        if (use_a) {
            ob.push_back(buf_result_a);
            ob.push_back(buf_end_status_a);
        } else {
            ob.push_back(buf_result_b);
            ob.push_back(buf_end_status_b);
        }
        q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events[i], &read_events[i][0]);
        if (use_a) {
            cbd_ptr[i].i = i;
            cbd_ptr[i].aggr_result_buf = aggr_result_buf_a;
            cbd_ptr[i].pu_end_status = pu_end_status_a;
            cbd_ptr[i].op = op;
            cbd_ptr[i].key_column = key_column;
            cbd_ptr[i].pld_column = pld_column;
            cbd_ptr[i].map0 = &map0;
            cbd_ptr[i].r = &ret;
            read_events[i][0].setCallback(CL_COMPLETE, aggr_kernel_finish, cbd_ptr + i);
        } else {
            cbd_ptr[i].i = i;
            cbd_ptr[i].aggr_result_buf = aggr_result_buf_b;
            cbd_ptr[i].pu_end_status = pu_end_status_b;
            cbd_ptr[i].op = op;
            cbd_ptr[i].key_column = key_column;
            cbd_ptr[i].pld_column = pld_column;
            cbd_ptr[i].map0 = &map0;
            cbd_ptr[i].r = &ret;
            read_events[i][0].setCallback(CL_COMPLETE, aggr_kernel_finish, cbd_ptr + i);
        }
    }

    // wait all to finish.
    std::cout << "Kernel Argument have been set\n";
    q.flush();
    q.finish();

    struct timeval tv3;
    gettimeofday(&tv3, 0);
    exec_us = tvdiff(&tv0, &tv3);
    std::cout << std::dec << "FPGA execution time of " << num_rep << " runs: " << exec_us << " usec\n"
              << "Average execution per run: " << exec_us / num_rep << " usec\n";

    for (int i = 0; i < num_rep; ++i) {
        cl_ulong ts, te;
        kernel_events[i][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &ts);
        kernel_events[i][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &te);
        unsigned long t = (te - ts) / 1000;
        printf("INFO: kernel %d: execution time %lu usec\n", i, t);
    }

    for (int i = 0; i < PU_STATUS_DEPTH; i++) {
        std::cout << std::hex << "read_config: pu_end_status_a[" << i << "]=" << pu_end_status_a[i] << std::endl;
        std::cout << std::hex << "read_config: pu_end_status_b[" << i << "]=" << pu_end_status_b[i] << std::endl;
    }

    std::cout << "ref_result_num=" << result_cnt << std::endl;

    free(aggr_result_buf_a);
    free(aggr_result_buf_b);

    ret ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    std::cout << "---------------------------------------------\n";

    return ret;

#endif
}
