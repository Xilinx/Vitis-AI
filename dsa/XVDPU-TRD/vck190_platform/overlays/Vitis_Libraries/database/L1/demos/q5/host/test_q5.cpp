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
#include "table_dt.hpp"
#include "utils.hpp"
#include "prepare.hpp"

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <sys/time.h>

#include "xf_utils_sw/logger.hpp"

// XXX work-around gmp.h issue.
#include <cstddef>

const int PU_NM = 8;

// FIXME temp hack to disable hls_math.h
#define XF_DATABSE_AGGREGATE_H
#define XF_DATABSE_GROUP_AGGREGATE_H
#include "q5kernel.hpp"

#include <ap_int.h>

#ifndef HLS_TEST
#include <xcl2.hpp>
#include <CL/cl_ext_xilinx.h>

#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

#define XCL_BANK0 XCL_BANK(0)
#define XCL_BANK1 XCL_BANK(1)
#define XCL_BANK2 XCL_BANK(2)
#define XCL_BANK3 XCL_BANK(3)
#define XCL_BANK4 XCL_BANK(4)
#define XCL_BANK5 XCL_BANK(5)
#define XCL_BANK6 XCL_BANK(6)
#define XCL_BANK7 XCL_BANK(7)
#define XCL_BANK8 XCL_BANK(8)
#define XCL_BANK9 XCL_BANK(9)
#define XCL_BANK10 XCL_BANK(10)
#define XCL_BANK11 XCL_BANK(11)
#define XCL_BANK12 XCL_BANK(12)
#define XCL_BANK13 XCL_BANK(13)
#define XCL_BANK14 XCL_BANK(14)
#define XCL_BANK15 XCL_BANK(15)

struct rlt_pair {
    std::string name;
    TPCH_INT nationkey;
    long long group_result;
};
std::vector<rlt_pair> query_result;
typedef struct print_buf_result_data_ {
    int i;
    TPCH_INT* v;
    TPCH_INT* price;    // discount
    TPCH_INT* discount; //
    rlt_pair* r;
} print_buf_result_data_t;

void CL_CALLBACK print_buf_result(cl_event event, cl_int cmd_exec_status, void* user_data) {
    print_buf_result_data_t* d = (print_buf_result_data_t*)user_data;

    TPCH_INT* key = d->v;
    TPCH_INT* price = d->price;
    TPCH_INT* discount = d->discount;

    // group-by
    int nm = *(d->v)++;
    for (int i = 0; i < 5; i++) {
        d->r[i].group_result = 0;
    }
    for (int i = 0; i < nm; i++) {
        for (int j = 0; j < 5; j++) {
            if (d->r[j].nationkey == key[i + 16]) {
                d->r[j].group_result += price[i + 16] * (100 - discount[i + 16]);
                break;
            }
        }
    }
    // order-by
    std::vector<rlt_pair> rows;
    for (int i = 0; i < 5; i++) {
        rows.push_back(d->r[i]);
    }
    struct {
        bool operator()(const rlt_pair& a, const rlt_pair& b) const { return a.group_result > b.group_result; }
    } rlt_head2tail;
    std::sort(rows.begin(), rows.end(), rlt_head2tail);

    printf("FGPA result %d:\n", d->i);
    for (int i = 0; i < 5; i++) {
        printf("Name %s: %lld.%lld\n", rows[i].name.c_str(), rows[i].group_result / 10000,
               rows[i].group_result % 10000);
        if (d->i == 0) {
            query_result.push_back(rows[i]);
        }
    }
}
#endif

template <typename T>
int load_dat(void* data, const std::string& name, const std::string& dir, size_t n) {
    if (!data) {
        return -1;
    }
    std::string fn = dir + "/" + name + ".dat";
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
    std::cout << "\n------------ TPC-H Query 5 (1G) -------------\n";

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    ArgParser parser(argc, argv);

    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string work_dir;
    if (!parser.getCmdOption("-work", work_dir)) {
        std::cout << "ERROR: work dir is not set!\n";
        return 1;
    }

    // call data generator
    const int sf = 1;
    std::string in_dir = prepare(work_dir, sf);

#ifdef HLS_TEST
    int num_rep = 1;
#else
    int num_rep = 1;
    std::string num_str;
    if (parser.getCmdOption("-rep", num_str)) {
        try {
            num_rep = std::stoi(num_str);
        } catch (...) {
            num_rep = 1;
        }
    }
    if (num_rep > 20) {
        num_rep = 20;
        std::cout << "WARNING: limited repeat to " << num_rep << " times\n.";
    }
#endif

    const size_t l_depth = L_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_l_orderkey = aligned_alloc<TPCH_INT>(l_depth);
    TPCH_INT* col_l_extendedprice = aligned_alloc<TPCH_INT>(l_depth);
    TPCH_INT* col_l_discount = aligned_alloc<TPCH_INT>(l_depth);
    TPCH_INT* col_l_suppkey = aligned_alloc<TPCH_INT>(l_depth);
    memset(col_l_orderkey, 0, 16);
    memset(col_l_extendedprice, 0, 16);
    memset(col_l_discount, 0, 16);
    memset(col_l_suppkey, 0, 16);

    const size_t o_depth = O_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_o_orderkey = aligned_alloc<TPCH_INT>(o_depth);
    TPCH_INT* col_o_custkey = aligned_alloc<TPCH_INT>(o_depth);
    TPCH_INT* col_o_orderdate = aligned_alloc<TPCH_INT>(o_depth);
    memset(col_o_orderkey, 0, 16);
    memset(col_o_custkey, 0, 16);
    memset(col_o_orderdate, 0, 16);

    const size_t c_depth = C_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_c_nationkey = aligned_alloc<TPCH_INT>(c_depth);
    TPCH_INT* col_c_custkey = aligned_alloc<TPCH_INT>(c_depth);
    memset(col_c_nationkey, 0, 16);
    memset(col_c_custkey, 0, 16);

    const size_t s_depth = S_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_s_suppkey = aligned_alloc<TPCH_INT>(s_depth);
    TPCH_INT* col_s_nationkey = aligned_alloc<TPCH_INT>(s_depth);
    memset(col_s_suppkey, 0, 16);
    memset(col_s_nationkey, 0, 16);

    const size_t r_depth = R_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_r_regionkey = aligned_alloc<TPCH_INT>(r_depth);
    std::array<char, REGION_LEN + 1> col_r_name[r_depth];

    const size_t n_depth = N_MAX_ROW + VEC_LEN * 2 - 1;
    TPCH_INT* col_n_nationkey = aligned_alloc<TPCH_INT>(n_depth);
    TPCH_INT* col_n_regionkey = aligned_alloc<TPCH_INT>(n_depth);
    std::array<char, REGION_LEN + 1> col_n_name[n_depth];

    const size_t result_depth = 180000 + VEC_LEN;
    TPCH_INT* out1_k = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out1_p1 = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out1_p2 = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out1_p3 = aligned_alloc<TPCH_INT>(result_depth);

    TPCH_INT* out2_k = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out2_p1 = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out2_p2 = aligned_alloc<TPCH_INT>(result_depth);
    TPCH_INT* out2_p3 = aligned_alloc<TPCH_INT>(result_depth);

    TPCH_INT* n_out_k = aligned_alloc<TPCH_INT>(32);
    TPCH_INT* in_dummy_1 = aligned_alloc<TPCH_INT>(16);
    TPCH_INT* in_dummy_2 = aligned_alloc<TPCH_INT>(16);
    TPCH_INT* in_dummy_3 = aligned_alloc<TPCH_INT>(16);
    memset(n_out_k, 0, 16);
    memset(in_dummy_1, 0, 16);
    memset(in_dummy_2, 0, 16);
    memset(in_dummy_3, 0, 16);
    std::cout << "Host map buffer has been allocated.\n";

    int l_nrow = L_MAX_ROW;
    int o_nrow = O_MAX_ROW;
    int r_nrow = R_MAX_ROW;
    int n_nrow = N_MAX_ROW;
    int c_nrow = C_MAX_ROW;
    int s_nrow = S_MAX_ROW;

    std::cout << "Lineitem " << l_nrow << " rows\n"
              << "Orders " << o_nrow << " rows\n"
              << "Region " << r_nrow << "rows\n"
              << "Nation " << n_nrow << "rows\n"
              << "Customer " << c_nrow << "rows\n"
              << "Supplier " << s_nrow << "rows\n";

    int err;

    err = load_dat<TPCH_INT>(col_l_orderkey + VEC_LEN, "l_orderkey", in_dir, l_nrow);
    if (err) return err;
    col_l_orderkey[0] = l_nrow;
    err = load_dat<TPCH_INT>(col_l_extendedprice + VEC_LEN, "l_extendedprice", in_dir, l_nrow);
    col_l_extendedprice[0] = l_nrow;
    if (err) return err;
    err = load_dat<TPCH_INT>(col_l_discount + VEC_LEN, "l_discount", in_dir, l_nrow);
    col_l_discount[0] = l_nrow;
    if (err) return err;
    err = load_dat<TPCH_INT>(col_l_suppkey + VEC_LEN, "l_suppkey", in_dir, l_nrow);
    col_l_suppkey[0] = l_nrow;
    if (err) return err;

    std::cout << "Lineitem table has been read from disk\n";

    err = load_dat<TPCH_INT>(col_o_orderkey + VEC_LEN, "o_orderkey", in_dir, o_nrow);
    col_o_orderkey[0] = o_nrow;
    if (err) return err;
    err = load_dat<TPCH_INT>(col_o_custkey + VEC_LEN, "o_custkey", in_dir, o_nrow);
    col_o_custkey[0] = o_nrow;
    if (err) return err;
    err = load_dat<TPCH_INT>(col_o_orderdate + VEC_LEN, "o_orderdate", in_dir, o_nrow);
    if (err) return err;
    col_o_orderdate[0] = o_nrow;

    std::cout << "Orders table has been read from disk\n";

    err = load_dat<TPCH_INT>(col_r_regionkey, "r_regionkey", in_dir, r_nrow);
    if (err) return err;
    err = load_dat<std::array<char, REGION_LEN + 1> >(col_r_name, "r_name", in_dir, r_nrow);
    if (err) return err;
    std::cout << "Region table has been read from disk\n";

    err = load_dat<TPCH_INT>(col_n_regionkey, "n_regionkey", in_dir, n_nrow);
    if (err) return err;
    err = load_dat<TPCH_INT>(col_n_nationkey, "n_nationkey", in_dir, n_nrow);
    if (err) return err;
    err = load_dat<std::array<char, REGION_LEN + 1> >(col_n_name, "n_name", in_dir, n_nrow);
    if (err) return err;
    std::cout << "Nation table has been read from disk\n";

    err = load_dat<TPCH_INT>(col_c_nationkey + VEC_LEN, "c_nationkey", in_dir, c_nrow);
    if (err) return err;
    col_c_nationkey[0] = c_nrow;
    err = load_dat<TPCH_INT>(col_c_custkey + VEC_LEN, "c_custkey", in_dir, c_nrow);
    if (err) return err;
    col_c_custkey[0] = c_nrow;
    std::cout << "Customer table has been read from disk\n";

    err = load_dat<TPCH_INT>(col_s_suppkey + VEC_LEN, "s_suppkey", in_dir, s_nrow);
    if (err) return err;
    col_s_suppkey[0] = s_nrow;
    err = load_dat<TPCH_INT>(col_s_nationkey + VEC_LEN, "s_nationkey", in_dir, s_nrow);
    if (err) return err;
    col_s_nationkey[0] = s_nrow;
    std::cout << "Supplier table has been read from disk\n";

    ap_uint<8 * KEY_SZ * 4>* stb_buf[PU_NM];
    for (int i = 0; i < PU_NM; i++) {
        stb_buf[i] = aligned_alloc<ap_uint<8 * KEY_SZ * 4> >(BUFF_DEPTH);
    }

    // filter r table and join r with n
    struct timeval tv_r_s, tv_r_e;
    gettimeofday(&tv_r_s, 0);
    const char* filter_con = "MIDDLE EAST";
    TPCH_INT* col_r_regionkey_filter = (TPCH_INT*)malloc(r_nrow * sizeof(TPCH_INT));
    int filter_out_nm = 0;
    rlt_pair result[5];
    for (int r = 0; r < r_nrow; ++r) {
        std::array<char, REGION_LEN + 1> r_name = col_r_name[r];
#ifdef DEBUG
        std::string s(r_name.data());
        std::cout << "r_name = " << s << std::endl;
#endif
        if (!strcmp(filter_con, r_name.data())) {
            col_r_regionkey_filter[filter_out_nm++] = col_r_regionkey[r];
        }
    }
    std::cout << "Filtered out " << filter_out_nm << std::endl;
    int n_joined_nm = 0;

    for (int n = 0; n < n_nrow; ++n) {
        for (int r = 0; r < filter_out_nm; ++r) {
            if (col_n_regionkey[n] == col_r_regionkey_filter[r]) {
                std::array<char, REGION_LEN + 1> n_name = col_n_name[n];
                std::string ss(n_name.data());
                result[n_joined_nm].name = ss;
                result[n_joined_nm].nationkey = col_n_nationkey[n];
                n_out_k[VEC_LEN + n_joined_nm++] = col_n_nationkey[n];
            }
        }
    }
    gettimeofday(&tv_r_e, 0);
    std::cout << "CPU execution time of R and N join " << tvdiff(&tv_r_s, &tv_r_e) / 1000 << " ms" << std::endl;
    n_out_k[0] = n_joined_nm;
    std::cout << "Region and Nation joined out " << n_joined_nm << std::endl;
    struct timeval tv_c_s, tv_c_e;
    gettimeofday(&tv_c_s, 0);
    int c_out_nm = 0;
    for (int i = 16; i < c_nrow + 16; ++i) {
        for (int j = 16; j < n_joined_nm + 16; ++j) {
            if (n_out_k[j] == col_c_nationkey[i]) {
                out1_k[c_out_nm] = col_c_custkey[j];
                out1_p1[c_out_nm] = col_c_nationkey[j];
                ++c_out_nm;
                break;
            }
        }
    }
    gettimeofday(&tv_c_e, 0);
    std::cout << "CPU execution time of C join " << tvdiff(&tv_c_s, &tv_c_e) / 1000 << " ms" << std::endl;
    std::cout << "Customer joined out " << c_out_nm << std::endl;

#ifdef HLS_TEST

    q5_hash_join((ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)n_out_k, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)in_dummy_1,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_c_nationkey,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_c_custkey, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)in_dummy_2,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)in_dummy_3, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_k,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p1, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p2,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p3, stb_buf[0], stb_buf[1], stb_buf[2], stb_buf[3],
                 stb_buf[4], stb_buf[5], stb_buf[6], stb_buf[7],
                 228, // 0+1*4+2*16+3*64
                 0, 3);
    int c_joined_nm = out1_k[0];
#ifdef DEBUG
    for (int i = 0; i < 10 /*c_joined_nm*/; i++) {
        // std::cout<<"c_custkey="<<out1_k[i+16]<<std::endl;
        std::cout << "c_nationkey=" << out1_p1[i + 16] << std::endl;
    }
#endif
    std::cout << "Customer joined out " << c_joined_nm << std::endl;
    q5_hash_join((ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_k, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p1,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_o_custkey,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_o_orderkey,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_o_orderdate, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)in_dummy_3,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_k, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p1,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p2, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p3, stb_buf[0],
                 stb_buf[1], stb_buf[2], stb_buf[3], stb_buf[4], stb_buf[5], stb_buf[6], stb_buf[7],
                 156, // 0+3*4+1*16+2*64)
                 1,
                 1); // enable filter
    int o_joined_nm = out2_k[0];
#ifdef DEBUG
    for (int i = 0; i < 10; ++i) {
        std::cout << "c_nationkey=" << out2_p1[i + VEC_LEN] << std::endl;
    }
#endif
    std::cout << "Orders joined out " << o_joined_nm << std::endl;
    q5_hash_join(
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_k, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p1,
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_l_orderkey, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_l_suppkey,
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_l_extendedprice, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_l_discount,
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_k, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p1,
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p2, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p3, stb_buf[0],
        stb_buf[1], stb_buf[2], stb_buf[3], stb_buf[4], stb_buf[5], stb_buf[6], stb_buf[7],
        228, // 0+1*4+2*16+3*64
        0, 0);
    int l_joined_nm = out1_k[0];
#ifdef DEBUE
    for (int i = 0; i < 10; ++i) {
        // std::cout<<"l_suppkey="<<out1_k[i+VEC_LEN]<<std::endl;
        std::cout << "c_nationkey=" << out1_p3[i + VEC_LEN] << std::endl;
    }
#endif
    std::cout << "Lineitem joined out " << l_joined_nm << std::endl;
    q5_hash_join((ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_s_suppkey,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_s_nationkey, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_k,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p1, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p2,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out1_p3, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_k,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p1, (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p2,
                 (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)out2_p3, stb_buf[0], stb_buf[1], stb_buf[2], stb_buf[3],
                 stb_buf[4], stb_buf[5], stb_buf[6], stb_buf[7],
                 210, // 2+0*4+1*16+3*64
                 0,
                 2); // use_two_key
    int s_joined_nm = out2_k[0];
#ifdef DEBUE
    for (int i = 0; i < 10; ++i) {
        std::cout << "c_nationkey=" << out2_k[i + VEC_LEN] << std::endl;
    }
#endif
    std::cout << "Supplier joined out " << s_joined_nm << std::endl;

#else // HLS_TEST

    // Get CL devices.
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];

    // Create context and command queue for selected device
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    cl::CommandQueue q(context, device,
                       // CL_QUEUE_PROFILING_ENABLE);
                       CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    std::cout << "Selected Device " << devName << "\n";

    cl::Program::Binaries xclBins = xcl::import_binary_file(xclbin_path);
    devices.resize(1);
    cl::Program program(context, devices, xclBins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel kernel0(program, "q5_hash_join", &err); // XXX must match
    logger.logCreateKernel(err);
    std::cout << "Kernel has been created\n";

    cl_mem_ext_ptr_t mext_n_out_k = {0, n_out_k, kernel0()};

    cl_mem_ext_ptr_t mext_c_nationkey = {2, col_c_nationkey, kernel0()};
    cl_mem_ext_ptr_t mext_c_custkey = {3, col_c_custkey, kernel0()};

    cl_mem_ext_ptr_t mext_o_custkey = {2, col_o_custkey, kernel0()};
    cl_mem_ext_ptr_t mext_o_orderkey = {3, col_o_orderkey, kernel0()};
    cl_mem_ext_ptr_t mext_o_orderdate = {4, col_o_orderdate, kernel0()};

    cl_mem_ext_ptr_t mext_l_orderkey = {2, col_l_orderkey, kernel0()};
    cl_mem_ext_ptr_t mext_l_suppkey = {3, col_l_suppkey, kernel0()};
    cl_mem_ext_ptr_t mext_l_extendedprice = {4, col_l_extendedprice, kernel0()};
    cl_mem_ext_ptr_t mext_l_discount = {5, col_l_discount, kernel0()};

    cl_mem_ext_ptr_t mext_s_suppkey = {0, col_s_suppkey, kernel0()};
    cl_mem_ext_ptr_t mext_s_nationkey = {1, col_s_nationkey, kernel0()};

    cl_mem_ext_ptr_t mext_out1_k = {6, out1_k, kernel0()};
    cl_mem_ext_ptr_t mext_out1_p1 = {7, out1_p1, kernel0()};
    cl_mem_ext_ptr_t mext_out1_p2 = {8, out1_p2, kernel0()};
    cl_mem_ext_ptr_t mext_out1_p3 = {9, out1_p3, kernel0()};

    cl_mem_ext_ptr_t mext_out2_k = {6, out2_k, kernel0()};
    cl_mem_ext_ptr_t mext_out2_p1 = {7, out2_p1, kernel0()};
    cl_mem_ext_ptr_t mext_out2_p2 = {8, out2_p2, kernel0()};
    cl_mem_ext_ptr_t mext_out2_p3 = {9, out2_p3, kernel0()};

    cl_mem_ext_ptr_t mext_dummy1 = {1, in_dummy_1, kernel0()};
    cl_mem_ext_ptr_t mext_dummy2 = {4, in_dummy_2, kernel0()};
    cl_mem_ext_ptr_t mext_dummy3 = {5, in_dummy_3, kernel0()};

    cl_mem_ext_ptr_t memExt[PU_NM];

    for (int i = 0; i < PU_NM; ++i) {
        memExt[i] = {10 + i, stb_buf[i], kernel0()};
    }

    // Map buffers
    // a

    cl::Buffer buf_n_out_k_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                             (size_t)(TPCH_INT_SZ * 32), &mext_n_out_k);

    cl::Buffer buf_c_custkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * c_depth), &mext_c_custkey);
    cl::Buffer buf_c_nationkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * c_depth), &mext_c_nationkey);

    cl::Buffer buf_o_custkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * o_depth), &mext_o_custkey);
    cl::Buffer buf_o_orderkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(TPCH_INT_SZ * o_depth), &mext_o_orderkey);
    cl::Buffer buf_o_orderdate_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * o_depth), &mext_o_orderdate);

    cl::Buffer buf_l_orderkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(TPCH_INT_SZ * l_depth), &mext_l_orderkey);
    cl::Buffer buf_l_suppkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * l_depth), &mext_l_suppkey);
    cl::Buffer buf_l_extendedprice_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(TPCH_INT_SZ * l_depth), &mext_l_extendedprice);
    cl::Buffer buf_l_discout_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * l_depth), &mext_l_discount);

    cl::Buffer buf_s_suppkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * s_depth), &mext_s_suppkey);
    cl::Buffer buf_s_nationkey_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * s_depth), &mext_s_nationkey);

    cl::Buffer buf_out1_k_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_k);
    cl::Buffer buf_out1_p1_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p1);
    cl::Buffer buf_out1_p2_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p2);
    cl::Buffer buf_out1_p3_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p3);

    cl::Buffer buf_out2_k_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_k);
    cl::Buffer buf_out2_p1_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p1);
    cl::Buffer buf_out2_p2_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p2);
    cl::Buffer buf_out2_p3_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p3);

    cl::Buffer buf_dummy1_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy1);
    cl::Buffer buf_dummy2_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy2);
    cl::Buffer buf_dummy3_a(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy3);

    // b (need to copy input)

    cl::Buffer buf_n_out_k_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                             (size_t)(TPCH_INT_SZ * 32), &mext_n_out_k);

    cl::Buffer buf_c_custkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * c_depth), &mext_c_custkey);
    cl::Buffer buf_c_nationkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * c_depth), &mext_c_nationkey);

    cl::Buffer buf_o_custkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * o_depth), &mext_o_custkey);
    cl::Buffer buf_o_orderkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(TPCH_INT_SZ * o_depth), &mext_o_orderkey);
    cl::Buffer buf_o_orderdate_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * o_depth), &mext_o_orderdate);

    cl::Buffer buf_l_orderkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                (size_t)(TPCH_INT_SZ * l_depth), &mext_l_orderkey);
    cl::Buffer buf_l_suppkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * l_depth), &mext_l_suppkey);
    cl::Buffer buf_l_extendedprice_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(TPCH_INT_SZ * l_depth), &mext_l_extendedprice);
    cl::Buffer buf_l_discout_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * l_depth), &mext_l_discount);

    cl::Buffer buf_s_suppkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (size_t)(TPCH_INT_SZ * s_depth), &mext_s_suppkey);
    cl::Buffer buf_s_nationkey_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                 (size_t)(TPCH_INT_SZ * s_depth), &mext_s_nationkey);

    cl::Buffer buf_out1_k_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_k);
    cl::Buffer buf_out1_p1_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p1);
    cl::Buffer buf_out1_p2_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p2);
    cl::Buffer buf_out1_p3_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out1_p3);

    cl::Buffer buf_out2_k_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_k);
    cl::Buffer buf_out2_p1_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p1);
    cl::Buffer buf_out2_p2_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p2);
    cl::Buffer buf_out2_p3_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                             (size_t)(TPCH_INT_SZ * result_depth), &mext_out2_p3);

    cl::Buffer buf_dummy1_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy1);
    cl::Buffer buf_dummy2_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy2);
    cl::Buffer buf_dummy3_b(context, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                            (size_t)(TPCH_INT_SZ * 16), &mext_dummy3);

    std::cout << "DDR buffers have been mapped/copy-and-mapped\n";
    cl::Buffer buff_a[PU_NM];
    cl::Buffer buff_b[PU_NM];
    for (int i = 0; i < PU_NM; i++) {
        buff_a[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                               (size_t)(KEY_SZ * BUFF_DEPTH), &memExt[i]);
        buff_b[i] = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR,
                               (size_t)(KEY_SZ * BUFF_DEPTH), &memExt[i]);
    }

    q.finish();

    std::vector<std::vector<cl::Event> > write_events(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events_c(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events_o(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events_l(num_rep);
    std::vector<std::vector<cl::Event> > kernel_events_s(num_rep);
    std::vector<std::vector<cl::Event> > read_events(num_rep);
    for (int i = 0; i < num_rep; ++i) {
        write_events[i].resize(1);
        kernel_events_c[i].resize(1);
        kernel_events_o[i].resize(1);
        kernel_events_l[i].resize(1);
        kernel_events_s[i].resize(1);
        read_events[i].resize(1);
    }

    std::vector<print_buf_result_data_t> cbd(num_rep);
    std::vector<print_buf_result_data_t>::iterator it = cbd.begin();
    print_buf_result_data_t* cbd_ptr = &(*it);

    struct timeval tv0;
    int exec_us;
    gettimeofday(&tv0, 0);
    for (int i = 0; i < num_rep; ++i) {
        int use_a = i & 1;
        // write data to DDR
        std::vector<cl::Memory> ib;
        if (use_a) {
            ib.push_back(buf_n_out_k_a);
            ib.push_back(buf_c_nationkey_a);
            ib.push_back(buf_c_custkey_a);

            ib.push_back(buf_o_orderkey_a);
            ib.push_back(buf_o_orderdate_a);
            ib.push_back(buf_o_custkey_a);

            ib.push_back(buf_l_suppkey_a);
            ib.push_back(buf_l_orderkey_a);
            ib.push_back(buf_l_extendedprice_a);
            ib.push_back(buf_l_discout_a);

            ib.push_back(buf_s_suppkey_a);
            ib.push_back(buf_s_nationkey_a);

            ib.push_back(buf_dummy1_a);
            ib.push_back(buf_dummy2_a);
            ib.push_back(buf_dummy3_a);
        } else {
            ib.push_back(buf_n_out_k_b);
            ib.push_back(buf_c_nationkey_b);
            ib.push_back(buf_c_custkey_b);

            ib.push_back(buf_o_orderkey_b);
            ib.push_back(buf_o_orderdate_b);
            ib.push_back(buf_o_custkey_b);

            ib.push_back(buf_l_suppkey_b);
            ib.push_back(buf_l_orderkey_b);
            ib.push_back(buf_l_extendedprice_b);
            ib.push_back(buf_l_discout_b);

            ib.push_back(buf_s_suppkey_b);
            ib.push_back(buf_s_nationkey_b);
            ib.push_back(buf_dummy1_b);
            ib.push_back(buf_dummy2_b);
            ib.push_back(buf_dummy3_b);
        }
        if (i > 1) {
            q.enqueueMigrateMemObjects(ib, 0, &read_events[i - 2], &write_events[i][0]);
        } else {
            q.enqueueMigrateMemObjects(ib, 0, nullptr, &write_events[i][0]);
        }
        // set args and enqueue kernel
        if (use_a) {
            int j = 0;
            kernel0.setArg(j++, buf_n_out_k_a);
            kernel0.setArg(j++, buf_dummy1_a);

            kernel0.setArg(j++, buf_c_nationkey_a);
            kernel0.setArg(j++, buf_c_custkey_a);
            kernel0.setArg(j++, buf_dummy2_a);
            kernel0.setArg(j++, buf_dummy3_a);

            kernel0.setArg(j++, buf_out1_k_a);
            kernel0.setArg(j++, buf_out1_p1_a);
            kernel0.setArg(j++, buf_out1_p2_a);
            kernel0.setArg(j++, buf_out1_p3_a);
            kernel0.setArg(j++, buff_a[0]);
            kernel0.setArg(j++, buff_a[1]);
            kernel0.setArg(j++, buff_a[2]);
            kernel0.setArg(j++, buff_a[3]);
            kernel0.setArg(j++, buff_a[4]);
            kernel0.setArg(j++, buff_a[5]);
            kernel0.setArg(j++, buff_a[6]);
            kernel0.setArg(j++, buff_a[7]);
            kernel0.setArg(18, 228);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 3);
            q.enqueueTask(kernel0, &write_events[i], &kernel_events_c[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_out1_k_a);
            kernel0.setArg(j++, buf_out1_p1_a);

            kernel0.setArg(j++, buf_o_custkey_a);
            kernel0.setArg(j++, buf_o_orderkey_a);
            kernel0.setArg(j++, buf_o_orderdate_a);
            kernel0.setArg(j++, buf_dummy3_a);

            kernel0.setArg(j++, buf_out2_k_a);
            kernel0.setArg(j++, buf_out2_p1_a);
            kernel0.setArg(j++, buf_out2_p2_a);
            kernel0.setArg(j++, buf_out2_p3_a);
            kernel0.setArg(18, 156);
            kernel0.setArg(19, 1);
            kernel0.setArg(20, 1);
            q.enqueueTask(kernel0, &kernel_events_c[i], &kernel_events_o[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_out2_k_a);
            kernel0.setArg(j++, buf_out2_p1_a);
            kernel0.setArg(j++, buf_l_orderkey_a);
            kernel0.setArg(j++, buf_l_suppkey_a);
            kernel0.setArg(j++, buf_l_extendedprice_a);
            kernel0.setArg(j++, buf_l_discout_a);

            kernel0.setArg(j++, buf_out1_k_a);
            kernel0.setArg(j++, buf_out1_p1_a);
            kernel0.setArg(j++, buf_out1_p2_a);
            kernel0.setArg(j++, buf_out1_p3_a);
            kernel0.setArg(18, 228);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 0);
            q.enqueueTask(kernel0, &kernel_events_o[i], &kernel_events_l[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_s_suppkey_a);
            kernel0.setArg(j++, buf_s_nationkey_a);
            kernel0.setArg(j++, buf_out1_k_a);
            kernel0.setArg(j++, buf_out1_p1_a);
            kernel0.setArg(j++, buf_out1_p2_a);
            kernel0.setArg(j++, buf_out1_p3_a);

            kernel0.setArg(j++, buf_out2_k_a);
            kernel0.setArg(j++, buf_out2_p1_a);
            kernel0.setArg(j++, buf_out2_p2_a);
            kernel0.setArg(j++, buf_out2_p3_a);
            kernel0.setArg(18, 210);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 2);
            q.enqueueTask(kernel0, &kernel_events_l[i], &kernel_events_s[i][0]);
        } else {
            int j = 0;
            kernel0.setArg(j++, buf_n_out_k_b);
            kernel0.setArg(j++, buf_dummy1_b);

            kernel0.setArg(j++, buf_c_nationkey_b);
            kernel0.setArg(j++, buf_c_custkey_b);
            kernel0.setArg(j++, buf_dummy2_b);
            kernel0.setArg(j++, buf_dummy3_b);

            kernel0.setArg(j++, buf_out1_k_b);
            kernel0.setArg(j++, buf_out1_p1_b);
            kernel0.setArg(j++, buf_out1_p2_b);
            kernel0.setArg(j++, buf_out1_p3_b);
            kernel0.setArg(j++, buff_b[0]);
            kernel0.setArg(j++, buff_b[1]);
            kernel0.setArg(j++, buff_b[2]);
            kernel0.setArg(j++, buff_b[3]);
            kernel0.setArg(j++, buff_b[4]);
            kernel0.setArg(j++, buff_b[5]);
            kernel0.setArg(j++, buff_b[6]);
            kernel0.setArg(j++, buff_b[7]);
            kernel0.setArg(18, 228);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 3);
            q.enqueueTask(kernel0, &write_events[i], &kernel_events_c[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_out1_k_b);
            kernel0.setArg(j++, buf_out1_p1_b);

            kernel0.setArg(j++, buf_o_custkey_b);
            kernel0.setArg(j++, buf_o_orderkey_b);
            kernel0.setArg(j++, buf_o_orderdate_b);
            kernel0.setArg(j++, buf_dummy3_b);

            kernel0.setArg(j++, buf_out2_k_b);
            kernel0.setArg(j++, buf_out2_p1_b);
            kernel0.setArg(j++, buf_out2_p2_b);
            kernel0.setArg(j++, buf_out2_p3_b);
            kernel0.setArg(18, 156);
            kernel0.setArg(19, 1);
            kernel0.setArg(20, 1);
            q.enqueueTask(kernel0, &kernel_events_c[i], &kernel_events_o[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_out2_k_b);
            kernel0.setArg(j++, buf_out2_p1_b);
            kernel0.setArg(j++, buf_l_orderkey_b);
            kernel0.setArg(j++, buf_l_suppkey_b);
            kernel0.setArg(j++, buf_l_extendedprice_b);
            kernel0.setArg(j++, buf_l_discout_b);

            kernel0.setArg(j++, buf_out1_k_b);
            kernel0.setArg(j++, buf_out1_p1_b);
            kernel0.setArg(j++, buf_out1_p2_b);
            kernel0.setArg(j++, buf_out1_p3_b);
            kernel0.setArg(18, 228);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 0);
            q.enqueueTask(kernel0, &kernel_events_o[i], &kernel_events_l[i][0]);
            j = 0;
            kernel0.setArg(j++, buf_s_suppkey_b);
            kernel0.setArg(j++, buf_s_nationkey_b);
            kernel0.setArg(j++, buf_out1_k_b);
            kernel0.setArg(j++, buf_out1_p1_b);
            kernel0.setArg(j++, buf_out1_p2_b);
            kernel0.setArg(j++, buf_out1_p3_b);

            kernel0.setArg(j++, buf_out2_k_b);
            kernel0.setArg(j++, buf_out2_p1_b);
            kernel0.setArg(j++, buf_out2_p2_b);
            kernel0.setArg(j++, buf_out2_p3_b);
            kernel0.setArg(18, 210);
            kernel0.setArg(19, 0);
            kernel0.setArg(20, 2);
            q.enqueueTask(kernel0, &kernel_events_l[i], &kernel_events_s[i][0]);
        }
        // read data from DDR
        std::vector<cl::Memory> ob;
        if (use_a) {
            ob.push_back(buf_out2_k_a);
            ob.push_back(buf_out2_p1_a);
            ob.push_back(buf_out2_p2_a);
        } else {
            ob.push_back(buf_out2_k_b);
            ob.push_back(buf_out2_p1_b);
            ob.push_back(buf_out2_p2_b);
        }
        q.enqueueMigrateMemObjects(ob, CL_MIGRATE_MEM_OBJECT_HOST, &kernel_events_s[i], &read_events[i][0]);

        if (use_a) {
            cbd_ptr[i].i = i;
            cbd_ptr[i].v = (TPCH_INT*)out2_k;
            cbd_ptr[i].price = (TPCH_INT*)out2_p1;
            cbd_ptr[i].discount = (TPCH_INT*)out2_p2;
            cbd_ptr[i].r = result;
            read_events[i][0].setCallback(CL_COMPLETE, print_buf_result, cbd_ptr + i);
        } else {
            cbd_ptr[i].i = i;
            cbd_ptr[i].v = (TPCH_INT*)out2_k;
            cbd_ptr[i].price = (TPCH_INT*)out2_p1;
            cbd_ptr[i].discount = (TPCH_INT*)out2_p2;
            cbd_ptr[i].r = result;
            read_events[i][0].setCallback(CL_COMPLETE, print_buf_result, cbd_ptr + i);
        }
    }

    // wait all to finish.
    q.flush();
    q.finish();

    struct timeval tv3;
    gettimeofday(&tv3, 0);
    exec_us = tvdiff(&tv0, &tv3);
    std::cout << "FPGA execution time of " << num_rep << " runs: " << exec_us / 1000 << " ms\n"
              << "Average execution per run: " << exec_us / num_rep / 1000 << " ms\n";

#endif // HLS_TEST
    std::cout << "---------------------------------------------\n\n";

    // golden compare
    if (l_nrow == 6001215 && o_nrow == 1500000 && r_nrow == 5 && n_nrow == 25 && c_nrow == 150000 && s_nrow == 10000) {
        err = strcmp(query_result[0].name.c_str(), "IRAQ");
        err += strcmp(query_result[1].name.c_str(), "SAUDI ARABIA");
        err += strcmp(query_result[2].name.c_str(), "EGYPT");
        err += strcmp(query_result[3].name.c_str(), "IRAN");
        err += strcmp(query_result[4].name.c_str(), "JORDAN");

        err += query_result[0].group_result == 582325532776 ? 0 : 1;
        err += query_result[1].group_result == 533350133675 ? 0 : 1;
        err += query_result[2].group_result == 532934636888 ? 0 : 1;
        err += query_result[3].group_result == 504877781438 ? 0 : 1;
        err += query_result[4].group_result == 488014573985 ? 0 : 1;

    } else {
        std::cout << "WARNING: unknown test size, result is not checked with provisioned golden data." << std::endl;
    }

    err ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    std::cout << "---------------------------------------------\n\n";

    return err;
}
