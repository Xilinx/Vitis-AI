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

// clang-format off
#include "table_dt.hpp"
#include "prepare.hpp"
#include "filter_test.hpp"
#include "filter_kernel.hpp"
#include "utils.hpp"
// clang-format on

// TODO change to a perper size.
#define BUF_CFG_DEPTH 64
#define BUF_L_DEPTH ((L_MAX_ROW + 1023) & (-1UL ^ 0x3ffUL))
#define BUF_O_DEPTH ((O_MAX_ROW + 1023) & (-1UL ^ 0x3ffUL))

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <sys/time.h>

#include "xf_utils_sw/logger.hpp"

#include "xclhost.hpp"
#include "CL/cl_ext_xilinx.h"
#include "cl_errcode.hpp"
#define XBANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

int ret = 0;

enum flt_debug_level { FLT_ERROR, FLT_WARNING, FLT_INFO, FLT_DEBUG, FLT_ALL };

const flt_debug_level debug_level = FLT_INFO;

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

int create_buffers(cl_context ctx,
                   cl_kernel kernel,
                   int i,
                   //
                   uint32_t* raw_filter_cfg,     //
                   DATE_T* col_l_shipdate,       //
                   MONEY_T* col_l_discount,      //
                   TPCH_INT* col_l_quantity,     //
                   DATE_T* col_l_commitdate,     //
                   MONEY_T* col_l_extendedprice, //
                   MONEY_T* col_revenue,         //
                   //
                   cl_mem* buf_filter_cfg,      //
                   cl_mem* buf_l_shipdate,      //
                   cl_mem* buf_l_discount,      //
                   cl_mem* buf_l_quantity,      //
                   cl_mem* buf_l_commitdate,    //
                   cl_mem* buf_l_extendedprice, //
                   cl_mem* buf_revenue,         //
                   //
                   int l_depth) {
    // prepare extended attribute for all buffers
    ;
    cl_mem_ext_ptr_t mext_filter_cfg = {0, raw_filter_cfg, kernel};
    cl_mem_ext_ptr_t mext_l_shipdate = {1, col_l_shipdate, kernel};
    cl_mem_ext_ptr_t mext_l_discount = {2, col_l_discount, kernel};
    cl_mem_ext_ptr_t mext_l_quantity = {3, col_l_quantity, kernel};
    cl_mem_ext_ptr_t mext_l_commitdate = {4, col_l_commitdate, kernel};
    cl_mem_ext_ptr_t mext_l_extendedprice = {5, col_l_extendedprice, kernel};
    cl_mem_ext_ptr_t mext_revenue = {7, col_revenue, kernel};

    cl_int err;

    *buf_filter_cfg = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (sizeof(uint32_t) * BUF_CFG_DEPTH), &mext_filter_cfg, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_shipdate = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(DATE_SZ * l_depth), &mext_l_shipdate, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_discount = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(MONEY_SZ * l_depth), &mext_l_discount, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_quantity = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(TPCH_INT_SZ * l_depth), &mext_l_quantity, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_commitdate = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                       (size_t)(DATE_SZ * l_depth), &mext_l_commitdate, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_extendedprice = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_t)(MONEY_SZ * l_depth), &mext_l_extendedprice, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_revenue = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                  (size_t)(MONEY_SZ * 2 * 1), &mext_revenue, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    return CL_SUCCESS;
}

typedef struct update_buffer_data_ {
    cl_event update_event;
    int i;
} update_buffer_data_t;

void CL_CALLBACK update_buffer(cl_event ev, cl_int st, void* d) {
    update_buffer_data_t* t = (update_buffer_data_t*)d;
    //
    struct timeval tv0, tv1;
    if (debug_level >= FLT_DEBUG) {
        gettimeofday(&tv0, 0);
    }
    //
    clSetUserEventStatus(t->update_event, CL_COMPLETE);
    //
    if (debug_level >= FLT_DEBUG) {
        gettimeofday(&tv1, 0);
        int exec_us = tvdiff(&tv0, &tv1);
        printf("INFO: update callback %d finished in %d usec.\n", t->i, exec_us);
    }
}

typedef struct print_revenue_data_ {
    MONEY_T* col_revenue;
    int row;
    int i;
} print_revenue_data_t;

void CL_CALLBACK print_buffer(cl_event ev, cl_int st, void* d) {
    print_revenue_data_t* t = (print_revenue_data_t*)d;
    MONEY_T* col_revenue = t->col_revenue;
    long long rv = *((long long*)col_revenue);
    printf("FPGA result %d: %lld.%lld\n", t->i, rv / 10000, rv % 10000);

    // compare FPGA results with CPU ref results
    if (t->row == 1000) {
        std::cout << "First 1000 rows from 1G data, reference value: 16575.2594\n";
        ret = (rv == 165752594) ? 0 : 1;
    } else if (t->row == L_MAX_ROW) {
        std::cout << "All rows from 1G data, reference value: 62102819.2435\n";
        ret = (rv == 621028192435) ? 0 : 1;
    }

    if (ret)
        std::cout << "FAIL: " << ret << " error(s) detected!" << std::endl;
    else if (t->row == 1000 || t->row == L_MAX_ROW)
        std::cout << "PASS!" << std::endl;
    else
        std::cout << "WARNING: unknown test size, result is not checked with provisioned golden data." << std::endl;
}

using xclhost::aligned_alloc;
using namespace xf::database::enums;

int main(int argc, const char* argv[]) {
    std::cout << "\n------------ TPC-H Query 6 Modified (1G) -------------\n";

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // cmd arg parser.
    ArgParser parser(argc, argv);

    std::string help;
    if (parser.getCmdOption("-h", help)) {
        std::cout << "Arguments:\n"
                     "   -h            show this help\n"
                     "   -xclbin FILE  path to xclbin\n"
                     "   -data DATADIR path to data folder\n"
                     "   -rep N        continously run for N times\n"
                     "   -mini M       load only M lines from input\n"
                     "------------------------------------------------------\n\n";
        return 0;
    }

    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }

    std::string work_dir;
    if (!parser.getCmdOption("-data", work_dir)) {
        std::cout << "ERROR: data dir is not set!\n";
        return 1;
    }

    int sf = 1;

    // call data generator
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

    int l_nrow;
    std::string do_mini;
    if (parser.getCmdOption("-mini", do_mini)) {
        try {
            l_nrow = std::stoi(do_mini);
        } catch (...) {
            l_nrow = 100;
        }
    } else {
        l_nrow = L_MAX_ROW;
    }
    std::cout << "Lineitem " << l_nrow << " rows\n";

    const size_t l_depth = L_MAX_ROW + VEC_LEN - 1;

    // same host buffer will back a pair of ping-pang OpenCL buffers.
    DATE_T* col_l_shipdate = aligned_alloc<DATE_T>(l_depth);
    MONEY_T* col_l_discount = aligned_alloc<MONEY_T>(l_depth);
    TPCH_INT* col_l_quantity = aligned_alloc<TPCH_INT>(l_depth);
    DATE_T* col_l_commitdate = aligned_alloc<DATE_T>(l_depth);
    MONEY_T* col_l_extendedprice = aligned_alloc<MONEY_T>(l_depth);

    std::cout << "Host input buffers have been allocated.\n";

    int erro;

    erro = load_dat<TPCH_INT>(col_l_shipdate, "l_shipdate", in_dir, l_nrow);
    if (erro) return erro;
    erro = load_dat<TPCH_INT>(col_l_discount, "l_discount", in_dir, l_nrow);
    if (erro) return erro;
    erro = load_dat<TPCH_INT>(col_l_quantity, "l_quantity", in_dir, l_nrow);
    if (erro) return erro;
    erro = load_dat<TPCH_INT>(col_l_commitdate, "l_commitdate", in_dir, l_nrow);
    if (erro) return erro;
    erro = load_dat<TPCH_INT>(col_l_extendedprice, "l_extendedprice", in_dir, l_nrow);
    if (erro) return erro;

    std::cout << "Lineitem table has been read from disk\n";

    uint32_t* config_bits = aligned_alloc<uint32_t>(BUF_CFG_DEPTH);

    // TODO avoid manually write these bits...
    int n = 0;
    // 19940101 <= l_shipdate < 19950101
    config_bits[n++] = (uint32_t)19940101UL;
    config_bits[n++] = (uint32_t)19950101UL;
    config_bits[n++] = 0UL | (FOP_GEU << FilterOpWidth) | (FOP_LTU);
    // 5 <= l_discount <= 7
    config_bits[n++] = (uint32_t)5L;
    config_bits[n++] = (uint32_t)7L;
    config_bits[n++] = 0UL | (FOP_GE << FilterOpWidth) | (FOP_LE);
    // l_quantity < 24
    config_bits[n++] = (uint32_t)0L;
    config_bits[n++] = (uint32_t)24L;
    config_bits[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_LT);
    // l_commitdate
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);

    uint32_t r = 0;
    int sh = 0;
    // l_shipdate -- l_discount
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // l_shipdate -- l_quantity
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // l_shipdate > l_commitdate
    r |= ((uint32_t)(FOP_GTU << sh));
    sh += FilterOpWidth;

    // l_discount -- l_quantity
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // l_discount -- l_commitdate
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // l_quantity -- l_commitdate
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    assert(sh < 32 && "need more than one dword for var to var ops");
    config_bits[n++] = r;

    // 4 true and 6 true
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)0UL;
    config_bits[n++] = (uint32_t)(1UL << 31);

    if (debug_level >= FLT_DEBUG) std::cout << "DEBUG: total config dwords: " << n << std::endl;

#ifdef HLS_TEST
    MONEY_T* col_revenue = aligned_alloc<MONEY_T>(2);
    filter_kernel(
        // config/op
        (ap_uint<32>*)config_bits,
        // input, condition columns
        (ap_uint<8 * KEY_SZ * VEC_LEN>*)col_l_shipdate, (ap_uint<8 * MONEY_SZ * VEC_LEN>*)col_l_discount,
        (ap_uint<8 * TPCH_INT_SZ * VEC_LEN>*)col_l_quantity, (ap_uint<8 * KEY_SZ * VEC_LEN>*)col_l_commitdate,
        // input, payload column
        (ap_uint<8 * MONEY_SZ * VEC_LEN>*)col_l_extendedprice,
        // input, size of workload
        l_nrow,
        // output
        (ap_uint<8 * MONEY_SZ * 2>*)col_revenue);
    long long rv = *((long long*)col_revenue);
    printf("FPGA result: %lld.%lld\n", rv / 10000, rv % 10000);
    free(col_revenue);
#else  // HLS_TEST
    // Create Program and Kernel
    cl_int err;

    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prog;

    err = xclhost::init_hardware(&ctx, &dev_id, &cq, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 "");
    if (clCheckError(err) != CL_SUCCESS) {
        return err;
    }

    err = xclhost::load_binary(&prog, ctx, dev_id, xclbin_path.c_str());
    if (clCheckError(err) != CL_SUCCESS) {
        return err;
    }
    // kernel
    cl_kernel kernel;

    kernel = clCreateKernel(prog, "filter_kernel", &err);
    logger.logCreateKernel(err);

    // two step ping-pang.
    const int step = 2;

    // input ping-pong buffers.
    enum {
        idx_config = 0,
        idx_l_shipdate,
        idx_l_discount,
        idx_l_quantity,
        idx_l_commitdate,
        idx_l_extendedprice,
        idx_end
    };
    cl_mem buf_input[step][idx_end];

    // output ping-pong buffers.
    MONEY_T* col_revenue[step];
    for (int i = 0; i < step; ++i) {
        col_revenue[i] = aligned_alloc<MONEY_T>(2);
    }
    cl_mem buf_output[step];

    for (int i = 0; i < step; ++i) {
        // XXX use the same host buffer for different cl mem obj.
        if (create_buffers(
                ctx, kernel, i,
                //
                config_bits, col_l_shipdate, col_l_discount, col_l_quantity, col_l_commitdate, col_l_extendedprice,
                //
                col_revenue[i],
                //
                &buf_input[i][idx_config], &buf_input[i][idx_l_shipdate], &buf_input[i][idx_l_discount],
                &buf_input[i][idx_l_quantity], &buf_input[i][idx_l_commitdate], &buf_input[i][idx_l_extendedprice],
                //
                &buf_output[i],
                //
                l_depth)) {
            printf("ERROR: input buffer array %d failed to be created.\n", i);
        } else {
            if (debug_level >= FLT_DEBUG) printf("DEBUG: input buffer array %d has been created.\n", i);
        }
    }

    // events
    cl_event* write_events = (cl_event*)malloc(sizeof(cl_event) * num_rep);
    cl_event* kernel_events = (cl_event*)malloc(sizeof(cl_event) * num_rep);
    cl_event* read_events = (cl_event*)malloc(sizeof(cl_event) * num_rep);

    // user events for cpu task.
    cl_event* update_events = (cl_event*)malloc(sizeof(cl_event) * num_rep);

    for (int i = 0; i < num_rep; ++i) {
        update_events[i] = clCreateUserEvent(ctx, &err);
        if (err != CL_SUCCESS) {
            printf("ERROR: fail to create update_events[%d]: %s.\n", i, clGetErrorString(err));
        }
    }

    // data for callback.
    update_buffer_data_t* ucbd = (update_buffer_data_t*)malloc(sizeof(update_buffer_data_t) * num_rep);

    print_revenue_data_t* pcbd = (print_revenue_data_t*)malloc(sizeof(print_revenue_data_t) * num_rep);

    // ------------------ tick tock   ------------------

    struct timeval tv0;
    int exec_us;
    gettimeofday(&tv0, 0);

    for (int i = 0; i < num_rep; ++i) {
        int pingpong = i % step;

        // enqueue move to device
        // wait for host side update done.
        if (i > (step - 1)) {
            clEnqueueMigrateMemObjects(cq, idx_end, buf_input[pingpong], 0, 1, &update_events[i - step],
                                       &write_events[i]);
        } else {
            clEnqueueMigrateMemObjects(cq, idx_end, buf_input[pingpong], 0, 0, NULL, &write_events[i]);
        }

        // enqueue kernel
        int ka = 0;
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_config]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_shipdate]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_discount]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_quantity]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_commitdate]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_extendedprice]);
        clSetKernelArg(kernel, ka++, sizeof(int), &l_nrow);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_output[pingpong]);
        // add to queue
        clEnqueueTask(cq, kernel, 1, &write_events[i], &kernel_events[i]);

        if (i < (num_rep - step)) {
            ucbd[i].update_event = update_events[i];
            ucbd[i].i = i;
            clSetEventCallback(kernel_events[i], CL_COMPLETE, update_buffer, &ucbd[i]);
        }

        // enqueue fetch result
        clEnqueueMigrateMemObjects(cq, 1, &buf_output[pingpong], CL_MIGRATE_MEM_OBJECT_HOST, 1, &kernel_events[i],
                                   &read_events[i]);
        pcbd[i].i = i;
        pcbd[i].row = l_nrow;
        pcbd[i].col_revenue = col_revenue[pingpong];
        clSetEventCallback(read_events[i], CL_COMPLETE, print_buffer, &pcbd[i]);
    }

    clFinish(cq);
    if (debug_level >= FLT_DEBUG) {
        printf("DEBUG: CL finished.\n");
    }

    // ------------------ tick tock   ------------------

    struct timeval tv1;
    gettimeofday(&tv1, 0);
    exec_us = tvdiff(&tv0, &tv1);
    printf("Total wall time of %d runs: %u usec\n", num_rep, exec_us);

    // free resources
    for (int i = 0; i < step; ++i) {
        clReleaseMemObject(buf_input[i][idx_config]);
        clReleaseMemObject(buf_input[i][idx_l_shipdate]);
        clReleaseMemObject(buf_input[i][idx_l_discount]);
        clReleaseMemObject(buf_input[i][idx_l_quantity]);
        clReleaseMemObject(buf_input[i][idx_l_commitdate]);
        clReleaseMemObject(buf_input[i][idx_l_extendedprice]);
        clReleaseMemObject(buf_output[i]);
    }
    for (int i = 0; i < num_rep; ++i) {
        clReleaseEvent(write_events[i]);
        clReleaseEvent(kernel_events[i]);
        clReleaseEvent(read_events[i]);
        clReleaseEvent(update_events[i]);
    }

    err = clReleaseProgram(prog);
    if (err != CL_SUCCESS) {
        printf("ERROR: fail to release program:%s.\n", clGetErrorString(err));
    }

    err = clReleaseCommandQueue(cq);
    if (err != CL_SUCCESS) {
        printf("ERROR: fail to release cmd queue:%s.\n", clGetErrorString(err));
    }

    err = clReleaseContext(ctx);
    if (err != CL_SUCCESS) {
        printf("ERROR: fail to release context:%s.\n", clGetErrorString(err));
    }

    free(write_events);
    free(kernel_events);
    free(read_events);
    free(update_events);

    for (int i = 0; i < step; ++i) {
        free(col_revenue[i]);
    }
#endif // !defined(HLS_TEST)

    // release host buffers.
    free(config_bits);

    free(col_l_shipdate);
    free(col_l_discount);
    free(col_l_quantity);
    free(col_l_commitdate);
    free(col_l_extendedprice);

    ret ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    std::cout << "------------------------------------------------------\n\n";
    return ret;
}
