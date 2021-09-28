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
#include "q5simplified.hpp"
#include "utils.hpp"
#include "test_q5s.hpp"
// clang-format on

#include <iomanip>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <sys/time.h>

#include "xf_utils_sw/logger.hpp"

#include "xclhost.hpp"
#include "cl_errcode.hpp"

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

#if !defined(Q5E2_HJ_PU_NM) || (Q5E2_HJ_PU_NM != 8)
#error "PU_NM is hard coded to 8 in host code"
#endif
const int PU_NM = Q5E2_HJ_PU_NM;

enum q5_debug_level { Q5_ERROR, Q5_WARNING, Q5_INFO, Q5_DEBUG, Q5_ALL };

const q5_debug_level debug_level = Q5_ERROR;

#define ORDERKEY_RAGNE (6000000)
#define HORIZ_PART ((ORDERKEY_MAX + ORDERKEY_RAGNE - 1) / ORDERKEY_RAGNE)

// extra space in partition buffers.
#define BUF_L_DEPTH (L_MAX_ROW / HORIZ_PART + VEC_LEN - 1 + 8000)
#define BUF_O_DEPTH (O_MAX_ROW / HORIZ_PART + VEC_LEN - 1 + 2000)

FILE* fo(std::string fn) {
    FILE* f = fopen(fn.c_str(), "rb");
    if (!f) {
        std::cerr << "ERROR: " << fn << " cannot be opened for binary read." << std::endl;
    }
    return f;
}
int64_t get_golden_sum(int l_row,
                       int* col_l_orderkey,
                       int* col_l_extendedprice,
                       int* col_l_discount,
                       int o_row,
                       int* col_o_orderkey,
                       int* col_o_orderdate) {
    int64_t sum = 0;
    int cnt = 0;
    std::unordered_multimap<uint32_t, uint32_t> ht1;
    {
        for (int i = 0; i < o_row; ++i) {
            int32_t k = col_o_orderkey[i];
            int32_t date = col_o_orderdate[i];
            // insert into hash table
            if (date >= 19940101 && date < 19950101) {
                ht1.insert(std::make_pair(k, date));
            }
        }
    }
    // read t once
    for (int i = 0; i < l_row; ++i) {
        int32_t k = col_l_orderkey[i];
        int32_t p = col_l_extendedprice[i];
        int32_t d = col_l_discount[i];
        // check hash table
        auto its = ht1.equal_range(k);
        for (auto it = its.first; it != its.second; ++it) {
            // std::cout << p << ", " << d << std::endl;
            sum += (p * (100 - d));
            ++cnt;
        }
    }
    return sum;
}

int create_buffers(cl_context ctx,
                   cl_kernel kernel,
                   int i, //
                   KEY_T* col_l_orderkey,
                   MONEY_T* col_l_extendedprice, //
                   MONEY_T* col_l_discount,      //
                   KEY_T* col_o_orderkey,
                   DATE_T* col_o_orderdate, //
                   cl_mem* buf_l_orderkey,
                   cl_mem* buf_l_extendedprice, //
                   cl_mem* buf_l_discount,      //
                   cl_mem* buf_o_orderkey,
                   cl_mem* buf_o_orderdate, //
                   int l_depth = BUF_L_DEPTH,
                   int o_depth = BUF_O_DEPTH) {
    // prepare extended attribute for all buffers

    cl_mem_ext_ptr_t mext_l_orderkey = {0, col_l_orderkey, kernel};
    cl_mem_ext_ptr_t mext_l_extendedprice = {1, col_l_extendedprice, kernel};
    cl_mem_ext_ptr_t mext_l_discount = {2, col_l_discount, kernel};
    cl_mem_ext_ptr_t mext_o_orderkey = {4, col_o_orderkey, kernel};
    cl_mem_ext_ptr_t mext_o_orderdate = {5, col_o_orderdate, kernel};

    cl_int err;

    *buf_l_orderkey = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(KEY_SZ * l_depth), &mext_l_orderkey, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_extendedprice = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_t)(MONEY_SZ * l_depth), &mext_l_extendedprice, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_l_discount = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(MONEY_SZ * l_depth), &mext_l_discount, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_o_orderkey = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                     (size_t)(KEY_SZ * o_depth), &mext_o_orderkey, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    *buf_o_orderdate = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      (size_t)(DATE_SZ * o_depth), &mext_o_orderdate, &err);
    if (clCheckError(err) != CL_SUCCESS) return err;

    return CL_SUCCESS;
}

typedef struct update_buffer_data_ {
    KEY_T* col_l_orderkey_d;
    MONEY_T* col_l_extendedprice_d;
    MONEY_T* col_l_discount_d;
    KEY_T* col_o_orderkey_d;
    DATE_T* col_o_orderdate_d;
    //
    KEY_T* col_l_orderkey;
    MONEY_T* col_l_extendedprice;
    MONEY_T* col_l_discount;
    KEY_T* col_o_orderkey;
    DATE_T* col_o_orderdate;
    //
    cl_event event_update;
    int i;
} update_buffer_data_t;

void CL_CALLBACK update_buffer(cl_event ev, cl_int st, void* d) {
    update_buffer_data_t* t = (update_buffer_data_t*)d;
    //
    struct timeval tv0;
    int exec_us;
    gettimeofday(&tv0, 0);
    //
    memcpy(t->col_l_orderkey_d, t->col_l_orderkey, KEY_SZ * BUF_L_DEPTH);
    memcpy(t->col_l_extendedprice_d, t->col_l_extendedprice, MONEY_SZ * BUF_L_DEPTH);
    memcpy(t->col_l_discount_d, t->col_l_discount, MONEY_SZ * BUF_L_DEPTH);
    memcpy(t->col_o_orderkey_d, t->col_o_orderkey, KEY_SZ * BUF_O_DEPTH);
    memcpy(t->col_o_orderdate_d, t->col_o_orderdate, DATE_SZ * BUF_O_DEPTH);
    //
    clSetUserEventStatus(t->event_update, CL_COMPLETE);
    //
    struct timeval tv1;
    gettimeofday(&tv1, 0);
    exec_us = tvdiff(&tv0, &tv1);
    if (debug_level >= Q5_INFO) printf("INFO: callback %d finishes in %d usec.\n", t->i, exec_us);
}

int main(int argc, const char* argv[]) {
    std::cout << "\n------------ TPC-H Query 5 Simplified (1~100G) -------------\n";

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

    int sf = 1;
    std::string sf_s;
    if (parser.getCmdOption("-sf", sf_s)) {
        try {
            sf = std::stoi(sf_s);
        } catch (...) {
            sf = 1;
        }
    }

    // call data generator
    std::string in_dir = prepare(work_dir, sf);

    KEY_T* col_l_orderkey[HORIZ_PART];
    MONEY_T* col_l_extendedprice[HORIZ_PART];
    MONEY_T* col_l_discount[HORIZ_PART];
    int l_nrow_part[HORIZ_PART];

    KEY_T* col_o_orderkey[HORIZ_PART];
    DATE_T* col_o_orderdate[HORIZ_PART];
    int o_nrow_part[HORIZ_PART];

    MONEY_T* part_result[HORIZ_PART];

    FILE* f_l_orderkey = fo(in_dir + "/l_orderkey.dat");
    FILE* f_l_extendedprice = fo(in_dir + "/l_extendedprice.dat");
    FILE* f_l_discount = fo(in_dir + "/l_discount.dat");
    FILE* f_o_orderkey = fo(in_dir + "/o_orderkey.dat");
    FILE* f_o_orderdate = fo(in_dir + "/o_orderdate.dat");

    KEY_T t_l_orderkey;
    MONEY_T t_l_extendedprice;
    MONEY_T t_l_discount;
    KEY_T t_o_orderkey;
    DATE_T t_o_orderdate;

    int l_nrow = 0;
    int o_nrow = 0;
    bool no_more = false;
    bool fit_in_one = false;
    bool overflow = false;

    for (int i = 0; i < HORIZ_PART; ++i) {
        KEY_T okey_max = ORDERKEY_RAGNE * (i + 1) + 1;
        if (debug_level >= Q5_DEBUG) printf("DEBUG: part %d, orderkey max = %d\n", i, okey_max);

        // alloc o
        col_o_orderkey[i] = aligned_alloc<KEY_T>(BUF_O_DEPTH);
        col_o_orderdate[i] = aligned_alloc<DATE_T>(BUF_O_DEPTH);
        // read o
        int j = 0;
        if (i > 0 && !no_more) {
            col_o_orderkey[i][j] = t_o_orderkey;
            col_o_orderdate[i][j] = t_o_orderdate;
            j++;
            o_nrow++;
        }
        // read new data
        while (o_nrow < O_MAX_ROW) {
            int rn = fread(&t_o_orderkey, sizeof(KEY_T), 1, f_o_orderkey);
            if (rn != 1) {
                no_more = true;
                break;
            }
            rn = fread(&t_o_orderdate, sizeof(DATE_T), 1, f_o_orderdate);
            if (rn != 1) {
                no_more = true;
                break;
            }
            if (t_o_orderkey < okey_max) {
                if (j < BUF_O_DEPTH) {
                    col_o_orderkey[i][j] = t_o_orderkey;
                    col_o_orderdate[i][j] = t_o_orderdate;
                } else {
                    overflow = true;
                }
                j++;
                o_nrow++;
            } else {
                break;
            }
        }
        o_nrow_part[i] = j;
        if (i == 0 && no_more) {
            fit_in_one = true;
        }
        if (debug_level >= Q5_DEBUG)
            printf("DEBUG: BUF_O_DEPTH=%ld, part %d: %d (%ld slots unused)\n", BUF_O_DEPTH, i, j, BUF_O_DEPTH - j);

        // alloc l
        col_l_orderkey[i] = aligned_alloc<KEY_T>(BUF_L_DEPTH);
        col_l_extendedprice[i] = aligned_alloc<MONEY_T>(BUF_L_DEPTH);
        col_l_discount[i] = aligned_alloc<MONEY_T>(BUF_L_DEPTH);
        // read l
        // data failed to be written to last part.
        j = 0;
        if (i > 0 && !no_more) {
            col_l_orderkey[i][j] = t_l_orderkey;
            col_l_extendedprice[i][j] = t_l_extendedprice;
            col_l_discount[i][j] = t_l_discount;
            j++;
            l_nrow++;
        }
        // read new data.
        while (l_nrow < L_MAX_ROW) {
            int rn = fread(&t_l_orderkey, sizeof(KEY_T), 1, f_l_orderkey);
            if (rn != 1) {
                no_more = true;
                break;
            }
            rn = fread(&t_l_extendedprice, sizeof(MONEY_T), 1, f_l_extendedprice);
            if (rn != 1) {
                no_more = true;
                break;
            }
            rn = fread(&t_l_discount, sizeof(MONEY_T), 1, f_l_discount);
            if (rn != 1) {
                no_more = true;
                break;
            }
            // test whether the data belong to the part
            if (t_l_orderkey < okey_max) {
                if (j < BUF_L_DEPTH) {
                    col_l_orderkey[i][j] = t_l_orderkey;
                    col_l_extendedprice[i][j] = t_l_extendedprice;
                    col_l_discount[i][j] = t_l_discount;
                } else {
                    overflow = true;
                }
                j++;
                l_nrow++;
            } else {
                break;
            }
        };
        l_nrow_part[i] = j;
        if (debug_level >= Q5_DEBUG)
            printf("DEBUG: BUF_L_DEPTH=%ld, part %d: %d (%ld slots unused)\n", BUF_L_DEPTH, i, j, BUF_L_DEPTH - j);

        // alloc result
        part_result[i] = aligned_alloc<MONEY_T>(2);
        memset(part_result[i], 0, sizeof(MONEY_T) * 2);
    }

    fclose(f_l_orderkey);
    fclose(f_l_extendedprice);
    fclose(f_l_discount);
    fclose(f_o_orderkey);
    fclose(f_o_orderdate);

    std::cout << "Lineitem " << l_nrow << " rows\n"
              << "Orders " << o_nrow << " rows\n";

    std::cout << "Host map buffer has been allocated.\n";

    if (overflow) {
        printf("ERROR: some buffer has overflow!\n");
        return 1;
    }

    // OPENCL HOST CODE AREA START

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

    kernel = clCreateKernel(prog, "q5simplified", &err);
    logger.logCreateKernel(err);

    // temp buffers.
    cl_mem buf_table[PU_NM];

    cl_mem_ext_ptr_t memExt[PU_NM];

    for (int j = 0; j < PU_NM; ++j) {
        memExt[j] = {7 + j, NULL, kernel};
        buf_table[j] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS,
                                      (size_t)(KEY_SZ * BUFF_DEPTH), &memExt[j], &err);
        if (err != CL_SUCCESS) {
            printf("ERROR: fail to create buf_table[%d]: %s.\n", j, clGetErrorString(err));
            return err;
        }
    }

    // output buffers
    cl_mem buf_result[HORIZ_PART];

    for (int i = 0; i < HORIZ_PART; ++i) {
        cl_mem_ext_ptr_t mext_result = {15, part_result[i], kernel};

        buf_result[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_WRITE_ONLY,
                                       (size_t)(MONEY_SZ * 2), &mext_result, &err);
        if (clCheckError(err) != CL_SUCCESS) return err;
    }

    const int step = 3;

    // input ping-pong buffers.
    enum { idx_l_orderkey = 0, idx_l_extendedprice, idx_l_discount, idx_o_orderkey, idx_o_orderdate, idx_end };
    cl_mem buf_input[step][idx_end];

    for (int i = 0; i < step; ++i) {
        int l_depth = fit_in_one ? l_nrow_part[0] : BUF_L_DEPTH;
        int o_depth = fit_in_one ? o_nrow_part[0] : BUF_O_DEPTH;
        if (create_buffers(ctx, kernel, i, col_l_orderkey[i], col_l_extendedprice[i], col_l_discount[i],
                           col_o_orderkey[i], col_o_orderdate[i], &buf_input[i][idx_l_orderkey],
                           &buf_input[i][idx_l_extendedprice], &buf_input[i][idx_l_discount],
                           &buf_input[i][idx_o_orderkey], &buf_input[i][idx_o_orderdate], l_depth, o_depth)) {
            printf("ERROR: input buffer[%d] failed to be created.\n", i);
        } else {
            if (debug_level >= Q5_DEBUG) printf("DEBUG: input buffer[%d] has been created.\n", i);
        }
    }

    // start working here.
    // events
    cl_event event_write[HORIZ_PART], event_kernel[HORIZ_PART], event_read[HORIZ_PART];

    // user events for cpu task.
    cl_event event_update[HORIZ_PART];

    for (int i = 0; i < HORIZ_PART; ++i) {
        event_update[i] = clCreateUserEvent(ctx, &err);
        if (err != CL_SUCCESS) {
            printf("ERROR: fail to create event_update[%d]: %s.\n", i, clGetErrorString(err));
        }
    }

    // data for callback.
    update_buffer_data_t cbdata[HORIZ_PART];

    // ------------------ tick tock   ------------------

    struct timeval tv0;
    int exec_us;
    gettimeofday(&tv0, 0);

    for (int i = 0; i < HORIZ_PART; ++i) {
        if (o_nrow_part[i] == 0) {
            continue;
        }

        int pingpong = i % step;

        // enqueue move to device
        // wait for host side update done.
        if (i > (step - 1)) {
            clEnqueueMigrateMemObjects(cq, 5, buf_input[pingpong], 0, 1, &event_update[i - step], &event_write[i]);
        } else {
            clEnqueueMigrateMemObjects(cq, 5, buf_input[pingpong], 0, 0, NULL, &event_write[i]);
        }

        // enqueue kernel
        int ka = 0;
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_orderkey]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_extendedprice]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_l_discount]);
        clSetKernelArg(kernel, ka++, sizeof(int), &l_nrow_part[i]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_o_orderkey]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_input[pingpong][idx_o_orderdate]);
        clSetKernelArg(kernel, ka++, sizeof(int), &o_nrow_part[i]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[0]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[1]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[2]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[3]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[4]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[5]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[6]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_table[7]);
        clSetKernelArg(kernel, ka++, sizeof(cl_mem), &buf_result[i]);
        // add to queue
        clEnqueueTask(cq, kernel, 1, &event_write[i], &event_kernel[i]);

        if (i < (HORIZ_PART - step) && o_nrow_part[i + step] != 0) {
            // start update as soon as kernel finished.
            cbdata[i].col_l_orderkey_d = col_l_orderkey[pingpong];
            cbdata[i].col_l_extendedprice_d = col_l_extendedprice[pingpong];
            cbdata[i].col_l_discount_d = col_l_discount[pingpong];
            cbdata[i].col_o_orderkey_d = col_o_orderkey[pingpong];
            cbdata[i].col_o_orderdate_d = col_o_orderdate[pingpong];
            //
            cbdata[i].col_l_orderkey = col_l_orderkey[i + step];
            cbdata[i].col_l_extendedprice = col_l_extendedprice[i + step];
            cbdata[i].col_l_discount = col_l_discount[i + step];
            cbdata[i].col_o_orderkey = col_o_orderkey[i + step];
            cbdata[i].col_o_orderdate = col_o_orderdate[i + step];
            //
            cbdata[i].event_update = event_update[i];
            cbdata[i].i = i;
            //
            clSetEventCallback(event_kernel[i], CL_COMPLETE, update_buffer, &cbdata[i]);
        }

        // enqueue fetch result
        clEnqueueMigrateMemObjects(cq, 1, &buf_result[i], CL_MIGRATE_MEM_OBJECT_HOST, 1, &event_kernel[i],
                                   &event_read[i]);
    }

    clFinish(cq);
    if (debug_level >= Q5_DEBUG) printf("DEBUG: CL finished.\n");

    // ------------------ tick tock   ------------------

    struct timeval tv1;
    gettimeofday(&tv1, 0);
    exec_us = tvdiff(&tv0, &tv1);

#if 0
  // print kernel time
  unsigned long kernel_time = 0;
  for (int i = 0; i < HORIZ_PART; ++i) {
    cl_ulong ts, te;
    clGetEventProfilingInfo(event_kernel[i], CL_PROFILING_COMMAND_START,
                            sizeof(ts), &ts, NULL);
    clGetEventProfilingInfo(event_kernel[i], CL_PROFILING_COMMAND_END,
                            sizeof(te), &te, NULL);
    unsigned long t = (te - ts) / 1000;
    kernel_time += t;
    printf("INFO: kernel %d: execution time %lu usec\n", i, t);
  }
  printf("Total kernel execution time %lu usec\n", kernel_time);
#endif

    // show result
    long long v = 0;
    for (int i = 0; i < HORIZ_PART; ++i) {
        long long pv = *((long long*)part_result[i]);
        if (o_nrow_part[i] != 0 && debug_level >= Q5_DEBUG)
            printf("Result of part %d: %lld.%04lld\n", i, pv / 10000, pv % 10000);
        v += pv;
    }
    printf("FPGA Result: %lld.%04lld (end-to-end time %d ms)\n", v / 10000, v % 10000, exec_us / 1000);

    // free resources
    for (int i = 0; i < step; ++i) {
        clReleaseMemObject(buf_input[i][idx_l_orderkey]);
        clReleaseMemObject(buf_input[i][idx_l_extendedprice]);
        clReleaseMemObject(buf_input[i][idx_l_discount]);
        clReleaseMemObject(buf_input[i][idx_o_orderkey]);
        clReleaseMemObject(buf_input[i][idx_o_orderdate]);
    }
    for (int i = 0; i < PU_NM; ++i) {
        clReleaseMemObject(buf_table[i]);
    }
    for (int i = 0; i < HORIZ_PART; ++i) {
        clReleaseMemObject(buf_result[i]);
        if (o_nrow_part[i] != 0) {
            clReleaseEvent(event_write[i]);
            clReleaseEvent(event_kernel[i]);
            clReleaseEvent(event_read[i]);
            clReleaseEvent(event_update[i]);
        }
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

    // OPENCL HOST CODE AREA END

    // compare golden result
    int64_t golden_sum = 0;
    for (int i = 0; i < HORIZ_PART; ++i) {
        golden_sum += get_golden_sum(l_nrow_part[i], col_l_orderkey[i], col_l_extendedprice[i], col_l_discount[i],
                                     o_nrow_part[i], col_o_orderkey[i], col_o_orderdate[i]);
    }
    printf("Golden: %lld.%04lld\n", golden_sum / 10000, golden_sum % 10000);

    err += (v == golden_sum) ? 0 : 1;

    err ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    std::cout << "-----------------------------------------------------------\n\n";
    return err;
}
