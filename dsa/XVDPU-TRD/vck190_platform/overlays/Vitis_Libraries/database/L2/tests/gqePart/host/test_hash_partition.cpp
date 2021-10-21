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

#ifndef HLS_TEST
// OpenCL C API utils
#include "xclhost.hpp"
#endif

#include "xf_utils_sw/logger.hpp"

#include "x_utils.hpp"

// L1
#include "xf_database/hash_lookup3.hpp"
// GQE L2
#include "xf_database/meta_table.hpp"
#include "xf_database/kernel_command.hpp"
// HLS
#include "ap_int.h"

#include "table_dt.hpp"

#include <sys/time.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>
#include <thread>
#include <unistd.h>
#include <map>

const int HASHWH = 2;
const int HASHWL = 8;

#ifdef HLS_TEST
extern "C" void gqePart(const int bucket_depth, // bucket depth

                        // table index indicate build table or join table
                        const int tab_index,

                        // the log partition number
                        const int log_part,

                        // input data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* din_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* din_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* din_col2,

                        ap_uint<64>* din_val,

                        // kernel config
                        ap_uint<512> din_krn_cfg[14],

                        // meta input buffer
                        ap_uint<512> din_meta[24],
                        // meta output buffer
                        ap_uint<512> dout_meta[24],

                        // output data columns
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col0,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col1,
                        ap_uint<8 * TPCH_INT_SZ * VEC_LEN>* dout_col2);

#endif

inline int tvdiff(const timeval& tv0, const timeval& tv1) {
    return (tv1.tv_sec - tv0.tv_sec) * 1000000 + (tv1.tv_usec - tv0.tv_usec);
}

inline int tvdiff(const timeval& tv0, const timeval& tv1, const char* name) {
    int exec_us = tvdiff(tv0, tv1);
    printf("%s: %d.%03d msec\n", name, (exec_us / 1000), (exec_us % 1000));
    return exec_us;
}

template <typename T>
int generate_data(T* data, int64_t range, size_t n) {
    if (!data) {
        return -1;
    }

    for (size_t i = 0; i < n; i++) {
        data[i] = (T)(rand() % range + 1);
    }

    return 0;
}

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

void gen_pass_fcfg(uint32_t cfg[]) {
    using namespace xf::database;
    int n = 0;

    // cond_1
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_2
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_3
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);
    // cond_4
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = 0UL | (FOP_DC << FilterOpWidth) | (FOP_DC);

    uint32_t r = 0;
    int sh = 0;
    // cond_1 -- cond_2
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_1 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_2 -- cond_3
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;
    // cond_2 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    // cond_3 -- cond_4
    r |= ((uint32_t)(FOP_DC << sh));
    sh += FilterOpWidth;

    cfg[n++] = r;

    // 4 true and 6 true
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)0UL;
    cfg[n++] = (uint32_t)(1UL << 31);
}

int main(int argc, const char* argv[]) {
    // cmd arg parser.
    x_utils::ArgParser parser(argc, argv);

#ifndef HLS_TEST
    std::string xclbin_path;
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
#endif

    std::string scale;
    int sim_scale = 1000;
    if (parser.getCmdOption("-scale", scale)) {
        try {
            sim_scale = std::stoi(scale);
        } catch (...) {
            sim_scale = 10000;
        }
    }

    int l_nrow = 100;
    // int l_nrow = L_MAX_ROW / sim_scale;
    std::cout << "Lineitem " << l_nrow << " rows\n";

    int log_part = 2;
    if (parser.getCmdOption("-log_part", scale)) {
        try {
            log_part = std::stoi(scale);
        } catch (...) {
            log_part = 2;
        }
    }
    if (log_part < 2) {
        std::cout << "ERROR: partition number only supports >= 4 !!" << std::endl;
        return -1;
    }

    // --------- partitioning Table L ----------
    // partition setups
    const int bucket_depth = 512;
    const int tab_index = 0;
    const int partition_num = 1 << log_part;

    // the col nrow of each section
    int tab_part_sec_nrow_each = l_nrow;
    int tab_part_sec_size = tab_part_sec_nrow_each * TPCH_INT_SZ;

    x_utils::MM mm;
    int err = 0;

    // data load from disk. due to table size, data read into several sections
    // L host side pinned buffers for partition kernel
    TPCH_INT* tab_part_in_col[3];
    for (int i = 0; i < 3; i++) {
        tab_part_in_col[i] = mm.aligned_alloc<TPCH_INT>(tab_part_sec_nrow_each);
        if (i < 1) {
            err += generate_data<TPCH_INT>((TPCH_INT*)(tab_part_in_col[i]), 1000, l_nrow);
        } else {
            for (int tt = 0; tt < l_nrow; tt++) {
                tab_part_in_col[i][tt] = tt;
            }
        }
    }
    if (err) {
        fprintf(stderr, "ERROR: failed to gen data.\n");
        return 1;
    }
    ap_uint<64>* valid_in_col = mm.aligned_alloc<ap_uint<64> >((tab_part_sec_nrow_each + 63) / 64);
    for (int i = 0; i < (l_nrow + 63) / 64; i++) {
        valid_in_col[i] = 0xffffffffffffffff;
    }
    std::cout << "       Key0          Key1         Payload\n";
    for (int i = 0; i < l_nrow; i++) {
        std::cout << std::setw(10) << tab_part_in_col[0][i] << ",   ";
        std::cout << std::setw(10) << tab_part_in_col[1][i] << ",   ";
        std::cout << std::setw(10) << tab_part_in_col[2][i] << std::endl;
    }
    std::cout << "finished dat loading/generating" << std::endl;

    // partition output data
    int tab_part_out_col_nrow_512_init = tab_part_sec_nrow_each * 2 / VEC_LEN;
    assert(tab_part_out_col_nrow_512_init > 0 && "Error: table output col size must > 0");
    // the depth of each partition in each col
    int tab_part_out_col_eachpart_nrow_512 = (tab_part_out_col_nrow_512_init + partition_num - 1) / partition_num;
    std::cout << "tab_part_out_col_eachpart_nrow_512: " << tab_part_out_col_eachpart_nrow_512 << std::endl;
    // total output data nrow, aligned by 512
    int tab_part_out_col_nrow_512 = partition_num * tab_part_out_col_eachpart_nrow_512;
    int tab_part_out_col_size = tab_part_out_col_nrow_512 * TPCH_INT_SZ * VEC_LEN;

    // partition_output data
    ap_uint<512>* tab_part_out_col[3];
    for (int i = 0; i < 3; i++) {
        tab_part_out_col[i] = mm.aligned_alloc<ap_uint<512> >(tab_part_out_col_nrow_512);
    }

    ap_uint<512>* krn_cfg_part = mm.aligned_alloc<ap_uint<512> >(14);
    // init
    memset(krn_cfg_part, 0, sizeof(ap_uint<512>) * 14);
    // tab A col enable
    krn_cfg_part[1].range(8, 6) = 1;
    // write out enable
    krn_cfg_part[1].range(14, 12) = 7;
    // tab A gen_row_id en
    krn_cfg_part[1].range(19, 19) = 1;
    // tab A valid_en
    krn_cfg_part[1].range(20, 20) = 1;

    // filter
    uint32_t cfg[53];
    gen_pass_fcfg(cfg);
    memcpy(&krn_cfg_part[6], cfg, sizeof(uint32_t) * 53);

    //--------------- metabuffer setup L -----------------
    xf::database::gqe::MetaTable meta_part_in;
    xf::database::gqe::MetaTable meta_part_out;
    meta_part_in.setSecID(0);
    meta_part_in.setColNum(1);
    meta_part_in.setCol(0, 0, tab_part_sec_nrow_each);
    meta_part_in.meta();

    // setup partition kernel used meta output
    meta_part_out.setColNum(2);
    meta_part_out.setPartition(partition_num, tab_part_out_col_eachpart_nrow_512);
    meta_part_out.setCol(0, 0, tab_part_out_col_nrow_512);
    // meta_part_out.setCol(1, 1, tab_part_out_col_nrow_512);
    meta_part_out.setCol(2, 2, tab_part_out_col_nrow_512);
    meta_part_out.meta();

#ifdef HLS_TEST
    gqePart(bucket_depth, tab_index, log_part, (ap_uint<512>*)tab_part_in_col[0], (ap_uint<512>*)tab_part_in_col[1],
            (ap_uint<512>*)tab_part_in_col[2], (ap_uint<64>*)valid_in_col, krn_cfg_part, meta_part_in.meta(),
            meta_part_out.meta(), tab_part_out_col[0], tab_part_out_col[1], tab_part_out_col[2]);

    int* nrow_per_part = meta_part_out.getPartLen();

    std::map<int64_t, int> key_part_map;

    for (int i = 0; i < partition_num; i++) {
        int offset = tab_part_out_col_eachpart_nrow_512 * i;
        int prow = nrow_per_part[i];
        std::cout << "Part " << i << " nrow: " << prow << std::endl;
        const int nread = (prow + 7) / 8;
        int abc = 0;

        if (tab_part_out_col_eachpart_nrow_512 * VEC_LEN < prow) {
            std::cout << "ERROR: the output buffer nrow for each partition: "
                      << (tab_part_out_col_eachpart_nrow_512 * VEC_LEN);
            std::cout << " is smaller than the resulting nrow: " << prow << std::endl;
            exit(1);
        }

        for (int n = 0; n < nread; n++) {
            const int len = (abc + VEC_LEN) > prow ? (prow - abc) : VEC_LEN;
            for (int m = 0; m < len; m++) {
                int64_t key = tab_part_out_col[0][offset + n](m * 64 + 63, m * 64);
                ap_uint<64> key2 = tab_part_out_col[1][offset + n](m * 64 + 63, m * 64);
                ap_uint<64> rowid = tab_part_out_col[2][offset + n](m * 64 + 63, m * 64);
                std::cout << std::setw(10) << (int64_t)key << ", ";
                std::cout << std::setw(10) << (int64_t)key2 << ", ";
                std::cout << std::setw(10) << (int64_t)rowid << std::endl;
                // pre-check whether same key are in the same partition
                if (key_part_map.find(key) != key_part_map.end()) {
                    // to make sure every key in each partition is orthogonal to those in the different partitions
                    if (i != key_part_map[key]) {
                        std::cout << "Find Error, Error key is " << key << std::endl;
                        nerror++;
                    }
                } else {
                    // new key found
                    key_part_map.insert(std::make_pair(key, i));
                }
            }
            abc += len;
        }
    }
    std::cout << "All partitions are checked.\nPASS!" << std::endl;
#else
    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);
    // setup OpenCL related
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cmq;
    cl_program prg;
    err += xclhost::init_hardware(&ctx, &dev_id, &cmq,
                                  CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, MSTR(XDEVICE));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL with " MSTR(XDEVICE) "\n");
        return err;
    }
    err = xclhost::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        return err;
    }
    // partition kernel
    cl_kernel partkernel;
    partkernel = clCreateKernel(prg, "gqePart", &err);
    // will not exit with failure by default
    logger.logCreateKernel(err);

    // ------------------------------------------
    // using jcmdclass = xf::database::gqe::JoinCommand;
    // jcmdclass jcmd = jcmdclass();

    // jcmd.setJoinType(xf::database::INNER_JOIN);

    // jcmd.Scan(0, {0, 1, 2});
    // // jcmd.setDualKeyOn();
    // ap_uint<512>* cfg_part = jcmd.getConfigBits();

    cl_mem_ext_ptr_t mext_tab_part_in_col[3];
    for (int i = 0; i < 3; ++i) {
        mext_tab_part_in_col[i] = {(3 + i), tab_part_in_col[i], partkernel};
    }

    cl_mem_ext_ptr_t mext_tab_part_out_col[3];
    for (int i = 0; i < 3; ++i) {
        mext_tab_part_out_col[i] = {(10 + i), tab_part_out_col[i], partkernel};
    }

    cl_mem_ext_ptr_t mext_valid_in_col = {6, valid_in_col, partkernel};

    cl_mem_ext_ptr_t mext_cfg_part = {7, krn_cfg_part, partkernel};

    // dev buffers, part in
    cl_mem buf_tab_part_in_col[3];
    for (int c = 0; c < 3; c++) {
        buf_tab_part_in_col[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                tab_part_sec_size, &mext_tab_part_in_col[c], &err);
    }

    cl_mem buf_valid_in_col = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             (l_nrow + 63) / 64 * sizeof(ap_uint<64>), &mext_valid_in_col, &err);

    // dev buffers, part out
    std::cout << "Input device buffer has been created" << std::endl;
    cl_mem buf_tab_part_out_col[3];
    for (int c = 0; c < 3; c++) {
        buf_tab_part_out_col[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                 tab_part_out_col_size, &mext_tab_part_out_col[c], &err);
    }
    std::cout << "Output device buffer has been created" << std::endl;

    cl_mem buf_cfg_part = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                         (sizeof(ap_uint<512>) * 14), &mext_cfg_part, &err);

    cl_mem_ext_ptr_t mext_meta_part_in, mext_meta_part_out;

    mext_meta_part_in = {8, meta_part_in.meta(), partkernel};
    mext_meta_part_out = {9, meta_part_out.meta(), partkernel};

    cl_mem buf_meta_part_in;
    buf_meta_part_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      (sizeof(ap_uint<512>) * 8), &mext_meta_part_in, &err);
    cl_mem buf_meta_part_out;
    buf_meta_part_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                       (sizeof(ap_uint<512>) * 24), &mext_meta_part_out, &err);

    //----------------------partition L run-----------------------------
    std::cout << "------------------- Partitioning L table -----------------" << std::endl;
    int j = 0;
    const int k_depth = 512;
    const int gen_row_id = 1;
    const int sec_id = 0;
    const int din_val_en = 0;
    clSetKernelArg(partkernel, j++, sizeof(int), &k_depth);
    clSetKernelArg(partkernel, j++, sizeof(int), &tab_index);
    clSetKernelArg(partkernel, j++, sizeof(int), &log_part);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_in_col[0]);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_in_col[1]);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_in_col[2]);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_valid_in_col);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_cfg_part);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_meta_part_in);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_meta_part_out);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_out_col[0]);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_out_col[1]);
    clSetKernelArg(partkernel, j++, sizeof(cl_mem), &buf_tab_part_out_col[2]);

    // partition h2d
    std::vector<cl_mem> part_in_vec;
    part_in_vec.push_back(buf_tab_part_in_col[0]);
    part_in_vec.push_back(buf_tab_part_in_col[1]);
    part_in_vec.push_back(buf_tab_part_in_col[2]);
    part_in_vec.push_back(buf_valid_in_col);
    part_in_vec.push_back(buf_meta_part_in);
    part_in_vec.push_back(buf_cfg_part);

    // partition d2h
    std::vector<cl_mem> part_out_vec;
    part_out_vec.push_back(buf_tab_part_out_col[0]);
    part_out_vec.push_back(buf_tab_part_out_col[1]);
    part_out_vec.push_back(buf_tab_part_out_col[2]);
    part_out_vec.push_back(buf_meta_part_out);
    clEnqueueMigrateMemObjects(cmq, 1, &buf_meta_part_out, 0, 0, nullptr, nullptr);

    clEnqueueMigrateMemObjects(cmq, part_in_vec.size(), part_in_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                               nullptr, nullptr);
    clEnqueueMigrateMemObjects(cmq, part_out_vec.size(), part_out_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED,
                               0, nullptr, nullptr);

    cl_event evt_part_h2d, evt_part_krn, evt_part_d2h;

    std::vector<cl_event> evt_part_h2d_dep;
    std::vector<cl_event> evt_part_krn_dep;
    std::vector<cl_event> evt_part_d2h_dep;

    timeval tv_part_start, tv_part_end;
    timeval tv1, tv2;
    gettimeofday(&tv_part_start, 0);
    clEnqueueMigrateMemObjects(cmq, part_in_vec.size(), part_in_vec.data(), 0, 0, nullptr, &evt_part_h2d);
    clFinish(cmq);
    gettimeofday(&tv1, 0);
    tvdiff(tv_part_start, tv1, "h2d");
    std::cout << "h2d done " << std::endl;
    clEnqueueTask(cmq, partkernel, 1, &evt_part_h2d, &evt_part_krn);
    clFinish(cmq);
    gettimeofday(&tv2, 0);
    tvdiff(tv1, tv2, "krn: ");
    std::cout << "krn done " << std::endl;
    clEnqueueMigrateMemObjects(cmq, part_out_vec.size(), part_out_vec.data(), 1, 1, &evt_part_krn, &evt_part_d2h);
    clFinish(cmq);
    gettimeofday(&tv1, 0);
    tvdiff(tv2, tv1, "d2h: ");
    std::cout << "d2h done " << std::endl;

    clFlush(cmq);
    clFinish(cmq);
    gettimeofday(&tv_part_end, 0);

    std::cout << "Checking result...." << std::endl;
    std::map<int, int> key_part_map;

    // get number of rows from each partition
    int* nrow_per_part = meta_part_out.getPartLen();
    int nerror = 0;
    for (int i = 0; i < partition_num; i++) {
        int offset = tab_part_out_col_eachpart_nrow_512 * i;
        int prow = nrow_per_part[i];
        std::cout << "Partition " << i << " nrow: " << prow << std::endl;

        if (tab_part_out_col_eachpart_nrow_512 * VEC_LEN < prow) {
            std::cout << "ERROR: the output buffer nrow for each partition: "
                      << (tab_part_out_col_eachpart_nrow_512 * VEC_LEN);
            std::cout << " is smaller than the resulting nrow: " << prow << std::endl;
            exit(1);
        }

        std::cout << "       Key0         Key1      Row-ID\n";
        const int nread = (prow + 7) / 8;
        int abc = 0;
        for (int n = 0; n < nread; n++) {
            const int len = (abc + VEC_LEN) > prow ? (prow - abc) : VEC_LEN;
            for (int m = 0; m < len; m++) {
                ap_uint<64> key = tab_part_out_col[0][offset + n](m * 64 + 63, m * 64);
                ap_uint<64> key2 = tab_part_out_col[1][offset + n](m * 64 + 63, m * 64);
                ap_uint<64> rowid = tab_part_out_col[2][offset + n](m * 64 + 63, m * 64);
                std::cout << std::setw(10) << (int64_t)key << ", ";
                std::cout << std::setw(10) << (int64_t)key2 << ", ";
                std::cout << std::setw(10) << (int64_t)rowid << std::endl;
                // pre-check whether same key are in the same partition
                if (key_part_map.find(key) != key_part_map.end()) {
                    // to make sure every key in each partition is orthogonal to those in the different partitions
                    if (i != key_part_map[key]) {
                        std::cout << "Find Error, Error key is " << key << std::endl;
                        nerror++;
                    }
                } else {
                    // new key found
                    key_part_map.insert(std::make_pair(key, i));
                }
            }
            abc += len;
        }
    }
    (nerror > 0) ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

#endif

    return nerror;
}
