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

#include "xf_utils_sw/logger.hpp"

// L2
#include "xf_database/meta_table.hpp"
// L3
#include "xf_database/gqe_table.hpp"
#include "xf_database/gqe_join.hpp"

#include <mutex>
#include <unistd.h>
#include <condition_variable>

//#define Valgrind_debug 1
#define USER_DEBUG 1
#define JOIN_PERF_PROFILE 1 1
//#define JOIN_PERF_PROFILE_2 1

#define TPCH_INT_SZ 8
#define VEC_LEN 8

#define XCL_BANK0 (XCL_MEM_TOPOLOGY | unsigned(32))
#define XCL_BANK1 (XCL_MEM_TOPOLOGY | unsigned(33))

namespace xf {
namespace database {
namespace gqe {

ErrCode Joiner::run(Table& tab_a,
                    std::string filter_a,
                    Table& tab_b,
                    std::string filter_b,
                    std::string join_str,
                    Table& tab_c,
                    std::string output_str,
                    int join_type,
                    JoinStrategyBase* strategyimp) {
    ErrCode error = ErrCode::SUCCESS;
    // strategy
    bool new_s = false;
    if (strategyimp == nullptr) {
        strategyimp = new JoinStrategyBase();
        new_s = true;
    }
    // update table C result nrow
    StrategySet params = strategyimp->getSolutionParams(tab_a, tab_b);
    // cfg
    if (params.sol == 0) {
        std::cout << "Using Solution 0: Direct Join" << std::endl;
        JoinConfig jcfg(tab_a, filter_a, tab_b, filter_b, join_str, tab_c, output_str, join_type);
        error = join_sol0(tab_a, tab_b, tab_c, jcfg, params);
    } else if (params.sol == 1) {
        std::cout << "Using Solution 1: Pipelined Join" << std::endl;
        JoinConfig jcfg(tab_a, filter_a, tab_b, filter_b, join_str, tab_c, output_str, join_type);
        error = join_sol1(tab_a, tab_b, tab_c, jcfg, params);
    } else if (params.sol == 2) {
        std::cout << "Using Solution 2: Partition + Join" << std::endl;
        PartJoinConfig pjcfg(tab_a, filter_a, tab_b, filter_b, join_str, tab_c, output_str, join_type);
        error = join_sol2(tab_a, tab_b, tab_c, pjcfg, params);
    } else {
        std::cout << "ERROR, the solution is not support!!";
    }

    ResetHostBuf();

    if (new_s) delete strategyimp;
    return error;
}

// direct join, 1x build + 1x probe
ErrCode Joiner::join_sol0(Table& tab_a, Table& tab_b, Table& tab_c, JoinConfig& jcfg, StrategySet params) {
    gqe::utils::MM mm;
    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    ap_uint<512>* table_cfg5s = jcfg.getJoinConfigBits();

    std::vector<std::vector<int8_t> > q5s_join_scan = jcfg.getShuffleScan();
    int64_t o_nrow = tab_a.getRowNum();
    int64_t l_nrow = tab_b.getRowNum();
    int64_t result_nrow = tab_c.getRowNum();
#ifdef USER_DEBUG
    std::cout << "o_nrow: " << o_nrow << std::endl;
    std::cout << "l_nrow: " << l_nrow << std::endl;
    std::cout << "result_nrow: " << result_nrow << std::endl;
#endif

    size_t out_valid_col_num = tab_c.getColNum();

#ifdef USER_DEBUG
    std::cout << "out_valid_col_num: " << out_valid_col_num << std::endl;
#endif

    // read params from user
    int sec_o = params.sec_o;
    int sec_l = params.sec_l;

    int64_t tab_o_col_type_size[3];
    int64_t tab_l_col_type_size[3];

    // the build table
    // the O table might be composed of multi-sections
    tab_a.checkSecNum(sec_o);
    size_t table_o_sec_num = tab_a.getSecNum();
    int* table_o_sec_nrow = new int[table_o_sec_num];
    if (sec_o == 0) {
        for (size_t sec = 0; sec < table_o_sec_num; sec++) {
            table_o_sec_nrow[sec] = tab_a.getSecRowNum(sec);
        }
    } else { // sec_o == 1
        table_o_sec_nrow[0] = o_nrow;
    }

    // the real sec size for each section
    int64_t table_o_sec_size[3][table_o_sec_num];
    char* table_o_user[3][table_o_sec_num];
    char* table_o_valid_user[table_o_sec_num];
    int64_t table_o_valid_sec_size[table_o_sec_num];
    for (int i = 0; i < 3; ++i) {
        int idx = (int)q5s_join_scan[0][i];
        if (idx != -1) {
            tab_o_col_type_size[i] = tab_a.getColTypeSize(idx);
            for (size_t j = 0; j < table_o_sec_num; j++) {
                table_o_sec_size[i][j] = table_o_sec_nrow[j] * tab_o_col_type_size[i];
                table_o_user[i][j] = tab_a.getColPointer(idx, 0, j);
            }
        } else {
            tab_o_col_type_size[i] = 8;
        }
    }
    for (size_t j = 0; j < table_o_sec_num; j++) {
        table_o_valid_user[j] = tab_a.getValColPointer(0, j);
        table_o_valid_sec_size[j] = (table_o_sec_nrow[j] + 7) / 8;
    }

    // the size of table o host buf
    int64_t table_o_size[3];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[0][i] != -1)
            table_o_size[i] = o_nrow * tab_o_col_type_size[i];
        else
            table_o_size[i] = VEC_LEN * 8;
    }

    // probe table
    // the L table might be composed of multi-sections
    tab_b.checkSecNum(sec_l);
    size_t table_l_sec_num = tab_b.getSecNum();
    int* table_l_sec_nrow = new int[table_l_sec_num];
    if (sec_l == 0) {
        for (size_t sec = 0; sec < table_l_sec_num; sec++) {
            table_l_sec_nrow[sec] = tab_b.getSecRowNum(sec);
        }
    } else { // sec_l == 1
        table_l_sec_nrow[0] = l_nrow;
    }

    // the real sec size for each section
    int64_t table_l_sec_size[3][table_l_sec_num];
    char* table_l_user[3][table_l_sec_num];
    char* table_l_valid_user[table_l_sec_num];
    int64_t table_l_valid_sec_size[table_l_sec_num];
    for (int i = 0; i < 3; i++) {
        int idx = (int)q5s_join_scan[1][i];
        if (idx != -1) {
            tab_l_col_type_size[i] = tab_b.getColTypeSize(idx);
            for (size_t j = 0; j < table_l_sec_num; j++) {
                table_l_sec_size[i][j] = table_l_sec_nrow[j] * tab_l_col_type_size[i];
                table_l_user[i][j] = tab_b.getColPointer(idx, 0, j);
            }
        }
    }
    for (size_t j = 0; j < table_l_sec_num; j++) {
        table_l_valid_user[j] = tab_b.getValColPointer(0, j);
        table_l_valid_sec_size[j] = (table_l_sec_nrow[j] + 7) / 8;
    }

    // the size of table l host buf
    int64_t table_l_size[3];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[1][i] != -1) {
            table_l_size[i] = l_nrow * tab_l_col_type_size[i];
        } else {
            table_l_size[i] = VEC_LEN * 8;
        }
    }

    // host buffer to be mapped with device buffer through OpenCL
    char* table_o[3];
    char* table_l[3];

    for (size_t i = 0; i < 3; i++) {
        table_o[i] = AllocHostBuf(1, table_o_size[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        table_l[i] = AllocHostBuf(1, table_l_size[i]);
    }

    char* din_valid_o = mm.aligned_alloc<char>((o_nrow + 7) / 8);
    char* din_valid_l = mm.aligned_alloc<char>((l_nrow + 7) / 8);

    // Num of vecs.
    const int size_apu_512 = 64;
    int64_t table_result_depth = (result_nrow + VEC_LEN - 1) / VEC_LEN; // 8 columns in one buffer
    int64_t table_out_col_type_size[4];
    int64_t table_result_size[4];
    // the pointer that points to tab_c cols
    char* table_out_user[4];
    // buffer used in join
    char* table_out[4];

    // size used for perf analyze
    double total_result_buf_size = 0;
    std::vector<int8_t> q5s_join_wr = jcfg.getShuffleWrite();
    for (size_t i = 0; i < 4; i++) {
        int shf_i = (int)q5s_join_wr[i];
#ifdef USER_DEBUG
        std::cout << "i: " << i << ", q5s_join_wr[i]: " << shf_i << std::endl;
#endif
        if (shf_i != -1) {
            table_out_user[i] = tab_c.getColPointer(shf_i);

            table_out_col_type_size[i] = tab_c.getColTypeSize(shf_i);
            table_result_size[i] = table_result_depth * VEC_LEN * table_out_col_type_size[i];
            total_result_buf_size += table_result_size[i];
        } else {
            table_result_size[i] = VEC_LEN;
            table_out_user[i] = mm.aligned_alloc<char>(VEC_LEN);
        }
    }
    total_result_buf_size = total_result_buf_size / 1024 / 1024;

    for (size_t i = 0; i < 4; i++) {
        std::cout << "table_result_size[" << i << "]: " << table_result_size[i] << std::endl;
        table_out[i] = AllocHostBuf(0, table_result_size[i]);
    }
    //--------------- metabuffer setup -----------------
    // using col0 and col1 buffer during build
    // setup build used meta input
    MetaTable meta_build_in;
    meta_build_in.setSecID(0);
    meta_build_in.setColNum(3); // unused, not affecting anything
    for (size_t i = 0; i < 3; i++) {
        meta_build_in.setCol(i, i, o_nrow);
    }

    // setup probe used meta input
    MetaTable meta_probe_in;
    meta_probe_in.setSecID(0);
    meta_probe_in.setColNum(3);
    for (size_t i = 0; i < 3; i++) {
        meta_probe_in.setCol(i, i, l_nrow);
    }

    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3. (When aggr is off)
    // when aggr is on, actually only using col0 is enough.
    // below example only illustrates the output buffers can be shuffled.
    // setup probe used meta output
    MetaTable meta_probe_out;
    meta_probe_out.setColNum(4);
    for (size_t i = 0; i < 4; i++) {
        meta_probe_out.setCol(i, i, result_nrow);
    }
    //--------------------------------------------
    //
    size_t build_probe_flag_0 = 0;
    size_t build_probe_flag_1 = 1;

    cl_int err;
    cl_kernel bkernel;
    bkernel = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);
    // probe kernel, pipeline used handle
    cl_kernel jkernel;
    jkernel = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);
    std::cout << "Kernel has been created\n";

    cl_mem_ext_ptr_t mext_cfg5s;
    cl_mem_ext_ptr_t mext_meta_build_in, mext_meta_probe_in, mext_meta_probe_out;

    mext_cfg5s = {XCL_BANK1, table_cfg5s, 0};

    mext_meta_build_in = {XCL_BANK1, meta_build_in.meta(), 0};
    mext_meta_probe_in = {XCL_BANK1, meta_probe_in.meta(), 0};
    mext_meta_probe_out = {XCL_BANK0, meta_probe_out.meta(), 0};

    // Map buffers
    cl_mem buf_table_o[3];
    cl_mem buf_table_l[3];
    cl_mem buf_table_out[4];
    cl_mem buf_cfg5s;
    cl_mem_ext_ptr_t mext_buf_valid_o = {XCL_BANK1, din_valid_o, 0};
    cl_mem buf_valid_o = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (o_nrow + 7) / 8 * sizeof(char), &mext_buf_valid_o, &err);

    cl_buffer_region sub_table_o_size[3];
    sub_table_o_size[0] = {buf_head[1][0], buf_size[1][0]};
    sub_table_o_size[1] = {buf_head[1][1], buf_size[1][1]};
    sub_table_o_size[2] = {buf_head[1][2], buf_size[1][2]};

    for (size_t i = 0; i < 3; i++) {
        buf_table_o[i] = clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                           CL_BUFFER_CREATE_TYPE_REGION, &sub_table_o_size[i], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
    }

    cl_buffer_region sub_table_l_size[3];
    sub_table_l_size[0] = {buf_head[1][3], buf_size[1][3]};
    sub_table_l_size[1] = {buf_head[1][4], buf_size[1][4]};
    sub_table_l_size[2] = {buf_head[1][5], buf_size[1][5]};
    for (size_t j = 0; j < 3; j++) {
        buf_table_l[j] = clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                           CL_BUFFER_CREATE_TYPE_REGION, &sub_table_l_size[j], &err);
    }

    cl_mem_ext_ptr_t mext_buf_valid_l = {XCL_BANK1, din_valid_l, 0};
    cl_mem buf_valid_l = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (l_nrow + 7) / 8 * sizeof(char), &mext_buf_valid_l, &err);

    cl_buffer_region sub_table_result_size[4];
    sub_table_result_size[0] = {buf_head[0][0], buf_size[0][0]};
    sub_table_result_size[1] = {buf_head[0][1], buf_size[0][1]};
    sub_table_result_size[2] = {buf_head[0][2], buf_size[0][2]};
    sub_table_result_size[3] = {buf_head[0][3], buf_size[0][3]};

    for (size_t j = 0; j < 4; j++) {
        buf_table_out[j] = clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                             CL_BUFFER_CREATE_TYPE_REGION, &sub_table_result_size[j], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
    }

    buf_cfg5s = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, (size_apu_512 * 14),
                               &mext_cfg5s, &err);
    if (err != CL_SUCCESS) {
        return MEM_ERROR;
    }

    cl_mem buf_meta_build_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (size_apu_512 * 8), &mext_meta_build_in, &err);
    if (err != CL_SUCCESS) {
        return MEM_ERROR;
    }

    cl_mem buf_meta_probe_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (size_apu_512 * 8), &mext_meta_probe_in, &err);
    if (err != CL_SUCCESS) {
        return MEM_ERROR;
    }
    cl_mem buf_meta_probe_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                               (size_apu_512 * 8), &mext_meta_probe_out, &err);
    if (err != CL_SUCCESS) {
        return MEM_ERROR;
    }

    std::cout << "buffers have been mapped.\n";

    // helper buffer sets
    // resident vec
    std::vector<cl_mem> non_loop_bufs;
    std::vector<cl_mem> resident_vec;
    resident_vec.push_back(buf_cfg5s);
    resident_vec.push_back(buf_meta_build_in);
    resident_vec.push_back(buf_meta_probe_in);
    resident_vec.push_back(buf_meta_probe_out);
    resident_vec.push_back(buf_valid_o);
    resident_vec.push_back(buf_valid_l);

    for (int i = 0; i < 3; i++) {
        non_loop_bufs.push_back(buf_table_o[i]);
    }
    non_loop_bufs.push_back(buf_cfg5s);
    non_loop_bufs.push_back(buf_valid_o);
    non_loop_bufs.push_back(buf_meta_build_in);
    non_loop_bufs.push_back(buf_meta_probe_out);

    std::vector<cl_mem> loop_in_bufs;
    for (int i = 0; i < 3; i++) {
        loop_in_bufs.push_back(buf_table_l[i]);
    }
    loop_in_bufs.push_back(buf_meta_probe_in);
    loop_in_bufs.push_back(buf_valid_l);

    std::vector<cl_mem> loop_out_bufs;
    for (int i = 0; i < 4; i++) {
        loop_out_bufs.push_back(buf_table_out[i]);
    }
    loop_out_bufs.push_back(buf_meta_probe_out);

    clEnqueueMigrateMemObjects(cq, resident_vec.size(), resident_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                               nullptr, nullptr);

    // set args and enqueue kernel
    int j = 0;
    clSetKernelArg(bkernel, j++, sizeof(size_t), &build_probe_flag_0);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_valid_o);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_cfg5s);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_build_in);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_probe_out);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[3]);
    for (int k = 0; k < PU_NM * 2; k++) {
        clSetKernelArg(bkernel, j++, sizeof(cl_mem), &dbuf_hbm[k]);
    }

    // set args and enqueue kernel
    j = 0;
    clSetKernelArg(jkernel, j++, sizeof(size_t), &build_probe_flag_1);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[0]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[1]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_l[2]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_valid_l);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_cfg5s);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_meta_probe_in);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_meta_probe_out);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[0]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[1]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[2]);
    clSetKernelArg(jkernel, j++, sizeof(cl_mem), &buf_table_out[3]);
    for (int k = 0; k < PU_NM * 2; k++) {
        clSetKernelArg(jkernel, j++, sizeof(cl_mem), &dbuf_hbm[k]);
    }

#ifdef JOIN_PERF_PROFILE
    gqe::utils::Timer timer;
#endif

    std::array<cl_event, 1> evt_tb_o;
    std::array<cl_event, 1> evt_bkrn;
    std::array<cl_event, 1> evt_tb_l;
    std::array<cl_event, 1> evt_pkrn;
    std::array<cl_event, 1> evt_tb_out;

// 1) copy Order table from host DDR to build kernel pinned host buffer
#ifdef JOIN_PERF_PROFILE
    timer.add(); // 0
#endif

    double memcpy_o_size = 0;
    int64_t tab_o_cpy_ptr[3] = {0};
    int64_t tab_o_cpy_val_ptr = 0;
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        for (size_t i = 0; i < 3; i++) {
            int idx = (int)(q5s_join_scan[0][i]);
            if (idx != -1) {
                memcpy(table_o[i] + tab_o_cpy_ptr[i], table_o_user[i][sec], table_o_sec_size[i][sec]);
                tab_o_cpy_ptr[i] += table_o_sec_size[i][sec];
                memcpy_o_size += table_o_sec_size[i][sec];
            }
        }
        if (tab_a.getRowIDEnableFlag() && tab_a.getValidEnableFlag()) {
            memcpy(din_valid_o + tab_o_cpy_val_ptr, table_o_valid_user[sec], table_o_valid_sec_size[sec]);
            tab_o_cpy_val_ptr += table_o_valid_sec_size[sec];

            memcpy_o_size += table_o_valid_sec_size[sec];
        }
    }

#ifdef JOIN_PERF_PROFILE
    timer.add(); // 1
#endif

    // 2) migrate order table data from host buffer to device buffer
    clEnqueueMigrateMemObjects(cq, non_loop_bufs.size(), non_loop_bufs.data(), 0, 0, nullptr, &evt_tb_o[0]);

    // 3) launch build kernel
    clEnqueueTask(cq, bkernel, 1, evt_tb_o.data(), &evt_bkrn[0]);
    clWaitForEvents(1, &evt_bkrn[0]);
// 4) copy L table from host DDR to build kernel pinned host buffer
#ifdef JOIN_PERF_PROFILE
    timer.add(); // 2
#endif

    double memcpy_l_size = 0;
    int64_t tab_l_cpy_ptr[3] = {0};
    int64_t tab_l_cpy_val_ptr = 0;
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        for (size_t i = 0; i < 3; i++) {
            int idx = (int)(q5s_join_scan[1][i]);
            if (idx != -1) {
                memcpy(table_l[i] + tab_l_cpy_ptr[i], table_l_user[i][sec], table_l_sec_size[i][sec]);
                tab_l_cpy_ptr[i] += table_l_sec_size[i][sec];
                memcpy_l_size += table_l_sec_size[i][sec];
            }
        }
        if (tab_b.getRowIDEnableFlag() && tab_b.getValidEnableFlag()) {
            memcpy(din_valid_l + tab_l_cpy_val_ptr, table_l_valid_user[sec], table_l_valid_sec_size[sec]);
            tab_l_cpy_val_ptr += table_l_valid_sec_size[sec];

            memcpy_l_size += table_l_valid_sec_size[sec];
        }
    }

#ifdef JOIN_PERF_PROFILE
    timer.add(); // 3
#endif

    // 5) migrate L table data from host buffer to device buffer
    clEnqueueMigrateMemObjects(cq, loop_in_bufs.size(), loop_in_bufs.data(), 0, 0, nullptr, &evt_tb_l[0]);

    // 6) launch probe kernel
    clEnqueueTask(cq, jkernel, 1, evt_tb_l.data(), &evt_pkrn[0]);

    // 7) migrate result data from device buffer to host buffer
    clEnqueueMigrateMemObjects(cq, loop_out_bufs.size(), loop_out_bufs.data(), CL_MIGRATE_MEM_OBJECT_HOST, 1,
                               evt_pkrn.data(), &evt_tb_out[0]);

    // 8) copy output data from pinned host buffer to user host buffer
    clWaitForEvents(1, &evt_tb_out[0]);

#ifdef JOIN_PERF_PROFILE
    timer.add(); // 4
#endif
    // get the results
    int64_t probe_out_nrow = meta_probe_out.getColLen();
    double memcpy_out_result_size = 0; // memcpy out data size
    for (size_t i = 0; i < 4; i++) {
        int shf_i = (int)q5s_join_wr[i];
        if (shf_i != -1) {
            memcpy(table_out_user[i], table_out[i], probe_out_nrow * table_out_col_type_size[i]);
            memcpy_out_result_size += probe_out_nrow * table_out_col_type_size[i];
        }
    }
    memcpy_out_result_size = memcpy_out_result_size / 1024 / 1024;

#ifdef JOIN_PERF_PROFILE
    timer.add(); // 5
#endif

// 9) calc and print the execution time of each phase
#ifdef JOIN_PERF_PROFILE
    cl_ulong start, end;
    double ev_ns;

    std::cout << std::endl << "============== execution time ==================" << std::endl;
    // 9.1) memcpy O
    std::cout << "1. memcpy left table size: " << (double)memcpy_o_size / 1024 / 1024 << " MB" << std::endl;
    std::cout << "1. memcpy left table time: " << timer.getMilliSec(0, 1) << " ms" << std::endl;
    std::cout << "1. memcpy left table throughput: "
              << (double)memcpy_o_size / 1024 / 1024 / 1024 / timer.getMilliSec(0, 1) * 1000 << " GB/s " << std::endl;

    // 9.2) migrate O
    clGetEventProfilingInfo(evt_tb_o[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_tb_o[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = (double)(end - start) / 1000000; // ns to ms
    std::cout << "2. migrate left table time: " << ev_ns << " ms" << std::endl;
    std::cout << "2. migrate left table throughput: " << (double)memcpy_o_size / 1024 / 1024 / 1024 / ev_ns * 1000
              << " GB/s " << std::endl;

    // 9.3) build kernel
    clGetEventProfilingInfo(evt_bkrn[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_bkrn[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = (double)(end - start) / 1000000;
    std::cout << "3. build kernel time: " << ev_ns << " ms" << std::endl;
    std::cout << "3. build kernel throughput: " << (double)memcpy_o_size / 1024 / 1024 / 1024 / ev_ns * 1000 << " GB/s "
              << std::endl;

    // 9.4) memcpy L
    std::cout << "4. memcpy right table size: " << (double)memcpy_l_size / 1024 / 1024 << " MB" << std::endl;
    std::cout << "4. memcpy right table time: " << timer.getMilliSec(2, 3) << " ms" << std::endl;
    std::cout << "4. memcpy right table throughput: "
              << (double)memcpy_l_size / 1024 / 1024 / 1024 / timer.getMilliSec(2, 3) * 1000 << " GB/s" << std::endl;

    // 9.5) migrate L
    clGetEventProfilingInfo(evt_tb_l[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_tb_l[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = (double)(end - start) / 1000000;
    std::cout << "5. migrate right table time: " << ev_ns << " ms" << std::endl;
    std::cout << "5. migrate right table throughput: " << (double)memcpy_l_size / 1024 / 1024 / 1024 / ev_ns * 1000
              << " GB/s " << std::endl;

    // 9.6) probe kernel
    clGetEventProfilingInfo(evt_pkrn[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_pkrn[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = (double)(end - start) / 1000000;
    std::cout << "6. probe kernel time: " << ev_ns << " ms" << std::endl;
    std::cout << "6. probe kernel throughput: " << (double)memcpy_l_size / 1024 / 1024 / 1024 / ev_ns * 1000 << " GB/s"
              << std::endl;

    // 9.7) migreate output
    clGetEventProfilingInfo(evt_tb_out[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_tb_out[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = (double)(end - start) / 1000000;
    std::cout << "7. migrate out time: " << ev_ns << " ms" << std::endl;
    std::cout << "7. migrate out throughput: " << total_result_buf_size / 1024 / ev_ns * 1000 << " GB/s " << std::endl;

    // 9.8) memcpy output
    std::cout << "8. memcpy result size: " << memcpy_out_result_size << " MB" << std::endl;
    std::cout << "8. memcpy result time: " << timer.getMilliSec(4, 5) << " ms" << std::endl;
    std::cout << "8. memcpy result throughput: " << memcpy_out_result_size / 1024 / timer.getMilliSec(4, 5) * 1000
              << " GB/s" << std::endl;

    // 9.9) e2e
    double total_input_size = (double)(memcpy_o_size + memcpy_l_size) / 1024 / 1024;
    std::cout << "9. end-to-end size: " << total_input_size << " MB" << std::endl;
    std::cout << "9. end-to-end time: " << timer.getMilliSec(0, 5) << " ms" << std::endl;
    std::cout << "9. end-to-end throughput: " << total_input_size / 1024 / timer.getMilliSec(0, 5) * 1000 << " GB/s "
              << std::endl;

    // =========== print result ===========
    printf("\n");

#endif

    // check the probe updated meta
    std::cout << "Output buffer has " << probe_out_nrow << " rows." << std::endl;
    tab_c.setRowNum(probe_out_nrow);

    delete[] table_o_sec_nrow;
    delete[] table_l_sec_nrow;
    //--------------release---------------
    for (int i = 0; i < 3; i++) {
        clReleaseMemObject(buf_table_o[i]);
    }
    for (int i = 0; i < 3; i++) {
        clReleaseMemObject(buf_table_l[i]);
    }
    for (int i = 0; i < 4; i++) {
        clReleaseMemObject(buf_table_out[i]);
    }

    clReleaseMemObject(buf_valid_o);
    clReleaseMemObject(buf_valid_l);

    clReleaseMemObject(buf_cfg5s);
    clReleaseMemObject(buf_meta_build_in);
    clReleaseMemObject(buf_meta_probe_in);
    clReleaseMemObject(buf_meta_probe_out);

    clReleaseEvent(evt_tb_o[0]);
    clReleaseEvent(evt_bkrn[0]);
    clReleaseEvent(evt_tb_l[0]);
    clReleaseEvent(evt_pkrn[0]);
    clReleaseEvent(evt_tb_out[0]);

    clReleaseKernel(bkernel);
    clReleaseKernel(jkernel);

    return SUCCESS;
}

ErrCode Joiner::join_sol1(Table& tab_a, Table& tab_b, Table& tab_c, JoinConfig& jcfg, StrategySet params) {
    gqe::utils::MM mm;

    gqe::utils::Timer tv_total;
    tv_total.add();

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    int64_t o_nrow = tab_a.getRowNum();
    int64_t l_nrow = tab_b.getRowNum();

    ap_uint<512>* table_cfg5s = jcfg.getJoinConfigBits();
    std::vector<std::vector<int8_t> > q5s_join_scan = jcfg.getShuffleScan();

    size_t o_valid_col_num = tab_a.getColNum();
    size_t l_valid_col_num = tab_b.getColNum();
    size_t out_valid_col_num = tab_c.getColNum();

    // table O
    int table_o_nrow = o_nrow;

    int64_t tab_o_col_type_size[3];
    int64_t tab_l_col_type_size[3];
    int64_t table_out_col_type_size[4];

    int sec_o = params.sec_o;
    // the O table might be composed of multi-sections
    tab_a.checkSecNum(sec_o);
    size_t table_o_sec_num = tab_a.getSecNum();
    int* table_o_sec_nrow = new int[table_o_sec_num];
    if (sec_o == 0) {
        for (size_t sec = 0; sec < table_o_sec_num; sec++) {
            table_o_sec_nrow[sec] = tab_a.getSecRowNum(sec);
        }
    } else {
        table_o_sec_nrow[0] = o_nrow;
    }

    // the real sec size for each section
    int64_t table_o_sec_size[3][table_o_sec_num];
    char* table_o_user[3][table_o_sec_num];
    for (int i = 0; i < 3; i++) {
        int idx = (int)q5s_join_scan[0][i];
        if (idx != -1) {
            tab_o_col_type_size[i] = tab_a.getColTypeSize(idx);
            for (size_t j = 0; j < table_o_sec_num; j++) {
                table_o_sec_size[i][j] = table_o_sec_nrow[j] * tab_o_col_type_size[i];
                table_o_user[i][j] = tab_a.getColPointer(idx, 0, j);
            }
        } else {
            tab_o_col_type_size[i] = 8;
        }
    }

    int64_t table_o_size[3];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[0][i] != -1) {
            table_o_size[i] = table_o_nrow * tab_o_col_type_size[i];
        } else {
            table_o_size[i] = VEC_LEN * 8;
        }
    }

    char* table_o_valid_user[table_o_sec_num];
    int64_t table_o_valid_sec_size[table_o_sec_num];
    if (tab_a.getValidEnableFlag()) {
        for (size_t j = 0; j < table_o_sec_num; j++) {
            table_o_valid_user[j] = tab_a.getValColPointer(sec_o, j);
            table_o_valid_sec_size[j] = (table_o_sec_nrow[j] + 7) / 8 * sizeof(char);
        }
    }

    // Table L
    // checks if we should bring the section number from json or calculate them locally (evenly divided)
    int sec_l = params.sec_l;
    tab_b.checkSecNum(sec_l);
    size_t table_l_sec_num = tab_b.getSecNum();
    int* table_l_sec_nrow = new int[table_l_sec_num];
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        table_l_sec_nrow[sec] = tab_b.getSecRowNum(sec);
    }

#ifdef USER_DEBUG
    std::cout << "table_l_sec_num: " << table_l_sec_num << std::endl;
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        std::cout << "Table L sec[" << sec << "] nrow: " << table_l_sec_nrow[sec] << std::endl;
    }
#endif

    // get max section L nrow
    int table_l_sec_nrow_max = 0;
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        if (table_l_sec_nrow[sec] > table_l_sec_nrow_max) {
            table_l_sec_nrow_max = table_l_sec_nrow[sec];
        }
    }
    // in case the data type is different for each col
    int64_t table_l_sec_size_max[3];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[1][i] != -1) {
            tab_l_col_type_size[i] = tab_b.getColTypeSize(q5s_join_scan[1][i]);
            table_l_sec_size_max[i] = table_l_sec_nrow_max * tab_l_col_type_size[i];
        } else {
            table_l_sec_size_max[i] = VEC_LEN * 8;
        }
    }

    // the real sec size for each section
    int64_t table_l_sec_size[3][table_l_sec_num];
    char* table_l_user[3][table_l_sec_num];
    char* table_l_valid_user[table_l_sec_num];
    int64_t table_l_valid_sec_size[table_l_sec_num];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[1][i] != -1) {
            for (size_t j = 0; j < table_l_sec_num; j++) {
                table_l_sec_size[i][j] = table_l_sec_nrow[j] * tab_l_col_type_size[i];
                table_l_user[i][j] = tab_b.getColPointer(q5s_join_scan[1][i], sec_l, j);
            }
        }
    }
    if (tab_b.getValidEnableFlag()) {
        for (size_t j = 0; j < table_l_sec_num; j++) {
            table_l_valid_user[j] = tab_b.getValColPointer(sec_l, j);
            table_l_valid_sec_size[j] = (table_l_sec_nrow[j] + 7) / 8 * sizeof(char);
        }
    }

    // host buffer to be mapped with device buffer through OpenCL
    char* table_o[3];
    for (size_t i = 0; i < 3; i++) {
        table_o[i] = AllocHostBuf(1, table_o_size[i]);
    }

    char* table_l[3][2];
    for (size_t i = 0; i < 3; i++) {
        table_l[i][0] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }
    for (size_t i = 0; i < 3; i++) {
        table_l[i][1] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }

    // Table C
    int64_t table_l_out_nrow = tab_c.getRowNum();
    std::cout << "table_l_out_nrow: " << table_l_out_nrow << std::endl;
    int64_t table_l_out_depth = (table_l_out_nrow + VEC_LEN - 1) / VEC_LEN;
    int64_t table_l_out_size[4];
    char* table_out_user[4];
    char* table_out[4][2];

    // get the cfg of sw wr shuffle
    std::vector<int8_t> q5s_join_wr = jcfg.getShuffleWrite();
    for (size_t i = 0; i < 4; i++) {
        int shf_i = (int)q5s_join_wr[i];
#ifdef USER_DEBUG
        std::cout << "i: " << i << ", q5s_join_wr[i]: " << shf_i << std::endl;
#endif
        if (shf_i != -1) {
            table_out_col_type_size[i] = tab_c.getColTypeSize(shf_i);
            table_l_out_size[i] = table_l_out_depth * VEC_LEN * table_out_col_type_size[i];

            table_out_user[i] = tab_c.getColPointer(shf_i);
        } else {
            table_l_out_size[i] = VEC_LEN * sizeof(char);
            table_out_user[i] = mm.aligned_alloc<char>(VEC_LEN);
        }
    }

    for (size_t i = 0; i < 4; i++) {
        table_out[i][0] = AllocHostBuf(0, table_l_out_size[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        table_out[i][1] = AllocHostBuf(0, table_l_out_size[i]);
    }

    //--------------- metabuffer setup -----------------
    // using col0 and col1 buffer during build
    // setup build used meta input, un-used columns are assigned to -1, as shown below.
    // 2 input columns data are valid, col0 and col1, not actually used cols can be marked as -1.
    MetaTable meta_build_in;
    meta_build_in.setSecID(0);
    meta_build_in.setColNum(3);
    for (size_t i = 0; i < 3; i++) {
        meta_build_in.setCol(i, i, table_o_nrow);
    }

    // setup probe used meta input
    MetaTable meta_probe_in[2];
    meta_probe_in[0].setSecID(0);
    meta_probe_in[1].setSecID(0);
    meta_probe_in[0].setColNum(3);
    meta_probe_in[1].setColNum(3);
    for (size_t i = 0; i < 3; i++) {
        /*
        meta_probe_in[0].setCol(i, i, table_l_sec_nrow[0]);
        meta_probe_in[1].setCol(i, i, table_l_sec_nrow[0]);
        */
        meta_probe_in[0].setCol(i, i, table_l_sec_nrow_max);
        meta_probe_in[1].setCol(i, i, table_l_sec_nrow_max);
    }

    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3. (When aggr is off)
    // when aggr is on, actually only using col0 is enough.
    // below example only illustrates the output buffers can be shuffled.
    // setup probe used meta output
    MetaTable meta_probe_out[2];
    meta_probe_out[0].setSecID(0);
    meta_probe_out[1].setSecID(0);
    meta_probe_out[0].setColNum(4);
    meta_probe_out[1].setColNum(4);
    for (size_t i = 0; i < 4; i++) {
        meta_probe_out[0].setCol(i, i, table_l_out_nrow);
        meta_probe_out[1].setCol(i, i, table_l_out_nrow);
    }

    //--------------------------------------------
    size_t build_probe_flag_0 = 0;
    size_t build_probe_flag_1 = 1;

    // Get CL devices.
    cl_int err;
    // build kernel
    cl_kernel bkernel;
    bkernel = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);
    // probe kernel, pipeline used handle
    cl_kernel pkernel[2];
    for (int i = 0; i < 2; i++) {
        pkernel[i] = clCreateKernel(prg, "gqeJoin", &err);
        logger.logCreateKernel(err);
    }
#ifdef USER_DEBUG
    std::cout << "Kernel has been created\n";
#endif

    cl_mem_ext_ptr_t mext_cfg5s;
    cl_mem_ext_ptr_t mext_meta_build_in, mext_meta_probe_in[2], mext_meta_probe_out[2];

    mext_meta_build_in = {XCL_BANK1, meta_build_in.meta(), 0};

    mext_meta_probe_in[0] = {XCL_BANK1, meta_probe_in[0].meta(), 0};
    mext_meta_probe_in[1] = {XCL_BANK1, meta_probe_in[1].meta(), 0};
    mext_meta_probe_out[0] = {XCL_BANK0, meta_probe_out[0].meta(), 0};
    mext_meta_probe_out[1] = {XCL_BANK0, meta_probe_out[1].meta(), 0};

    mext_cfg5s = {XCL_BANK1, table_cfg5s, 0};

    char* din_valid_o = mm.aligned_alloc<char>((o_nrow + 7) / 8);
    cl_mem_ext_ptr_t mext_buf_valid_o = {XCL_BANK1, din_valid_o, 0};
    cl_mem buf_valid_o = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (o_nrow + 7) / 8 * sizeof(char), &mext_buf_valid_o, &err);

    // Map buffers
    cl_mem buf_table_o[3];
    cl_mem buf_table_l[3][2];
    cl_mem buf_table_out[4][2];
    cl_mem buf_cfg5s;

    cl_buffer_region sub_table_o_size[3];
    sub_table_o_size[0] = {buf_head[1][0], buf_size[1][0]};
    sub_table_o_size[1] = {buf_head[1][1], buf_size[1][1]};
    sub_table_o_size[2] = {buf_head[1][2], buf_size[1][2]};

    for (size_t i = 0; i < 3; i++) {
        buf_table_o[i] = clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                           CL_BUFFER_CREATE_TYPE_REGION, &sub_table_o_size[i], &err);
    }

    cl_buffer_region sub_table_l_size[6];
    sub_table_l_size[0] = {buf_head[1][3], buf_size[1][3]};
    sub_table_l_size[1] = {buf_head[1][4], buf_size[1][4]};
    sub_table_l_size[2] = {buf_head[1][5], buf_size[1][5]};
    sub_table_l_size[3] = {buf_head[1][6], buf_size[1][6]};
    sub_table_l_size[4] = {buf_head[1][7], buf_size[1][7]};
    sub_table_l_size[5] = {buf_head[1][8], buf_size[1][8]};

    for (size_t j = 0; j < 3; j++) {
        buf_table_l[j][0] = clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION, &sub_table_l_size[j], &err);
        buf_table_l[j][1] = clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                              CL_BUFFER_CREATE_TYPE_REGION, &sub_table_l_size[3 + j], &err);
    }

    char* din_valid_l[2];
    cl_mem_ext_ptr_t mext_buf_valid_l[2];
    cl_mem buf_valid_l[2];
    for (int i = 0; i < 2; i++) {
        din_valid_l[i] = mm.aligned_alloc<char>((table_l_sec_nrow_max + 7) / 8);
        mext_buf_valid_l[i] = {XCL_BANK1, din_valid_l[i], 0};
        buf_valid_l[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        (table_l_sec_nrow_max + 7) / 8 * sizeof(char), &mext_buf_valid_l[i], &err);
    }

    cl_buffer_region sub_table_result_size[8];
    sub_table_result_size[0] = {buf_head[0][0], buf_size[0][0]};
    sub_table_result_size[1] = {buf_head[0][1], buf_size[0][1]};
    sub_table_result_size[2] = {buf_head[0][2], buf_size[0][2]};
    sub_table_result_size[3] = {buf_head[0][3], buf_size[0][3]};
    sub_table_result_size[4] = {buf_head[0][4], buf_size[0][4]};
    sub_table_result_size[5] = {buf_head[0][5], buf_size[0][5]};
    sub_table_result_size[6] = {buf_head[0][6], buf_size[0][6]};
    sub_table_result_size[7] = {buf_head[0][7], buf_size[0][7]};

    // the table_l_out_size is already re-sized by output-sw-shuffle
    for (size_t j = 0; j < 4; j++) {
        buf_table_out[j][0] = clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                                CL_BUFFER_CREATE_TYPE_REGION, &sub_table_result_size[j], &err);
        buf_table_out[j][1] = clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                                CL_BUFFER_CREATE_TYPE_REGION, &sub_table_result_size[4 + j], &err);
    }

    buf_cfg5s = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                               (sizeof(ap_uint<512>) * 14), &mext_cfg5s, &err);

    // meta buffers
    cl_mem buf_meta_build_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (sizeof(ap_uint<512>) * 8), &mext_meta_build_in, &err);

    cl_mem buf_meta_probe_in[2];
    buf_meta_probe_in[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (sizeof(ap_uint<512>) * 8), &mext_meta_probe_in[0], &err);

    buf_meta_probe_in[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (sizeof(ap_uint<512>) * 8), &mext_meta_probe_in[1], &err);
    cl_mem buf_meta_probe_out[2];
    buf_meta_probe_out[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (sizeof(ap_uint<512>) * 8), &mext_meta_probe_out[0], &err);
    buf_meta_probe_out[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (sizeof(ap_uint<512>) * 8), &mext_meta_probe_out[1], &err);
#ifdef USER_DEBUG
    std::cout << "Temp buffers have been mapped.\n";
#endif

    // helper buffer sets
    std::vector<cl_mem> resident_vec;
    resident_vec.push_back(buf_cfg5s);
    resident_vec.push_back(buf_valid_o);
    resident_vec.push_back(buf_valid_l[0]);
    resident_vec.push_back(buf_valid_l[1]);
    resident_vec.push_back(buf_meta_build_in);
    resident_vec.push_back(buf_meta_probe_in[0]);
    resident_vec.push_back(buf_meta_probe_in[1]);
    resident_vec.push_back(buf_meta_probe_out[0]);
    resident_vec.push_back(buf_meta_probe_out[1]);

    std::vector<cl_mem> non_loop_bufs;
    for (int i = 0; i < 3; i++) {
        non_loop_bufs.push_back(buf_table_o[i]);
    }
    non_loop_bufs.push_back(buf_cfg5s);
    non_loop_bufs.push_back(buf_valid_o);
    non_loop_bufs.push_back(buf_meta_build_in);
    non_loop_bufs.push_back(buf_meta_probe_out[0]);
    non_loop_bufs.push_back(buf_meta_probe_out[1]);

    std::vector<cl_mem> loop_in_bufs[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            loop_in_bufs[k].push_back(buf_table_l[i][k]);
        }
        loop_in_bufs[k].push_back(buf_meta_probe_in[k]);
        loop_in_bufs[k].push_back(buf_valid_l[k]);
    }

    std::vector<cl_mem> loop_out_bufs[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 4; i++) {
            loop_out_bufs[k].push_back(buf_table_out[i][k]);
        }
        loop_out_bufs[k].push_back(buf_meta_probe_out[k]);
    }

    // make sure all buffers are resident on device
    clEnqueueMigrateMemObjects(cq, resident_vec.size(), resident_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                               nullptr, nullptr);

    // set build kernel args
    int j = 0;
    clSetKernelArg(bkernel, j++, sizeof(size_t), &build_probe_flag_0);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_valid_o);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_cfg5s);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_build_in);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_probe_out[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[0][0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[1][0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[2][0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_out[3][0]);
    for (int k = 0; k < PU_NM * 2; k++) {
        clSetKernelArg(bkernel, j++, sizeof(cl_mem), &dbuf_hbm[k]);
    }

    // set probe kernel args
    for (int i = 0; i < 2; i++) {
        j = 0;
        clSetKernelArg(pkernel[i], j++, sizeof(size_t), &build_probe_flag_1);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_l[0][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_l[1][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_l[2][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_valid_l[i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_cfg5s);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_meta_probe_in[i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_meta_probe_out[i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_out[0][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_out[1][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_out[2][i]);
        clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &buf_table_out[3][i]);
        for (int k = 0; k < PU_NM * 2; k++) {
            clSetKernelArg(pkernel[i], j++, sizeof(cl_mem), &dbuf_hbm[k]);
        }
    }

// =================== starting the pipelined task =========================
#ifdef JOIN_PERF_PROFILE
    gqe::utils::Timer tv_build_memcpyin;
    gqe::utils::Timer tv_probe_memcpyin[table_l_sec_num];
    gqe::utils::Timer tv_probe_memcpyout[table_l_sec_num];
#endif
    gqe::utils::Timer tv;

    std::vector<cl_event> evt_build_h2d;
    std::vector<cl_event> evt_build_krn;
    evt_build_h2d.resize(1);
    evt_build_krn.resize(1);

    tv.add(); // 0

//--------------- build --------------------
// copy and migrate shared O to device
#ifdef JOIN_PERF_PROFILE
    tv_build_memcpyin.add(); // 0
#endif
    int64_t tab_o_cpy_ptr[3] = {0};
    int64_t tab_o_cpy_val_ptr = 0;
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        for (size_t i = 0; i < 3; i++) {
            int idx = (int)(q5s_join_scan[0][i]);
            if (idx != -1) {
                memcpy(table_o[i] + tab_o_cpy_ptr[i], table_o_user[i][sec], table_o_sec_size[i][sec]);
                tab_o_cpy_ptr[i] += table_o_sec_size[i][sec];
            }
        }

        if (tab_a.getRowIDEnableFlag() && tab_a.getValidEnableFlag()) {
            memcpy(din_valid_o + tab_o_cpy_val_ptr, table_o_valid_user[sec], table_o_valid_sec_size[sec]);
            tab_o_cpy_val_ptr += table_o_valid_sec_size[sec];
        }
    }
#ifdef JOIN_PERF_PROFILE
    tv_build_memcpyin.add(); // 1
#endif

    // h2d
    clEnqueueMigrateMemObjects(cq, non_loop_bufs.size(), non_loop_bufs.data(), 0, 0, nullptr, &evt_build_h2d[0]);

    // launch build kernel
    clEnqueueTask(cq, bkernel, evt_build_h2d.size(), evt_build_h2d.data(), &evt_build_krn[0]);

    clWaitForEvents(evt_build_krn.size(), evt_build_krn.data());

    // ------------------- probe -------------------
    // dep events
    std::vector<std::vector<cl_event> > evt_probe_h2d(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_probe_krn(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_probe_d2h(table_l_sec_num);
    for (size_t i = 0; i < table_l_sec_num; ++i) {
        evt_probe_h2d[i].resize(1);
        evt_probe_krn[i].resize(1);
        evt_probe_d2h[i].resize(1);
    }
    // kernel dep events, to guarantee the kernel time is accurate
    std::vector<std::vector<cl_event> > evt_dep(table_l_sec_num);
    evt_dep[0].resize(1);
    for (size_t i = 1; i < table_l_sec_num; ++i) {
        evt_dep[i].resize(2);
    }

    // loop: copy result, copy and migrate L, sched migrate L, kernel, and migrate back result.
    int64_t nrow_all_results[table_l_sec_num];
    int64_t probe_out_nrow_accu = 0;
    for (size_t i = 0; i < table_l_sec_num + 2; ++i) {
        int k_id = i % 2;

        // 1) copy L section data from host DDR to pinned host DDR
        if (i > 1) {
            // if run loop/sec_l s >1, need to wait the i-2 loop kernel finish, then
            // memcpy to host input buffer
            clWaitForEvents(evt_probe_krn[i - 2].size(), evt_probe_krn[i - 2].data());
        }
        if (i < table_l_sec_num) {
#ifdef JOIN_PERF_PROFILE
            tv_probe_memcpyin[i].add(); // 0
#endif
            for (size_t j = 0; j < 3; j++) {
                int idx = (int)(q5s_join_scan[1][j]);
                if (idx != -1) {
                    memcpy(table_l[j][k_id], table_l_user[j][i], table_l_sec_size[j][i]);
                }
            }
            if (tab_b.getRowIDEnableFlag() && tab_b.getValidEnableFlag()) {
                memcpy(din_valid_l[k_id], table_l_valid_user[i], table_l_valid_sec_size[i]);
            }
#ifdef JOIN_PERF_PROFILE
            tv_probe_memcpyin[i].add(); // 1
#endif
        }
        // 2) migrate section L data from host DDR to dev DDR
        if (i < table_l_sec_num) {
            for (size_t k = 0; k < 3; k++) {
                meta_probe_in[k_id].setCol(k, k, table_l_sec_nrow[i]);
            }
            meta_probe_in[k_id].setSecID(i);
            meta_probe_in[k_id].meta();
            // migrate h2d
            if (i > 1) {
                clEnqueueMigrateMemObjects(cq, loop_in_bufs[k_id].size(), loop_in_bufs[k_id].data(), 0,
                                           evt_probe_krn[i - 2].size(), evt_probe_krn[i - 2].data(),
                                           &evt_probe_h2d[i][0]);
            } else {
                clEnqueueMigrateMemObjects(cq, loop_in_bufs[k_id].size(), loop_in_bufs[k_id].data(), 0, 0, nullptr,
                                           &evt_probe_h2d[i][0]);
            }
        }
        // 5) memcpy the output data back to user host buffer, for i-2 round
        if (i > 1) {
            clWaitForEvents(evt_probe_d2h[i - 2].size(), evt_probe_d2h[i - 2].data());
            // get the output nrow
            int64_t probe_out_nrow = meta_probe_out[k_id].getColLen();
#ifdef USER_DEBUG
            std::cout << "i: " << i - 2 << ", Output buffer has " << probe_out_nrow << " rows." << std::endl;
#endif
            nrow_all_results[i - 2] = probe_out_nrow;
// memcpy only valid results back
#ifdef JOIN_PERF_PROFILE
            tv_probe_memcpyout[i - 2].add(); // 0
#endif
            for (size_t j = 0; j < 4; j++) {
                int shf_i = (int)q5s_join_wr[j];
                if (shf_i != -1) {
                    memcpy(table_out_user[j] + (int64_t)probe_out_nrow_accu * table_out_col_type_size[j],
                           table_out[j][k_id], (int64_t)probe_out_nrow * table_out_col_type_size[j]);
                }
            }
            probe_out_nrow_accu += probe_out_nrow;
#ifdef JOIN_PERF_PROFILE
            tv_probe_memcpyout[i - 2].add(); // 1
#endif
        }
        if (i < table_l_sec_num) {
            // 3) launch kernel
            clWaitForEvents(evt_build_krn.size(), evt_build_krn.data());
            evt_dep[i][0] = evt_probe_h2d[i][0];
            if (i > 0) {
                evt_dep[i][1] = evt_probe_krn[i - 1][0];
            }
            clEnqueueTask(cq, pkernel[k_id], evt_dep[i].size(), evt_dep[i].data(), &evt_probe_krn[i][0]);

            // 4) migrate result data from device buffer to host buffer
            clEnqueueMigrateMemObjects(cq, loop_out_bufs[k_id].size(), loop_out_bufs[k_id].data(),
                                       CL_MIGRATE_MEM_OBJECT_HOST, evt_probe_krn[i].size(), evt_probe_krn[i].data(),
                                       &evt_probe_d2h[i][0]);
        }
    }
    tv.add(); // 1

    tv_total.add();
    auto tvtime_total = tv_total.getMilliSec();

    std::cout << "total time for map join solution 1 = " << (double)tvtime_total << " ms" << std::endl;

    // =================== Print results =========================

    // compute time
    long kernel_ex_time = 0;
    cl_ulong start, end;
    long ev_ns;
    clGetEventProfilingInfo(evt_build_krn[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
    clGetEventProfilingInfo(evt_build_krn[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
    ev_ns = end - start;
    kernel_ex_time += ev_ns;
    for (size_t i = 0; i < table_l_sec_num; i++) {
        clGetEventProfilingInfo(evt_probe_krn[i][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_probe_krn[i][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        ev_ns = end - start;
        kernel_ex_time += ev_ns;
    }
    // compute result
    int64_t out_nrow_sum = 0;
    for (size_t i = 0; i < table_l_sec_num; i++) {
        // check the probe updated meta
        int64_t out_nrow = nrow_all_results[i];
        out_nrow_sum += out_nrow;
#ifdef USER_DEBUG
        std::cout << "GQE result, sec: " << i << ", nrow: " << out_nrow << std::endl;
#endif
    }
    tab_c.setRowNum(out_nrow_sum);

    double in1_bytes = (double)o_nrow * sizeof(int64_t) * o_valid_col_num / 1024 / 1024;
    double in2_bytes = (double)l_nrow * sizeof(int64_t) * l_valid_col_num / 1024 / 1024;
    double out_bytes = (double)out_nrow_sum * sizeof(int64_t) * out_valid_col_num / 1024 / 1024;

    std::cout << "-----------------------Input/Output Info-----------------------" << std::endl;
    std::cout << "Table" << std::setw(20) << "Column Number" << std::setw(30) << "Row Number" << std::endl;
    std::cout << "L" << std::setw(24) << o_valid_col_num << std::setw(30) << o_nrow << std::endl;
    std::cout << "R" << std::setw(24) << l_valid_col_num << std::setw(30) << l_nrow << std::endl;
    std::cout << "LxR" << std::setw(22) << out_valid_col_num << std::setw(30) << out_nrow_sum << std::endl;
    std::cout << "-----------------------Data Transfer Info-----------------------" << std::endl;
    std::cout << "H2D size (Left Table) = " << in1_bytes << " MB" << std::endl;
    std::cout << "H2D size (Right Table) = " << in2_bytes << " MB" << std::endl;
    std::cout << "D2H size = " << out_bytes << " MB" << std::endl;

    std::cout << "-----------------------Performance Info-----------------------" << std::endl;
    double all_bytes = (double)(in1_bytes + in2_bytes) / 1024;
    printf("Total kernel execution time : %ld.%03ld msec\n", kernel_ex_time / 1000000,
           (kernel_ex_time % 1000000) / 1000);
    double tvtime = tv.getMilliSec();
    printf("End-to-end time: %lf msec\n", tvtime);
    printf("End-to-end throughput: %lf GB/s\n", all_bytes / (tvtime / 1000));

    delete[] table_o_sec_nrow;
    delete[] table_l_sec_nrow;

    //--------------release---------------
    for (int i = 0; i < 3; i++) {
        clReleaseMemObject(buf_table_o[i]);
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l[i][k]);
        }
    }
    for (int i = 0; i < 4; i++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_out[i][k]);
        }
    }

    clReleaseMemObject(buf_valid_o);
    clReleaseMemObject(buf_valid_l[0]);
    clReleaseMemObject(buf_valid_l[1]);

    clReleaseMemObject(buf_cfg5s);

    clReleaseMemObject(buf_meta_build_in);
    clReleaseMemObject(buf_meta_probe_in[0]);
    clReleaseMemObject(buf_meta_probe_in[1]);
    clReleaseMemObject(buf_meta_probe_out[0]);
    clReleaseMemObject(buf_meta_probe_out[1]);

    clReleaseEvent(evt_build_h2d[0]);
    clReleaseEvent(evt_build_krn[0]);
    for (size_t i = 0; i < table_l_sec_num; i++) {
        clReleaseEvent(evt_probe_h2d[i][0]);
        clReleaseEvent(evt_probe_krn[i][0]);
        clReleaseEvent(evt_probe_d2h[i][0]);
    }

    clReleaseKernel(bkernel);
    clReleaseKernel(pkernel[0]);
    clReleaseKernel(pkernel[1]);

    return SUCCESS;
}

struct queue_struct_join {
    // the sec index
    int sec;
    // the partition index
    int p;
    // the nrow setup of MetaTable, only the last round nrow is different to
    // per_slice_nrow in probe
    int64_t meta_nrow;
    // updating meta info (nrow) for each partition&slice, due to async, this
    // change is done in threads
    MetaTable* meta;
    // dependency event num
    int num_event_wait_list;
    // dependency events
    cl_event* event_wait_list;
    // user event to trace current memcpy operation
    cl_event* event;
    // the valid col index
    std::vector<int> col_idx;
    // memcpy src locations

    char* ptr_src[4];
    // ----- part o memcpy in used -----
    // data size of memcpy in
    int type_size[4];
    int64_t size[4];
    // memcpy dst locations
    char* ptr_dst[4];
    // ----- part o memcpy out used -----
    int partition_num;
    // the allocated size (nrow) of each partititon out buffer
    int64_t part_max_nrow_512;
    // memcpy dst locations, used in part memcpy out
    char*** part_ptr_dst;
    // ----- probe memcpy used -----
    int slice;
    // the nrow of first (slice_num - 1) rounds, only valid in probe memcpy in
    int64_t per_slice_nrow;
    // buf_head addr
    int64_t buf_head[4];
    cl_command_queue cq;
    cl_mem dbuf;
};
//
class threading_pool {
   public:
    const int size_apu_512 = 64;
    std::thread part_o_in_t;
    std::thread part_o_d2h_t;
    std::thread part_o_out_t;
    std::thread part_l_in_ping_t;
    std::thread part_l_in_pong_t;
    std::thread part_l_d2h_t;
    std::thread part_l_out_ping_t;
    std::thread part_l_out_pong_t;
    std::thread build_in_t;
    std::thread probe_in_ping_t;
    std::thread probe_in_pong_t;
    std::thread probe_d2h_t;
    std::thread probe_out_t;

    std::mutex m;
    std::condition_variable cv;
    int cur;

    std::queue<queue_struct_join> q0;      // part o memcpy in used queue
    std::queue<queue_struct_join> q1_d2h;  // part o d2h used queue
    std::queue<queue_struct_join> q1;      // part o memcpy out used queue
    std::queue<queue_struct_join> q2_ping; // part l memcpy in used queue
    std::queue<queue_struct_join> q2_pong; // part l memcpy in used queue
    std::queue<queue_struct_join> q3_d2h;  // part l d2h in used queue
    std::queue<queue_struct_join> q3_ping; // part l memcpy out used queue
    std::queue<queue_struct_join> q3_pong; // part l memcpy out used queue
    std::queue<queue_struct_join> q4;      // build memcpy in used queue
    std::queue<queue_struct_join> q5_ping; // probe memcpy in used queue
    std::queue<queue_struct_join> q5_pong; // probe memcpy in used queue
    std::queue<queue_struct_join> q6_d2h;  // probe d2h out used queue
    std::queue<queue_struct_join> q6;      // probe memcpy out used queue

    // the flag indicate each thread is running
    std::atomic<bool> q0_run;
    std::atomic<bool> q1_d2h_run;
    std::atomic<bool> q1_run;
    std::atomic<bool> q2_ping_run;
    std::atomic<bool> q2_pong_run;
    std::atomic<bool> q3_ping_run;
    std::atomic<bool> q3_d2h_run;
    std::atomic<bool> q3_pong_run;
    std::atomic<bool> q4_run;
    std::atomic<bool> q5_run_ping;
    std::atomic<bool> q5_run_pong;
    std::atomic<bool> q6_d2h_run;
    std::atomic<bool> q6_run;

    // the nrow of each partition
    int64_t o_new_part_offset[256];
    std::atomic<int64_t> l_new_part_offset[256];
    int64_t probe_out_nrow_accu = 0;
    int64_t toutrow[256][32];

    // the buffer size of each output partition of Tab L.
    int64_t l_partition_out_col_part_nrow_max;
    // the buffer size of each output partition of Tab O.
    int64_t o_partition_out_col_part_nrow_max;

    // constructor
    threading_pool(){};

    // table O memcpy in thread
    // -----------------------------------------------------------------------
    // memcpy(table_o_partition_in_col[0][kid], table_o_user_col0_sec[sec],
    // sizeof(TPCH_INT) * table_o_sec_depth[sec]);
    // memcpy(table_o_partition_in_col[1][kid], table_o_user_col1_sec[sec],
    // sizeof(TPCH_INT) * table_o_sec_depth[sec]);
    // meta_o_partition_in[kid].setColNum(2);
    // meta_o_partition_in[kid].setCol(0, 0, table_o_sec_depth[sec]);
    // meta_o_partition_in[kid].setCol(1, 1, table_o_sec_depth[sec]);
    // meta_o_partition_in[kid].meta();
    // -----------------------------------------------------------------------
    void part_o_memcpy_in_t() {
        while (q0_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part o memcpy in " << std::endl;
#endif
            while (!q0.empty()) {
                queue_struct_join q = q0.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                    }
                }

                q.meta->setSecID(q.sec);
                q.meta->setColNum(col_num);
                for (int i = 0; i < col_num; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element in queue after processing it.
                q0.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();
                double q_total_size = 0;
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        q_total_size += q.size[i];
                    }
                }
                double input_memcpy_size = q_total_size / 1024 / 1024;

                std::cout << "Tab O sec: " << q.sec << " memcpy in, size: " << input_memcpy_size
                          << " MB, time: " << tvtime
                          << " ms, throughput: " << input_memcpy_size / 1024 / (tvtime / 1000) << "GB/s" << std::endl;
#endif
            }
        }
    };

    // table O d2h thread
    void part_o_d2h_out_t() {
        while (q1_d2h_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part o d2h" << std::endl;
#endif
            while (!q1_d2h.empty()) {
                queue_struct_join q = q1_d2h.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int partition_num = q.partition_num;
                std::vector<cl_mem> part_o_sub_vec;
                // create the real data size sub-buffers
                cl_mem buf_part_o_real_out[3][partition_num];
                cl_buffer_region sub_part_o_out_real_size[3][partition_num];
                int* nrow_per_part_o = q.meta->getPartLen();
                cl_int err;
                int col_num = q.col_idx.size();
                for (int p = 0; p < partition_num; p++) {
                    // std::cout << "P: " << p << ", nrow: " << nrow_per_part_o[p] << std::endl;
                    if (nrow_per_part_o[p] > 0) {
                        for (int col = 0; col < col_num; col++) {
                            int idx = q.col_idx[col];
                            if (idx != -1) {
                                sub_part_o_out_real_size[col][p] = {
                                    q.buf_head[col] + p * q.part_max_nrow_512 * sizeof(ap_uint<512>),
                                    nrow_per_part_o[p] * sizeof(int64_t)};

                                buf_part_o_real_out[col][p] = clCreateSubBuffer(
                                    q.dbuf, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                    &sub_part_o_out_real_size[col][p], &err);
                                part_o_sub_vec.push_back(buf_part_o_real_out[col][p]);
                            }
                        }
                    }
                }
                cl_event d2h_evt;
                if (part_o_sub_vec.size() > 0) {
                    clEnqueueMigrateMemObjects(q.cq, part_o_sub_vec.size(), part_o_sub_vec.data(),
                                               CL_MIGRATE_MEM_OBJECT_HOST, 0, nullptr, &d2h_evt);
                    clWaitForEvents(1, &d2h_evt);
                }

                // relase thread used sub-buffer and events
                for (int p = 0; p < partition_num; p++) {
                    if (nrow_per_part_o[p] > 0) {
                        for (int col = 0; col < col_num; col++) {
                            int idx = q.col_idx[col];
                            if (idx != -1) clReleaseMemObject(buf_part_o_real_out[col][p]);
                        }
                    }
                }
                if (part_o_sub_vec.size() > 0) {
                    clReleaseEvent(d2h_evt);
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q1_d2h.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();
                int valid_col = 0;
                for (int i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }
                int64_t output_d2h_nrow = 0;
                for (int p = 0; p < q.partition_num; p++) {
                    output_d2h_nrow += nrow_per_part_o[p];
                }

                double output_d2h_size = (double)output_d2h_nrow * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "Tab O sec: " << q.sec << " real d2h, size: " << output_d2h_size << " MB, with "
                          << q.partition_num * valid_col << " times, total time: " << tvtime
                          << " ms, throughput: " << output_d2h_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    };

    // table O memcpy out thread
    void part_o_memcpy_out_t() {
        while (q1_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part o memcpy out" << std::endl;
#endif
            while (!q1.empty()) {
                queue_struct_join q = q1.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif

                int64_t o_partition_out_col_part_depth = q.part_max_nrow_512;

                int* nrow_per_part_o = q.meta->getPartLen();

                for (int p = 0; p < q.partition_num; p++) {
                    // casting to 64-bit
                    int64_t sec_partitioned_res_part_nrow = nrow_per_part_o[p];
                    // std::cout << "Part O, sec: " << q.sec << ", partition: " << p << ", nrow: " <<
                    // sec_partitioned_res_part_nrow << std::endl;
                    // error out when the partition out buf size is smaller than output nrow
                    if (sec_partitioned_res_part_nrow > o_partition_out_col_part_nrow_max) {
                        std::cerr << "partition out nrow: " << sec_partitioned_res_part_nrow
                                  << ", buffer size: " << o_partition_out_col_part_nrow_max << std::endl;
                        std::cerr << "ERROR: Table O output partition size is smaller than required!" << std::endl;
                        exit(1);
                    }
                    int64_t offset = o_new_part_offset[p];
                    o_new_part_offset[p] += sec_partitioned_res_part_nrow;
                    int col_num = q.col_idx.size();
                    for (int i = 0; i < col_num; i++) {
                        int idx = q.col_idx[i];
                        if (idx != -1) {
                            memcpy(q.part_ptr_dst[p][i] + offset * q.type_size[i],
                                   q.ptr_src[i] + p * o_partition_out_col_part_depth * (size_apu_512),

                                   q.type_size[i] * sec_partitioned_res_part_nrow);
                        }
                    }
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q1.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();
                int valid_col = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }
                int64_t output_memcpy_nrow = 0;
                for (int p = 0; p < q.partition_num; p++) {
                    output_memcpy_nrow += nrow_per_part_o[p];
                }

                double output_memcpy_size = (double)output_memcpy_nrow * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "Tab O sec: " << q.sec << " memcpy out, size: " << output_memcpy_size << " MB, with "
                          << q.partition_num * valid_col << " times, total time: " << tvtime
                          << " ms, throughput: " << output_memcpy_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    };
    // table L memcpy in thread
    void part_l_memcpy_in_ping_t() {
        while (q2_ping_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part l memcpy in" << std::endl;
#endif
            while (!q2_ping.empty()) {
                queue_struct_join q = q2_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                    }
                }

                q.meta->setSecID(q.sec);
                q.meta->setColNum(3);
                for (int i = 0; i < 3; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q2_ping.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double input_memcpy_size = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) input_memcpy_size += q.size[i];
                }

                input_memcpy_size = input_memcpy_size / 1024 / 1024;

                std::cout << "Tab L sec: " << q.sec << " memcpy in, size: " << input_memcpy_size
                          << " MB, time: " << tvtime
                          << " ms, throughput: " << input_memcpy_size / 1024 / (tvtime / 1000) << "GB/s" << std::endl;
#endif
            }
        }
    };

    // table L memcpy in thread
    void part_l_memcpy_in_pong_t() {
        while (q2_pong_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part l memcpy in pong" << std::endl;
#endif
            while (!q2_pong.empty()) {
                queue_struct_join q = q2_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                    }
                }

                q.meta->setSecID(q.sec);
                q.meta->setColNum(3);
                for (int i = 0; i < 3; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }

                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q2_pong.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double input_memcpy_size = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) input_memcpy_size += q.size[i];
                }

                input_memcpy_size = input_memcpy_size / 1024 / 1024;

                std::cout << "Tab L sec: " << q.sec << " memcpy in, size: " << input_memcpy_size
                          << " MB, time: " << tvtime
                          << " ms, throughput: " << input_memcpy_size / 1024 / (tvtime / 1000) << "GB/s" << std::endl;
#endif
            }
        }
    };

    // table L d2h thread
    void part_l_d2h_out_t() {
        while (q3_d2h_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part l d2h" << std::endl;
#endif
            while (!q3_d2h.empty()) {
                queue_struct_join q = q3_d2h.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int partition_num = q.partition_num;
                std::vector<cl_mem> part_l_sub_vec;
                // create the real data size sub-buffers
                cl_mem buf_part_l_real_out[3][partition_num];
                cl_buffer_region sub_part_l_out_real_size[3][partition_num];
                int* nrow_per_part_l = q.meta->getPartLen();
                cl_int err;
                int col_num = q.col_idx.size();
                for (int p = 0; p < partition_num; p++) {
                    // std::cout << "Table L P: " << p << ", nrow: " << nrow_per_part_l[p] << std::endl;
                    if (nrow_per_part_l[p] > 0) {
                        for (int col = 0; col < col_num; col++) {
                            int idx = q.col_idx[col];
                            if (idx != -1) {
                                sub_part_l_out_real_size[col][p] = {
                                    q.buf_head[col] + p * q.part_max_nrow_512 * sizeof(ap_uint<512>),
                                    nrow_per_part_l[p] * sizeof(int64_t)};

                                buf_part_l_real_out[col][p] = clCreateSubBuffer(
                                    q.dbuf, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                                    &sub_part_l_out_real_size[col][p], &err);

                                part_l_sub_vec.push_back(buf_part_l_real_out[col][p]);
                            }
                        }
                    }
                }

                cl_event d2h_evt;
                if (part_l_sub_vec.size() > 0) {
                    clEnqueueMigrateMemObjects(q.cq, part_l_sub_vec.size(), part_l_sub_vec.data(),
                                               CL_MIGRATE_MEM_OBJECT_HOST, 0, nullptr, &d2h_evt);

                    clWaitForEvents(1, &d2h_evt);
                }

                // relase thread used sub-buffer and events
                for (int p = 0; p < partition_num; p++) {
                    if (nrow_per_part_l[p] > 0) {
                        for (int col = 0; col < col_num; col++) {
                            int idx = q.col_idx[col];
                            if (idx != -1) clReleaseMemObject(buf_part_l_real_out[col][p]);
                        }
                    }
                }
                if (part_l_sub_vec.size() > 0) {
                    clReleaseEvent(d2h_evt);
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q3_d2h.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();
                int valid_col = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }
                int64_t output_d2h_nrow = 0;
                for (int p = 0; p < q.partition_num; p++) {
                    output_d2h_nrow += nrow_per_part_l[p];
                }

                double output_d2h_size = (double)output_d2h_nrow * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "Tab L sec: " << q.sec << " real d2h, size: " << output_d2h_size << " MB, with "
                          << q.partition_num * valid_col << " times, total time: " << tvtime
                          << " ms, throughput: " << output_d2h_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    };

    // table L memcpy out thread
    void part_l_memcpy_out_ping_t() {
        while (q3_ping_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part l memcpy out ping" << std::endl;
#endif
            while (!q3_ping.empty()) {
                queue_struct_join q = q3_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int64_t l_partition_out_col_part_depth = q.part_max_nrow_512;

                int* nrow_per_part_l = q.meta->getPartLen();
                int64_t offset[q.partition_num];

                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return cur == q.sec; });

                    for (int p = 0; p < q.partition_num; ++p) {
                        offset[p] = l_new_part_offset[p];
                        l_new_part_offset[p] += nrow_per_part_l[p];
                    }
                    // let the other thread proceed to next round
                    cur++;
                    cv.notify_one();
                }

                for (int p = 0; p < q.partition_num; ++p) {
                    int64_t sec_partitioned_res_part_nrow = nrow_per_part_l[p];
                    // std::cout << "Part L, sec: " << q.sec << ", partition: " << p << ", nrow: " <<
                    // sec_partitioned_res_part_nrow << std::endl;

                    if (sec_partitioned_res_part_nrow > l_partition_out_col_part_nrow_max) {
                        std::cerr << "partition out nrow: " << sec_partitioned_res_part_nrow
                                  << ", buffer size: " << l_partition_out_col_part_nrow_max << std::endl;
                        std::cerr << "ERROR: Table L output partition size is smaller than "
                                     "required!"
                                  << std::endl;
                        exit(1);
                    }

                    int col_num = q.col_idx.size();
                    for (int i = 0; i < col_num; i++) {
                        int idx = q.col_idx[i];
                        if (idx != -1) {
                            memcpy(q.part_ptr_dst[p][i] + offset[p] * q.type_size[i],
                                   q.ptr_src[i] + p * l_partition_out_col_part_depth * size_apu_512,
                                   q.type_size[i] * sec_partitioned_res_part_nrow);
                        }
                    }
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q3_ping.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                int valid_col = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }

                int64_t output_memcpy_nrow = 0;
                for (int p = 0; p < q.partition_num; p++) {
                    output_memcpy_nrow += nrow_per_part_l[p];
                }

                double output_memcpy_size = (double)output_memcpy_nrow * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "Tab L sec: " << q.sec << " memcpy out, size: " << output_memcpy_size << " MB, with "
                          << q.partition_num * valid_col << " times, total time: " << tvtime
                          << " ms, throughput: " << output_memcpy_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    };

    // table L memcpy out thread
    void part_l_memcpy_out_pong_t() {
        while (q3_pong_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "part l memcpy out pong" << std::endl;
#endif
            while (!q3_pong.empty()) {
                queue_struct_join q = q3_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int64_t l_partition_out_col_part_depth = q.part_max_nrow_512;

                int* nrow_per_part_l = q.meta->getPartLen();
                int64_t offset[q.partition_num];

                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return cur == q.sec; });

                    for (int p = 0; p < q.partition_num; ++p) {
                        offset[p] = l_new_part_offset[p];
                        l_new_part_offset[p] += nrow_per_part_l[p];
                    }
                    // let the other thread proceed to next round
                    cur++;
                    cv.notify_one();
                }

                for (int p = 0; p < q.partition_num; ++p) {
                    int64_t sec_partitioned_res_part_nrow = nrow_per_part_l[p];
                    // std::cout << "Part L, sec: " << q.sec << ", partition: " << p << ", nrow: " <<
                    // sec_partitioned_res_part_nrow << std::endl;
                    if (sec_partitioned_res_part_nrow > l_partition_out_col_part_nrow_max) {
                        std::cerr << "partition out nrow: " << sec_partitioned_res_part_nrow
                                  << ", buffer size: " << l_partition_out_col_part_nrow_max << std::endl;
                        std::cerr << "ERROR: Table L output partition size is smaller than "
                                     "required!"
                                  << std::endl;
                        exit(1);
                    }

                    int col_num = q.col_idx.size();
                    for (int i = 0; i < col_num; i++) {
                        int idx = q.col_idx[i];
                        if (idx != -1) {
                            memcpy(q.part_ptr_dst[p][i] + offset[p] * q.type_size[i],
                                   q.ptr_src[i] + p * l_partition_out_col_part_depth * size_apu_512,
                                   q.type_size[i] * sec_partitioned_res_part_nrow);
                        }
                    }
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q3_pong.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                int valid_col = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }

                int64_t output_memcpy_nrow = 0;
                for (int p = 0; p < q.partition_num; p++) {
                    output_memcpy_nrow += nrow_per_part_l[p];
                }

                double output_memcpy_size = (double)output_memcpy_nrow * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "Tab L sec: " << q.sec << " memcpy out, size: " << output_memcpy_size << " MB, with "
                          << q.partition_num * valid_col << " times, total time: " << tvtime
                          << " ms, throughput: " << output_memcpy_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    };
    // build memcpy in thread
    // memcpy(table_o_build_in_col[0], table_o_new_part_col[p][0],
    // table_o_build_in_size);
    // memcpy(table_o_build_in_col[1], table_o_new_part_col[p][1],
    // table_o_build_in_size);
    void build_memcpy_in_t() {
        while (q4_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "build memcpy in" << std::endl;
#endif
            while (!q4.empty()) {
                queue_struct_join q = q4.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                    }
                }

                q.meta->setColNum(3);
                for (int i = 0; i < 3; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q4.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double data_size = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) data_size += q.size[i];
                }

                data_size = data_size / 1024 / 1024;
                std::cout << "Tab O p: " << q.p << " build memcpy in, size: " << data_size << " MB, time: " << tvtime
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000) << " GB/s" << std::endl;
#endif
            }
        }
    };
    // probe memcpy in thread
    // memcpy(reinterpret_cast<int*>(table_l_probe_in_col[0][sid]),
    //        reinterpret_cast<int*>(table_l_new_part_col[p][0]) + per_slice_nrow
    //        * slice,
    //        table_l_probe_in_slice_nrow_sid_size);
    // memcpy(reinterpret_cast<int*>(table_l_probe_in_col[1][sid]),
    //        reinterpret_cast<int*>(table_l_new_part_col[p][1]) + per_slice_nrow
    //        * slice,
    //        table_l_probe_in_slice_nrow_sid_size);
    // memcpy(reinterpret_cast<int*>(table_l_probe_in_col[2][sid]),
    //        reinterpret_cast<int*>(table_l_new_part_col[p][2]) + per_slice_nrow
    //        * slice,
    //        table_l_probe_in_slice_nrow_sid_size);
    void probe_memcpy_in_ping_t() {
        while (q5_run_ping) {
#if Valgrind_debug
            sleep(1);
            std::cout << "probe memcpy in" << std::endl;
#endif
            while (!q5_ping.empty()) {
                queue_struct_join q = q5_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i] + q.per_slice_nrow * q.slice * q.type_size[i], q.size[i]);
                    }
                }

                q.meta->setColNum(3);
                for (int i = 0; i < 3; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q5_ping.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double data_size = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) data_size += q.size[i];
                }

                data_size = data_size / 1024 / 1024;
                std::cout << "Tab L p: " << q.p << " probe memcpy in, size: " << data_size << " MB, time: " << tvtime
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000) << " GB/s" << std::endl;
#endif
            }
        }
    }
    void probe_memcpy_in_pong_t() {
        while (q5_run_pong) {
#if Valgrind_debug
            sleep(1);
            std::cout << "probe memcpy in pong" << std::endl;
#endif
            while (!q5_pong.empty()) {
                queue_struct_join q = q5_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        memcpy(q.ptr_dst[i], q.ptr_src[i] + q.per_slice_nrow * q.slice * q.type_size[i], q.size[i]);
                    }
                }

                q.meta->setColNum(3);
                for (int i = 0; i < 3; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q5_pong.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double data_size = 0;
                for (size_t i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) data_size += q.size[i];
                }

                data_size = data_size / 1024 / 1024;
                std::cout << "Tab L p: " << q.p << " probe memcpy in, size: " << data_size << " MB, time: " << tvtime
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000) << " GB/s" << std::endl;
#endif
            }
        }
    }

    // d2h thread
    void probe_d2h_out_t() {
        while (q6_d2h_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "probe d2h " << std::endl;
#endif
            while (!q6_d2h.empty()) {
                queue_struct_join q = q6_d2h.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                // create the real data size sub-buffers
                std::vector<cl_mem> probe_sub_vec;
                cl_mem buf_probe_real_out[4];
                cl_buffer_region sub_probe_out_real_size[4];
                int64_t nrow_probe_out = q.meta->getColLen();
                cl_int err;
                int col_num = q.col_idx.size();
                if (nrow_probe_out > 0) {
                    for (int col = 0; col < col_num; col++) {
                        int idx = q.col_idx[col];
                        if (idx != -1) {
                            sub_probe_out_real_size[col] = {q.buf_head[col], nrow_probe_out * sizeof(int64_t)};

                            buf_probe_real_out[col] =
                                clCreateSubBuffer(q.dbuf, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE,
                                                  CL_BUFFER_CREATE_TYPE_REGION, &sub_probe_out_real_size[col], &err);
                            probe_sub_vec.push_back(buf_probe_real_out[col]);
                        }
                    }
                }

                cl_event d2h_evt;
                if (probe_sub_vec.size() > 0) {
                    clEnqueueMigrateMemObjects(q.cq, probe_sub_vec.size(), probe_sub_vec.data(),
                                               CL_MIGRATE_MEM_OBJECT_HOST, 0, nullptr, &d2h_evt);
                    clWaitForEvents(1, &d2h_evt);
                }

                // relase thread used sub-buffer and events
                if (nrow_probe_out > 0) {
                    for (int col = 0; col < col_num; col++) {
                        int idx = q.col_idx[col];
                        if (idx != -1) clReleaseMemObject(buf_probe_real_out[col]);
                    }
                    clReleaseEvent(d2h_evt);
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                q6_d2h.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();
                int valid_col = 0;
                for (int i = 0; i < q.col_idx.size(); i++) {
                    if (q.col_idx[i] != -1) valid_col++;
                }

                double output_d2h_size = (double)nrow_probe_out * valid_col * sizeof(int64_t) / 1024 / 1024;
                std::cout << "probe p: " << q.p << ", slice: " << q.slice << " real d2h, size: " << output_d2h_size
                          << " MB, total time: " << tvtime
                          << " ms, throughput: " << output_d2h_size / 1024 / ((double)tvtime / 1000) << "GB/s"
                          << std::endl;
#endif
            }
        }
    }

    // probe memcpy out thread
    // only copy necessary output rows back to the user final output space.
    void probe_memcpy_out_t() {
        while (q6_run) {
#if Valgrind_debug
            sleep(1);
            std::cout << "probe memcpy out " << std::endl;
#endif
            while (!q6.empty()) {
                queue_struct_join q = q6.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

#if JOIN_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                // save output data nrow
                int64_t probe_out_nrow = q.meta->getColLen();
                // std::cout << "HJ, p: " << q.p << ", nrow: " << probe_out_nrow << std::endl;
                // for (int nn = 0; nn < probe_out_nrow; nn++) {
                //    std::cout << ((int64_t*)(q.ptr_src[0]))[nn] << ", ";
                //    std::cout << ((int64_t*)(q.ptr_src[1]))[nn] << std::endl;
                //    // std::cout << ((int64_t*)(q.ptr_src[2]))[nn] << std::endl;
                //}
                toutrow[q.p][q.slice] = probe_out_nrow;
                int64_t h_dst_size = q.part_max_nrow_512 * size_apu_512;
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    int idx = q.col_idx[i];
                    if (idx != -1) {
                        int64_t u_dst_size = q.size[i]; // user buffer size
                        int64_t pout_size = probe_out_nrow * q.type_size[i];
                        if (pout_size > h_dst_size) { // host buffer is not enough
                            std::cerr << "Error in checking probe memcpy out size: host buffer size(" << h_dst_size
                                      << ") < output size(" << pout_size << ")" << std::endl;
                            std::cerr << "Please using the JoinStrategyManualSet strategy,"
                                      << " and set enough buffer size for output table" << std::endl;

                            exit(1);
                        }
                        if (probe_out_nrow_accu * q.type_size[i] + pout_size > u_dst_size) {
                            std::cerr << "Error in checking probe memcpy out size: user buffer size(" << u_dst_size
                                      << ") < output size(" << (probe_out_nrow_accu * q.type_size[i] + pout_size) << ")"
                                      << std::endl;
                            std::cerr << "Please using the JoinStrategyManualSet strategy, "
                                      << "and set enough buffer size for output table" << std::endl;

                            exit(1);
                        }
                        memcpy(q.ptr_dst[i] + probe_out_nrow_accu * q.type_size[i], q.ptr_src[i], pout_size);
                    }
                }

                // save the accumulate output nrow, to record the data offset
                probe_out_nrow_accu += probe_out_nrow;
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                q6.pop();

#if JOIN_PERF_PROFILE_2
                tv.add(); // 1
                double tvtime = tv.getMilliSec();

                double data_size = 0;
                for (int i = 0; i < col_num; i++) {
                    if (q.col_idx[i] != -1) {
                        data_size += (double)probe_out_nrow * q.type_size[i] / 1024 / 1024;
                    }
                }
                std::cout << "Tab L p: " << q.p << ", s: " << q.slice << ", probe memcpy out, size: " << data_size
                          << " MB, time: " << tvtime
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000) << " GB/s" << std::endl;

#endif
            }
        }
    }

    // initialize the table O partition threads
    void parto_init() {
        memset(o_new_part_offset, 0, sizeof(int64_t) * 256);
        memset(l_new_part_offset, 0, sizeof(int64_t) * 256);
        for (int i = 0; i < 256; i++) {
            memset(toutrow[i], 0, sizeof(int64_t) * 32);
        }

        // start the part o memcpy in thread and non-stop running
        q0_run = 1;
        part_o_in_t = std::thread(&threading_pool::part_o_memcpy_in_t, this);

        q1_d2h_run = 1;
        part_o_d2h_t = std::thread(&threading_pool::part_o_d2h_out_t, this);

        // start the part o memcpy out thread and non-stop running
        q1_run = 1;
        part_o_out_t = std::thread(&threading_pool::part_o_memcpy_out_t, this);
    }

    // initialize the table L partition threads
    void partl_init() {
        // start the part o memcpy in thread and non-stop running
        q2_ping_run = 1;
        part_l_in_ping_t = std::thread(&threading_pool::part_l_memcpy_in_ping_t, this);

        // start the part o memcpy in thread and non-stop running
        q2_pong_run = 1;
        part_l_in_pong_t = std::thread(&threading_pool::part_l_memcpy_in_pong_t, this);

        // start the part L d2h in thread and non-stop running
        q3_d2h_run = 1;
        part_l_d2h_t = std::thread(&threading_pool::part_l_d2h_out_t, this);

        // start the part o memcpy in thread and non-stop running
        q3_ping_run = 1;
        part_l_out_ping_t = std::thread(&threading_pool::part_l_memcpy_out_ping_t, this);

        // start the part o memcpy in thread and non-stop running
        q3_pong_run = 1;
        part_l_out_pong_t = std::thread(&threading_pool::part_l_memcpy_out_pong_t, this);

        cur = 0;
        cv.notify_all();
    }

    // initialize the hash join threads
    void hj_init() {
        // start the build memcpy in thread and non-stop running
        q4_run = 1;
        build_in_t = std::thread(&threading_pool::build_memcpy_in_t, this);

        // start the probe memcpy in thread and non-stop running
        q5_run_ping = 1;
        probe_in_ping_t = std::thread(&threading_pool::probe_memcpy_in_ping_t, this);

        // start the probe memcpy in thread and non-stop running
        q5_run_pong = 1;
        probe_in_pong_t = std::thread(&threading_pool::probe_memcpy_in_pong_t, this);

        // start the probe d2h thread and non-stop running
        q6_d2h_run = 1;
        probe_d2h_t = std::thread(&threading_pool::probe_d2h_out_t, this);
        // start the probe memcpy out thread and non-stop running
        q6_run = 1;
        probe_out_t = std::thread(&threading_pool::probe_memcpy_out_t, this);
    };
};

ErrCode Joiner::join_sol2(Table& tab_a, Table& tab_b, Table& tab_c, PartJoinConfig& pjcfg, StrategySet params) {
    gqe::utils::MM mm;
    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    ap_uint<512>* q5s_cfg_part = pjcfg.getPartConfigBits();
    ap_uint<512>* q5s_cfg_join = pjcfg.getJoinConfigBits();

    std::vector<std::vector<int8_t> > q5s_part_scan = pjcfg.getShuffleScanPart();
    std::vector<std::vector<int8_t> > q5s_join_scan = pjcfg.getShuffleScanHJ();
    // read params from user
    int sec_o = params.sec_o;
    int sec_l = params.sec_l;
    int slice_num = params.slice_num;
    int log_part = params.log_part;
    // get total row number and valid col number
    int64_t o_nrow = tab_a.getRowNum();
    int64_t l_nrow = tab_b.getRowNum();
    int o_valid_col_num = tab_a.getColNum();
    int l_valid_col_num = tab_b.getColNum();
    int out_valid_col_num = tab_c.getColNum();

    // get the part O output col num, used in buffer define and  memcpy out
    std::vector<std::vector<int8_t> > part_wr_col_out;
    part_wr_col_out = pjcfg.getShuffleWritePart();

    // start threading pool threads
    threading_pool pool;

    pool.parto_init();
    // -------------------------setup partition O ----------------------------
    // Assuming table O can not be put to FPGA DDR, divided into two sections
    tab_a.checkSecNum(sec_o);
    size_t table_o_sec_num = tab_a.getSecNum();
    int* table_o_sec_depth = mm.aligned_alloc<int>(table_o_sec_num);
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        table_o_sec_depth[sec] = tab_a.getSecRowNum(sec);
    }
#if 0
    std::cout << "table_o_sec_num: " << table_o_sec_num << std::endl;
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        std::cout << "table_o_sec_depth[" << sec << "] nrow: " << table_o_sec_depth[sec] << std::endl;
    }
#endif

    // Get each col type
    // get each section byte size column * sec_num
    //
    int* table_o_sec_size[3];
    for (int i = 0; i < 3; i++) {
        table_o_sec_size[i] = mm.aligned_alloc<int>(table_o_sec_num);
    }
    int table_o_col_types[3];

    for (int j = 0; j < 3; j++) {
        int idx = (int)q5s_part_scan[0][j];
        if (idx != -1) {
            table_o_col_types[j] = tab_a.getColTypeSize(idx);
            for (size_t i = 0; i < table_o_sec_num; i++) {
                table_o_sec_size[j][i] = table_o_sec_depth[i] * table_o_col_types[j];
            }
        } else {
            table_o_col_types[j] = 8;
        }
    }
    // get max seciction buffer depth and byte size
    int table_o_sec_depth_max = 0;
    // in case the data type is different for each col
    int64_t table_o_sec_size_max[3];
    for (size_t i = 0; i < table_o_sec_num; i++) {
        if (table_o_sec_depth[i] > table_o_sec_depth_max) {
            table_o_sec_depth_max = table_o_sec_depth[i];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (q5s_part_scan[0][i] != -1) {
            table_o_sec_size_max[i] = (int64_t)table_o_sec_depth_max * table_o_col_types[i];
        } else {
            table_o_sec_size_max[i] = 64;
        }
    }

    char** table_o_valid_user = mm.aligned_alloc<char*>(table_o_sec_num);
    int64_t* table_o_valid_sec_size = mm.aligned_alloc<int64_t>(table_o_sec_num);
    for (size_t i = 0; i < table_o_sec_num; i++) {
        table_o_valid_user[i] = tab_a.getValColPointer(sec_o, i);
        table_o_valid_sec_size[i] = sizeof(char) * (table_o_sec_depth[i] + 7) / 8;
    }

    // data load from disk. due to table size, data read into two sections
    char** table_o_user_col_sec[3];
    for (int i = 0; i < 3; i++) {
        table_o_user_col_sec[i] = mm.aligned_alloc<char*>(table_o_sec_num);
    }
    for (int j = 0; j < 3; j++) {
        int idx = (int)q5s_part_scan[0][j];
        if (idx != -1) {
            for (size_t i = 0; i < table_o_sec_num; ++i) {
                table_o_user_col_sec[j][i] = tab_a.getColPointer(idx, sec_o, i);
            }
        } else {
            for (size_t i = 0; i < table_o_sec_num; ++i) {
                table_o_user_col_sec[j][i] = mm.aligned_alloc<char>(32);
            }
        }
    }

    // host side pinned buffers for partition kernel
    char* table_o_partition_in_col[3][2]; // 3cols, ping-pong

    for (int i = 0; i < 3; i++) {
        table_o_partition_in_col[i][0] = AllocHostBuf(1, table_o_sec_size_max[i]); // DDR1: 0-1-2
    }
    for (int i = 0; i < 3; i++) {
        table_o_partition_in_col[i][1] = AllocHostBuf(1, table_o_sec_size_max[i]); // DDR1: 3-4-5
    }
    const int size_apu_512 = 64;
    // partition setups
    const int k_depth = 512;
    int log_partition_num = log_part;
    int partition_num = 1 << log_partition_num;
    // partition output col size,  setup the proper size by multiple 1.5
    int o_partition_out_col_depth_init = (table_o_sec_depth_max * params.coef_expansion_partO + VEC_LEN - 1) / VEC_LEN;
    assert(o_partition_out_col_depth_init > 0 && "Table O output col size must > 0");
    int o_partition_out_col_part_depth = (o_partition_out_col_depth_init + partition_num - 1) / partition_num;

    // the depth of each partition in each col.
    pool.o_partition_out_col_part_nrow_max = o_partition_out_col_part_depth * VEC_LEN;
    // update depth to make sure the buffer size is aligned by partititon_num *
    // o_partition_out_col_part_depth;
    int o_partition_out_col_depth = partition_num * o_partition_out_col_part_depth;

    // partition output data
    char* table_o_partition_out_col[3][2];

    for (int i = 0; i < 3; i++) {
        table_o_partition_out_col[i][0] = AllocHostBuf(0, o_partition_out_col_depth * size_apu_512); // DDR0: 0-1-2
    }
    for (int i = 0; i < 3; i++) {
        table_o_partition_out_col[i][1] = AllocHostBuf(0, o_partition_out_col_depth * size_apu_512); // DDR0: 3-4-5
    }

    //--------------- metabuffer setup O -----------------
    // using col0 and col1 buffer during build,setup partition kernel used meta
    // input
    MetaTable meta_o_partition_in[2];
    for (int k = 0; k < 2; k++) {
        meta_o_partition_in[k].setSecID(0);
        meta_o_partition_in[k].setColNum(3);
        for (int i = 0; i < 3; i++) {
            meta_o_partition_in[k].setCol(i, i, table_o_sec_depth[0]);
        }
        meta_o_partition_in[k].meta();
    }

    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3.
    // setup partition kernel used meta output
    MetaTable meta_o_partition_out[2];
    for (int k = 0; k < 2; k++) {
        meta_o_partition_out[k].setColNum(3);
        meta_o_partition_out[k].setPartition(partition_num, o_partition_out_col_part_depth);
        for (int i = 0; i < 3; i++) {
            meta_o_partition_out[k].setCol(i, i, o_partition_out_col_depth);
        }
        meta_o_partition_out[k].meta();
    }

    cl_int err;
    // partition kernel settings
    cl_kernel partkernel_O[2], partkernel_L[2];
    partkernel_O[0] = clCreateKernel(prg, "gqePart", &err);
    logger.logCreateKernel(err);
    partkernel_O[1] = clCreateKernel(prg, "gqePart", &err);
    logger.logCreateKernel(err);

    partkernel_L[0] = clCreateKernel(prg, "gqePart", &err);
    logger.logCreateKernel(err);
    partkernel_L[1] = clCreateKernel(prg, "gqePart", &err);
    logger.logCreateKernel(err);
#ifdef USER_DEBUG
    std::cout << "Kernel has been created\n";
#endif

    cl_mem_ext_ptr_t mext_meta_o_partition_in[2], mext_meta_o_partition_out[2];

    mext_meta_o_partition_in[0] = {XCL_BANK1, meta_o_partition_in[0].meta(), 0};
    mext_meta_o_partition_in[1] = {XCL_BANK1, meta_o_partition_in[1].meta(), 0};

    mext_meta_o_partition_out[0] = {XCL_BANK0, meta_o_partition_out[0].meta(), 0};
    mext_meta_o_partition_out[1] = {XCL_BANK0, meta_o_partition_out[1].meta(), 0};

    cl_mem_ext_ptr_t mext_cfg5s_part = {XCL_BANK1, q5s_cfg_part, 0};

    int din_val_o_len = (table_o_sec_depth_max + 7) / 8;
    char* din_valid_o[2];
    din_valid_o[0] = mm.aligned_alloc<char>(din_val_o_len);
    din_valid_o[1] = mm.aligned_alloc<char>(din_val_o_len);
    cl_mem_ext_ptr_t mext_buf_valid_o[2];
    cl_mem buf_valid_o[2];
    mext_buf_valid_o[0] = {XCL_BANK1, din_valid_o[0], 0};
    mext_buf_valid_o[1] = {XCL_BANK1, din_valid_o[1], 0};
    buf_valid_o[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                    din_val_o_len * sizeof(char), &mext_buf_valid_o[0], &err);
    buf_valid_o[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                    din_val_o_len * sizeof(char), &mext_buf_valid_o[1], &err);

    // dev buffers
    cl_mem buf_table_o_partition_in_col[3][2];
    cl_buffer_region sub_table_o_sec_size[6];
    sub_table_o_sec_size[0] = {buf_head[1][0], buf_size[1][0]};
    sub_table_o_sec_size[1] = {buf_head[1][1], buf_size[1][1]};
    sub_table_o_sec_size[2] = {buf_head[1][2], buf_size[1][2]};
    sub_table_o_sec_size[3] = {buf_head[1][3], buf_size[1][3]};
    sub_table_o_sec_size[4] = {buf_head[1][4], buf_size[1][4]};
    sub_table_o_sec_size[5] = {buf_head[1][5], buf_size[1][5]};

    for (int i = 0; i < 3; i++) {
        buf_table_o_partition_in_col[i][0] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_o_sec_size[i], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
        buf_table_o_partition_in_col[i][1] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_o_sec_size[i + 3], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
    }

    cl_mem buf_table_o_partition_out_col[3][2];
    cl_buffer_region sub_part_out_col_size[6];
    sub_part_out_col_size[0] = {buf_head[0][0], buf_size[0][0]};
    sub_part_out_col_size[1] = {buf_head[0][1], buf_size[0][1]};
    sub_part_out_col_size[2] = {buf_head[0][2], buf_size[0][2]};
    sub_part_out_col_size[3] = {buf_head[0][3], buf_size[0][3]};
    sub_part_out_col_size[4] = {buf_head[0][4], buf_size[0][4]};
    sub_part_out_col_size[5] = {buf_head[0][5], buf_size[0][5]};

    for (int i = 0; i < 3; i++) {
        buf_table_o_partition_out_col[i][0] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_part_out_col_size[i], &err);
        buf_table_o_partition_out_col[i][1] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_part_out_col_size[i + 3], &err);
    }
    cl_mem buf_cfg5s_part = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                           (size_apu_512 * 14), &mext_cfg5s_part, &err);
    cl_mem buf_meta_o_partition_in[2];
    buf_meta_o_partition_in[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                (size_apu_512 * 8), &mext_meta_o_partition_in[0], &err);
    buf_meta_o_partition_in[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                (size_apu_512 * 8), &mext_meta_o_partition_in[1], &err);

    cl_mem buf_meta_o_partition_out[2];
    buf_meta_o_partition_out[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                 (size_apu_512 * 24), &mext_meta_o_partition_out[0], &err);
    buf_meta_o_partition_out[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                 (size_apu_512 * 24), &mext_meta_o_partition_out[1], &err);

    // make meta/cfg resident
    std::vector<cl_mem> part_o_resident_vec;
    part_o_resident_vec.push_back(buf_cfg5s_part);
    part_o_resident_vec.push_back(buf_valid_o[0]);
    part_o_resident_vec.push_back(buf_valid_o[1]);
    part_o_resident_vec.push_back(buf_meta_o_partition_in[0]);
    part_o_resident_vec.push_back(buf_meta_o_partition_in[1]);
    part_o_resident_vec.push_back(buf_meta_o_partition_out[0]);
    part_o_resident_vec.push_back(buf_meta_o_partition_out[1]);
    cl_event evt_part_o_resident;
    clEnqueueMigrateMemObjects(cq, part_o_resident_vec.size(), part_o_resident_vec.data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, &evt_part_o_resident);
    clWaitForEvents(1, &evt_part_o_resident);

//----------end Order table partition---------
//
//-----------partition O run-----------
#ifdef USER_DEBUG
    std::cout << "-----------Partitioning O table ---------" << std::endl;
#endif
    // set args and enqueue kernel
    const int idx_o = 0;
    int j = 0;
    for (int k = 0; k < 2; k++) {
        j = 0;
        clSetKernelArg(partkernel_O[k], j++, sizeof(int), &k_depth);
        clSetKernelArg(partkernel_O[k], j++, sizeof(int), &idx_o);
        clSetKernelArg(partkernel_O[k], j++, sizeof(int), &log_partition_num);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_in_col[0][k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_in_col[1][k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_in_col[2][k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_valid_o[k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_cfg5s_part);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_meta_o_partition_in[k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_meta_o_partition_out[k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_out_col[0][k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_out_col[1][k]);
        clSetKernelArg(partkernel_O[k], j++, sizeof(cl_mem), &buf_table_o_partition_out_col[2][k]);
    }

    // partition h2d
    std::vector<cl_mem> partition_o_in_vec[2];
    for (int k = 0; k < 2; k++) {
        partition_o_in_vec[k].push_back(buf_table_o_partition_in_col[0][k]);
        partition_o_in_vec[k].push_back(buf_table_o_partition_in_col[1][k]);
        partition_o_in_vec[k].push_back(buf_table_o_partition_in_col[2][k]);
        partition_o_in_vec[k].push_back(buf_meta_o_partition_in[k]);
        partition_o_in_vec[k].push_back(buf_cfg5s_part);
        partition_o_in_vec[k].push_back(buf_valid_o[k]);
    }

    // partition d2h
    std::vector<cl_mem> partition_o_out_vec[2];
    for (int k = 0; k < 2; k++) {
        partition_o_out_vec[k].push_back(buf_meta_o_partition_out[k]);
    }

    cl_event evt_meta_o_partition_out[2];
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_o_partition_out[0], 0, 0, nullptr, &evt_meta_o_partition_out[0]);
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_o_partition_out[1], 0, 0, nullptr, &evt_meta_o_partition_out[1]);
    clWaitForEvents(2, evt_meta_o_partition_out);

    // create user partition res cols
    // all sections partition 0 output to same 8-col buffers
    char*** table_o_new_part_col = mm.aligned_alloc<char**>(partition_num);

    // combine sec0_partition0, sec1_parttion0, ...secN_partition0 in 1 buffer. The depth is
    int64_t o_new_part_depth = (int64_t)o_partition_out_col_part_depth * table_o_sec_num;

    for (int p = 0; p < partition_num; ++p) {
        table_o_new_part_col[p] = mm.aligned_alloc<char*>(3);
        for (int i = 0; i < 3; ++i) {
            if (part_wr_col_out[0][i] != -1) {
                table_o_new_part_col[p][i] = mm.aligned_alloc<char>(o_new_part_depth * size_apu_512);
                memset(table_o_new_part_col[p][i], 0, o_new_part_depth * size_apu_512);
            } else {
                table_o_new_part_col[p][i] = mm.aligned_alloc<char>(size_apu_512);
            }
        }
    }

    // create events
    std::vector<std::vector<cl_event> > evt_part_o_h2d(table_o_sec_num);
    std::vector<std::vector<cl_event> > evt_part_o_krn(table_o_sec_num);
    std::vector<std::vector<cl_event> > evt_part_o_meta_d2h(table_o_sec_num);
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        evt_part_o_h2d[sec].resize(1);
        evt_part_o_krn[sec].resize(1);
        evt_part_o_meta_d2h[sec].resize(1);
    }

    gqe::utils::Timer tv_opart;
    double tvtime_opart;

    std::vector<std::vector<cl_event> > evt_part_o_h2d_dep(table_o_sec_num);
    evt_part_o_h2d_dep[0].resize(1);
    for (size_t i = 1; i < table_o_sec_num; ++i) {
        if (i == 1) {
            evt_part_o_h2d_dep[i].resize(1);
        } else {
            evt_part_o_h2d_dep[i].resize(2);
        }
    }
    std::vector<std::vector<cl_event> > evt_part_o_krn_dep(table_o_sec_num);
    evt_part_o_krn_dep[0].resize(1);
    for (size_t i = 1; i < table_o_sec_num; ++i) {
        if (i == 1)
            evt_part_o_krn_dep[i].resize(2);
        else
            evt_part_o_krn_dep[i].resize(4);
    }
    std::vector<std::vector<cl_event> > evt_part_o_d2h_dep(table_o_sec_num);
    std::vector<std::vector<cl_event> > evt_part_o_meta_d2h_dep(table_o_sec_num);
    std::vector<std::vector<cl_event> > evt_part_o_memcpy_out_dep(table_o_sec_num);
    evt_part_o_d2h_dep[0].resize(1);
    evt_part_o_meta_d2h_dep[0].resize(1);
    evt_part_o_memcpy_out_dep[0].resize(2);
    for (size_t i = 1; i < table_o_sec_num; ++i) {
        if (i == 1) {
            evt_part_o_meta_d2h_dep[i].resize(1);
            evt_part_o_d2h_dep[i].resize(1);
            evt_part_o_memcpy_out_dep[i].resize(2);
        } else {
            evt_part_o_meta_d2h_dep[i].resize(3);
            evt_part_o_d2h_dep[i].resize(2);
            evt_part_o_memcpy_out_dep[i].resize(2);
        }
    }

    // define parto memcpy in user events
    std::vector<std::vector<cl_event> > evt_part_o_memcpy_in(table_o_sec_num);
    for (size_t i = 0; i < table_o_sec_num; i++) {
        evt_part_o_memcpy_in[i].resize(1);
        evt_part_o_memcpy_in[i][0] = clCreateUserEvent(ctx, &err);
    }
    std::vector<std::vector<cl_event> > evt_part_o_d2h(table_o_sec_num);
    std::vector<std::vector<cl_event> > evt_part_o_memcpy_out(table_o_sec_num);
    for (size_t i = 0; i < table_o_sec_num; i++) {
        evt_part_o_d2h[i].resize(1);
        evt_part_o_memcpy_out[i].resize(1);
        evt_part_o_d2h[i][0] = clCreateUserEvent(ctx, &err);
        evt_part_o_memcpy_out[i][0] = clCreateUserEvent(ctx, &err);
    }

    std::vector<queue_struct_join> parto_min(table_o_sec_num);
    std::vector<queue_struct_join> parto_d2h(table_o_sec_num);
    std::vector<queue_struct_join> parto_mout(table_o_sec_num);

    tv_opart.add(); // 0

    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        int kid = sec % 2;
        std::cout << sec << " of " << table_o_sec_num << std::endl;

        // 1) memcpy in
        std::cout << "0.0 begin " << std::endl;
        parto_min[sec].sec = sec;
        parto_min[sec].event = &evt_part_o_memcpy_in[sec][0];
        parto_min[sec].meta_nrow = table_o_sec_depth[sec];
        parto_min[sec].meta = &meta_o_partition_in[kid];
        std::cout << "0.1 done" << std::endl;
        for (int i = 0; i < 3; i++) {
            int idx = (int)q5s_part_scan[0][i];
            parto_min[sec].col_idx.push_back(idx);

            if (idx != -1) {
                parto_min[sec].size[i] = table_o_sec_size[i][sec];
                parto_min[sec].type_size[i] = table_o_col_types[i];
                parto_min[sec].ptr_src[i] = table_o_user_col_sec[i][sec];
                parto_min[sec].ptr_dst[i] = table_o_partition_in_col[i][kid];
            }
        }
        std::cout << "0.2 done" << std::endl;

        if (tab_a.getRowIDEnableFlag() && tab_a.getValidEnableFlag()) {
            parto_min[sec].col_idx.push_back(3);
            parto_min[sec].size[3] = table_o_valid_sec_size[sec];
            parto_min[sec].type_size[3] = sizeof(int64_t);
            parto_min[sec].ptr_src[3] = table_o_valid_user[sec];
            parto_min[sec].ptr_dst[3] = din_valid_o[kid];
        }
        std::cout << "0.3 done" << std::endl;
        if (sec > 1) {
            parto_min[sec].num_event_wait_list = evt_part_o_h2d[sec - 2].size();
            parto_min[sec].event_wait_list = evt_part_o_h2d[sec - 2].data();
        } else {
            parto_min[sec].num_event_wait_list = 0;
            parto_min[sec].event_wait_list = nullptr;
        }
        std::cout << "0.4 done" << std::endl;
        pool.q0.push(parto_min[sec]);
        std::cout << "1 done" << std::endl;
        // 2) h2d
        evt_part_o_h2d_dep[sec][0] = evt_part_o_memcpy_in[sec][0];
        std::cout << "1.1 done" << std::endl;
        if (sec > 1) {
            evt_part_o_h2d_dep[sec][1] = evt_part_o_krn[sec - 2][0];
        }
        std::cout << "1.2 done" << std::endl;
        clEnqueueMigrateMemObjects(cq, partition_o_in_vec[kid].size(), partition_o_in_vec[kid].data(), 0,
                                   evt_part_o_h2d_dep[sec].size(), evt_part_o_h2d_dep[sec].data(),
                                   &evt_part_o_h2d[sec][0]);
        std::cout << "2 done" << std::endl;
        // 3) kernel
        evt_part_o_krn_dep[sec][0] = evt_part_o_h2d[sec][0];
        if (sec > 0) {
            evt_part_o_krn_dep[sec][1] = evt_part_o_krn[sec - 1][0];
        }
        if (sec > 1) {
            evt_part_o_krn_dep[sec][2] = evt_part_o_meta_d2h[sec - 2][0];
            evt_part_o_krn_dep[sec][3] = evt_part_o_d2h[sec - 2][0];
        }
        clEnqueueTask(cq, partkernel_O[kid], evt_part_o_krn_dep[sec].size(), evt_part_o_krn_dep[sec].data(),
                      &evt_part_o_krn[sec][0]);

        std::cout << "3 done" << std::endl;
        // 4) meta d2h, transfer partition meta results back
        evt_part_o_meta_d2h_dep[sec][0] = evt_part_o_krn[sec][0];
        if (sec > 1) {
            evt_part_o_meta_d2h_dep[sec][1] = evt_part_o_memcpy_out[sec - 2][0];
            evt_part_o_meta_d2h_dep[sec][2] = evt_part_o_d2h[sec - 2][0];
        }
        clEnqueueMigrateMemObjects(cq, partition_o_out_vec[kid].size(), partition_o_out_vec[kid].data(), 1,
                                   evt_part_o_meta_d2h_dep[sec].size(), evt_part_o_meta_d2h_dep[sec].data(),
                                   &evt_part_o_meta_d2h[sec][0]);

        std::cout << "4 done" << std::endl;
        // 5) d2h, transfer real data in each partition out buffer, only transfer valid data with sub-buffer
        evt_part_o_d2h_dep[sec][0] = evt_part_o_meta_d2h[sec][0];
        if (sec > 1) {
            evt_part_o_d2h_dep[sec][1] = evt_part_o_memcpy_out[sec - 2][0];
        }
        parto_d2h[sec].sec = sec;
        parto_d2h[sec].partition_num = partition_num;
        parto_d2h[sec].part_max_nrow_512 = o_partition_out_col_part_depth;
        parto_d2h[sec].event = &evt_part_o_d2h[sec][0];
        parto_d2h[sec].meta = &meta_o_partition_out[kid];
        for (size_t i = 0; i < part_wr_col_out[0].size(); i++) {
            int o_i = part_wr_col_out[0][i];
            parto_d2h[sec].col_idx.push_back(o_i);
            if (o_i != -1) parto_d2h[sec].buf_head[i] = buf_head[0][i + kid * 3];
        }
        parto_d2h[sec].num_event_wait_list = evt_part_o_d2h_dep[sec].size();
        parto_d2h[sec].event_wait_list = evt_part_o_d2h_dep[sec].data();
        parto_d2h[sec].cq = cq;
        parto_d2h[sec].dbuf = dbuf_ddr0;
        pool.q1_d2h.push(parto_d2h[sec]);

        std::cout << "5 done" << std::endl;
        // 6) memcpy out
        // memcpy out evt dep
        evt_part_o_memcpy_out_dep[sec][0] = evt_part_o_meta_d2h[sec][0];
        evt_part_o_memcpy_out_dep[sec][1] = evt_part_o_d2h[sec][0];

        parto_mout[sec].sec = sec;
        parto_mout[sec].partition_num = partition_num;
        parto_mout[sec].part_max_nrow_512 = o_partition_out_col_part_depth;
        parto_mout[sec].event = &evt_part_o_memcpy_out[sec][0];
        parto_mout[sec].meta = &meta_o_partition_out[kid];
        for (size_t i = 0; i < part_wr_col_out[0].size(); i++) {
            int o_i = part_wr_col_out[0][i];
            parto_mout[sec].col_idx.push_back(o_i);
            if (o_i != -1) {
                parto_mout[sec].ptr_src[i] = table_o_partition_out_col[i][kid];
                parto_mout[sec].type_size[i] = sizeof(int64_t);
            }
        }
        parto_mout[sec].part_ptr_dst = table_o_new_part_col;
        parto_mout[sec].num_event_wait_list = evt_part_o_memcpy_out_dep[sec].size();
        parto_mout[sec].event_wait_list = evt_part_o_memcpy_out_dep[sec].data();
        pool.q1.push(parto_mout[sec]);
        std::cout << "6 done" << std::endl;
    }
    clWaitForEvents(evt_part_o_memcpy_out[table_o_sec_num - 1].size(),
                    evt_part_o_memcpy_out[table_o_sec_num - 1].data());
    if (table_o_sec_num > 1) {
        clWaitForEvents(evt_part_o_memcpy_out[table_o_sec_num - 2].size(),
                        evt_part_o_memcpy_out[table_o_sec_num - 2].data());
    }

    tv_opart.add(); // 1
    pool.q0_run = 0;
    pool.q1_d2h_run = 0;
    pool.q1_run = 0;
    pool.part_o_in_t.join();
    pool.part_o_d2h_t.join();
    pool.part_o_out_t.join();

    for (int p = 0; p < partition_num; p++) {
        std::cout << "Table O, part: " << p << ", nrow: " << pool.o_new_part_offset[p] << std::endl;
#if 0
        for (int n = 0; n < pool.o_new_part_offset[p]; n++) {
            for (int c = 0; c < 3; c++) {
                if (part_wr_col_out[0][c] != -1) {
                    int64_t dat = ((int64_t*)(table_o_new_part_col[p][c]))[n];
                    if (c == 2)
                        std::cout << "col " << c << ": (" << (dat >> 32) << ", " << (dat & 0xffffffff) << "), ";
                    else
                        std::cout << "col " << c << ": " << dat << ", ";
                }
            }
            std::cout << std::endl;
        }
        std::cout << "----------------" << std::endl;
#endif
    }

    // profiling
    {
        cl_ulong start, end;
        double ev_ns;

        for (size_t sec = 0; sec < table_o_sec_num; sec++) {
            //// part o h2d
            clGetEventProfilingInfo(evt_part_o_h2d[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_part_o_h2d[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partO sec: " << sec << " h2d time: " << ev_ns << " ms" << std::endl;
            // parto kernel
            clGetEventProfilingInfo(evt_part_o_krn[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_part_o_krn[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partO sec: " << sec << " krn time: " << ev_ns << " ms" << std::endl;
            // parto d2h
            clGetEventProfilingInfo(evt_part_o_meta_d2h[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
                                    NULL);
            clGetEventProfilingInfo(evt_part_o_meta_d2h[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,
                                    NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partO sec: " << sec << " meta d2h time: " << ev_ns << " ms" << std::endl;
        }
    }

    // print the execution times for O table partition
    tvtime_opart = tv_opart.getMilliSec();
    double o_input_memcpy_size = 0;
    for (size_t sec = 0; sec < table_o_sec_num; sec++) {
        for (int i = 0; i < 3; i++) {
            int idx = (int)q5s_part_scan[0][i];
            if (idx != -1) {
                o_input_memcpy_size += (double)table_o_sec_size[i][sec];
            }
        }
    }
    o_input_memcpy_size = o_input_memcpy_size / 1024 / 1024;

#ifdef USER_DEBUG
    std::cout << "----------- finished O table partition---------------" << std::endl << std::endl;
#endif

    pool.partl_init();
    // Assuming table L can not be put to FPGA DDR, divided into many sections
    tab_b.checkSecNum(sec_l);
    size_t table_l_sec_num = tab_b.getSecNum();
    int* table_l_sec_depth = mm.aligned_alloc<int>(table_l_sec_num);
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        table_l_sec_depth[sec] = tab_b.getSecRowNum(sec);
    }
#if 0
    std::cout << "table_l_sec_num: " << table_l_sec_num << std::endl;
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        std::cout << "table_l_sec_depth[" << sec << "] nrow: " << table_l_sec_depth[sec] << std::endl;
    }
#endif

    int64_t* table_l_sec_size[3];
    for (int i = 0; i < 3; i++) {
        table_l_sec_size[i] = mm.aligned_alloc<int64_t>(table_l_sec_num);
    }
    int table_l_col_types[3];
    for (int j = 0; j < 3; j++) {
        int idx = (int)q5s_part_scan[1][j];
        if (idx != -1) {
            table_l_col_types[j] = tab_b.getColTypeSize(idx);
            for (size_t i = 0; i < table_l_sec_num; i++) {
                table_l_sec_size[j][i] = (int64_t)table_l_sec_depth[i] * table_l_col_types[j];
            }
        } else {
            table_l_col_types[j] = 8;
        }
    }

    // get max seciction buffer depth and byte size
    int table_l_sec_depth_max = 0;
    int64_t table_l_sec_size_max[3];
    for (size_t i = 0; i < table_l_sec_num; i++) {
        if (table_l_sec_depth[i] > table_l_sec_depth_max) {
            table_l_sec_depth_max = table_l_sec_depth[i];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (q5s_part_scan[1][i] != -1) {
            table_l_sec_size_max[i] = (int64_t)table_l_sec_depth_max * table_l_col_types[i];
        } else {
            table_l_sec_size_max[i] = 64;
        }
    }

    char** table_l_valid_user = mm.aligned_alloc<char*>(table_l_sec_num);
    int64_t* table_l_valid_sec_size = mm.aligned_alloc<int64_t>(table_l_sec_num);
    for (size_t i = 0; i < table_l_sec_num; i++) {
        table_l_valid_user[i] = tab_b.getValColPointer(sec_l, i);
        table_l_valid_sec_size[i] = sizeof(char) * (table_l_sec_depth[i] + 7) / 8;
    }

    // data load from disk. due to table size, data read into several sections
    char** table_l_user_col_sec[3];
    for (int i = 0; i < 3; i++) {
        table_l_user_col_sec[i] = mm.aligned_alloc<char*>(table_l_sec_num);
    }
    for (int j = 0; j < 3; j++) {
        int idx = (int)q5s_part_scan[1][j];
        if (idx != -1) {
            for (size_t i = 0; i < table_l_sec_num; ++i) {
                table_l_user_col_sec[j][i] = tab_b.getColPointer(idx, sec_l, i);
            }
        } else {
            for (size_t i = 0; i < table_l_sec_num; ++i) {
                table_l_user_col_sec[j][i] = mm.aligned_alloc<char>(32);
            }
        }
    }

    // L host side pinned buffers for partition kernel
    char* table_l_partition_in_col[3][2];

    for (int i = 0; i < 3; i++) {
        table_l_partition_in_col[i][0] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }
    for (int i = 0; i < 3; i++) {
        table_l_partition_in_col[i][1] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }

    // partition output col size,  setup the proper size by multiple 1.5
    int l_partition_out_col_depth_init = (table_l_sec_depth_max * params.coef_expansion_partL + VEC_LEN - 1) / VEC_LEN;
    assert(l_partition_out_col_depth_init > 0 && "Table L output col size must > 0");
    // the depth of each partition in each col.
    int l_partition_out_col_part_depth = (l_partition_out_col_depth_init + partition_num - 1) / partition_num;
    pool.l_partition_out_col_part_nrow_max = l_partition_out_col_part_depth * VEC_LEN;

    // update depth to make sure the buffer size is aligned by partititon_num *
    // l_partition_out_col_part_depth;
    int64_t l_partition_out_col_depth = partition_num * l_partition_out_col_part_depth;

    // partition output data
    char* table_l_partition_out_col[3][2];

    for (int i = 0; i < 3; i++) {
        table_l_partition_out_col[i][0] = AllocHostBuf(0, l_partition_out_col_depth * size_apu_512);
    }
    for (int i = 0; i < 3; i++) {
        table_l_partition_out_col[i][1] = AllocHostBuf(0, l_partition_out_col_depth * size_apu_512);
    }

    //--------------- metabuffer setup L -----------------
    MetaTable meta_l_partition_in[2];
    for (int k = 0; k < 2; k++) {
        meta_l_partition_in[k].setSecID(0);
        meta_l_partition_in[k].setColNum(3);
        for (int i = 0; i < 3; i++) {
            meta_l_partition_in[k].setCol(i, i, table_l_sec_depth[0]);
        }
        meta_l_partition_in[k].meta();
    }

    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3.
    // setup partition kernel used meta output
    MetaTable meta_l_partition_out[2];
    for (int k = 0; k < 2; k++) {
        meta_l_partition_out[k].setColNum(3);
        meta_l_partition_out[k].setPartition(partition_num, l_partition_out_col_part_depth);
        for (int i = 0; i < 3; i++) {
            meta_l_partition_out[k].setCol(i, i, l_partition_out_col_depth);
        }
        meta_l_partition_out[k].meta();
    }

    cl_mem_ext_ptr_t mext_meta_l_partition_in[2], mext_meta_l_partition_out[2];

    mext_meta_l_partition_in[0] = {XCL_BANK1, meta_l_partition_in[0].meta(), 0};
    mext_meta_l_partition_in[1] = {XCL_BANK1, meta_l_partition_in[1].meta(), 0};

    mext_meta_l_partition_out[0] = {XCL_BANK0, meta_l_partition_out[0].meta(), 0};
    mext_meta_l_partition_out[1] = {XCL_BANK0, meta_l_partition_out[1].meta(), 0};

    // dev buffers
    cl_mem buf_table_l_partition_in_col[3][2];
    cl_buffer_region sub_table_l_sec_size[6];
    sub_table_l_sec_size[0] = {buf_head[1][6], buf_size[1][6]};
    sub_table_l_sec_size[1] = {buf_head[1][7], buf_size[1][7]};
    sub_table_l_sec_size[2] = {buf_head[1][8], buf_size[1][8]};
    sub_table_l_sec_size[3] = {buf_head[1][9], buf_size[1][9]};
    sub_table_l_sec_size[4] = {buf_head[1][10], buf_size[1][10]};
    sub_table_l_sec_size[5] = {buf_head[1][11], buf_size[1][11]};

    for (int i = 0; i < 3; i++) {
        buf_table_l_partition_in_col[i][0] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_sec_size[i], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
        buf_table_l_partition_in_col[i][1] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_sec_size[i + 3], &err);
        if (err != CL_SUCCESS) {
            return MEM_ERROR;
        }
    }

    int din_val_l_len = (table_l_sec_depth_max + 7) / 8;
    char* din_valid_l[2];
    din_valid_l[0] = mm.aligned_alloc<char>(din_val_l_len);
    din_valid_l[1] = mm.aligned_alloc<char>(din_val_l_len);

    cl_mem_ext_ptr_t mext_buf_valid_l[2];
    cl_mem buf_valid_l[2];
    for (int i = 0; i < 2; i++) {
        mext_buf_valid_l[i] = {XCL_BANK1, din_valid_l[i], 0};
        buf_valid_l[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        din_val_l_len * sizeof(char), &mext_buf_valid_l[i], &err);
    }

    cl_mem buf_table_l_partition_out_col[3][2];
    cl_buffer_region sub_part_out_l_col_size[6];
    sub_part_out_l_col_size[0] = {buf_head[0][6], buf_size[0][6]};
    sub_part_out_l_col_size[1] = {buf_head[0][7], buf_size[0][7]};
    sub_part_out_l_col_size[2] = {buf_head[0][8], buf_size[0][8]};
    sub_part_out_l_col_size[3] = {buf_head[0][9], buf_size[0][9]};
    sub_part_out_l_col_size[4] = {buf_head[0][10], buf_size[0][10]};
    sub_part_out_l_col_size[5] = {buf_head[0][11], buf_size[0][11]};

    for (int i = 0; i < 3; i++) {
        buf_table_l_partition_out_col[i][0] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_part_out_l_col_size[i], &err);
        buf_table_l_partition_out_col[i][1] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_part_out_l_col_size[i + 3], &err);
    }
    cl_mem buf_meta_l_partition_in[2];
    buf_meta_l_partition_in[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                (size_apu_512 * 8), &mext_meta_l_partition_in[0], &err);
    buf_meta_l_partition_in[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                                (size_apu_512 * 8), &mext_meta_l_partition_in[1], &err);

    cl_mem buf_meta_l_partition_out[2];
    buf_meta_l_partition_out[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                 (size_apu_512 * 24), &mext_meta_l_partition_out[0], &err);
    buf_meta_l_partition_out[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                                 (size_apu_512 * 24), &mext_meta_l_partition_out[1], &err);
    // make meta/cfg resident
    std::vector<cl_mem> part_l_resident_vec;
    part_l_resident_vec.push_back(buf_meta_l_partition_in[0]);
    part_l_resident_vec.push_back(buf_meta_l_partition_in[1]);
    part_l_resident_vec.push_back(buf_meta_l_partition_out[0]);
    part_l_resident_vec.push_back(buf_meta_l_partition_out[1]);
    part_l_resident_vec.push_back(buf_valid_l[0]);
    part_l_resident_vec.push_back(buf_valid_l[1]);
    cl_event evt_part_l_resident;
    clEnqueueMigrateMemObjects(cq, part_l_resident_vec.size(), part_l_resident_vec.data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, &evt_part_l_resident);
    clWaitForEvents(1, &evt_part_l_resident);

//------------end of partition L setup------------

//-----------partition L run-----------
#ifdef USER_DEBUG
    std::cout << "------------Partitioning L table -----------" << std::endl;
#endif
    const int idx_l = 1;
    for (int k = 0; k < 2; k++) {
        j = 0;
        clSetKernelArg(partkernel_L[k], j++, sizeof(int), &k_depth);
        clSetKernelArg(partkernel_L[k], j++, sizeof(int), &idx_l);
        clSetKernelArg(partkernel_L[k], j++, sizeof(int), &log_partition_num);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_in_col[0][k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_in_col[1][k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_in_col[2][k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_valid_l[k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_cfg5s_part);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_meta_l_partition_in[k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_meta_l_partition_out[k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_out_col[0][k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_out_col[1][k]);
        clSetKernelArg(partkernel_L[k], j++, sizeof(cl_mem), &buf_table_l_partition_out_col[2][k]);
    }

    // partition h2d
    std::vector<cl_mem> partition_l_in_vec[2];
    for (int k = 0; k < 2; k++) {
        partition_l_in_vec[k].push_back(buf_table_l_partition_in_col[0][k]);
        partition_l_in_vec[k].push_back(buf_table_l_partition_in_col[1][k]);
        partition_l_in_vec[k].push_back(buf_table_l_partition_in_col[2][k]);
        partition_l_in_vec[k].push_back(buf_meta_l_partition_in[k]);
        partition_l_in_vec[k].push_back(buf_cfg5s_part);
        partition_l_in_vec[k].push_back(buf_valid_l[k]);
    }

    // partition d2h
    std::vector<cl_mem> partition_l_out_vec[2];
    for (int k = 0; k < 2; k++) {
        partition_l_out_vec[k].push_back(buf_meta_l_partition_out[k]);
    }
    cl_event evt_buf_meta_l_partition_out[2];
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_l_partition_out[0], 0, 0, nullptr, &evt_buf_meta_l_partition_out[0]);
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_l_partition_out[1], 0, 0, nullptr, &evt_buf_meta_l_partition_out[1]);
    clWaitForEvents(2, evt_buf_meta_l_partition_out);

    // create user partition res cols
    // all sections partition 0 output to same 8-col buffers
    char*** table_l_new_part_col = mm.aligned_alloc<char**>(partition_num);

    // combine sec0_partition0, sec1_parttion0, ...secN_partition0 in 1 buffer.
    int64_t l_new_part_depth = l_partition_out_col_part_depth * table_l_sec_num;

    for (int p = 0; p < partition_num; ++p) {
        table_l_new_part_col[p] = mm.aligned_alloc<char*>(3);
        for (int i = 0; i < 3; ++i) {
            if (i < (int)(part_wr_col_out[1].size())) {
                table_l_new_part_col[p][i] = mm.aligned_alloc<char>(l_new_part_depth * size_apu_512);
                memset(table_l_new_part_col[p][i], 0, l_new_part_depth * size_apu_512);
            } else {
                table_l_new_part_col[p][i] = mm.aligned_alloc<char>(size_apu_512);
            }
        }
    }
    // record the new_part table offset to write partition i of all sections

    std::vector<std::vector<cl_event> > evt_part_l_h2d(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_part_l_krn(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_part_l_meta_d2h(table_l_sec_num);

    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        evt_part_l_h2d[sec].resize(1);
        evt_part_l_krn[sec].resize(1);
        evt_part_l_meta_d2h[sec].resize(1);
    }

    std::vector<std::vector<cl_event> > evt_part_l_h2d_dep(table_l_sec_num);
    evt_part_l_h2d_dep[0].resize(1);
    for (size_t i = 1; i < table_l_sec_num; ++i) {
        if (i == 1)
            evt_part_l_h2d_dep[i].resize(1);
        else
            evt_part_l_h2d_dep[i].resize(2);
    }
    std::vector<std::vector<cl_event> > evt_part_l_krn_dep(table_l_sec_num);
    evt_part_l_krn_dep[0].resize(1);
    for (size_t i = 1; i < table_l_sec_num; ++i) {
        if (i == 1)
            evt_part_l_krn_dep[i].resize(2);
        else
            evt_part_l_krn_dep[i].resize(4);
    }
    std::vector<std::vector<cl_event> > evt_part_l_d2h_dep(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_part_l_meta_d2h_dep(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_part_l_memcpy_out_dep(table_l_sec_num);
    evt_part_l_d2h_dep[0].resize(1);
    evt_part_l_meta_d2h_dep[0].resize(1);
    evt_part_l_memcpy_out_dep[0].resize(2);
    for (size_t i = 1; i < table_l_sec_num; ++i) {
        if (i == 1) {
            evt_part_l_meta_d2h_dep[i].resize(1);
            evt_part_l_d2h_dep[i].resize(1);
            evt_part_l_memcpy_out_dep[i].resize(2);
        } else {
            evt_part_l_meta_d2h_dep[i].resize(3);
            evt_part_l_d2h_dep[i].resize(2);
            evt_part_l_memcpy_out_dep[i].resize(2);
        }
    }

    // define partl memcpy in user events
    std::vector<std::vector<cl_event> > evt_part_l_memcpy_in(table_l_sec_num);
    for (size_t i = 0; i < table_l_sec_num; i++) {
        evt_part_l_memcpy_in[i].resize(1);
        evt_part_l_memcpy_in[i][0] = clCreateUserEvent(ctx, &err);
    }

    std::vector<std::vector<cl_event> > evt_part_l_d2h(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_part_l_memcpy_out(table_l_sec_num);
    for (size_t i = 0; i < table_l_sec_num; i++) {
        evt_part_l_d2h[i].resize(1);
        evt_part_l_memcpy_out[i].resize(1);
        evt_part_l_d2h[i][0] = clCreateUserEvent(ctx, &err);
        evt_part_l_memcpy_out[i][0] = clCreateUserEvent(ctx, &err);
    }

    std::vector<queue_struct_join> partl_min(table_l_sec_num);
    std::vector<queue_struct_join> partl_d2h(table_l_sec_num);
    std::vector<queue_struct_join> partl_mout(table_l_sec_num);

    gqe::utils::Timer tv_lpart;
    tv_lpart.add(); // 0
    for (size_t sec = 0; sec < table_l_sec_num; sec++) {
        int kid = sec % 2;
        // 1) memcpy in
        partl_min[sec].sec = sec;
        partl_min[sec].event = &evt_part_l_memcpy_in[sec][0];
        partl_min[sec].meta_nrow = table_l_sec_depth[sec];
        partl_min[sec].meta = &meta_l_partition_in[kid];
        for (int i = 0; i < 3; i++) {
            int idx = (int)q5s_part_scan[1][i];
            partl_min[sec].col_idx.push_back(idx);

            if (idx != -1) {
                partl_min[sec].size[i] = table_l_sec_size[i][sec];
                partl_min[sec].type_size[i] = table_l_col_types[i];
                partl_min[sec].ptr_src[i] = table_l_user_col_sec[i][sec];
                partl_min[sec].ptr_dst[i] = table_l_partition_in_col[i][kid];
            }
        }

        if (tab_b.getRowIDEnableFlag() && tab_b.getValidEnableFlag()) {
            partl_min[sec].col_idx.push_back(3);
            std::cout << "partl enable rowid and validation flag" << std::endl;
            partl_min[sec].size[3] = table_l_valid_sec_size[sec];
            partl_min[sec].type_size[3] = sizeof(int64_t);
            partl_min[sec].ptr_src[3] = table_l_valid_user[sec];
            partl_min[sec].ptr_dst[3] = din_valid_l[kid];
        }
        if (sec > 1) {
            partl_min[sec].num_event_wait_list = evt_part_l_h2d[sec - 2].size();
            partl_min[sec].event_wait_list = evt_part_l_h2d[sec - 2].data();
        } else {
            partl_min[sec].num_event_wait_list = 0;
            partl_min[sec].event_wait_list = nullptr;
        }
        if (kid == 0) pool.q2_ping.push(partl_min[sec]);
        if (kid == 1) pool.q2_pong.push(partl_min[sec]);

        // 2) h2d
        evt_part_l_h2d_dep[sec][0] = evt_part_l_memcpy_in[sec][0];
        if (sec > 1) {
            evt_part_l_h2d_dep[sec][1] = evt_part_l_krn[sec - 2][0];
        }
        clEnqueueMigrateMemObjects(cq, partition_l_in_vec[kid].size(), partition_l_in_vec[kid].data(), 0,
                                   evt_part_l_h2d_dep[sec].size(), evt_part_l_h2d_dep[sec].data(),
                                   &evt_part_l_h2d[sec][0]);

        // 3) kernel
        evt_part_l_krn_dep[sec][0] = evt_part_l_h2d[sec][0];
        if (sec > 0) {
            evt_part_l_krn_dep[sec][1] = evt_part_l_krn[sec - 1][0];
        }
        if (sec > 1) {
            evt_part_l_krn_dep[sec][2] = evt_part_l_meta_d2h[sec - 2][0];
            evt_part_l_krn_dep[sec][3] = evt_part_l_d2h[sec - 2][0];
        }
        clEnqueueTask(cq, partkernel_L[kid], evt_part_l_krn_dep[sec].size(), evt_part_l_krn_dep[sec].data(),
                      &evt_part_l_krn[sec][0]);

        // 4) d2h, transfer partiton results back
        evt_part_l_meta_d2h_dep[sec][0] = evt_part_l_krn[sec][0];
        if (sec > 1) {
            evt_part_l_meta_d2h_dep[sec][1] = evt_part_l_memcpy_out[sec - 2][0];
            evt_part_l_meta_d2h_dep[sec][2] = evt_part_l_d2h[sec - 2][0];
        }
        clEnqueueMigrateMemObjects(cq, partition_l_out_vec[kid].size(), partition_l_out_vec[kid].data(), 1,
                                   evt_part_l_meta_d2h_dep[sec].size(), evt_part_l_meta_d2h_dep[sec].data(),
                                   &evt_part_l_meta_d2h[sec][0]);

        // 5)a thread that wait d2h finish and do real output data d2h
        evt_part_l_d2h_dep[sec][0] = evt_part_l_meta_d2h[sec][0];
        if (sec > 1) {
            evt_part_l_d2h_dep[sec][1] = evt_part_l_memcpy_out[sec - 2][0];
        }
        partl_d2h[sec].sec = sec;
        partl_d2h[sec].partition_num = partition_num;
        partl_d2h[sec].part_max_nrow_512 = l_partition_out_col_part_depth;
        partl_d2h[sec].event = &evt_part_l_d2h[sec][0];
        partl_d2h[sec].meta = &meta_l_partition_out[kid];

        for (size_t i = 0; i < part_wr_col_out[1].size(); i++) {
            int l_i = part_wr_col_out[1][i];
            partl_d2h[sec].col_idx.push_back(l_i);
            if (l_i != -1) partl_d2h[sec].buf_head[i] = buf_head[0][6 + i + kid * 3];
        }
        partl_d2h[sec].num_event_wait_list = evt_part_l_d2h_dep[sec].size();
        partl_d2h[sec].event_wait_list = evt_part_l_d2h_dep[sec].data();
        partl_d2h[sec].cq = cq;
        partl_d2h[sec].dbuf = dbuf_ddr0;
        pool.q3_d2h.push(partl_d2h[sec]);

        // memcpy out evt dep
        evt_part_l_memcpy_out_dep[sec][0] = evt_part_l_meta_d2h[sec][0];
        evt_part_l_memcpy_out_dep[sec][1] = evt_part_l_d2h[sec][0];

        // 6) memcpy out
        partl_mout[sec].sec = sec;
        partl_mout[sec].partition_num = partition_num;
        partl_mout[sec].part_max_nrow_512 = l_partition_out_col_part_depth;
        partl_mout[sec].event = &evt_part_l_memcpy_out[sec][0];
        partl_mout[sec].meta = &meta_l_partition_out[kid];
        for (size_t i = 0; i < part_wr_col_out[1].size(); i++) {
            int o_i = part_wr_col_out[1][i];
            partl_mout[sec].col_idx.push_back(o_i);
            if (o_i != -1) {
                partl_mout[sec].ptr_src[i] = table_l_partition_out_col[i][kid];
                partl_mout[sec].type_size[i] = sizeof(int64_t); // TODO: table_l_col_types[i];
            }
        }
        partl_mout[sec].part_ptr_dst = table_l_new_part_col;
        partl_mout[sec].num_event_wait_list = evt_part_l_memcpy_out_dep[sec].size();
        partl_mout[sec].event_wait_list = evt_part_l_memcpy_out_dep[sec].data();

        if (kid == 0) pool.q3_ping.push(partl_mout[sec]);
        if (kid == 1) pool.q3_pong.push(partl_mout[sec]);
    }
    clWaitForEvents(evt_part_l_memcpy_out[table_l_sec_num - 1].size(),
                    evt_part_l_memcpy_out[table_l_sec_num - 1].data());
    if (table_l_sec_num > 1) {
        clWaitForEvents(evt_part_l_memcpy_out[table_l_sec_num - 2].size(),
                        evt_part_l_memcpy_out[table_l_sec_num - 2].data());
    }
    tv_lpart.add(); // 1
    pool.q2_ping_run = 0;
    pool.q2_pong_run = 0;

    pool.q3_d2h_run = 0;
    pool.q3_ping_run = 0;
    pool.q3_pong_run = 0;
    pool.part_l_in_ping_t.join();
    pool.part_l_in_pong_t.join();
    pool.part_l_d2h_t.join();
    pool.part_l_out_ping_t.join();
    pool.part_l_out_pong_t.join();

    for (int p = 0; p < partition_num; p++) {
        std::cout << "Table L, part: " << p << ", nrow: " << pool.l_new_part_offset[p] << std::endl;
#if 0
            for (int n = 0; n < pool.l_new_part_offset[p]; n++) {
                for (int c = 0; c < part_wr_col_out[1].size(); c++) {
                    int idx = part_wr_col_out[1][c];
                    if (idx != -1) {
                        std::cout << "col " << c << ": " << ((int64_t*)(table_l_new_part_col[p][c]))[n] << ", ";
                    }
                }
                std::cout << std::endl;
            }
#endif
    }
    double tvtime_lpart = tv_lpart.getMilliSec();
    double l_input_memcpy_size = 0;
    for (size_t j = 0; j < table_l_sec_num; j++) {
        for (int i = 0; i < 3; i++) {
            int idx = (int)q5s_part_scan[1][i];
            if (idx != -1) {
                l_input_memcpy_size += table_l_sec_size[i][j];
            }
        }
    }
    l_input_memcpy_size = l_input_memcpy_size / 1024 / 1024;

    // profiling
    {
        cl_ulong start, end;
        double ev_ns;

        for (size_t sec = 0; sec < table_l_sec_num; sec++) {
            //// part o h2d
            clGetEventProfilingInfo(evt_part_l_h2d[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_part_l_h2d[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partL sec: " << sec << " h2d time: " << ev_ns << " ms" << std::endl;
            // parto kernel
            clGetEventProfilingInfo(evt_part_l_krn[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_part_l_krn[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partL sec: " << sec << " krn time: " << ev_ns << " ms" << std::endl;

            // parto d2h
            clGetEventProfilingInfo(evt_part_l_meta_d2h[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
                                    NULL);
            clGetEventProfilingInfo(evt_part_l_meta_d2h[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end,
                                    NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "partL sec: " << sec << " meta d2h time: " << ev_ns << " ms" << std::endl;
        }
    }
#ifdef USER_DEBUG
    std::cout << "----------- finished L table partition-------------" << std::endl << std::endl;
#endif

    //----------------setup hash join---------------
    //--------build o_part_0-and probe with l_part_0---------
    //
    pool.hj_init();
    // build-probe need to be run for each partition pair. TO use the same host/device buffer, the max data size
    // among partitions must be obtained. Then buffer allocations are using the max data size
    int64_t table_o_build_in_nrow_max = 0;
    for (int p = 0; p < partition_num; p++) {
        table_o_build_in_nrow_max = std::max(table_o_build_in_nrow_max, pool.o_new_part_offset[p]);
    }
    int64_t table_o_build_in_depth_max = (table_o_build_in_nrow_max + VEC_LEN - 1) / VEC_LEN;
    int64_t table_o_build_in_size_max = table_o_build_in_depth_max * size_apu_512;

    int64_t table_l_probe_in_nrow_max = 0;
    for (int p = 0; p < partition_num; p++) {
        int64_t tmp = pool.l_new_part_offset[p];
        table_l_probe_in_nrow_max = std::max(table_l_probe_in_nrow_max, tmp);
    }
    int64_t table_l_probe_in_depth_max = (table_l_probe_in_nrow_max + VEC_LEN - 1) / VEC_LEN;
    // slice probe in size
    int64_t table_l_probe_in_slice_depth = (table_l_probe_in_depth_max + slice_num - 1) / slice_num;
    int64_t table_l_probe_in_slice_size = table_l_probe_in_slice_depth * size_apu_512;
    int64_t l_result_nrow = tab_c.getRowNum() / partition_num;

    int64_t table_l_probe_out_nrow = l_result_nrow;
    int64_t table_l_probe_out_depth = (table_l_probe_out_nrow + VEC_LEN - 1) / VEC_LEN;

    // slice probe out size
    int64_t table_l_probe_out_slice_nrow = table_l_probe_out_nrow / slice_num;
    int64_t table_l_probe_out_slice_depth = (table_l_probe_out_depth + slice_num - 1) / slice_num;
    int64_t table_l_probe_out_slice_size = table_l_probe_out_slice_depth * size_apu_512;

    char* table_o_build_in_col[3];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[0][i] != -1)
            table_o_build_in_col[i] = AllocHostBuf(1, table_o_build_in_size_max);
        else
            table_o_build_in_col[i] = AllocHostBuf(1, VEC_LEN);
    }

    // define probe kernel pinned host buffers, input
    char* table_l_probe_in_col[3][2];
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[1][i] != -1)
            table_l_probe_in_col[i][0] = AllocHostBuf(1, table_l_probe_in_slice_size);
        else
            table_l_probe_in_col[i][0] = AllocHostBuf(1, VEC_LEN);
    }
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[1][i] != -1)
            table_l_probe_in_col[i][1] = AllocHostBuf(1, table_l_probe_in_slice_size);
        else
            table_l_probe_in_col[i][1] = AllocHostBuf(1, VEC_LEN);
    }

    // define probe kernel pinned host buffers, output
    char* table_l_probe_out_col[4][2];

    std::vector<int8_t> q5s_join_wr = pjcfg.getShuffleWriteHJ();
    // define the final output and memset 0
    char* table_out_col[4];
    int64_t table_out_col_type[4];
    int64_t table_out_slice_size[4];
    for (int i = 0; i < 4; i++) {
#ifdef USER_DEBUG
        std::cout << "i: " << i << ", q5s_join_wr[i]: " << (int)q5s_join_wr[i] << std::endl;
#endif
        int shf_i = (int)q5s_join_wr[i];
        if (shf_i != -1) {
            // input/output col type are int64
            table_out_col_type[i] = tab_c.getColTypeSize(shf_i);
            table_out_slice_size[i] = table_l_probe_out_slice_size;
            table_out_col[i] = tab_c.getColPointer(shf_i);
        } else {
            table_out_slice_size[i] = VEC_LEN;
            table_out_col[i] = mm.aligned_alloc<char>(VEC_LEN);
        }
    }

    for (size_t i = 0; i < 4; i++) {
        table_l_probe_out_col[i][0] = AllocHostBuf(0, table_out_slice_size[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        table_l_probe_out_col[i][1] = AllocHostBuf(0, table_out_slice_size[i]);
    }

    //--------------- metabuffer setup -----------------
    // using col0 and col1 buffer during build
    // setup build used meta input
    // set to max here, can be updated in the iteration hash build-probe
    MetaTable meta_build_in;
    meta_build_in.setColNum(3);
    for (int i = 0; i < 3; i++) {
        meta_build_in.setCol(i, i, table_o_build_in_nrow_max);
    }

    // setup probe used meta input
    MetaTable meta_probe_in[2];
    for (int k = 0; k < 2; k++) {
        meta_probe_in[k].setColNum(3);
        for (int i = 0; i < 3; i++) {
            meta_probe_in[k].setCol(i, i, table_l_probe_in_nrow_max);
        }
    }
    //
    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3. (When aggr is off)
    // when aggr is on, actually only using col0 is enough.
    // below example only illustrates the output buffers can be shuffled.
    // setup probe used meta output
    MetaTable meta_probe_out[2];
    for (int k = 0; k < 2; k++) {
        meta_probe_out[k].setColNum(4);
        for (int i = 0; i < 4; i++) {
            meta_probe_out[k].setCol(i, i, table_l_probe_out_slice_nrow);
        }
    }

    //--------------------------------------------

    // build kernel
    cl_kernel bkernel;
    bkernel = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);
    // probe kernel
    cl_kernel pkernel[2];
    pkernel[0] = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);
    pkernel[1] = clCreateKernel(prg, "gqeJoin", &err);
    logger.logCreateKernel(err);

    size_t build_probe_flag_0 = 0;
    size_t build_probe_flag_1 = 1;

    cl_mem_ext_ptr_t mext_cfg5s_hj;
    cl_mem_ext_ptr_t mext_meta_build_in, mext_meta_probe_in[2], mext_meta_probe_out[2];

    mext_meta_build_in = {XCL_BANK1, meta_build_in.meta(), 0};
    mext_meta_probe_in[0] = {XCL_BANK1, meta_probe_in[0].meta(), 0};
    mext_meta_probe_in[1] = {XCL_BANK1, meta_probe_in[1].meta(), 0};
    mext_meta_probe_out[0] = {XCL_BANK0, meta_probe_out[0].meta(), 0};
    mext_meta_probe_out[1] = {XCL_BANK0, meta_probe_out[1].meta(), 0};

    mext_cfg5s_hj = {XCL_BANK1, q5s_cfg_join, 0};

    cl_mem buf_table_o_build_in_col[3];
    cl_buffer_region sub_table_o_build_in_size[3];
    sub_table_o_build_in_size[0] = {buf_head[1][12], buf_size[1][12]};
    sub_table_o_build_in_size[1] = {buf_head[1][13], buf_size[1][13]};
    sub_table_o_build_in_size[2] = {buf_head[1][14], buf_size[1][14]};

    for (int i = 0; i < 3; i++) {
        buf_table_o_build_in_col[i] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_o_build_in_size[i], &err);
    }
    cl_mem buf_table_l_probe_in_col[3][2];
    cl_buffer_region sub_table_l_probe_in_size[6];
    sub_table_l_probe_in_size[0] = {buf_head[1][15], buf_size[1][15]};
    sub_table_l_probe_in_size[1] = {buf_head[1][16], buf_size[1][16]};
    sub_table_l_probe_in_size[2] = {buf_head[1][17], buf_size[1][17]};
    sub_table_l_probe_in_size[3] = {buf_head[1][18], buf_size[1][18]};
    sub_table_l_probe_in_size[4] = {buf_head[1][19], buf_size[1][19]};
    sub_table_l_probe_in_size[5] = {buf_head[1][20], buf_size[1][20]};

    for (int i = 0; i < 3; i++) {
        buf_table_l_probe_in_col[i][0] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_in_size[i], &err);
        buf_table_l_probe_in_col[i][1] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_in_size[3 + i], &err);
    }
    // the table_out_slice_size is already re-sized by output-sw-shuffle
    cl_mem buf_table_l_probe_out_col[4][2];
    cl_buffer_region sub_table_l_probe_out_size[8];
    sub_table_l_probe_out_size[0] = {buf_head[0][12], buf_size[0][12]};
    sub_table_l_probe_out_size[1] = {buf_head[0][13], buf_size[0][13]};
    sub_table_l_probe_out_size[2] = {buf_head[0][14], buf_size[0][14]};
    sub_table_l_probe_out_size[3] = {buf_head[0][15], buf_size[0][15]};
    sub_table_l_probe_out_size[4] = {buf_head[0][16], buf_size[0][16]};
    sub_table_l_probe_out_size[5] = {buf_head[0][17], buf_size[0][17]};
    sub_table_l_probe_out_size[6] = {buf_head[0][18], buf_size[0][18]};
    sub_table_l_probe_out_size[7] = {buf_head[0][19], buf_size[0][19]};

    for (int i = 0; i < 4; i++) {
        buf_table_l_probe_out_col[i][0] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_out_size[i], &err);
        buf_table_l_probe_out_col[i][1] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_out_size[4 + i], &err);
    }

    cl_mem buf_cfg5s_hj = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                         (size_apu_512 * 14), &mext_cfg5s_hj, &err);

    cl_mem buf_meta_build_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                              (size_apu_512 * 8), &mext_meta_build_in, &err);

    cl_mem buf_meta_probe_in[2];
    buf_meta_probe_in[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_apu_512 * 8), &mext_meta_probe_in[0], &err);
    buf_meta_probe_in[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_apu_512 * 8), &mext_meta_probe_in[1], &err);
    cl_mem buf_meta_probe_out[2];
    buf_meta_probe_out[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (size_apu_512 * 8), &mext_meta_probe_out[0], &err);
    buf_meta_probe_out[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (size_apu_512 * 8), &mext_meta_probe_out[1], &err);

    char* din_valid_hj = mm.aligned_alloc<char>(VEC_LEN);
    cl_mem_ext_ptr_t mext_buf_valid_hj = {XCL_BANK1, din_valid_hj, 0};
    cl_mem buf_valid_hj = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                         VEC_LEN * sizeof(char), &mext_buf_valid_hj, &err);
    // make meta/cfg resident
    std::vector<cl_mem> hj_resident_vec;
    hj_resident_vec.push_back(buf_cfg5s_hj);
    hj_resident_vec.push_back(buf_meta_build_in);
    hj_resident_vec.push_back(buf_meta_probe_in[0]);
    hj_resident_vec.push_back(buf_meta_probe_in[1]);
    hj_resident_vec.push_back(buf_meta_probe_out[0]);
    hj_resident_vec.push_back(buf_meta_probe_out[1]);
    cl_event evt_hj_resident_vec;
    clEnqueueMigrateMemObjects(cq, hj_resident_vec.size(), hj_resident_vec.data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, &evt_hj_resident_vec);
    clWaitForEvents(1, &evt_hj_resident_vec);

//-----------end of hash join setup------------

#ifdef USER_DEBUG
    std::cout << "-------------HASH JOIN for each partition------------" << std::endl;
#endif
    // build kernel h2d
    std::vector<cl_mem> build_in_vec;
    for (int i = 0; i < 3; i++) {
        if (q5s_join_scan[0][i] != -1) {
            build_in_vec.push_back(buf_table_o_build_in_col[i]);
        }
    }
    build_in_vec.push_back(buf_cfg5s_hj);
    build_in_vec.push_back(buf_meta_build_in);

    // probe kernel h2d
    std::vector<cl_mem> probe_in_vec[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            if (q5s_join_scan[0][i] != -1) {
                probe_in_vec[k].push_back(buf_table_l_probe_in_col[i][k]);
            }
        }
        probe_in_vec[k].push_back(buf_meta_probe_in[k]);
    }

    // probe kernel d2h
    std::vector<cl_mem> probe_out_vec[2];
    for (int k = 0; k < 2; k++) {
        probe_out_vec[k].push_back(buf_meta_probe_out[k]);
    }

    // set kernel args
    // bkernel
    j = 0;
    clSetKernelArg(bkernel, j++, sizeof(size_t), &build_probe_flag_0);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o_build_in_col[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o_build_in_col[1]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_o_build_in_col[2]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_valid_hj);

    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_cfg5s_hj);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_build_in);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_meta_probe_out[0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][0]); // no output for build
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][0]);
    clSetKernelArg(bkernel, j++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][0]);
    for (int t = 0; t < PU_NM * 2; t++) {
        clSetKernelArg(bkernel, j++, sizeof(cl_mem), &dbuf_hbm[t]);
    }

    // pkernel
    for (int k = 0; k < 2; k++) {
        j = 0;
        clSetKernelArg(pkernel[k], j++, sizeof(size_t), &build_probe_flag_1);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_in_col[0][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_in_col[1][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_in_col[2][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_valid_hj);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_cfg5s_hj);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_meta_probe_in[k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_meta_probe_out[k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_out_col[1][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_out_col[2][k]);
        clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &buf_table_l_probe_out_col[3][k]);
        for (int t = 0; t < PU_NM * 2; t++) {
            clSetKernelArg(pkernel[k], j++, sizeof(cl_mem), &dbuf_hbm[t]);
        }
    }
    // define cl_event used for build and probe
    std::vector<std::vector<cl_event> > evt_build_h2d(partition_num);
    std::vector<std::vector<cl_event> > evt_build_krn(partition_num);
    std::vector<std::vector<cl_event> > evt_probe_h2d(partition_num * slice_num);
    std::vector<std::vector<cl_event> > evt_probe_krn(partition_num * slice_num);
    std::vector<std::vector<cl_event> > evt_probe_meta_d2h(partition_num * slice_num);

    for (int e_i = 0; e_i < partition_num * slice_num; e_i++) {
        evt_probe_h2d[e_i].resize(1);
        evt_probe_krn[e_i].resize(1);
        evt_probe_meta_d2h[e_i].resize(1);
    }
    for (int p = 0; p < partition_num; p++) {
        evt_build_h2d[p].resize(1);
        evt_build_krn[p].resize(1);
    }

    std::vector<std::vector<cl_event> > evt_probe_krn_dep(partition_num * slice_num);
    evt_probe_krn_dep[0].resize(2);
    evt_probe_krn_dep[1].resize(3);
    for (int e_i = 2; e_i < partition_num * slice_num; e_i++) {
        evt_probe_krn_dep[e_i].resize(5);
    }

    // define user events used for memcpy functions
    std::vector<std::vector<cl_event> > evt_build_memcpy_in(partition_num);
    for (int p = 0; p < partition_num; p++) {
        evt_build_memcpy_in[p].resize(1);
        evt_build_memcpy_in[p][0] = clCreateUserEvent(ctx, &err);
    }

    // define dependence events for build and probe
    std::vector<std::vector<cl_event> > evt_build_h2d_dep(partition_num);
    evt_build_h2d_dep[0].resize(1);
    for (int p = 1; p < partition_num; p++) {
        evt_build_h2d_dep[p].resize(2);
    }

    std::vector<std::vector<cl_event> > evt_build_krn_dep(partition_num);
    evt_build_krn_dep[0].resize(1);
    evt_build_krn_dep[1].resize(3);
    for (int p = 2; p < partition_num; p++) {
        evt_build_krn_dep[p].resize(4);
    }

    std::vector<std::vector<cl_event> > evt_probe_memcpy_in(partition_num * slice_num);
    for (int e_i = 0; e_i < partition_num * slice_num; e_i++) {
        evt_probe_memcpy_in[e_i].resize(1);
        evt_probe_memcpy_in[e_i][0] = clCreateUserEvent(ctx, &err);
    }

    std::vector<std::vector<cl_event> > evt_probe_h2d_dep(partition_num * slice_num);
    evt_probe_h2d_dep[0].resize(1);
    evt_probe_h2d_dep[1].resize(1);
    for (int e_i = 2; e_i < partition_num * slice_num; e_i++) {
        evt_probe_h2d_dep[e_i].resize(2);
    }

    std::vector<std::vector<cl_event> > evt_probe_meta_d2h_dep(partition_num * slice_num);
    std::vector<std::vector<cl_event> > evt_probe_d2h_dep(partition_num * slice_num);
    std::vector<std::vector<cl_event> > evt_probe_memcpy_out_dep(partition_num * slice_num);
    evt_probe_meta_d2h_dep[0].resize(1);
    evt_probe_meta_d2h_dep[1].resize(1);
    evt_probe_d2h_dep[0].resize(1);
    evt_probe_d2h_dep[1].resize(1);
    evt_probe_memcpy_out_dep[0].resize(2);
    evt_probe_memcpy_out_dep[1].resize(2);
    for (int e_i = 2; e_i < partition_num * slice_num; e_i++) {
        evt_probe_meta_d2h_dep[e_i].resize(3);
        evt_probe_d2h_dep[e_i].resize(2);
        evt_probe_memcpy_out_dep[e_i].resize(2);
    }

    std::vector<std::vector<cl_event> > evt_probe_d2h(partition_num * slice_num);
    std::vector<std::vector<cl_event> > evt_probe_memcpy_out(partition_num * slice_num);
    for (int e_i = 0; e_i < partition_num * slice_num; e_i++) {
        evt_probe_d2h[e_i].resize(1);
        evt_probe_memcpy_out[e_i].resize(1);
        evt_probe_d2h[e_i][0] = clCreateUserEvent(ctx, &err);
        evt_probe_memcpy_out[e_i][0] = clCreateUserEvent(ctx, &err);
    }

    gqe::utils::Timer tv_hj;

    // define callback function memcpy in/out used struct objects
    std::vector<queue_struct_join> build_min(partition_num);
    std::vector<std::vector<queue_struct_join> > probe_min(partition_num);
    for (int i = 0; i < partition_num; i++) {
        probe_min[i].resize(slice_num);
    }
    std::vector<std::vector<queue_struct_join> > probe_d2h(partition_num);
    for (int i = 0; i < partition_num; i++) {
        probe_d2h[i].resize(slice_num);
    }
    std::vector<std::vector<queue_struct_join> > probe_mout(partition_num);
    for (int i = 0; i < partition_num; i++) {
        probe_mout[i].resize(slice_num);
    }

    tv_hj.add(); // 0
    // to fully pipeline the build-probe processes among different partitions, counter e_i is used, which equals
    // partition_i * slice_i
    cl_event evt_buf_meta_probe_out[2];
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_probe_out[0], 0, 0, nullptr, &evt_buf_meta_probe_out[0]);
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_probe_out[1], 0, 0, nullptr, &evt_buf_meta_probe_out[1]);
    clWaitForEvents(2, evt_buf_meta_probe_out);

    int e_i = 0;
    for (int p = 0; p < partition_num; p++) {
        int64_t table_o_build_in_nrow = pool.o_new_part_offset[p]; // 0 is partition 0
        int64_t table_o_build_in_depth = (table_o_build_in_nrow + VEC_LEN - 1) / VEC_LEN;
        int64_t table_o_build_in_size = table_o_build_in_depth * size_apu_512;

        int64_t table_l_probe_in_nrow = pool.l_new_part_offset[p]; // 0 is partition 0
        int64_t per_slice_nrow = (table_l_probe_in_nrow + slice_num - 1) / slice_num;

        // assert(table_l_probe_in_nrow > slice_num);
        // assert(per_slice_nrow > slice_num);

        int64_t* table_l_probe_in_slice_nrow = mm.aligned_alloc<int64_t>(slice_num);

        for (int slice = 0; slice < slice_num; slice++) {
            table_l_probe_in_slice_nrow[slice] = per_slice_nrow;
            if (slice == slice_num - 1) {
                table_l_probe_in_slice_nrow[slice] = table_l_probe_in_nrow - slice * per_slice_nrow;
            }
        }

        //---------------build kernel run-------------------
        // 1) copy Order table from host DDR to build kernel pinned host buffer
        build_min[p].p = p;
        build_min[p].event = &evt_build_memcpy_in[p][0];
        build_min[p].meta_nrow = table_o_build_in_nrow;
        build_min[p].meta = &meta_build_in;
        for (int i = 0; i < 3; i++) {
            int idx = q5s_join_scan[0][i];
            build_min[p].col_idx.push_back(idx);
            if (idx != -1) {
                build_min[p].ptr_src[i] = table_o_new_part_col[p][q5s_join_scan[0][i]];
                build_min[p].ptr_dst[i] = table_o_build_in_col[i];
                build_min[p].size[i] = table_o_build_in_size;
            }
        }
        if (p > 0) {
            build_min[p].num_event_wait_list = evt_build_h2d[p - 1].size();
            build_min[p].event_wait_list = evt_build_h2d[p - 1].data();
        } else {
            build_min[p].num_event_wait_list = 0;
            build_min[p].event_wait_list = nullptr;
        }
        pool.q4.push(build_min[p]);

        // 2) migrate order table data from host buffer to device buffer
        evt_build_h2d_dep[p][0] = evt_build_memcpy_in[p][0];
        if (p > 0) {
            evt_build_h2d_dep[p][1] = evt_build_krn[p - 1][0];
        }
        clEnqueueMigrateMemObjects(cq, build_in_vec.size(), build_in_vec.data(), 0, evt_build_h2d_dep[p].size(),
                                   evt_build_h2d_dep[p].data(), &evt_build_h2d[p][0]);

        // 3) launch build kernel
        evt_build_krn_dep[p][0] = evt_build_h2d[p][0];
        if (p > 0) {
            evt_build_krn_dep[p][1] = evt_build_krn[p - 1][0];
            evt_build_krn_dep[p][2] = evt_probe_krn[e_i - 1][0];
        }
        if (p > 1) {
            evt_build_krn_dep[p][3] = evt_probe_krn[e_i - 2][0];
        }
        clEnqueueTask(cq, bkernel, evt_build_krn_dep[p].size(), evt_build_krn_dep[p].data(), &evt_build_krn[p][0]);

        //------------------probe kernel run in pipeline------------------
        int64_t table_l_probe_in_slice_nrow_sid;
        // int table_l_probe_in_slice_nrow_sid_size;

        // the idx of event
        for (int slice = 0; slice < slice_num; slice++) {
            int sid = e_i % 2;
            // the real nrow for each slice, only the last round is different to
            // per_slice_nrow
            table_l_probe_in_slice_nrow_sid = table_l_probe_in_slice_nrow[slice];
            // table_l_probe_in_slice_nrow_sid_size = table_l_probe_in_slice_nrow_sid * sizeof(int);

            // setup probe used meta input
            // 4) copy L table from host DDR to build kernel pinned host buffer
            probe_min[p][slice].per_slice_nrow = per_slice_nrow; // number in each slice
            probe_min[p][slice].p = p;
            probe_min[p][slice].slice = slice;
            probe_min[p][slice].event = &evt_probe_memcpy_in[e_i][0];
            probe_min[p][slice].meta_nrow = table_l_probe_in_slice_nrow_sid;
            probe_min[p][slice].meta = &meta_probe_in[sid];
            for (int i = 0; i < 3; i++) {
                int idx = q5s_join_scan[1][i];
                probe_min[p][slice].col_idx.push_back(idx);
                if (idx != -1) {
                    probe_min[p][slice].ptr_src[i] = table_l_new_part_col[p][q5s_join_scan[1][i]];
                    probe_min[p][slice].ptr_dst[i] = table_l_probe_in_col[i][sid];
                    probe_min[p][slice].type_size[i] = sizeof(int64_t);
                    probe_min[p][slice].size[i] = probe_min[p][slice].type_size[i] * table_l_probe_in_slice_nrow_sid;
                }
            }
            if (e_i > 1) {
                probe_min[p][slice].num_event_wait_list = evt_probe_h2d[e_i - 2].size();
                probe_min[p][slice].event_wait_list = evt_probe_h2d[e_i - 2].data();
            } else {
                probe_min[p][slice].num_event_wait_list = 0;
                probe_min[p][slice].event_wait_list = nullptr;
            }
            if (sid == 0) pool.q5_ping.push(probe_min[p][slice]);
            if (sid == 1) pool.q5_pong.push(probe_min[p][slice]);

            // 5) migrate L table data from host buffer to device buffer
            evt_probe_h2d_dep[e_i][0] = evt_probe_memcpy_in[e_i][0];
            if (e_i > 1) {
                evt_probe_h2d_dep[e_i][1] = evt_probe_krn[e_i - 2][0];
            }
            clEnqueueMigrateMemObjects(cq, probe_in_vec[sid].size(), probe_in_vec[sid].data(), 0,
                                       evt_probe_h2d_dep[e_i].size(), evt_probe_h2d_dep[e_i].data(),
                                       &evt_probe_h2d[e_i][0]);

            // 6) launch probe kernel
            evt_probe_krn_dep[e_i][0] = evt_probe_h2d[e_i][0];

            evt_probe_krn_dep[e_i][1] = evt_build_krn[p][0];
            if (e_i > 0) {
                evt_probe_krn_dep[e_i][2] = evt_probe_krn[e_i - 1][0];
            }
            if (e_i > 1) {
                evt_probe_krn_dep[e_i][3] = evt_probe_d2h[e_i - 2][0];
                evt_probe_krn_dep[e_i][4] = evt_probe_meta_d2h[e_i - 2][0];
            }
            clEnqueueTask(cq, pkernel[sid], evt_probe_krn_dep[e_i].size(), evt_probe_krn_dep[e_i].data(),
                          &evt_probe_krn[e_i][0]);

            // 7) migrate result data from device buffer to host buffer
            evt_probe_meta_d2h_dep[e_i][0] = evt_probe_krn[e_i][0];
            if (e_i > 1) {
                evt_probe_meta_d2h_dep[e_i][1] = evt_probe_memcpy_out[e_i - 2][0];
                evt_probe_meta_d2h_dep[e_i][2] = evt_probe_d2h[e_i - 2][0];
            }
            clEnqueueMigrateMemObjects(cq, probe_out_vec[sid].size(), probe_out_vec[sid].data(),
                                       CL_MIGRATE_MEM_OBJECT_HOST, evt_probe_meta_d2h_dep[e_i].size(),
                                       evt_probe_meta_d2h_dep[e_i].data(), &evt_probe_meta_d2h[e_i][0]);

            // 8) probe d2h
            evt_probe_d2h_dep[e_i][0] = evt_probe_meta_d2h[e_i][0];
            if (e_i > 1) {
                evt_probe_d2h_dep[e_i][1] = evt_probe_memcpy_out[e_i - 2][0];
            }
            probe_d2h[p][slice].p = p;
            probe_d2h[p][slice].slice = slice;
            probe_d2h[p][slice].event = &evt_probe_d2h[e_i][0];
            probe_d2h[p][slice].meta = &meta_probe_out[sid];
            probe_d2h[p][slice].part_max_nrow_512 = table_l_probe_out_slice_depth;

            for (int i = 0; i < 4; i++) {
                int shf_i = (int)q5s_join_wr[i];
                probe_d2h[p][slice].col_idx.push_back(shf_i);
                if (shf_i != -1) {
                    probe_d2h[p][slice].buf_head[i] = buf_head[0][12 + i + sid * 4];
                }
            }
            probe_d2h[p][slice].num_event_wait_list = evt_probe_d2h_dep[e_i].size();
            probe_d2h[p][slice].event_wait_list = evt_probe_d2h_dep[e_i].data();
            probe_d2h[p][slice].cq = cq;
            probe_d2h[p][slice].dbuf = dbuf_ddr0;
            pool.q6_d2h.push(probe_d2h[p][slice]);

            evt_probe_memcpy_out_dep[e_i][0] = evt_probe_meta_d2h[e_i][0];
            evt_probe_memcpy_out_dep[e_i][1] = evt_probe_d2h[e_i][0];
            // 9) memcpy the output data back to user host buffer
            probe_mout[p][slice].p = p;
            probe_mout[p][slice].slice = slice;
            probe_mout[p][slice].event = &evt_probe_memcpy_out[e_i][0];
            probe_mout[p][slice].meta = &meta_probe_out[sid];
            probe_mout[p][slice].part_max_nrow_512 = table_l_probe_out_slice_depth;
            for (int i = 0; i < 4; i++) {
                int shf_i = (int)q5s_join_wr[i];
                probe_mout[p][slice].col_idx.push_back(shf_i);
                if (shf_i != -1) {
                    probe_mout[p][slice].ptr_dst[i] = table_out_col[i];
                    probe_mout[p][slice].ptr_src[i] = table_l_probe_out_col[i][sid];
                    probe_mout[p][slice].type_size[i] = table_out_col_type[i];
                    probe_mout[p][slice].size[i] = table_out_col_type[i] * tab_c.getRowNum();
                }
            }
            probe_mout[p][slice].num_event_wait_list = evt_probe_memcpy_out_dep[e_i].size();
            probe_mout[p][slice].event_wait_list = evt_probe_memcpy_out_dep[e_i].data();
            pool.q6.push(probe_mout[p][slice]);

            e_i++;
        }
    }

    clWaitForEvents(1, evt_probe_memcpy_out[e_i - 1].data());
    clWaitForEvents(1, evt_probe_memcpy_out[e_i - 2].data());
    tv_hj.add(); // 1

    pool.q4_run = 0;
    pool.q5_run_ping = 0;
    pool.q5_run_pong = 0;
    pool.q6_d2h_run = 0;
    pool.q6_run = 0;

    pool.build_in_t.join();
    pool.probe_in_ping_t.join();
    pool.probe_in_pong_t.join();
    pool.probe_d2h_t.join();
    pool.probe_out_t.join();

    int64_t out_nrow_sum = 0;
    for (int p = 0; p < partition_num; p++) {
        for (int slice = 0; slice < slice_num; slice++) {
#ifdef USER_DEBUG
            std::cout << "GQE result p: " << p << ", s: " << slice << ", nrow: " << pool.toutrow[p][slice] << std::endl;
#endif
            out_nrow_sum += pool.toutrow[p][slice];
        }
    }
    tab_c.setRowNum(out_nrow_sum);

    //---------------------------

    // profiling
    {
        cl_ulong start, end;
        double ev_ns;

        for (int p = 0; p < partition_num; p++) {
            // build h2d kernel
            clGetEventProfilingInfo(evt_build_h2d[p][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_build_h2d[p][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "build h2d time: " << ev_ns << " ms" << std::endl;
            // build kernel
            clGetEventProfilingInfo(evt_build_krn[p][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_build_krn[p][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "build krn time: " << ev_ns << " ms" << std::endl;
        }
        ev_ns = 0;
        for (int i = 0; i < e_i; i++) {
            // probe h2d
            clGetEventProfilingInfo(evt_probe_h2d[i][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_probe_h2d[i][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "probe h2d time: " << ev_ns << " ms" << std::endl;
            // probe krn
            clGetEventProfilingInfo(evt_probe_krn[i][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
            clGetEventProfilingInfo(evt_probe_krn[i][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "probe krn time: " << ev_ns << " ms" << std::endl;
            // probe d2h
            clGetEventProfilingInfo(evt_probe_meta_d2h[i][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start,
                                    NULL);
            clGetEventProfilingInfo(evt_probe_meta_d2h[i][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
            ev_ns = (double)(end - start) / 1000000; // ns to ms
            std::cout << "probe meta d2h time: " << ev_ns << " ms" << std::endl;
        }
    }
    //------------------------------------------------------------------------
    //-----------------print the execution time of each part------------------

    double tvtime = 0;

    double hj_total_size = o_input_memcpy_size + l_input_memcpy_size;
    tvtime = tv_hj.getMilliSec();
    std::cout << "partO time: " << (double)tvtime_opart << "ms" << std::endl;
    std::cout << "partL time: " << (double)tvtime_lpart << "ms" << std::endl;
    std::cout << "hj time: " << (double)tvtime << "ms" << std::endl;
    double total_time = tvtime + tvtime_lpart + tvtime_opart;
    double out_bytes = (double)out_nrow_sum * sizeof(int) * out_valid_col_num / 1024 / 1024;

    std::cout << "-----------------------Input/Output Info-----------------------" << std::endl;
    std::cout << "Table" << std::setw(20) << "Column Number" << std::setw(30) << "Row Number" << std::endl;
    std::cout << "L" << std::setw(24) << o_valid_col_num << std::setw(30) << o_nrow << std::endl;
    std::cout << "R" << std::setw(24) << l_valid_col_num << std::setw(30) << l_nrow << std::endl;

    std::cout << "LxR" << std::setw(22) << out_valid_col_num << std::setw(30) << out_nrow_sum << std::endl;
    std::cout << "-----------------------Data Transfer Info-----------------------" << std::endl;
    std::cout << "H2D size (Left Table) = " << o_input_memcpy_size << " MB" << std::endl;
    std::cout << "H2D size (Right Table) = " << l_input_memcpy_size << " MB" << std::endl;
    std::cout << "HJ size (Left+Right Table) = " << hj_total_size << " MB" << std::endl;
    std::cout << "D2H size = " << out_bytes << " MB" << std::endl;

    std::cout << "-----------------------Performance Info-----------------------" << std::endl;
    std::cout << (double)tvtime_opart
              << " ms, throughput: " << o_input_memcpy_size / 1024 / ((double)tvtime_opart / 1000) << " GB/s"
              << std::endl;
    std::cout << (double)tvtime_lpart
              << " ms, throughput: " << l_input_memcpy_size / 1024 / ((double)tvtime_lpart / 1000) << " GB/s"

              << std::endl;
    std::cout << "End-to-end JOIN time: ";
    std::cout << (double)total_time << " ms, throughput: " << hj_total_size / 1024 / ((double)total_time / 1000)
              << " GB/s" << std::endl;

    //--------------release---------------
    for (size_t i = 0; i < table_o_sec_num; i++) {
        clReleaseEvent(evt_part_o_memcpy_in[i][0]);
        clReleaseEvent(evt_part_o_h2d[i][0]);
        clReleaseEvent(evt_part_o_krn[i][0]);
        clReleaseEvent(evt_part_o_meta_d2h[i][0]);
        clReleaseEvent(evt_part_o_d2h[i][0]);
        clReleaseEvent(evt_part_o_memcpy_out[i][0]);
    }

    for (size_t i = 0; i < table_l_sec_num; i++) {
        clReleaseEvent(evt_part_l_memcpy_in[i][0]);
        clReleaseEvent(evt_part_l_h2d[i][0]);
        clReleaseEvent(evt_part_l_krn[i][0]);
        clReleaseEvent(evt_part_l_meta_d2h[i][0]);
        clReleaseEvent(evt_part_l_d2h[i][0]);
        clReleaseEvent(evt_part_l_memcpy_out[i][0]);
    }
    for (int p = 0; p < partition_num; p++) {
        clReleaseEvent(evt_build_memcpy_in[p][0]);
        clReleaseEvent(evt_build_h2d[p][0]);
        clReleaseEvent(evt_build_krn[p][0]);
    }
    for (int e_i = 0; e_i < partition_num * slice_num; e_i++) {
        clReleaseEvent(evt_probe_memcpy_in[e_i][0]);
        clReleaseEvent(evt_probe_h2d[e_i][0]);
        clReleaseEvent(evt_probe_krn[e_i][0]);
        clReleaseEvent(evt_probe_meta_d2h[e_i][0]);
        clReleaseEvent(evt_probe_d2h[e_i][0]);
        clReleaseEvent(evt_probe_memcpy_out[e_i][0]);
    }
    clReleaseEvent(evt_part_o_resident);
    clReleaseEvent(evt_meta_o_partition_out[0]);
    clReleaseEvent(evt_meta_o_partition_out[1]);
    clReleaseEvent(evt_part_l_resident);
    clReleaseEvent(evt_buf_meta_l_partition_out[0]);
    clReleaseEvent(evt_buf_meta_l_partition_out[1]);
    clReleaseEvent(evt_hj_resident_vec);
    clReleaseEvent(evt_buf_meta_probe_out[0]);
    clReleaseEvent(evt_buf_meta_probe_out[1]);
    //---------part o-------
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_o_partition_in_col[c][k]);
            clReleaseMemObject(buf_table_o_partition_out_col[c][k]);
        }
    }
    clReleaseMemObject(buf_cfg5s_part);
    for (int k = 0; k < 2; k++) {
        clReleaseMemObject(buf_meta_o_partition_in[k]);
        clReleaseMemObject(buf_meta_o_partition_out[k]);
    }

    //--------part l-----
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l_partition_in_col[c][k]);
            clReleaseMemObject(buf_table_l_partition_out_col[c][k]);
        }
    }
    for (int k = 0; k < 2; k++) {
        clReleaseMemObject(buf_meta_l_partition_in[k]);
        clReleaseMemObject(buf_meta_l_partition_out[k]);
    }

    //--------hj------
    for (int c = 0; c < 3; c++) {
        clReleaseMemObject(buf_table_o_build_in_col[c]);
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l_probe_in_col[c][k]);
        }
    }
    for (int c = 0; c < 4; c++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l_probe_out_col[c][k]);
        }
    }

    clReleaseMemObject(buf_valid_o[0]);
    clReleaseMemObject(buf_valid_o[1]);
    clReleaseMemObject(buf_valid_l[0]);
    clReleaseMemObject(buf_valid_l[1]);

    clReleaseMemObject(buf_valid_hj);
    clReleaseMemObject(buf_cfg5s_hj);
    clReleaseMemObject(buf_meta_build_in);
    clReleaseMemObject(buf_meta_probe_in[0]);
    clReleaseMemObject(buf_meta_probe_in[1]);
    clReleaseMemObject(buf_meta_probe_out[0]);
    clReleaseMemObject(buf_meta_probe_out[1]);
    for (int k = 0; k < 2; k++) {
        clReleaseKernel(partkernel_O[k]);
        clReleaseKernel(partkernel_L[k]);
        clReleaseKernel(pkernel[k]);
    }
    clReleaseKernel(bkernel);
    return SUCCESS;
}

} // database
} // gqe
} // xf
