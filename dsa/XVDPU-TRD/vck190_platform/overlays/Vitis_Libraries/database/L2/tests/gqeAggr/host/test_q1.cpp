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

// OpenCL C API utils
#include "xclhost.hpp"
#include "x_utils.hpp"
#include "xf_utils_sw/logger.hpp"
// GQE L2
#include "xf_database/meta_table.hpp"
#include "xf_database/aggr_command.hpp"
// HLS
#include <ap_int.h>

#include "table_dt.hpp"
#include "q1.hpp"

#include <sys/time.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstdio>

const int PU_NM = 8;
const int VEC_SCAN = 8; // 256-bit column.

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

class MM {
   private:
    size_t _total;
    std::vector<void*> _pvec;

   public:
    MM() : _total(0) {}
    ~MM() {
        for (void* p : _pvec) {
            if (p) free(p);
        }
    }
    size_t size() const { return _total; }
    template <typename T>
    T* aligned_alloc(std::size_t num) {
        void* ptr = nullptr;
        size_t sz = num * sizeof(T);
        if (posix_memalign(&ptr, 4096, sz)) throw std::bad_alloc();
        _pvec.push_back(ptr);
        _total += sz;
        printf("align_alloc %lu Bytes\n", sz);
        return reinterpret_cast<T*>(ptr);
    }
};

int main(int argc, const char* argv[]) {
    std::cout << "\n--------- TPC-H Query 1 (1G) ---------\n";

    // cmd arg parser.
    x_utils::ArgParser parser(argc, argv);

    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    if (!parser.getCmdOption("-xclbin", xclbin_path)) {
        std::cout << "ERROR: xclbin path is not set!\n";
        return 1;
    }
    std::string scale;
    int sim_scale = 1;
    if (parser.getCmdOption("-scale", scale)) {
        try {
            sim_scale = std::stoi(scale);
        } catch (...) {
            sim_scale = 10000;
        }
    }

    int32_t lineitem_n = 6001215 / sim_scale;

    std::cout << "NOTE:running in sf" << scale << " data\n.";

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

    //--------------- metabuffer setup -----------------
    // setup meta input and output
    xf::database::gqe::MetaTable meta_aggr_in;
    meta_aggr_in.setColNum(8);
    meta_aggr_in.setCol(0, 0, lineitem_n);
    meta_aggr_in.setCol(1, 1, lineitem_n);
    meta_aggr_in.setCol(2, 2, lineitem_n);
    meta_aggr_in.setCol(3, 3, lineitem_n);
    meta_aggr_in.setCol(4, 4, lineitem_n);
    meta_aggr_in.setCol(5, 5, lineitem_n);
    meta_aggr_in.setCol(6, 6, lineitem_n);
    meta_aggr_in.setCol(7, 7, lineitem_n);

    int result_nrow = 20000;
    xf::database::gqe::MetaTable meta_aggr_out;
    meta_aggr_out.setColNum(8);
    meta_aggr_out.setCol(0, 0, result_nrow);
    meta_aggr_out.setCol(1, 1, result_nrow);
    meta_aggr_out.setCol(2, 2, result_nrow);
    meta_aggr_out.setCol(3, 3, result_nrow);
    meta_aggr_out.setCol(4, 4, result_nrow);
    meta_aggr_out.setCol(5, 5, result_nrow);
    meta_aggr_out.setCol(6, 6, result_nrow);
    meta_aggr_out.setCol(7, 7, result_nrow);

    using namespace xf::common::utils_sw;
    Logger logger(std::cout, std::cerr);

    // setup OpenCL related stuff
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cmq;
    cl_program prg;

    err = xclhost::init_hardware(&ctx, &dev_id, &cmq,
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

    // build kernel
    cl_kernel agg_kernel = clCreateKernel(prg, "gqeAggr", &err);
    logger.logCreateKernel(err);

    std::cout << "Kernel has been created\n";

    MM mm;

    // one config is enough
    using acmdclass = xf::database::gqe::AggrCommand;
    acmdclass acmd = acmdclass();
    acmd.Scan({3, 4, 5, 6, 0, 1, 2});
    acmd.setEvaluation(0, "strm1*(-strm2+c2)", {0, 100}, xf::database::gqe::sf100);
    acmd.setShuffle0({0, 1, 2, 3, 4, 5, 6, 8});
    acmd.setEvaluation(1, "strm1*(-strm2+c2)*(strm3+c3)", {0, 100, 100}, xf::database::gqe::sf10k);
    acmd.setShuffle1({0, 1, 8, 3, 4, 5, 6, 7});
    acmd.setFilter("d<=19980902");
    acmd.setShuffle2({4, 5});
    acmd.setShuffle3({6, 0, 1, 7, 2, 6, 0});

    acmd.setGroupAggr(0, xf::database::enums::AOP_MEAN);
    acmd.setGroupAggr(1, xf::database::enums::AOP_MEAN);
    acmd.setGroupAggr(2, xf::database::enums::AOP_MEAN);
    acmd.setGroupAggr(3, xf::database::enums::AOP_SUM);
    acmd.setGroupAggr(4, xf::database::enums::AOP_SUM);
    acmd.setGroupAggr(5, xf::database::enums::AOP_SUM);
    acmd.setGroupAggr(6, xf::database::enums::AOP_SUM);
    acmd.setGroupAggr(7, xf::database::enums::AOP_COUNT);

    acmd.setMerge(2, {0, 1});
    acmd.setMerge(4, {7});

    acmd.setWriteCol({0, 1, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15});

    ap_uint<32>* table_cfg = acmd.getConfigBits();
    ap_uint<32>* table_cfg_out = acmd.getConfigOutBits();

    int tb_in_col_depth = (lineitem_n + VEC_LEN - 1) / VEC_LEN;
    int tb_in_col_size = sizeof(ap_uint<512>) * tb_in_col_depth;

    ap_uint<512>* table_in_col0 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col1 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col2 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col3 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col4 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col5 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col6 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);
    ap_uint<512>* table_in_col7 = mm.aligned_alloc<ap_uint<512> >(tb_in_col_depth);

    size_t table_result_depth = (result_nrow + VEC_LEN - 1) / VEC_LEN;
    size_t table_result_size = table_result_depth * sizeof(ap_uint<512>);
    ap_uint<512>* table_out_col[16];
    for (int i = 0; i < 16; i++) {
        table_out_col[i] = mm.aligned_alloc<ap_uint<512> >(table_result_depth);
    }
    for (int i = 0; i < 16; i++) {
        memset(table_out_col[i], 0, table_result_size);
    }

    int error = 0;
    error += generate_data((int*)(table_in_col0), 50, lineitem_n);
    error += generate_data((int*)(table_in_col1), 50, lineitem_n);
    error += generate_data((int*)(table_in_col2), 50, lineitem_n);
    error += generate_data((int*)(table_in_col3), 50, lineitem_n);
    error += generate_data((int*)(table_in_col4), 50, lineitem_n);
    error += generate_data((int*)(table_in_col5), 50, lineitem_n);
    error += generate_data((int*)(table_in_col6), 50, lineitem_n);
    error += generate_data((int*)(table_in_col7), 50, lineitem_n);
    if (error) {
        fprintf(stderr, "ERROR: failed to load dat file.\n");
        return 1;
    }

    cl_mem_ext_ptr_t mext_table_in_col[8];
    cl_mem_ext_ptr_t mext_meta_aggr_in, mext_meta_aggr_out, mext_cfg, mext_cfg_out;
    cl_mem_ext_ptr_t mext_table_out[16], memExt[8];

    mext_table_in_col[0] = {0, table_in_col0, agg_kernel};
    mext_table_in_col[1] = {1, table_in_col1, agg_kernel};
    mext_table_in_col[2] = {2, table_in_col2, agg_kernel};
    mext_table_in_col[3] = {3, table_in_col3, agg_kernel};
    mext_table_in_col[4] = {4, table_in_col4, agg_kernel};
    mext_table_in_col[5] = {5, table_in_col5, agg_kernel};
    mext_table_in_col[6] = {6, table_in_col6, agg_kernel};
    mext_table_in_col[7] = {7, table_in_col7, agg_kernel};

    mext_meta_aggr_in = {8, meta_aggr_in.meta(), agg_kernel};
    mext_meta_aggr_out = {9, meta_aggr_out.meta(), agg_kernel};

    mext_cfg = {26, table_cfg, agg_kernel};
    mext_cfg_out = {27, table_cfg_out, agg_kernel};
    for (int i = 0; i < 16; ++i) {
        mext_table_out[i] = {(10 + i), table_out_col[i], agg_kernel};
    }

    for (int i = 0; i < 8; i++) {
        memExt[i] = {28 + i, nullptr, agg_kernel};
    }

    cl_mem buf_tb_in_col[8];
    for (int i = 0; i < 8; ++i) {
        buf_tb_in_col[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          tb_in_col_size, &mext_table_in_col[i], &err);
    }
    cl_mem buf_meta_aggr_in = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             (sizeof(ap_uint<512>) * 24), &mext_meta_aggr_in, &err);

    cl_mem buf_meta_aggr_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                              (sizeof(ap_uint<512>) * 24), &mext_meta_aggr_out, &err);
    cl_mem buf_cfg = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                    size_t(4 * 128), &mext_cfg, &err);
    cl_mem buf_cfg_out = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                        size_t(4 * 128), &mext_cfg_out, &err);
    cl_mem buf_tb_out_col[16];
    for (int i = 0; i < 16; ++i) {
        buf_tb_out_col[i] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           table_result_size, &mext_table_out[i], &err);
    }
    cl_mem buf_tmp[8];
    for (int i = 0; i < 8; i++) {
        buf_tmp[i] = clCreateBuffer(ctx, CL_MEM_READ_WRITE | CL_MEM_HOST_NO_ACCESS | CL_MEM_EXT_PTR_XILINX,
                                    (size_t)(8 * S_BUFF_DEPTH), &memExt[i], &err);
    }

    // set args and enqueue kernel
    int j = 0;
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[3]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[4]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[5]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[6]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[0]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[1]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[2]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_in_col[7]);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_meta_aggr_in);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_meta_aggr_out);
    for (int k = 0; k < 16; k++) {
        clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tb_out_col[k]);
    }
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_cfg);
    clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_cfg_out);
    for (int k = 0; k < 8; k++) {
        clSetKernelArg(agg_kernel, j++, sizeof(cl_mem), &buf_tmp[k]);
    }

    std::vector<cl_mem> in_vec;
    in_vec.push_back(buf_tb_in_col[0]);
    in_vec.push_back(buf_tb_in_col[1]);
    in_vec.push_back(buf_tb_in_col[2]);
    in_vec.push_back(buf_tb_in_col[3]);
    in_vec.push_back(buf_tb_in_col[4]);
    in_vec.push_back(buf_tb_in_col[5]);
    in_vec.push_back(buf_tb_in_col[6]);
    in_vec.push_back(buf_tb_in_col[7]);
    in_vec.push_back(buf_meta_aggr_in);
    in_vec.push_back(buf_meta_aggr_out);
    in_vec.push_back(buf_cfg);

    std::vector<cl_mem> out_vec;
    for (int i = 0; i < 16; ++i) {
        out_vec.push_back(buf_tb_out_col[i]);
    }
    out_vec.push_back(buf_meta_aggr_out);

    std::array<cl_event, 1> evt_h2d;
    std::array<cl_event, 1> evt_krn;
    std::array<cl_event, 1> evt_d2h;

    // step 1 h2d
    clEnqueueMigrateMemObjects(cmq, in_vec.size(), in_vec.data(), 0, 0, nullptr, &evt_h2d[0]);

    // step 2 run kernel
    clEnqueueTask(cmq, agg_kernel, 1, evt_h2d.data(), &evt_krn[0]);

    // step 3 d2h
    clEnqueueMigrateMemObjects(cmq, out_vec.size(), out_vec.data(), CL_MIGRATE_MEM_OBJECT_HOST, 1, evt_krn.data(),
                               &evt_d2h[0]);
    clFinish(cmq);
    std::cout << "finished data transfer d2h" << std::endl;
    int nrow = meta_aggr_out.getColLen();
    std::cout << "output nrow = " << nrow << std::endl;

    // re-use q1sort function to check results
    Table tk0("tk0", 20000, 16, "");
    Table tk1("tk1", 20000, 16, "");
    tk0.allocateHost();
    tk1.allocateHost();

    std::cout << "tbresultdepth = " << std::dec << table_result_depth << std::endl;
    size_t table_out_depth = (result_nrow + 2 * VEC_LEN - 1) / VEC_LEN;
    // tbout 16cols convert to tk0
    for (int i = 0; i < 16; i++) {
        memcpy(tk0.data + table_out_depth * i, table_out_col[i], sizeof(ap_uint<512>) * table_out_depth);
    }

    // step4 : kernel-join
    struct timeval tv_r_1, tv_r_e;
    gettimeofday(&tv_r_1, 0);
    q1Sort(tk0, tk1, meta_aggr_out);
    gettimeofday(&tv_r_e, 0);
    cl_ulong start1, end1;
    clGetEventProfilingInfo(evt_krn[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start1, NULL);
    clGetEventProfilingInfo(evt_krn[0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end1, NULL);

    long kerneltime1 = (end1 - start1) / 1000000;
    std::cout << std::dec << "Kernel execution time " << kerneltime1 << " ms" << std::endl;
    std::cout << std::dec << "Sort time " << x_utils::tvdiff(tv_r_1, tv_r_e) / 1000 << " ms" << std::endl;

    // q1Print(tk1);
    std::cout << "Golden result: -------------------------------------" << std::endl;

    Table tbg("tbg", 20000, 20, "");
    tbg.allocateHost();
    cpuQ1((int*)table_in_col0, (int*)table_in_col1, (int*)table_in_col2, (int*)table_in_col3, (int*)table_in_col4,
          (int*)table_in_col5, (int*)table_in_col6, (int*)table_in_col7, lineitem_n, tbg);

    int nerror = check_result(tk1, tbg);

    (nerror > 0) ? logger.error(Logger::Message::TEST_FAIL) : logger.info(Logger::Message::TEST_PASS);

    return nerror;
}
