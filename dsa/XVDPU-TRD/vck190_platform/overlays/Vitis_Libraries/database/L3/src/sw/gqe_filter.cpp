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

// L2
#include "xf_database/meta_table.hpp"
// L3
#include "xf_database/gqe_join_strategy.hpp" // for StrategySet
#include "xf_database/gqe_table.hpp"
#include "xf_database/gqe_filter.hpp"
#include "xf_database/gqe_bloomfilter.hpp"

#define USER_DEBUG 1
#define FILTER_PERF_PROFILE 1
//#define FILTER_PERF_PROFILE_2 1

#define TPCH_INT_SZ 8
#define VEC_LEN 8

#define XCL_BANK0 (XCL_MEM_TOPOLOGY | unsigned(32))
#define XCL_BANK1 (XCL_MEM_TOPOLOGY | unsigned(33))

namespace xf {
namespace database {
namespace gqe {

// async meta info needed for input/output
struct queue_struct_filter {
    // the sec index
    int sec;
    // the nrow setup of MetaTable
    int64_t meta_nrow;
    // updating meta info (nrow) for each sec, due to async, this
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
    // memcpy src locations for input/output
    char* ptr_src[4];
    // type size of memcpy in/out
    int type_size[4];
    // data size of memcpy in/out
    int64_t size[4];
    // memcpy dst locations for input/output
    char* ptr_dst[4];
};

// thread pool for performing paralleled memcpy
class threading_pool {
   public:
    // number of bytes in ap_uint<512>
    const int size_apu_512 = 64;
    std::thread probe_in_ping_t;
    std::thread probe_in_pong_t;
    std::thread probe_out_ping_t;
    std::thread probe_out_pong_t;

    std::mutex m;
    std::condition_variable cv;
    int cur;

    std::queue<queue_struct_filter> in_ping;  // probe memcpy in used queue
    std::queue<queue_struct_filter> in_pong;  // probe memcpy in used queue
    std::queue<queue_struct_filter> out_ping; // probe memcpy out used queue
    std::queue<queue_struct_filter> out_pong; // probe memcpy out used queue

    // the flag indicate each thread is running
    std::atomic<bool> in_ping_run;
    std::atomic<bool> in_pong_run;
    std::atomic<bool> out_ping_run;
    std::atomic<bool> out_pong_run;

    // total number of rows filtered out
    int64_t probe_out_nrow_accu = 0;
    // number of rows output for each section
    int64_t toutrow[256];

    // constructor
    threading_pool(){};

    void probe_memcpy_in_ping_t() {
        while (in_ping_run) {
#if Valgrind_debug
            sleep(1);
#endif
            while (!in_ping.empty()) {
                queue_struct_filter q = in_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    if (q.col_idx[i] != -1) memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                }

                q.meta->setColNum(col_num);
                for (int i = 0; i < col_num; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                in_ping.pop();
            }
        }
    }
    void probe_memcpy_in_pong_t() {
        while (in_pong_run) {
#if Valgrind_debug
            sleep(1);
#endif
            while (!in_pong.empty()) {
                queue_struct_filter q = in_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    if (q.col_idx[i] != -1) memcpy(q.ptr_dst[i], q.ptr_src[i], q.size[i]);
                }

                q.meta->setColNum(col_num);
                for (int i = 0; i < col_num; i++) {
                    q.meta->setCol(i, i, q.meta_nrow);
                }
                q.meta->meta();

                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the first element after processing it.
                in_pong.pop();
            }
        }
    }

    // probe memcpy out thread
    // only copy necessary output rows back to the user final output space.
    void probe_memcpy_out_ping_t() {
        while (out_ping_run) {
#if Valgrind_debug
            sleep(1);
#endif
            while (!out_ping.empty()) {
                queue_struct_filter q = out_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

                int64_t total_curr_nrow;
                // gets number of rows to be output
                int64_t probe_out_nrow = q.meta->getColLen();

                // until the other thread has updated probe_out_nrow_accu
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return cur == q.sec; });

                    total_curr_nrow = probe_out_nrow_accu;
                    // save the accumulate output nrow
                    probe_out_nrow_accu += probe_out_nrow;
                    // let the other thread proceed to next round
                    cur++;
                    cv.notify_one();
                }

#if FILTER_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                // save output data nrow
                toutrow[q.sec] = probe_out_nrow;

                int col_num = q.col_idx.size();

                for (int i = 0; i < col_num; i++) {
                    if (q.col_idx[i] != -1) {
                        int64_t u_dst_size = q.size[i];                      // user buffer size
                        int64_t pout_size = probe_out_nrow * q.type_size[i]; // current section size needs to be output
                        if (total_curr_nrow * q.type_size[i] + pout_size > u_dst_size) {
                            std::cerr << "Error in checking probe memcpy out size: user buffer size(" << u_dst_size
                                      << ") < output size(" << (total_curr_nrow * q.type_size[i] + pout_size) << ")"
                                      << std::endl;
                            std::cerr << "Please set enough buffer size for output table " << std::endl;
                            exit(1);
                        }
                        memcpy(q.ptr_dst[i] + total_curr_nrow * q.type_size[i], q.ptr_src[i], pout_size);
                    }
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);

                // remove the first element after processing it.
                out_ping.pop();

#if FILTER_PERF_PROFILE_2
                tv.add(); // 1
                int col_num = q.col_idx.size();
                double tvtime = tv.getMilliSec();
                double data_size = (double)pout_size * col_num / 1024 / 1024;
                std::cout << "Tab L sec: " << q.sec << ", probe memcpy out, size: " << data_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;

#endif
            }
        }
    }
    void probe_memcpy_out_pong_t() {
        while (out_pong_run) {
#if Valgrind_debug
            sleep(1);
#endif
            while (!out_pong.empty()) {
                queue_struct_filter q = out_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);

                int64_t total_curr_nrow;
                // gets number of rows to be output
                int64_t probe_out_nrow = q.meta->getColLen();

                // until the other thread has updated probe_out_nrow_accu
                {
                    std::unique_lock<std::mutex> lk(m);
                    cv.wait(lk, [&] { return cur == q.sec; });

                    total_curr_nrow = probe_out_nrow_accu;
                    // save the accumulate output nrow
                    probe_out_nrow_accu += probe_out_nrow;
                    // let the other thread proceed to next round
                    cur++;
                    cv.notify_one();
                }

#if FILTER_PERF_PROFILE_2
                gqe::utils::Timer tv;
                tv.add(); // 0
#endif
                // save output data nrow
                toutrow[q.sec] = probe_out_nrow;
                int col_num = q.col_idx.size();
                for (int i = 0; i < col_num; i++) {
                    if (q.col_idx[i] != -1) {
                        int64_t u_dst_size = q.size[i];                      // user buffer size
                        int64_t pout_size = probe_out_nrow * q.type_size[i]; // current section size needs to be output
                        if (total_curr_nrow * q.type_size[i] + pout_size > u_dst_size) {
                            std::cerr << "Error in checking probe memcpy out size: user buffer size(" << u_dst_size
                                      << ") < output size(" << (total_curr_nrow * q.type_size[i] + pout_size) << ")"
                                      << std::endl;
                            std::cerr << "Please set enough buffer size for output table " << std::endl;
                            exit(1);
                        }
                        memcpy(q.ptr_dst[i] + total_curr_nrow * q.type_size[i], q.ptr_src[i], pout_size);
                    }
                }

                clSetUserEventStatus(q.event[0], CL_COMPLETE);

                // remove the first element after processing it.
                out_pong.pop();

#if FILTER_PERF_PROFILE_2
                tv.add(); // 1
                int col_num = q.col_idx.size();
                double tvtime = tv.getMilliSec();
                double data_size = (double)pout_size * col_num / 1024 / 1024;
                std::cout << "Tab L sec: " << q.sec << ", probe memcpy out, size: " << data_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throughput: " << data_size / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;

#endif
            }
        }
    }

    // initialize the filter threads
    void filter_init() {
        // start the memcpy in thread and non-stop running
        in_ping_run = 1;
        probe_in_ping_t = std::thread(&threading_pool::probe_memcpy_in_ping_t, this);

        // start the memcpy in thread and non-stop running
        in_pong_run = 1;
        probe_in_pong_t = std::thread(&threading_pool::probe_memcpy_in_pong_t, this);

        // start the memcpy out thread and non-stop running
        out_ping_run = 1;
        probe_out_ping_t = std::thread(&threading_pool::probe_memcpy_out_ping_t, this);

        // start the part o memcpy in thread and non-stop running
        out_pong_run = 1;
        probe_out_pong_t = std::thread(&threading_pool::probe_memcpy_out_pong_t, this);

        cur = 0;
        cv.notify_all();
    }

}; // end class threading_pool

// performs pipelined N x bloom-filter probe
ErrCode Filter::filter_sol(Table& tab_in,
                           Table& tab_out,
                           BloomFilterConfig& fcfg,
                           uint64_t bf_size_in_bits,
                           ap_uint<256>** hash_table,
                           StrategySet params) {
    gqe::utils::MM mm;
    // get filter kernel config
    ap_uint<512>* q5s_cfg_filter = fcfg.getFilterConfigBits();
#ifdef USER_DEBUG
    std::cout << "cfg---------------cfg" << std::endl;
    for (int i = 0; i < 14; i++) {
        std::cout << std::dec << "No." << i << ": " << std::hex << q5s_cfg_filter[i] << std::endl;
    }
    std::cout << "cfg--------end------cfg" << std::endl;
#endif

    // get sw_shuffle_scan config
    std::vector<int8_t> q5s_filter_scan = fcfg.getShuffleScan();
    // read number of sections from user
    // only sec_l from user needed in gqeFilter
    int sec_l = params.sec_l;
    // get bloom-filter size in bytes
    uint64_t bf_size = bf_size_in_bits / 8;
    // copy hash-table of bloom-filter to HBM pinned host buffer
    for (int i = 0; i < PU_NM; i++) {
        memcpy(hbuf_hbm[i], hash_table[i], bf_size / PU_NM);
    }
    // get total row number and valid col number
    int64_t l_nrow = tab_in.getRowNum();
    int l_valid_col_num = tab_in.getColNum();
    // int l_valid_col_num = q5s_filter_scan.size();
    int out_valid_col_num = tab_out.getColNum();
#ifdef USER_DEBUG
    std::cout << "Number of sections sec_l: " << std::dec << sec_l << std::endl;
    std::cout << "Total number of rows in input table l_nrow: " << l_nrow << std::endl;
#endif

    // checks if we should bring the section number from json or calculate them locally (evenly divided)
    tab_in.checkSecNum(sec_l);
    std::cout << "Finish table division\n";
    int table_l_sec_num = tab_in.getSecNum();
    int* table_l_sec_depth = new int[table_l_sec_num];
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        table_l_sec_depth[sec] = tab_in.getSecRowNum(sec);
    }
// check each section number
// //////////////////////////////////////////////////
#ifdef USER_DEBUG
    std::cout << "table_l_sec_num: " << table_l_sec_num << std::endl;
    for (int i = 0; i < table_l_sec_num; i++) {
        std::cout << "table_l_sec_depth: " << table_l_sec_depth[i] << std::endl;
    }
#endif
    // //////////////////////////////////////////////////
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        if (table_l_sec_depth[sec] < table_l_sec_num) {
            std::cerr << "Error: Input table section size is smaller than section number!!!";
            std::cerr << "sec size of input table: " << table_l_sec_depth[sec] << ", ";
            std::cerr << "sec number of input table: " << table_l_sec_num << std::endl;
            exit(1);
        }
    }

    // calculates the size of each section from each column (in bytes)
    int64_t table_l_sec_size[3][table_l_sec_num];
    int table_l_col_types[3];
    for (int j = 0; j < 3; j++) {
        int idx = (int)q5s_filter_scan[j];
        if (idx != -1)
            table_l_col_types[j] = tab_in.getColTypeSize(idx);
        else
            table_l_col_types[j] = 8;

        for (int i = 0; i < table_l_sec_num; i++) {
            table_l_sec_size[j][i] = (int64_t)table_l_sec_depth[i] * table_l_col_types[j];
        }
    }
    // the max data size among different sections must be obtained.
    // Then buffer allocations are using the max data size
    int table_l_sec_depth_max = 0;
    int64_t table_l_sec_size_max[3];
    for (int i = 0; i < table_l_sec_num; i++) {
        if (table_l_sec_depth[i] > table_l_sec_depth_max) {
            table_l_sec_depth_max = table_l_sec_depth[i];
        }
    }
    for (int i = 0; i < 3; i++) {
        if (q5s_filter_scan[i] != -1) {
            table_l_sec_size_max[i] = (int64_t)table_l_sec_depth_max * table_l_col_types[i];
        } else {
            table_l_sec_size_max[i] = 64;
        }
    }

    // data load from disk. due to table size, data read into several sections
    char* table_l_user_col_sec[3][table_l_sec_num];
    for (int j = 0; j < 3; j++) {
        if (q5s_filter_scan[j] != -1) {
            for (int i = 0; i < table_l_sec_num; ++i) {
                table_l_user_col_sec[j][i] = tab_in.getColPointer(q5s_filter_scan[j], table_l_sec_num, i);
            }
        } else {
            for (int i = 0; i < table_l_sec_num; ++i) {
                table_l_user_col_sec[j][i] = mm.aligned_alloc<char>(8);
                memset(table_l_user_col_sec[j][i], 0, 8);
            }
        }
    }

    //
    //----------------setup bloom filter---------------
    //

    threading_pool pool;
    pool.filter_init();

    // define probe kernel pinned host buffers, input
    // 3 for 3 columns, 2 for ping & pong
    char* table_l_probe_in_col[3][2];
    for (int i = 0; i < 3; i++) {
        table_l_probe_in_col[i][0] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }
    for (int i = 0; i < 3; i++) {
        table_l_probe_in_col[i][1] = AllocHostBuf(1, table_l_sec_size_max[i]);
    }

    // define probe kernel pinned host buffers, output
    // 4 for 4 coumns at max, 2 for ping & pong
    char* table_l_probe_out_col[4][2];

    // get sw_shuffle_write config
    std::vector<int8_t> q5s_filter_wr = fcfg.getShuffleWrite();
    // define the final output and memset 0
    char* table_out_col[4];
    int64_t table_out_col_type[4];
    int64_t table_out_sec_size[4];
    int64_t table_l_probe_out_sec_size = 0;
    for (int j = 0; j < 3; j++) {
        if (q5s_filter_scan[j] != -1) {
            if (table_l_sec_size_max[j] > table_l_probe_out_sec_size) {
                table_l_probe_out_sec_size = table_l_sec_size_max[j];
            }
        }
    }
    for (int i = 0; i < 4; i++) {
        if (q5s_filter_wr[i] != -1) {
#ifdef USER_DEBUG
            std::cout << "i: " << i << ", q5s_filter_wr[i]: " << (int)q5s_filter_wr[i] << std::endl;
#endif
            int shf_i = (int)q5s_filter_wr[i];
            // input/output col type are int64
            table_out_col_type[i] = tab_out.getColTypeSize(shf_i);
            table_out_sec_size[i] = table_l_probe_out_sec_size;
            table_out_col[i] = tab_out.getColPointer(shf_i);
        } else {
            table_out_sec_size[i] = VEC_LEN;
            table_out_col[i] = mm.aligned_alloc<char>(VEC_LEN);
        }
    }

    for (size_t i = 0; i < 4; i++) {
        table_l_probe_out_col[i][0] = AllocHostBuf(0, table_out_sec_size[i]);
    }
    for (size_t i = 0; i < 4; i++) {
        table_l_probe_out_col[i][1] = AllocHostBuf(0, table_out_sec_size[i]);
    }

    //--------------- metabuffer setup -----------------
    // set to max here, can be updated in the iteration
    // setup probe used meta input
    MetaTable meta_probe_in[2];
    for (int k = 0; k < 2; k++) {
        meta_probe_in[k].setColNum(3);
        for (int i = 0; i < 3; i++) {
            meta_probe_in[k].setCol(i, i, table_l_sec_depth_max);
        }
    }
    //
    // ouput col0,1,2,3 buffers data, with order: 0 1 2 3.
    // below example only illustrates the output buffers can be shuffled.
    // setup probe used meta output
    MetaTable meta_probe_out[2];
    for (int k = 0; k < 2; k++) {
        meta_probe_out[k].setColNum(4);
        for (int i = 0; i < 4; i++) {
            meta_probe_out[k].setCol(i, i, table_l_sec_depth_max);
        }
    }

    //--------------------------------------------

    cl_int err;
    // probe kernel
    cl_kernel pkernel[2];
    pkernel[0] = clCreateKernel(prg, "gqeJoin", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create pkernel[0]\n");
        exit(1);
    }
    pkernel[1] = clCreateKernel(prg, "gqeJoin", &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create pkernel[1]\n");
        exit(1);
    }

    // 0 for build (unused), 1 for probe
    size_t build_probe_flag_0 = 0;
    size_t build_probe_flag_1 = 1;

    cl_mem_ext_ptr_t mext_cfg5s_ft;
    mext_cfg5s_ft = {XCL_BANK1, q5s_cfg_filter, 0};

    char* din_valid = mm.aligned_alloc<char>(VEC_LEN);
    cl_mem_ext_ptr_t mext_buf_valid = {XCL_BANK1, din_valid, 0};
    cl_mem buf_valid = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                      VEC_LEN * sizeof(char), &mext_buf_valid, &err);

    cl_mem_ext_ptr_t mext_meta_probe_in[2], mext_meta_probe_out[2];
    mext_meta_probe_in[0] = {XCL_BANK1, meta_probe_in[0].meta(), 0};   // pkernel[0]};
    mext_meta_probe_in[1] = {XCL_BANK1, meta_probe_in[1].meta(), 0};   // pkernel[1]};
    mext_meta_probe_out[0] = {XCL_BANK0, meta_probe_out[0].meta(), 0}; // pkernel[0]};
    mext_meta_probe_out[1] = {XCL_BANK0, meta_probe_out[1].meta(), 0}; // pkernel[1]};

    cl_mem buf_table_l_probe_in_col[3][2];
    cl_buffer_region sub_table_l_probe_in_size[6];
    sub_table_l_probe_in_size[0] = {buf_head[1][0], buf_size[1][0]};
    sub_table_l_probe_in_size[1] = {buf_head[1][1], buf_size[1][1]};
    sub_table_l_probe_in_size[2] = {buf_head[1][2], buf_size[1][2]};
    sub_table_l_probe_in_size[3] = {buf_head[1][3], buf_size[1][3]};
    sub_table_l_probe_in_size[4] = {buf_head[1][4], buf_size[1][4]};
    sub_table_l_probe_in_size[5] = {buf_head[1][5], buf_size[1][5]};

    for (int i = 0; i < 3; i++) {
        buf_table_l_probe_in_col[i][0] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_in_size[i], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create buffer buf_table_l_probe_in_col[%d][0]\n", i);
            exit(1);
        }
        buf_table_l_probe_in_col[i][1] =
            clCreateSubBuffer(dbuf_ddr1, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_in_size[3 + i], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create buffer buf_table_l_probe_in_col[%d][1]\n", i);
            exit(1);
        }
    }
    // the table_out_sec_size is already re-sized by output-sw-shuffle
    cl_mem buf_table_l_probe_out_col[4][2];
    cl_buffer_region sub_table_l_probe_out_size[8];
    sub_table_l_probe_out_size[0] = {buf_head[0][0], buf_size[0][0]};
    sub_table_l_probe_out_size[1] = {buf_head[0][1], buf_size[0][1]};
    sub_table_l_probe_out_size[2] = {buf_head[0][2], buf_size[0][2]};
    sub_table_l_probe_out_size[3] = {buf_head[0][3], buf_size[0][3]};
    sub_table_l_probe_out_size[4] = {buf_head[0][4], buf_size[0][4]};
    sub_table_l_probe_out_size[5] = {buf_head[0][5], buf_size[0][5]};
    sub_table_l_probe_out_size[6] = {buf_head[0][6], buf_size[0][6]};
    sub_table_l_probe_out_size[7] = {buf_head[0][7], buf_size[0][7]};

    for (int i = 0; i < 4; i++) {
        buf_table_l_probe_out_col[i][0] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_out_size[i], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create buffer buf_table_l_probe_out_col[%d][0]\n", i);
            exit(1);
        }
        buf_table_l_probe_out_col[i][1] =
            clCreateSubBuffer(dbuf_ddr0, CL_MEM_EXT_PTR_XILINX | CL_MEM_READ_WRITE, CL_BUFFER_CREATE_TYPE_REGION,
                              &sub_table_l_probe_out_size[4 + i], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create buffer buf_table_l_probe_out_col[%d][1]\n", i);
            exit(1);
        }
    }

    const int size_apu_512 = 64;
    cl_mem buf_cfg5s_ft = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                         (size_apu_512 * 14), &mext_cfg5s_ft, &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create buffer buf_cfg5s_ft\n");
        exit(1);
    }

    cl_mem buf_meta_probe_in[2];
    buf_meta_probe_in[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_apu_512 * 8), &mext_meta_probe_in[0], &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create buffer buf_meta_probe_in[0]\n");
        exit(1);
    }
    buf_meta_probe_in[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          (size_apu_512 * 8), &mext_meta_probe_in[1], &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create buffer buf_meta_probe_in[1]\n");
        exit(1);
    }
    cl_mem buf_meta_probe_out[2];
    buf_meta_probe_out[0] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (size_apu_512 * 8), &mext_meta_probe_out[0], &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create buffer buf_meta_probe_out[0]\n");
        exit(1);
    }
    buf_meta_probe_out[1] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                           (size_apu_512 * 8), &mext_meta_probe_out[1], &err);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to create buffer buf_meta_probe_out[1]\n");
        exit(1);
    }
//-----------end of gqeFilter setup------------

#ifdef USER_DEBUG
    std::cout << "-------------Bloom-filtering for each section------------" << std::endl;
#endif

    // meta/cfg buffer resident
    std::vector<cl_mem> resident_vec;
    resident_vec.push_back(buf_cfg5s_ft);
    resident_vec.push_back(buf_meta_probe_in[0]);
    resident_vec.push_back(buf_meta_probe_in[1]);
    resident_vec.push_back(buf_meta_probe_out[0]);
    resident_vec.push_back(buf_meta_probe_out[1]);

    // make sure buffers resident on dev
    clEnqueueMigrateMemObjects(cq, resident_vec.size(), resident_vec.data(), CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0,
                               nullptr, nullptr);

    // probe kernel h2d
    std::vector<cl_mem> probe_in_vec[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 3; i++) {
            if (q5s_filter_scan[i] != -1) {
                probe_in_vec[k].push_back(buf_table_l_probe_in_col[i][k]);
            }
        }
        probe_in_vec[k].push_back(buf_meta_probe_in[k]);
        probe_in_vec[k].push_back(buf_cfg5s_ft);
    }

    // probe kernel d2h
    std::vector<cl_mem> probe_out_vec[2];
    for (int k = 0; k < 2; k++) {
        for (int i = 0; i < 4; i++) {
            probe_out_vec[k].push_back(buf_table_l_probe_out_col[i][k]);
        }
        probe_out_vec[k].push_back(buf_meta_probe_out[k]);
    }

    // buffers for storing hash-table of bloom-filter
    std::vector<cl_mem> in_hbms;
    for (int c = 0; c < PU_NM; c++) {
        in_hbms.push_back(dbuf_hbm[c]);
    }

    // set kernel args
    // pkernel
    for (int k = 0; k < 2; k++) {
        int idx = 0;
        clSetKernelArg(pkernel[k], idx++, sizeof(size_t), &build_probe_flag_1);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_in_col[0][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_in_col[1][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_in_col[2][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_valid);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_cfg5s_ft);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_meta_probe_in[k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_meta_probe_out[k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_out_col[0][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_out_col[1][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_out_col[2][k]);
        clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &buf_table_l_probe_out_col[3][k]);
        for (int t = 0; t < PU_NM * 2; t++) {
            clSetKernelArg(pkernel[k], idx++, sizeof(cl_mem), &dbuf_hbm[t]);
        }
    }
    // define cl_event used for probe
    std::vector<std::vector<cl_event> > evt_probe_h2d(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_probe_krn(table_l_sec_num);
    std::vector<std::vector<cl_event> > evt_probe_d2h(table_l_sec_num);

    for (int sec = 0; sec < table_l_sec_num; sec++) {
        evt_probe_h2d[sec].resize(1);
        evt_probe_krn[sec].resize(1);
        evt_probe_d2h[sec].resize(1);
    }

    // define dependent cl_event for probe
    std::vector<std::vector<cl_event> > evt_probe_h2d_dep(table_l_sec_num);
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        if (sec == 0) {
            evt_probe_h2d_dep[0].resize(1);
        } else if (sec == 1) {
            evt_probe_h2d_dep[1].resize(1);
        } else {
            evt_probe_h2d_dep[sec].resize(2);
        }
    }

    std::vector<std::vector<cl_event> > evt_probe_krn_dep(table_l_sec_num);
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        if (sec == 0) {
            evt_probe_krn_dep[0].resize(1);
        } else if (sec == 1) {
            evt_probe_krn_dep[1].resize(2);
        } else {
            evt_probe_krn_dep[sec].resize(3);
        }
    }

    std::vector<std::vector<cl_event> > evt_probe_d2h_dep(table_l_sec_num);
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        if (sec == 0) {
            evt_probe_d2h_dep[0].resize(1);
        } else if (sec == 1) {
            evt_probe_d2h_dep[1].resize(1);
        } else {
            evt_probe_d2h_dep[sec].resize(2);
        }
    }

    // define probe memcpy in/out user events
    std::vector<std::vector<cl_event> > evt_probe_memcpy_in(table_l_sec_num);
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        evt_probe_memcpy_in[sec].resize(1);
        evt_probe_memcpy_in[sec][0] = clCreateUserEvent(ctx, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create user event evt_probe_memcpy_in[%d][0]\n", sec);
            exit(1);
        }
    }

    std::vector<std::vector<cl_event> > evt_probe_memcpy_out(table_l_sec_num);
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        evt_probe_memcpy_out[sec].resize(1);
        evt_probe_memcpy_out[sec][0] = clCreateUserEvent(ctx, &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: fail to create user event evt_probe_memcpy_out[%d][0]\n", sec);
            exit(1);
        }
    }

    // --------------------------- <1> ------------------------------
    // migrate hast-table into HBMs
    gqe::utils::Timer tv_in_hbm;
    tv_in_hbm.add();
    clEnqueueMigrateMemObjects(cq, in_hbms.size(), in_hbms.data(), 0, 0, nullptr, nullptr);
    clFinish(cq);
    tv_in_hbm.add();

    // --------------------------- <2> ------------------------------
    // performs pipelined bloom-filtering
    gqe::utils::Timer tv_ft;

    // define callback function memcpy in/out used struct objects
    queue_struct_filter probe_min[table_l_sec_num];
    queue_struct_filter probe_mout[table_l_sec_num];

    tv_ft.add(); // 0

    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_probe_out[0], 0, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, 1, &buf_meta_probe_out[1], 0, 0, nullptr, nullptr);

    //------------------probe kernel run in pipeline------------------
    for (int sec = 0; sec < table_l_sec_num; sec++) {
        // ping-pong switcher
        int sid = sec % 2;

        // setup probe used meta input
        // 1) copy L table from host DDR to probe kernel pinned host buffer
        probe_min[sec].sec = sec;
        probe_min[sec].event = &evt_probe_memcpy_in[sec][0];
        probe_min[sec].meta_nrow = table_l_sec_depth[sec];
        probe_min[sec].meta = &meta_probe_in[sid];
        for (int i = 0; i < 3; i++) {
            int idx = q5s_filter_scan[i];
            probe_min[sec].col_idx.push_back(idx);
            if (idx != -1) {
                probe_min[sec].ptr_src[i] = table_l_user_col_sec[i][sec];
                probe_min[sec].ptr_dst[i] = table_l_probe_in_col[i][sid];
                probe_min[sec].type_size[i] = table_l_col_types[i];
                probe_min[sec].size[i] = table_l_sec_depth[sec] * table_l_col_types[i];
            }
        }
        if (sec > 1) {
            probe_min[sec].num_event_wait_list = evt_probe_h2d[sec - 2].size();
            probe_min[sec].event_wait_list = evt_probe_h2d[sec - 2].data();
        } else {
            probe_min[sec].num_event_wait_list = 0;
            probe_min[sec].event_wait_list = nullptr;
        }
        if (sid == 0) pool.in_ping.push(probe_min[sec]);
        if (sid == 1) pool.in_pong.push(probe_min[sec]);

        // 2) migrate L table data from host buffer to device buffer
        evt_probe_h2d_dep[sec][0] = evt_probe_memcpy_in[sec][0];
        if (sec > 1) {
            evt_probe_h2d_dep[sec][1] = evt_probe_krn[sec - 2][0];
        }
        clEnqueueMigrateMemObjects(cq, probe_in_vec[sid].size(), probe_in_vec[sid].data(), 0,
                                   evt_probe_h2d_dep[sec].size(), evt_probe_h2d_dep[sec].data(),
                                   &evt_probe_h2d[sec][0]);
        // 3) launch probe kernel
        evt_probe_krn_dep[sec][0] = evt_probe_h2d[sec][0];
        if (sec > 0) {
            evt_probe_krn_dep[sec][1] = evt_probe_krn[sec - 1][0];
        }
        if (sec > 1) {
            evt_probe_krn_dep[sec][2] = evt_probe_d2h[sec - 2][0];
        }
        clEnqueueTask(cq, pkernel[sid], evt_probe_krn_dep[sec].size(), evt_probe_krn_dep[sec].data(),
                      &evt_probe_krn[sec][0]);

        // 4) migrate result data from device buffer to host buffer
        evt_probe_d2h_dep[sec][0] = evt_probe_krn[sec][0];
        if (sec > 1) {
            evt_probe_d2h_dep[sec][1] = evt_probe_memcpy_out[sec - 2][0];
        }
        clEnqueueMigrateMemObjects(cq, probe_out_vec[sid].size(), probe_out_vec[sid].data(), CL_MIGRATE_MEM_OBJECT_HOST,
                                   evt_probe_d2h_dep[sec].size(), evt_probe_d2h_dep[sec].data(),
                                   &evt_probe_d2h[sec][0]);

        // 5) memcpy the output data back to user host buffer
        probe_mout[sec].sec = sec;
        probe_mout[sec].event = &evt_probe_memcpy_out[sec][0];
        probe_mout[sec].meta_nrow = meta_probe_out[sid].getColLen();
        probe_mout[sec].meta = &meta_probe_out[sid];
        for (int i = 0; i < 4; i++) {
            int shf_i = (int)q5s_filter_wr[i];
            probe_mout[sec].col_idx.push_back(shf_i);
            if (shf_i != -1) {
                probe_mout[sec].ptr_dst[i] = table_out_col[i];
                probe_mout[sec].ptr_src[i] = table_l_probe_out_col[i][sid];
                probe_mout[sec].type_size[i] = table_out_col_type[i];
                probe_mout[sec].size[i] = table_out_col_type[i] * tab_out.getRowNum();
            }
        }
        probe_mout[sec].num_event_wait_list = evt_probe_d2h[sec].size();
        probe_mout[sec].event_wait_list = evt_probe_d2h[sec].data();
        if (sid == 0) pool.out_ping.push(probe_mout[sec]);
        if (sid == 1) pool.out_pong.push(probe_mout[sec]);
    }

    clWaitForEvents(evt_probe_memcpy_out[table_l_sec_num - 1].size(), evt_probe_memcpy_out[table_l_sec_num - 1].data());
    if (table_l_sec_num > 1) {
        clWaitForEvents(evt_probe_memcpy_out[table_l_sec_num - 2].size(),
                        evt_probe_memcpy_out[table_l_sec_num - 2].data());
    }
    tv_ft.add(); // 1

    // stop the sub-threads
    pool.in_ping_run = 0;
    pool.in_pong_run = 0;
    pool.out_ping_run = 0;
    pool.out_pong_run = 0;
    pool.probe_in_ping_t.join();
    pool.probe_in_pong_t.join();
    pool.probe_out_ping_t.join();
    pool.probe_out_pong_t.join();

    // calculate the results
    int64_t out_nrow_sum = 0;
    for (int sec = 0; sec < table_l_sec_num; sec++) {
#ifdef USER_DEBUG
        printf("GQE result sec: %d has %d rows\n", sec, pool.toutrow[sec]);
#endif
        out_nrow_sum += pool.toutrow[sec];
    }
    tab_out.setRowNum(out_nrow_sum);

    //------------------------------------------------------------------------
    //-----------------print the execution time of each part------------------
    double l_input_memcpy_size = 0;
    for (int i = 0; i < 3; i++) {
        if (q5s_filter_scan[i] != -1) {
            for (int j = 0; j < table_l_sec_num; j++) {
                l_input_memcpy_size += table_l_sec_size[i][j];
            }
        }
    }
    l_input_memcpy_size = l_input_memcpy_size / 1024 / 1024;

    double total_time = 0;
    total_time = tv_ft.getMilliSec();
    std::cout << "bloom-filtering time: " << (double)total_time << "ms" << std::endl;
    double out_bytes = (double)out_nrow_sum * sizeof(uint64_t) * out_valid_col_num / 1024 / 1024;

    std::cout << "-----------------------Input/Output Info-----------------------" << std::endl;
    std::cout << "Table" << std::setw(20) << "Column Number" << std::setw(30) << "Row Number" << std::endl;
    std::cout << "In" << std::setw(23) << l_valid_col_num << std::setw(30) << l_nrow << std::endl;
    std::cout << "Out" << std::setw(22) << out_valid_col_num << std::setw(30) << out_nrow_sum << std::endl;
    std::cout << "-----------------------Data Transfer Info-----------------------" << std::endl;
    std::cout << "H2D size = " << l_input_memcpy_size << " MB" << std::endl;
    std::cout << "D2H size = " << out_bytes << " MB" << std::endl;

    std::cout << "-----------------------Performance Info-----------------------" << std::endl;
    std::cout << "End-to-end Bloom-filtering time: ";
    std::cout << (double)total_time << " ms, throughput: " << l_input_memcpy_size / 1024 / ((double)total_time / 1000)
              << " GB/s" << std::endl;

    std::cout << "--------------release---------------\n";
    //--------bloom-filter------
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l_probe_in_col[c][k]);
        }
    }
    for (int c = 0; c < 3; c++) {
        for (int k = 0; k < 2; k++) {
            clReleaseMemObject(buf_table_l_probe_out_col[c][k]);
        }
    }
    clReleaseMemObject(buf_cfg5s_ft);
    clReleaseMemObject(buf_meta_probe_in[0]);
    clReleaseMemObject(buf_meta_probe_in[1]);
    clReleaseMemObject(buf_meta_probe_out[0]);
    clReleaseMemObject(buf_meta_probe_out[1]);
    std::cout << "--------------buffers released---------------\n";

    for (int sec = 0; sec < table_l_sec_num; sec++) {
        clReleaseEvent(evt_probe_memcpy_in[sec][0]);
        clReleaseEvent(evt_probe_h2d[sec][0]);
        clReleaseEvent(evt_probe_krn[sec][0]);
        clReleaseEvent(evt_probe_d2h[sec][0]);
        clReleaseEvent(evt_probe_memcpy_out[sec][0]);
    }
    std::cout << "--------------events released---------------\n";

    for (int k = 0; k < 2; k++) {
        clReleaseKernel(pkernel[k]);
    }
    std::cout << "--------------kernels released---------------\n";

    return SUCCESS;
}

ErrCode Filter::run(Table& tab_in,
                    std::string input_str,
                    BloomFilter& bf_in,
                    std::string filter_condition,
                    Table& tab_out,
                    std::string output_str,
                    StrategySet params) {
    ErrCode error_code;
    // gets information on input bloom-filter
    uint64_t in_bf_size = bf_in.getBloomFilterSize();
    ap_uint<256>** in_hash_table = bf_in.getHashTable();
    // gets shuffle/kernel configs for running
    BloomFilterConfig fcfg(tab_in, filter_condition, input_str, in_bf_size, tab_out, output_str);
    // performs bloom-filtering
    error_code = filter_sol(tab_in, tab_out, fcfg, in_bf_size, in_hash_table, params);
    return error_code;
}

} // namespace gqe
} // namespace database
} // namespace xf
