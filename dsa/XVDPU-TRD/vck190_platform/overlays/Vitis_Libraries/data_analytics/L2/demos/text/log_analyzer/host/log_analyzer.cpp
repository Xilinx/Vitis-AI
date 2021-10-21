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
#include "log_analyzer.hpp"
#include "log_analyzer_config.hpp"
#include "oniguruma.h"

namespace xf {
namespace search {
// constructor
// load binary and program
logAnalyzer::logAnalyzer(std::string xclbin) {
    xclbin_path = xclbin;
    err = xclhost::init_hardware(&ctx, &dev_id, &cq, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 MSTR(XDEVICE));
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL with " MSTR(XDEVICE) "\n");
        exit(1);
    }

    err = xclhost::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        exit(1);
    }
}

// de-constructor
logAnalyzer::~logAnalyzer() {
    err = clReleaseProgram(prg);
    if (err != CL_SUCCESS) {
        std::cout << "deconstructor" << std::endl;
        exit(1);
    }

    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
}

struct queue_struct {
    // the slice index
    int slc;
    // dependency event num
    int num_event_wait_list;
    // dependency events
    cl_event* event_wait_list;
    // user event to trace current operation
    cl_event* event;
    // line number of message for current slice
    uint32_t slc_lnm;
    // size of input for current slice
    uint32_t slc_sz;
    // destination of message pointer
    uint64_t* msg_ptr_dst;
    // destination of length pointer
    uint16_t* len_ptr_dst;
    // source of result pointer
    uint8_t* ptr_src;
};
class threading_pool {
   public:
    std::thread mcpy_in_ping_t;  // thread for memcpy in of input file;
    std::thread mcpy_in_pong_t;  // thread for memcpy int of input file;
    std::thread mcpy_out_ping_t; // thread for memcpy out of result;
    std::thread mcpy_out_pong_t; // thread for memcpy out of result;

    // task queue for part_in_t
    std::queue<queue_struct> q0_ping;
    // task queue for part_in_t
    std::queue<queue_struct> q0_pong;
    // task queue for memcpy_out_t
    std::queue<queue_struct> q1_ping;
    // task queue for memcpy_out_t
    std::queue<queue_struct> q1_pong;
    // flag to indicate each thread is running
    std::atomic<bool> q0_ping_run;
    std::atomic<bool> q0_pong_run;
    std::atomic<bool> q1_ping_run;
    std::atomic<bool> q1_pong_run;

    std::atomic<uint64_t> out_offt;
    std::atomic<uint64_t> msg_offt;
    std::atomic<uint64_t> len_offt;
    // pointer to msg buffer
    uint64_t* msg_ptr;
    // pointer to len buffer
    uint16_t* len_ptr;
    // pointer to output buffer
    uint8_t* out_ptr;

    // constructor
    threading_pool(){};

    // de-constructor
    ~threading_pool() {}
    // initialize the threads
    void init(uint64_t* msg_ptr, uint16_t* len_ptr, uint8_t* out_ptr) {
        this->msg_ptr = msg_ptr;
        this->len_ptr = len_ptr;
        this->out_ptr = out_ptr;
        // start the memcpy in of input log thread and non-stop
        q0_ping_run = true;
        mcpy_in_ping_t = std::thread(&threading_pool::func_mcpy_in_ping_t, this);
        mcpy_in_ping_t.detach();

        q0_pong_run = true;
        mcpy_in_pong_t = std::thread(&threading_pool::func_mcpy_in_pong_t, this);
        mcpy_in_pong_t.detach();

        // start the memcpy out of result thread and non-stop
        q1_ping_run = true;
        mcpy_out_ping_t = std::thread(&threading_pool::func_mcpy_out_ping_t, this);
        mcpy_out_ping_t.detach();

        q1_pong_run = true;
        mcpy_out_pong_t = std::thread(&threading_pool::func_mcpy_out_pong_t, this);
        mcpy_out_pong_t.detach();

        out_offt = 0;
        msg_offt = 0;
        len_offt = 0;
    }
    // input log file memcpy in thread
    void func_mcpy_in_ping_t() {
        while (q0_ping_run) {
            while (!q0_ping.empty()) {
                queue_struct q = q0_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);
#ifdef LOG_ANAY_RERY_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                uint64_t cur_m_oft = msg_offt;
                msg_offt += q.slc_sz;
                memcpy(q.msg_ptr_dst + 1, msg_ptr + cur_m_oft, q.slc_sz * sizeof(msg_ptr[0]));
                uint64_t cur_l_oft = len_offt;
                len_offt += q.slc_lnm;
                memcpy(q.len_ptr_dst + 2, len_ptr + cur_l_oft, q.slc_lnm * sizeof(len_ptr[0]));
#ifdef LOG_ANAY_RERY_PROFILE
                // update
                printf("slc = %d, lnm = %d, slice_sz = %d, start_pos = %d\n", q.slc, q.slc_lnm, q.slc_sz, cur_m_oft);
#endif

                q.msg_ptr_dst[0] = (uint64_t)(q.slc_sz + 1);
                q.len_ptr_dst[0] = (q.slc_lnm + 2) / 65536;
                q.len_ptr_dst[1] = (q.slc_lnm + 2) % 65536;
                // set the status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request
                q0_ping.pop();
#ifdef LOG_ANAY_RERY_PROFILE
                gettimeofday(&tv_end, 0);
                double sz_in_byte =
                    (double)(q.slc_sz * sizeof(msg_ptr[0]) + q.slc_lnm * sizeof(len_ptr[0])) / 1024 / 1024;
                double tvtime = x_utils::tvdiff(tv_start, tv_end);
                std::cout << "Input log slc: " << q.slc << ", memcpy in, size: " << sz_in_byte
                          << " MB, time: " << tvtime / 1000
                          << " ms, throughput: " << sz_in_byte / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;
#endif
            }
        }
    }
    // input log file memcpy in thread
    void func_mcpy_in_pong_t() {
        while (q0_pong_run) {
            while (!q0_pong.empty()) {
                queue_struct q = q0_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);
#ifdef LOG_ANAY_RERY_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                uint64_t cur_m_oft = msg_offt;
                msg_offt += q.slc_sz;
                memcpy(q.msg_ptr_dst + 1, msg_ptr + cur_m_oft, q.slc_sz * sizeof(msg_ptr[0]));
                uint64_t cur_l_oft = len_offt;
                len_offt += q.slc_lnm;
                memcpy(q.len_ptr_dst + 2, len_ptr + cur_l_oft, q.slc_lnm * sizeof(len_ptr[0]));
#ifdef LOG_ANAY_RERY_PROFILE
                // update
                printf("slc = %d, lnm = %d, slice_sz = %d, start_pos = %d\n", q.slc, q.slc_lnm, q.slc_sz, cur_m_oft);
#endif

                q.msg_ptr_dst[0] = (uint64_t)(q.slc_sz + 1);
                q.len_ptr_dst[0] = (q.slc_lnm + 2) / 65536;
                q.len_ptr_dst[1] = (q.slc_lnm + 2) % 65536;
                // set the status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request
                q0_pong.pop();
#ifdef LOG_ANAY_RERY_PROFILE
                gettimeofday(&tv_end, 0);
                double sz_in_byte =
                    (double)(q.slc_sz * sizeof(msg_ptr[0]) + q.slc_lnm * sizeof(len_ptr[0])) / 1024 / 1024;
                double tvtime = x_utils::tvdiff(tv_start, tv_end);
                std::cout << "Input log slc: " << q.slc << ", memcpy in, size: " << sz_in_byte
                          << " MB, time: " << tvtime / 1000
                          << " ms, throughput: " << sz_in_byte / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;
#endif
            }
        }
    }

    // post-process merge the result and memcpy them for pinned buffer to DDR buffer
    void func_mcpy_out_ping_t() {
        while (q1_ping_run) {
            while (!q1_ping.empty()) {
                queue_struct q = q1_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);
#ifdef LOG_ANAY_RERY_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                uint64_t out_sz = 0;
                memcpy(&out_sz, q.ptr_src, 8);
#ifdef LOG_ANAY_RERY_PROFILE
                printf("slc = %d, out_sz = %d\n", q.slc, out_sz);
#endif

                out_sz = out_sz - 256 / 8;

                uint64_t out_pos = out_offt + 8;
                // update the global offset
                out_offt += out_sz;
                // copy result to output buffer
                memcpy(out_ptr + out_pos, q.ptr_src + 256 / 8, out_sz * sizeof(q.ptr_src[0]));
                // set status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request from queue
                q1_ping.pop();
#ifdef LOG_ANAY_RERY_PROFILE
                gettimeofday(&tv_end, 0);
                double tvtime = x_utils::tvdiff(tv_start, tv_end);
                double memcpy_size = (double)out_sz / 1024 / 1024;
                std::cout << "Output slc: " << q.slc << ", memcpy out, size: " << memcpy_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throghput: " << memcpy_size / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;
#endif
            }
        }
    }
    // post-process, merge the result and memcpy them for pinned buffer to DDR buffer
    void func_mcpy_out_pong_t() {
        while (q1_pong_run) {
            while (!q1_pong.empty()) {
                queue_struct q = q1_pong.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);
#ifdef LOG_ANAY_RERY_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                uint64_t out_sz = 0;
                memcpy(&out_sz, q.ptr_src, 8);
#ifdef LOG_ANAY_RERY_PROFILE
                printf("slc = %d, out_sz = %d\n", q.slc, out_sz);
#endif

                out_sz = out_sz - 256 / 8;
                uint64_t out_pos = out_offt + 8;
                // update the global offset
                out_offt += out_sz;
                // copy result to output buffer
                memcpy(out_ptr + out_pos, q.ptr_src + 256 / 8, out_sz * sizeof(q.ptr_src[0]));
                // set status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request from queue
                q1_pong.pop();
#ifdef LOG_ANAY_RERY_PROFILE
                gettimeofday(&tv_end, 0);
                double tvtime = x_utils::tvdiff(tv_start, tv_end);
                double memcpy_size = (double)out_sz / 1024 / 1024;
                std::cout << "Output slc: " << q.slc << ", memcpy out, size: " << memcpy_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throghput: " << memcpy_size / 1024 / ((double)tvtime / 1000000) << " GB/s"
                          << std::endl;
#endif
            }
        }
    }
}; // end of class threading_pool

ErrCode logAnalyzer::analyze_all(uint64_t* cfg_buff,
                                 uint64_t* msg_buff,
                                 uint16_t* msg_len_buff,
                                 // buffer for GeoIP search
                                 uint64_t* net_high16,
                                 uint512* net_low21,
                                 // buffer for geo info in JSON format
                                 uint8_t* geo_buff,
                                 uint64_t* geo_len_buff,
                                 uint8_t* out_buff) {
    uint32_t cpgp_nm = cfg.getCpgpNm();
    x_utils::MM mm;

    // calculate the slction number
    timeval tv_start, tv_end;
    gettimeofday(&tv_start, 0);

    uint16_t* lnm_per_slc = mm.aligned_alloc<uint16_t>(MAX_SLC_NM);
    uint32_t* sz_per_slc = mm.aligned_alloc<uint32_t>(MAX_SLC_NM);

    uint32_t max_slice_lnm = 0;
    uint32_t slc_num = findSliceNum(msg_len_buff, msg_lnm, &max_slice_lnm, lnm_per_slc, sz_per_slc);

    gettimeofday(&tv_end, 0);
    double tvtime = x_utils::tvdiff(tv_start, tv_end);
    fprintf(stdout, "The log file is partition into %d slice with max_slice_lnm %d and  takes %f ms.\n", slc_num,
            max_slice_lnm, tvtime / 1000);

    // start threading pool threads
    threading_pool pool;
    pool.init(msg_buff, msg_len_buff, out_buff);
    // Aussuming the input log is very large,  it is divided into several slctions to improve throughput by overlap data
    // transfer and kernel exectuion
    // define memcpy in user events
    std::vector<std::vector<cl_event> > evt_memcpy_in(slc_num);
    for (unsigned int i = 0; i < slc_num; ++i) {
        evt_memcpy_in[i].resize(1);
        evt_memcpy_in[i][0] = clCreateUserEvent(ctx, &err);
    }
    // define memcpy out user events
    std::vector<std::vector<cl_event> > evt_memcpy_out(slc_num);
    for (unsigned int i = 0; i < slc_num; ++i) {
        evt_memcpy_out[i].resize(1);
        evt_memcpy_out[i][0] = clCreateUserEvent(ctx, &err);
    }
    // get kernel number
    std::string re_krnl_name = "reEngineKernel";
    cl_uint re_cu_num;
    {
        cl_kernel k = clCreateKernel(prg, re_krnl_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create kernel.\n");
            return DEV_ERR;
        }
        clGetKernelInfo(k, CL_KERNEL_COMPUTE_UNIT_COUNT, sizeof(re_cu_num), &re_cu_num, nullptr);
        std::cout << "DEBUG: " << re_krnl_name << " has " << re_cu_num << " CU(s)" << std::endl;
        clReleaseKernel(k);
    }
    std::string geo_krnl_name = "GeoIP_kernel";
    cl_uint geo_cu_num;
    {
        cl_kernel k = clCreateKernel(prg, geo_krnl_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create kernel.\n");
            return DEV_ERR;
        }
        clGetKernelInfo(k, CL_KERNEL_COMPUTE_UNIT_COUNT, sizeof(geo_cu_num), &geo_cu_num, nullptr);
        std::cout << "DEBUG: " << geo_krnl_name << " has " << geo_cu_num << " CU(s)" << std::endl;
        clReleaseKernel(k);
    }
    assert(re_cu_num % geo_cu_num == 0);
    std::string wj_krnl_name = "WJ_kernel";
    cl_uint wj_cu_num;
    {
        cl_kernel k = clCreateKernel(prg, wj_krnl_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create kernel.\n");
            return DEV_ERR;
        }
        clGetKernelInfo(k, CL_KERNEL_COMPUTE_UNIT_COUNT, sizeof(wj_cu_num), &wj_cu_num, nullptr);
        std::cout << "DEBUG: " << wj_krnl_name << " has " << wj_cu_num << " CU(s)" << std::endl;
        clReleaseKernel(k);
    }
    assert(re_cu_num % wj_cu_num == 0);
    // host side pinned buffers for reEngine
    std::vector<std::vector<uint64_t*> > msg_in_slice(3);
    std::vector<std::vector<uint16_t*> > len_in_slice(3);
    std::vector<std::vector<uint8_t*> > out_slice(3);
    for (int k = 0; k < 3; ++k) {
        // input buffer
        msg_in_slice[k].resize(re_cu_num);
        len_in_slice[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            msg_in_slice[k][c] = mm.aligned_alloc<uint64_t>((SLICE_MSG_SZ / 8) + 1);
            len_in_slice[k][c] = mm.aligned_alloc<uint16_t>(max_slice_lnm + 2);
        }
        // output buffer
        // output size is 4 timer than input size
        out_slice[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            out_slice[k][c] = mm.aligned_alloc<uint8_t>(8 * SLICE_MSG_SZ);
        }
    }
    // create kernel
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    std::vector<std::vector<cl_kernel> > re_krnls(3);
    for (int i = 0; i < 3; ++i) {
        re_krnls[i].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            std::string krnl_full_name = re_krnl_name + ":{" + re_krnl_name + "_" + std::to_string(c + 1) + "}";
            re_krnls[i][c] = clCreateKernel(prg, krnl_full_name.c_str(), &err);
            logger.logCreateKernel(err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create re kernel[%d].\n", c + 1);
                return DEV_ERR;
            }
        }
    }
    std::vector<std::vector<cl_kernel> > geo_krnls(3);
    for (int i = 0; i < 3; ++i) {
        geo_krnls[i].resize(re_cu_num);
        for (int j = 0; j < re_cu_num / geo_cu_num; ++j) {
            for (cl_uint c = 0; c < geo_cu_num; ++c) {
                std::string krnl_full_name = geo_krnl_name + ":{" + geo_krnl_name + "_" + std::to_string(c + 1) + "}";
                geo_krnls[i][j * geo_cu_num + c] = clCreateKernel(prg, krnl_full_name.c_str(), &err);
                logger.logCreateKernel(err);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "ERROR: failed to create geo kernel[%d].\n", c + 1);
                    return DEV_ERR;
                }
            }
        }
    }
    std::vector<std::vector<cl_kernel> > wj_krnls(3);
    for (int i = 0; i < 3; ++i) {
        wj_krnls[i].resize(re_cu_num);
        for (int j = 0; j < re_cu_num / wj_cu_num; ++j) {
            for (cl_uint c = 0; c < wj_cu_num; ++c) {
                std::string krnl_full_name = wj_krnl_name + ":{" + wj_krnl_name + "_" + std::to_string(c + 1) + "}";
                wj_krnls[i][j * wj_cu_num + c] = clCreateKernel(prg, krnl_full_name.c_str(), &err);
                logger.logCreateKernel(err);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "ERROR: failed to create wj kernel[%d].\n", c + 1);
                    return DEV_ERR;
                }
            }
        }
    }
    // create CL 3 ping-pong buffer for reEngine
    std::vector<cl_mem_ext_ptr_t> mext_cfg(re_cu_num);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_msg(3);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_len(3);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_re_out(3);
    for (cl_uint c = 0; c < re_cu_num; ++c) {
        mext_cfg[c] = {0, cfg_buff, re_krnls[0][c]};
    }
    for (int k = 0; k < 3; ++k) {
        mext_msg[k].resize(re_cu_num);
        mext_len[k].resize(re_cu_num);
        mext_re_out[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            mext_msg[k][c] = {1, msg_in_slice[k][c], re_krnls[k][c]};
            mext_len[k][c] = {2, len_in_slice[k][c], re_krnls[k][c]};
            // pure device buffer
            mext_re_out[k][c] = {3, nullptr, re_krnls[k][c]};
        }
    }
    // device buffer for reEngine
    std::vector<cl_mem> reCfgBuff(re_cu_num);
    std::vector<std::vector<cl_mem> > reMsgBuff(3);
    std::vector<std::vector<cl_mem> > reLenBuff(3);
    std::vector<std::vector<cl_mem> > reOutBuff(3);
    for (int k = 0; k < 3; ++k) {
        reMsgBuff[k].resize(re_cu_num);
        reLenBuff[k].resize(re_cu_num);
        reOutBuff[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            reMsgBuff[k][c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             sizeof(uint64_t) * ((SLICE_MSG_SZ / 8) + 1), &mext_msg[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create msg buffer\n");
                return MEM_ERR;
            }
        }
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            reLenBuff[k][c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             sizeof(uint16_t) * (max_slice_lnm + 2), &mext_len[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create len buffer\n");
                return MEM_ERR;
            }
        }
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            reOutBuff[k][c] =
                clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE,
                               sizeof(uint32_t) * ((cpgp_nm + 1) * max_slice_lnm + 1), &mext_re_out[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create out buffer\n");
                return MEM_ERR;
            }
        }
    }
    for (cl_uint c = 0; c < re_cu_num; c++) {
        reCfgBuff[c] =
            clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                           sizeof(uint64_t) * (INSTR_DEPTH + CCLASS_NM * 4 + 2 + CPGP_NM * 5), &mext_cfg[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create cfg buffer\n");
            return MEM_ERR;
        }
    }
    // create CL buffer for geo kernel
    std::vector<cl_mem_ext_ptr_t> mext_neth16(geo_cu_num);
    std::vector<cl_mem_ext_ptr_t> mext_netl21(geo_cu_num);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_geo_out(3);

    for (cl_uint c = 0; c < geo_cu_num; ++c) {
        mext_neth16[c] = {4, net_high16, geo_krnls[0][c]};
        mext_netl21[c] = {5, net_low21, geo_krnls[0][c]};
    }
    for (int k = 0; k < 3; ++k) {
        mext_geo_out[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            // pure device buffer
            mext_geo_out[k][c] = {7, nullptr, geo_krnls[0][c]};
        }
    }
    // Device buffer
    std::vector<cl_mem> geoNetH16Buff(geo_cu_num);
    std::vector<cl_mem> geoNetL21Buff(geo_cu_num);

    std::vector<std::vector<cl_mem> > geoOutBuff(3);
    for (int k = 0; k < 3; ++k) {
        geoOutBuff[k].resize(re_cu_num);
        for (int j = 0; j < re_cu_num / geo_cu_num; ++j) {
            for (cl_uint c = 0; c < geo_cu_num; ++c) {
                geoOutBuff[k][j * geo_cu_num + c] =
                    clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_HOST_NO_ACCESS | CL_MEM_READ_WRITE,
                                   sizeof(uint32_t) * (max_slice_lnm + 2), &mext_geo_out[k][j * geo_cu_num + c], &err);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "ERROR: failed to create geo out buffer\n");
                    return MEM_ERR;
                }
            }
        }
    }
    for (cl_uint c = 0; c < geo_cu_num; c++) {
        geoNetH16Buff[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          sizeof(uint64_t) * 65536, &mext_neth16[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create geo netH16 buffer\n");
            return MEM_ERR;
        }
    }
    for (cl_uint c = 0; c < geo_cu_num; c++) {
        geoNetL21Buff[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                          sizeof(uint512) * (GEO_DB_LNM / 16), &mext_netl21[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create geo netL21 buffer\n");
            return MEM_ERR;
        }
    }
    // create CL buffer for WJ kernel
    std::vector<cl_mem_ext_ptr_t> mext_geo_dt(wj_cu_num);
    std::vector<cl_mem_ext_ptr_t> mext_geo_len(wj_cu_num);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_wj_out(3);

    for (cl_uint c = 0; c < wj_cu_num; ++c) {
        mext_geo_dt[c] = {4, geo_buff, wj_krnls[0][c]};
        mext_geo_len[c] = {5, geo_len_buff, wj_krnls[0][c]};
    }
    for (int k = 0; k < 3; ++k) {
        mext_wj_out[k].resize(re_cu_num);
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            // device buffer
            mext_wj_out[k][c] = {7, out_slice[k][c], wj_krnls[0][c]};
        }
    }
    // Device buffer
    std::vector<cl_mem> wjDatBuff(wj_cu_num);
    std::vector<cl_mem> wjLenBuff(wj_cu_num);
    std::vector<std::vector<cl_mem> > wjOutBuff(3);
    for (int k = 0; k < 3; ++k) {
        wjOutBuff[k].resize(re_cu_num);
        for (int j = 0; j < re_cu_num / wj_cu_num; ++j) {
            for (cl_uint c = 0; c < wj_cu_num; ++c) {
                wjOutBuff[k][j * wj_cu_num + c] =
                    clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                                   sizeof(uint8_t) * SLICE_MSG_SZ * 8, &mext_wj_out[k][j * wj_cu_num + c], &err);
                if (err != CL_SUCCESS) {
                    fprintf(stderr, "ERROR: failed to create wj out buffer\n");
                    return MEM_ERR;
                }
            }
        }
    }
    for (cl_uint c = 0; c < wj_cu_num; c++) {
        // allocate 1G space
        wjDatBuff[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      1024 * 1024 * 1024, &mext_geo_dt[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create geo data buffer\n");
            return MEM_ERR;
        }
    }
    for (cl_uint c = 0; c < wj_cu_num; c++) {
        wjLenBuff[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      sizeof(uint64_t) * (geo_lnm + 1), &mext_geo_len[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create geo len buffer\n");
            return MEM_ERR;
        }
    }
    // make sure all buffers are resident on device
    std::vector<cl_mem> tot_in_bufs[3];
    for (int k = 0; k < 3; k++) {
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            tot_in_bufs[k].push_back(reMsgBuff[k][c]);
            tot_in_bufs[k].push_back(reLenBuff[k][c]);
        }
    }
    std::vector<cl_mem> tot_out_bufs[3];
    for (int k = 0; k < 3; k++) {
        for (cl_uint c = 0; c < re_cu_num; ++c) {
            tot_out_bufs[k].push_back(wjOutBuff[k][c]);
        }
    }
    clEnqueueMigrateMemObjects(cq, tot_in_bufs[0].size(), tot_in_bufs[0].data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_in_bufs[1].size(), tot_in_bufs[1].data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_in_bufs[2].size(), tot_in_bufs[2].data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);

    clEnqueueMigrateMemObjects(cq, tot_out_bufs[0].size(), tot_out_bufs[0].data(), 0, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_out_bufs[1].size(), tot_out_bufs[1].data(), 0, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_out_bufs[2].size(), tot_out_bufs[2].data(), 0, 0, nullptr, nullptr);
    // set kernel's arguements for reEngine
    for (cl_uint c = 0; c < re_cu_num; ++c) {
        for (int k = 0; k < 3; ++k) {
            clSetKernelArg(re_krnls[k][c], 0, sizeof(cl_mem), &reCfgBuff[c]);
            clSetKernelArg(re_krnls[k][c], 1, sizeof(cl_mem), &reMsgBuff[k][c]);
            clSetKernelArg(re_krnls[k][c], 2, sizeof(cl_mem), &reLenBuff[k][c]);
            clSetKernelArg(re_krnls[k][c], 3, sizeof(cl_mem), &reOutBuff[k][c]);
        }
    }
    uint32_t ipPos = (0 << 16) + 13;
    for (int j = 0; j < re_cu_num / geo_cu_num; ++j) {
        for (cl_uint c = 0; c < geo_cu_num; ++c) {
            for (int k = 0; k < 3; ++k) {
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 0, sizeof(uint32_t), &ipPos);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 1, sizeof(cl_mem), &reMsgBuff[k][j * geo_cu_num + c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 2, sizeof(cl_mem), &reLenBuff[k][j * geo_cu_num + c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 3, sizeof(cl_mem), &reOutBuff[k][j * geo_cu_num + c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 4, sizeof(cl_mem), &geoNetH16Buff[c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 5, sizeof(cl_mem), &geoNetL21Buff[c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 6, sizeof(cl_mem), &geoNetL21Buff[c]);
                clSetKernelArg(geo_krnls[k][j * geo_cu_num + c], 7, sizeof(cl_mem), &geoOutBuff[k][j * geo_cu_num + c]);
            }
        }
    }
    for (int j = 0; j < re_cu_num / wj_cu_num; ++j) {
        for (cl_uint c = 0; c < wj_cu_num; ++c) {
            for (int k = 0; k < 3; ++k) {
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 0, sizeof(cl_mem), &reCfgBuff[c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 1, sizeof(cl_mem), &reMsgBuff[k][j * wj_cu_num + c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 2, sizeof(cl_mem), &reLenBuff[k][j * wj_cu_num + c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 3, sizeof(cl_mem), &reOutBuff[k][j * wj_cu_num + c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 4, sizeof(cl_mem), &wjDatBuff[c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 5, sizeof(cl_mem), &wjLenBuff[c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 6, sizeof(cl_mem), &geoOutBuff[k][j * wj_cu_num + c]);
                clSetKernelArg(wj_krnls[k][j * wj_cu_num + c], 7, sizeof(cl_mem), &wjOutBuff[k][j * wj_cu_num + c]);
            }
        }
    }
    queue_struct mcpy_in_q[slc_num];
    queue_struct mcpy_out_q[slc_num];
    std::vector<std::vector<cl_event> > evt_h2d_vec(slc_num);
    std::vector<std::vector<cl_event> > evt_h2d(slc_num);
    for (unsigned int slc = 0; slc < slc_num; slc++) {
        if (slc >= re_cu_num * 3)
            evt_h2d_vec[slc].resize(2);
        else
            evt_h2d_vec[slc].resize(1);
        evt_h2d[slc].resize(1);
    }
    // events for reKernel
    std::vector<std::vector<cl_event> > evt_re_krnl_vec(slc_num);
    std::vector<std::vector<cl_event> > evt_re_krnl(slc_num);
    for (unsigned int slc = 0; slc < slc_num; slc++) {
        if (slc >= re_cu_num * 3)
            evt_re_krnl_vec[slc].resize(3);
        else if (slc >= re_cu_num)
            evt_re_krnl_vec[slc].resize(2);
        else
            evt_re_krnl_vec[slc].resize(1);
        evt_re_krnl[slc].resize(1);
    }
    // event for geoKernel
    std::vector<std::vector<cl_event> > evt_geo_krnl_vec(slc_num);
    std::vector<std::vector<cl_event> > evt_geo_krnl(slc_num);
    for (unsigned int slc = 0; slc < slc_num; slc++) {
        if (slc >= geo_cu_num * 3)
            evt_geo_krnl_vec[slc].resize(3);
        else if (slc >= geo_cu_num)
            evt_geo_krnl_vec[slc].resize(2);
        else
            evt_geo_krnl_vec[slc].resize(1);
        evt_geo_krnl[slc].resize(1);
    }
    // event for wjKernel
    std::vector<std::vector<cl_event> > evt_wj_krnl_vec(slc_num);
    std::vector<std::vector<cl_event> > evt_wj_krnl(slc_num);
    for (unsigned int slc = 0; slc < slc_num; slc++) {
        if (slc >= wj_cu_num * 3)
            evt_wj_krnl_vec[slc].resize(3);
        else if (slc >= wj_cu_num)
            evt_wj_krnl_vec[slc].resize(2);
        else
            evt_wj_krnl_vec[slc].resize(1);
        evt_wj_krnl[slc].resize(1);
    }

    std::vector<std::vector<cl_event> > evt_d2h_vec(slc_num);
    std::vector<std::vector<cl_event> > evt_d2h(slc_num);
    for (unsigned int slc = 0; slc < slc_num; slc++) {
        if (slc >= re_cu_num * 3)
            evt_d2h_vec[slc].resize(2);
        else
            evt_d2h_vec[slc].resize(1);
        evt_d2h[slc].resize(1);
    }

    timeval re_start, re_end;
    // cfg buffer migiration, only do this once
    std::vector<cl_mem> in_cfg_vec;
    for (cl_uint c = 0; c < re_cu_num; ++c) {
        in_cfg_vec.push_back(reCfgBuff[c]);
    }
    for (cl_uint c = 0; c < geo_cu_num; ++c) {
        in_cfg_vec.push_back(geoNetH16Buff[c]);
        in_cfg_vec.push_back(geoNetL21Buff[c]);
    }
    for (cl_uint c = 0; c < wj_cu_num; ++c) {
        in_cfg_vec.push_back(wjDatBuff[c]);
        in_cfg_vec.push_back(wjLenBuff[c]);
    }
    clEnqueueMigrateMemObjects(cq, in_cfg_vec.size(), in_cfg_vec.data(), 0, 0, nullptr, nullptr);
    clFinish(cq);
    gettimeofday(&re_start, 0);
    for (unsigned int slc = 0; slc < slc_num; ++slc) {
        // 3 ping-pong, which is used.
        int kid = (slc / re_cu_num) % 3;

        // which cu is used
        int cu_id = slc % re_cu_num;

        // which thread is used
        int mcpy_kid = slc % 2;
        // 1) memcpy_in
        mcpy_in_q[slc].slc = slc;
        mcpy_in_q[slc].msg_ptr_dst = msg_in_slice[kid][cu_id];
        mcpy_in_q[slc].len_ptr_dst = len_in_slice[kid][cu_id];
        mcpy_in_q[slc].slc_lnm = lnm_per_slc[slc];
        mcpy_in_q[slc].slc_sz = sz_per_slc[slc];

        if (slc >= re_cu_num * 3) {
            mcpy_in_q[slc].num_event_wait_list = evt_wj_krnl[slc - re_cu_num * 3].size();
            mcpy_in_q[slc].event_wait_list = evt_wj_krnl[slc - re_cu_num * 3].data();
        } else {
            mcpy_in_q[slc].num_event_wait_list = 0;
            mcpy_in_q[slc].event_wait_list = nullptr;
        }

        mcpy_in_q[slc].event = &evt_memcpy_in[slc][0];

        if (mcpy_kid == 0)
            pool.q0_ping.push(mcpy_in_q[slc]);
        else
            pool.q0_ping.push(mcpy_in_q[slc]);

        // clWaitForEvents(evt_memcpy_in[slc].size(), evt_memcpy_in[slc].data());
        // 2) H2D Migrate
        std::vector<cl_mem> in_vec;
        in_vec.push_back(reMsgBuff[kid][cu_id]);
        in_vec.push_back(reLenBuff[kid][cu_id]);

        evt_h2d_vec[slc][0] = evt_memcpy_in[slc][0];
        if (slc >= re_cu_num * 3) {
            evt_h2d_vec[slc][1] = evt_re_krnl[slc - 3 * re_cu_num][0];
        }
        clEnqueueMigrateMemObjects(cq, in_vec.size(), in_vec.data(), 0, evt_h2d_vec[slc].size(),
                                   evt_h2d_vec[slc].data(), &evt_h2d[slc][0]);
        // clFinish(cq);
        // printf("H2D transfer done\n");
        // 3) re kernel launch
        evt_re_krnl_vec[slc][0] = evt_h2d[slc][0];
        if (slc >= re_cu_num) {
            evt_re_krnl_vec[slc][1] = evt_re_krnl[slc - re_cu_num][0];
        }
        if (slc >= 3 * re_cu_num) {
            evt_re_krnl_vec[slc][2] = evt_wj_krnl[slc - 3 * re_cu_num][0];
        }
        clEnqueueTask(cq, re_krnls[kid][cu_id], evt_re_krnl_vec[slc].size(), evt_re_krnl_vec[slc].data(),
                      &evt_re_krnl[slc][0]);
        // clFinish(cq);
        // printf("reEngine done\n");
        // 4) geo kernel launch
        evt_geo_krnl_vec[slc][0] = evt_re_krnl[slc][0];
        if (slc >= geo_cu_num) {
            evt_geo_krnl_vec[slc][1] = evt_geo_krnl[slc - geo_cu_num][0];
        }
        if (slc >= 3 * geo_cu_num) {
            evt_geo_krnl_vec[slc][2] = evt_wj_krnl[slc - 3 * geo_cu_num][0];
        }
        clEnqueueTask(cq, geo_krnls[kid][cu_id], evt_geo_krnl_vec[slc].size(), evt_geo_krnl_vec[slc].data(),
                      &evt_geo_krnl[slc][0]);
        // clFinish(cq);
        // printf("geo kernel done\n");
        // 5) WJ kernel launch
        evt_wj_krnl_vec[slc][0] = evt_geo_krnl[slc][0];
        // wait the kernel is done
        if (slc >= wj_cu_num) {
            evt_wj_krnl_vec[slc][1] = evt_wj_krnl[slc - wj_cu_num][0];
        }
        // wait the d2h is done
        if (slc >= 3 * wj_cu_num) {
            evt_wj_krnl_vec[slc][2] = evt_d2h[slc - 3 * wj_cu_num][0];
        }
        clEnqueueTask(cq, wj_krnls[kid][cu_id], evt_wj_krnl_vec[slc].size(), evt_wj_krnl_vec[slc].data(),
                      &evt_wj_krnl[slc][0]);
        // clFinish(cq);
        // printf("wj kernel done\n");
        // 4) d2h, transfer partiion result back
        evt_d2h_vec[slc][0] = evt_wj_krnl[slc][0];
        if (slc >= 3 * re_cu_num) {
            evt_d2h_vec[slc][1] = evt_memcpy_out[slc - 3 * re_cu_num][0];
        }
        std::vector<cl_mem> out_vec;
        out_vec.push_back(wjOutBuff[kid][cu_id]);
        clEnqueueMigrateMemObjects(cq, out_vec.size(), out_vec.data(), CL_MIGRATE_MEM_OBJECT_HOST,
                                   evt_d2h_vec[slc].size(), evt_d2h_vec[slc].data(), &evt_d2h[slc][0]);
        // clFinish(cq);
        // printf("D2H done\n");

        // 6)memcpy out and post-process
        mcpy_out_q[slc].slc = slc;
        mcpy_out_q[slc].event = &evt_memcpy_out[slc][0];
        mcpy_out_q[slc].ptr_src = out_slice[kid][cu_id];
        mcpy_out_q[slc].num_event_wait_list = evt_d2h[slc].size();
        mcpy_out_q[slc].event_wait_list = evt_d2h[slc].data();
        if (mcpy_kid == 0)
            pool.q1_pong.push(mcpy_out_q[slc]);
        else
            pool.q1_pong.push(mcpy_out_q[slc]);
        // clWaitForEvents(evt_memcpy_out[slc].size(), evt_memcpy_out[slc].data());
    }
    // clWaitForEvents(evt_memcpy_out[2].size(), evt_memcpy_out[2].data());
    clWaitForEvents(evt_memcpy_out[slc_num - 1].size(), evt_memcpy_out[slc_num - 1].data());
    if (slc_num > 1) clWaitForEvents(evt_memcpy_out[slc_num - 2].size(), evt_memcpy_out[slc_num - 2].data());
    // stop all the threads
    pool.q0_ping_run = 0;
    pool.q1_ping_run = 0;
    pool.q0_pong_run = 0;
    pool.q1_pong_run = 0;
    gettimeofday(&re_end, 0);
    uint64_t out_sz = pool.out_offt;
    memcpy(out_buff, &out_sz, 8);
#ifdef LOG_ANAY_RERY_PROFILE
    cl_ulong start, end;
    long evt_ns;
    for (int slc = 0; slc < slc_num; ++slc) {
        double input_memcpy_size = (double)(SLICE_MSG_SZ + max_slice_lnm * 2) / 1024 / 1024;
        clGetEventProfilingInfo(evt_h2d[slc][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_h2d[slc][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "slc: " << slc << ", h2d, size: " << input_memcpy_size << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_memcpy_size / 1024 / ((double)evt_ns / 1000000000) << " GB/s"
                  << std::endl;

        double input_msg_size = (double)SLICE_MSG_SZ / 1024 / 1024;
        clGetEventProfilingInfo(evt_re_krnl[slc][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_re_krnl[slc][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "slc: " << slc << ", re-krnl, size: " << input_msg_size
                  << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_msg_size / 1024 / ((double)evt_ns / 1000000000) << " GB/s"
                  << std::endl;

        clGetEventProfilingInfo(evt_geo_krnl[slc][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_geo_krnl[slc][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "slc: " << slc << ", geo-krnl, size: " << input_msg_size
                  << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_msg_size / 1024 / ((double)evt_ns / 1000000000) << " GB/s"
                  << std::endl;

        clGetEventProfilingInfo(evt_wj_krnl[slc][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_wj_krnl[slc][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "slc: " << slc << ", wj-krnl, size: " << input_msg_size
                  << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_msg_size / 1024 / ((double)evt_ns / 1000000000) << " GB/s"
                  << std::endl;

        clGetEventProfilingInfo(evt_d2h[slc][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_d2h[slc][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        double output_memcpy_size = (double)(SLICE_MSG_SZ * 4) / 1024 / 1024;
        std::cout << "slc: " << slc << ", d2h, size: " << output_memcpy_size
                  << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << output_memcpy_size / 1024 / ((double)evt_ns / 1000000000) << " GB/s"
                  << std::endl;
    }
#endif

    double re_tvtime = x_utils::tvdiff(re_start, re_end);
    // TODO
    double total_log_size = pool.msg_offt * 8 / 1024 / 1024;
    std::cout << "logAnalyzer pipelined, time: " << (double)re_tvtime / 1000 << " ms, size: " << total_log_size
              << " MB, throughput: " << total_log_size / 1024 / ((double)re_tvtime / 1000000) << " GB/s" << std::endl;
    std::cout << "-----------------------------Finished logAnalyzer pipelined "
                 "test----------------------------------------------"
              << std::endl
              << std::endl;
    // DEVICE buffer
    return SUCCESS;
}
ErrCode logAnalyzer::analyze(uint64_t* msg_buff,
                             uint16_t* msg_len_buff,
                             uint32_t msg_lnm,
                             uint8_t* geo_db_buff,
                             uint32_t* geo_oft_buff,
                             uint32_t geo_lnm,
                             uint8_t* out_buff) {
    this->msg_lnm = msg_lnm;
    this->geo_lnm = geo_lnm;
    // convert geoIP data to requried format
    uint64_t* net_high16 = mm.aligned_alloc<uint64_t>(65536);
    uint512* net_low21 = mm.aligned_alloc<uint512>(GEO_DB_LNM / 16);
    geoIPConvert(geo_db_buff, geo_oft_buff, net_high16, net_low21);

    // covert geoIP data to JSON format
    uint8_t* geo_buff = mm.aligned_alloc<uint8_t>(1024 * 1024 * 1024);
    uint64_t* geo_len_buff = mm.aligned_alloc<uint64_t>(geo_lnm + 1);
    geoCSV2JSON(geo_db_buff, geo_oft_buff, geo_buff, geo_len_buff);

    // call the analyze_add to do grok, geoIP search and JSON dump
    uint64_t* cfg_buff = cfg.getConfigBits();
    return analyze_all(cfg_buff, msg_buff, msg_len_buff, net_high16, net_low21, geo_buff, geo_len_buff, out_buff);
}
int logAnalyzer::geoCSV2JSON(uint8_t* geo_db_buff, uint32_t* geo_oft_buff, uint8_t* geo_buff, uint64_t* geo_len_buff) {
    // get the fiedldName
    std::vector<std::string> fieldName;

    uint64_t index = 0;
    for (unsigned int l = 0; l < geo_lnm; l++) {
        std::string geo_str;
        size_t geo_sz = geo_oft_buff[l + 1] - geo_oft_buff[l];
        geo_str.assign((char*)&geo_db_buff[geo_oft_buff[l]], geo_sz);

        std::string onejson = geo_str.substr(geo_str.find('#') + 1, geo_str.size());
        onejson += "\n";
        // std::cout << "l=" << l << ", index=" << index << ", onejson.size()=" << onejson.size()
        //          << ", onejson=" << onejson;
        memcpy(geo_buff + index, onejson.data(), onejson.size());
        geo_len_buff[l] = index + ((index + onejson.size()) << 32);
        index += onejson.size();
        assert(index < 1024 * 1024 * 1024);
    }
    std::cout << "required geo buffer size " << index << std::endl;
    return 0;
}
int logAnalyzer::geoIPConvert(uint8_t* geo_db_buff, uint32_t* geo_oft_buff, uint64_t* net_high16, uint512* net_low21) {
    std::cout << "geoIPConvert\n";
    // the value of last a1a2
    int netsHigh16Cnt = -1;
    // store the row-number for each High-16 IP
    unsigned int* netsHigh16_tmp = mm.aligned_alloc<unsigned int>(65536);
    // low-16 bit IP and 5-bit mask
    unsigned int* netsLow21_tmp = mm.aligned_alloc<unsigned int>(GEO_DB_LNM);

    // store the flag for 0.0 to 255.255
    unsigned char* netsBFlag = mm.aligned_alloc<unsigned char>(65536);
    // initialize the netsBFlag buffer
    for (uint64_t i = 0; i < 65536; i++) {
        netsBFlag[i] = 0;
    }
    // parse the IP and convert to int
    // remove the first line
    uint32_t last_ip = 0;
    std::string geo_str;
    for (unsigned int i = 0; i < geo_lnm; i++) {
        // extract the net IP sub-string and store it in a temporary buffer
        size_t geo_sz = geo_oft_buff[i + 1] - geo_oft_buff[i];
        geo_str.assign((char*)&geo_db_buff[geo_oft_buff[i]], geo_sz);
        // std::cout << geo_str << std::endl;

        std::string net_str = geo_str.substr(0, geo_str.find(' '));
        // std::cout << "net_str=" << net_str << std::endl;
        uint32_t ip = std::stol(net_str.substr(0, net_str.find('/')));
        uint32_t b = std::stoi(net_str.substr(net_str.find('/') + 1, net_str.size()));
        // std::cout << "ip=" << ip << ",b=" << b << std::endl;

        unsigned int a1a2 = ip / 65536;
        unsigned int a3a4 = ip % 65536;
        // std::cout << "a1a2=" << a1a2 << ",a3a4=" << a3a4 << std::endl;
        netsLow21_tmp[i] = a3a4 + (b - 1) * 0x10000;

        // get the mask value
        if (b <= 16) {
            for (int ib = 0; ib < (1 << (16 - b)); ib++) {
                netsBFlag[a1a2 + ib] = 1;
            }
        }
        for (long d = netsHigh16Cnt; d < a1a2; d++) {
            // store the row number
            // the high 64-bit store the row number
            // the low 1-bit store the flag
            netsHigh16_tmp[d + 1] = i;
            if ((netsBFlag[d + 1] == 1) && (d + 1 != a1a2))
                net_high16[d + 1] = ((i - 1) << 1) + netsBFlag[d + 1];
            else
                net_high16[d + 1] = (i << 1) + netsBFlag[d + 1];
        }
        // the last a1a2
        netsHigh16Cnt = a1a2;
    }

    // set the left value to be same with last one.
    for (unsigned int i = netsHigh16Cnt + 1; i < 65536; i++) {
        netsHigh16_tmp[i] = geo_lnm;
        net_high16[i] = geo_lnm * 2;
    }
    // number of 512-bit data
    uint64_t cnt512 = 0;

    // record the last index of newLow buffer
    unsigned int indexLow = 0;

    // the high 32-bit is the address of a3a4
    // the low 32-bit is the row number
    net_high16[0] += (cnt512 << 32);
    uint512 tmp = 0;
    for (unsigned int i = 1; i <= netsHigh16Cnt + 1; i++) {
        unsigned int offsetLow = 0;

        // if 0 < the row number with same a1a2 < 24*16 (16 is burst length???)
        // std::cout << "netsHigh16_tmp[" << i << "]=" << netsHigh16_tmp[i] << ", netsHigh16_tmp[" << i - 1
        //          << "]=" << netsHigh16_tmp[i - 1] << std::endl;
        if (netsHigh16_tmp[i] - netsHigh16_tmp[i - 1] <= Bank2 * TH1 && netsHigh16_tmp[i] != netsHigh16_tmp[i - 1]) {
            // the data width is 512, it could store 24 21-bit data
            // the number of 512-bit data
            unsigned int cntNet = (netsHigh16_tmp[i] - netsHigh16_tmp[i - 1] + Bank2 - 1) / Bank2;

            unsigned int lastIndex = indexLow;

            // the number of 512-bit data
            cnt512 += cntNet;
            // the high 504-bits stores 24 21-bit data and the low 8-bits store the number of data
            for (unsigned int j = netsHigh16_tmp[i - 1]; j < netsHigh16_tmp[i]; j++) {
                if (offsetLow == Bank2) {
                    tmp.range(0, 7) = Bank2;
                    net_low21[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 21 + 20 + 8, offsetLow * 21 + 8) = netsLow21_tmp[j];
                offsetLow++;
            }
            tmp.range(0, 7) = offsetLow;
            net_low21[indexLow++] = tmp;
            tmp = 0;

            // start
            assert(indexLow - lastIndex == cntNet);
            // if the row number with same a1a2 > 24*16
        } else if (netsHigh16_tmp[i] != netsHigh16_tmp[i - 1]) {
            // burst read number
            unsigned int cntNet = (netsHigh16_tmp[i] - netsHigh16_tmp[i - 1] + Bank2 * TH2 - 1) / (Bank2 * TH2);
            // the index number
            // 512-bit store 32 16-bit index
            // the index store the start a3a4 value for each read block
            unsigned int idxCnt = (cntNet + Bank1 - 2) / Bank1;
            cnt512 += idxCnt;

            unsigned int lastIndex = indexLow;
            for (unsigned int j = 1; j < cntNet; j++) {
                if (offsetLow == Bank1) {
                    net_low21[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 16 + 15, offsetLow * 16) = netsLow21_tmp[netsHigh16_tmp[i - 1] + Bank2 * TH2 * j];
                offsetLow++;
            }
            net_low21[indexLow++] = tmp;
            tmp = 0;
            offsetLow = 0;

            assert(indexLow - lastIndex == idxCnt);
            if (indexLow - lastIndex != idxCnt)
                std::cout << "netsHigh16_tmp[" << i - 1 << "]=" << netsHigh16_tmp[i - 1] << ",netsHigh16_tmp[" << i
                          << "]=" << netsHigh16_tmp[i] << ",idxCnt=" << idxCnt << ",lastIndex=" << lastIndex
                          << ",indexLow=" << indexLow << ",cntNet=" << cntNet << ",calcu error\n";

            lastIndex = indexLow;
            // store the a3a4 and mask value
            // calculate the number of 512-bit data
            cntNet = (netsHigh16_tmp[i] - netsHigh16_tmp[i - 1] + Bank2 - 1) / Bank2;
            cnt512 += cntNet;
            for (unsigned int j = netsHigh16_tmp[i - 1]; j < netsHigh16_tmp[i]; j++) {
                if (offsetLow == Bank2) {
                    tmp.range(0, 7) = Bank2;
                    net_low21[indexLow++] = tmp;
                    tmp = 0;
                    offsetLow = 0;
                }
                tmp.range(offsetLow * 21 + 20 + 8, offsetLow * 21 + 8) = netsLow21_tmp[j];
                offsetLow++;
            }
            tmp.range(7, 0) = offsetLow;
            net_low21[indexLow++] = tmp;
            tmp = 0;
            assert(indexLow - lastIndex == cntNet);
        }
        // the address of netLowbuff for a1a2.
        // std::cout << "i=" << i << ", cnt512=" << cnt512 << ", net_high16=" << net_high16[i] << std::endl;
        net_high16[i] += (cnt512 << 32);
    }
    std::cout << "netsLow21 actual use buffer size is " << indexLow << std::endl;

    // the left one use the same value with last one.
    for (unsigned int i = netsHigh16Cnt + 2; i < 65536; i++) {
        net_high16[i] += (cnt512 << 32);
    }
    return 0;
}

uint32_t logAnalyzer::findSliceNum(
    uint16_t* len_buff, uint32_t lnm, uint32_t* slice_lnm, uint16_t* lnm_per_slc, uint32_t* sz_per_slc) {
    uint32_t slc_sz = 0;
    uint32_t slc_nm = 0;

    uint32_t start_lnm = 0;
    uint32_t end_lnm = 0;

    uint32_t tmp_slice_nm = 0;

    for (unsigned int i = 0; i < lnm; ++i) {
        if (len_buff[i] < MSG_SZ) {
            uint32_t tmp_slc_sz = slc_sz + (len_buff[i] + 7) / 8;
            if (tmp_slc_sz > SLICE_MSG_SZ / 8) {
                start_lnm = end_lnm;
                end_lnm = i;

                if (end_lnm - start_lnm > tmp_slice_nm) tmp_slice_nm = end_lnm - start_lnm;

                lnm_per_slc[slc_nm] = end_lnm - start_lnm;
                sz_per_slc[slc_nm] = slc_sz;
                slc_nm++;
                slc_sz = (len_buff[i] + 7) / 8;
            } else if (i == lnm - 1) {
                start_lnm = end_lnm;
                end_lnm = lnm;
                lnm_per_slc[slc_nm] = end_lnm - start_lnm;
                sz_per_slc[slc_nm] = tmp_slc_sz;
                slc_nm++;
                if (end_lnm - start_lnm > tmp_slice_nm) tmp_slice_nm = end_lnm - start_lnm;
            } else {
                slc_sz = tmp_slc_sz;
            }
        } else {
            fprintf(stderr, "ERROR: the length of %dth input message exceed maximum %d\n", i, MSG_SZ);
        }
    }
    *slice_lnm = tmp_slice_nm + 2;
    return slc_nm;
}
ErrCode logAnalyzer::compile(std::string pattern) {
    return cfg.compile(pattern);
}
} // namespace search
} // namespace xf
