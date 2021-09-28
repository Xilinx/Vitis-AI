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
#include "xf_data_analytics/text/regex_engine.hpp"

extern "C" {
#include "oniguruma.h"
}

namespace xf {
namespace data_analytics {
namespace text {
namespace re {

// constructor
// load binary and program
RegexEngine::RegexEngine(const std::string& xclbin,
                         const int dev_index,
                         const int instr_depth,
                         const int char_class_num,
                         const int capture_grp_num,
                         const int msg_size,
                         const int max_slice_size,
                         const int max_slice_num)
    : reCfg(instr_depth, char_class_num, capture_grp_num),
      kInstrDepth(instr_depth),
      kCharClassNum(char_class_num),
      kCaptureGrpNum(capture_grp_num),
      kMsgSize(msg_size),
      kMaxSliceSize(max_slice_size),
      kMaxSliceNum(max_slice_num) {
    xclbin_path = xclbin;
    err = details::init_hardware(&ctx, &dev_id, &cq, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE,
                                 dev_index);
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to init OpenCL with device index %d\n", dev_index);
        exit(1);
    }

    err = details::load_binary(&prg, ctx, dev_id, xclbin_path.c_str());
    if (err != CL_SUCCESS) {
        fprintf(stderr, "ERROR: fail to program PL\n");
        exit(1);
    }
}

// de-constructor
RegexEngine::~RegexEngine() {
    err = clReleaseProgram(prg);
    if (err != CL_SUCCESS) {
        std::cout << "deconstructor" << std::endl;
        exit(1);
    }

    clReleaseCommandQueue(cq);
    clReleaseContext(ctx);
}

struct queue_struct {
    // the sec index
    int sec;
    // dependency event num
    int num_event_wait_list;
    // dependency events
    cl_event* event_wait_list;
    // user event to trace current operation
    cl_event* event;
    // line number of message for current slice
    uint32_t lnm;
    // position of input for current slice
    uint32_t pos;
    // destination of message pointer
    uint64_t* msg_ptr_dst;
    // destination of length pointer
    uint16_t* len_ptr_dst;
    // source of result pointer
    uint32_t* ptr_src;
};

class threading_pool {
   private:
    const int kMsgSize;
    const int kMaxSliceSize;
    const int kMaxSliceNum;

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
    // pointer to msg buffer
    const uint64_t* msg_ptr;
    // pointer to len buffer
    uint16_t* len_ptr;
    // pointer to offset buffer
    uint32_t* offt_ptr;
    // pointer to output buffer
    uint32_t* out_ptr;

    uint32_t total_lnm;
    uint32_t cpgp_nm;

    // constructor
    threading_pool(const int msg_size, const int max_slice_size, const int max_slice_num)
        : kMsgSize(msg_size), kMaxSliceSize(max_slice_size), kMaxSliceNum(max_slice_num) {}

    // for post process
    regex_t* reg;
    OnigRegion* region;

    // de-constructor
    ~threading_pool() {
        onig_region_free(region, 1);
        onig_free(reg);
        onig_end();
    }
    // initialize the threads
    void init(const uint64_t* msg_ptr,
              uint32_t* offt_ptr,
              uint16_t* len_ptr,
              uint32_t* out_ptr,
              uint32_t total_lnm,
              uint32_t cpgp_nm,
              std::string pattern) {
        this->msg_ptr = msg_ptr;
        this->len_ptr = len_ptr;
        this->offt_ptr = offt_ptr;
        this->out_ptr = out_ptr;
        this->total_lnm = total_lnm;
        this->cpgp_nm = cpgp_nm;
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
        // initialize oniguruma regex
        region = onig_region_new();
        OnigEncoding use_encs[1];
        use_encs[0] = ONIG_ENCODING_ASCII;
        onig_initialize(use_encs, sizeof(use_encs) / sizeof(use_encs[0]));
        UChar* pattern_c = (UChar*)pattern.c_str();
        OnigErrorInfo einfo;
        int r = onig_new(&reg, pattern_c, pattern_c + strlen((char*)pattern_c), ONIG_OPTION_DEFAULT,
                         ONIG_ENCODING_ASCII, ONIG_SYNTAX_DEFAULT, &einfo);
    }
    // input log file memcpy in thread
    void func_mcpy_in_ping_t() {
        while (q0_ping_run) {
            while (!q0_ping.empty()) {
                queue_struct q = q0_ping.front();
                clWaitForEvents(q.num_event_wait_list, q.event_wait_list);
#ifdef RE_PERF_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                size_t slice_sz = 0;
                for (int i = 0; i < q.lnm; ++i) {
                    size_t tmp = slice_sz;
                    if (len_ptr[q.pos + i] < kMsgSize) {
                        tmp += (len_ptr[q.pos + i] + 7) / 8;
                    }
                    // reach the end of file or stop when reach the limitation of slice size
                    if (q.pos + i >= total_lnm || tmp > kMaxSliceSize / 8) {
                        break;
                    } else {
                        // if the message length exceed the maxinum, set len = 0;
                        if (len_ptr[q.pos + i] > kMsgSize) {
                            q.len_ptr_dst[i + 2] = 0;
                        } else {
                            q.len_ptr_dst[i + 2] = len_ptr[q.pos + i];
                            memcpy(q.msg_ptr_dst + slice_sz + 1, msg_ptr + offt_ptr[q.pos + i], len_ptr[q.pos + i]);
                        }
                        slice_sz = tmp;
                    }
                }
                // update
                // printf("sec = %d, lnm = %d, slice_sz = %d, start_pos = %d\n", q.sec, q.lnm, slice_sz, q.pos);
                q.msg_ptr_dst[0] = (uint64_t)(slice_sz + 1);
                q.len_ptr_dst[0] = (q.lnm + 2) / 65536;
                q.len_ptr_dst[1] = (q.lnm + 2) % 65536;
                // set the status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request
                q0_ping.pop();
#ifdef RE_PERF_PROFILE
                gettimeofday(&tv_end, 0);
                double sz_in_byte = (double)(slice_sz * sizeof(msg_ptr[0]) + q.lnm * sizeof(len_ptr[0])) / 1024 / 1024;
                double tvtime = details::tvdiff(tv_start, tv_end);
                std::cout << "Input log sec: " << q.sec << ", memcpy in, size: " << sz_in_byte
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
#ifdef RE_PERF_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                size_t slice_sz = 0;
                for (int i = 0; i < q.lnm; ++i) {
                    size_t tmp = slice_sz;
                    if (len_ptr[q.pos + i] < kMsgSize) {
                        tmp += (len_ptr[q.pos + i] + 7) / 8;
                    }
                    // reach the end of file or stop when reach the limitation of slice size
                    if (q.pos + i >= total_lnm || tmp > kMaxSliceSize / 8) {
                        break;
                    } else {
                        // if the message length exceed the maxinum, set len = 0;
                        if (len_ptr[q.pos + i] > kMsgSize) {
                            q.len_ptr_dst[i + 2] = 0;
                        } else {
                            q.len_ptr_dst[i + 2] = len_ptr[q.pos + i];
                            memcpy(q.msg_ptr_dst + slice_sz + 1, msg_ptr + offt_ptr[q.pos + i], len_ptr[q.pos + i]);
                        }
                        slice_sz = tmp;
                    }
                }
                // update
                // printf("sec = %d, lnm = %d, slice_sz = %d, start_pos = %d\n", q.sec, q.lnm, slice_sz, q.pos);
                q.msg_ptr_dst[0] = (uint64_t)(slice_sz + 1);
                q.len_ptr_dst[0] = (q.lnm + 2) / 65536;
                q.len_ptr_dst[1] = (q.lnm + 2) % 65536;
                // set the status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request
                q0_pong.pop();
#ifdef RE_PERF_PROFILE
                gettimeofday(&tv_end, 0);
                double sz_in_byte = (double)(slice_sz * sizeof(msg_ptr[0]) + q.lnm * sizeof(len_ptr[0])) / 1024 / 1024;
                double tvtime = details::tvdiff(tv_start, tv_end);
                std::cout << "Input log sec: " << q.sec << ", memcpy in, size: " << sz_in_byte
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
#ifdef RE_PERF_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                unsigned char* max_str = (unsigned char*)malloc(65536);
                for (unsigned int i = 0; i < q.lnm; ++i) {
                    uint8_t result = q.ptr_src[i * (cpgp_nm + 1) + 1];
                    // step 1: stack overflow or large message
                    if (result == 2 || result == 3) {
                        // step 2: find the position and length message
                        size_t msg_pos = offt_ptr[q.pos + i];
                        const uint64_t* msg = &msg_ptr[msg_pos];
                        uint16_t msg_len = len_ptr[q.pos + i];
                        memcpy(max_str, msg, msg_len);
                        max_str[msg_len] = '\0';
                        UChar* str = (UChar*)max_str;
                        unsigned char* end = str + strlen((char*)str);
                        unsigned char* start = str;
                        unsigned char* range = end;
                        int r = onig_search(reg, str, end, start, range, region, ONIG_OPTION_NONE);
                        // printf("[DEBUG], post_proc: %d, r: %d\n", *(q.start_pos) + i, r);
                        // step 4: insert the result back to out_ptr
                        if (r == 0) {
                            q.ptr_src[i * (cpgp_nm + 1) + 1] = 1;
                            for (int j = 0; j < cpgp_nm; ++j) {
                                uint32_t out = region->end[j] * 65536 + region->beg[j];
                                q.ptr_src[i * (cpgp_nm + 1) + 2 + j] = out;
                            }
                        } else if (r == ONIG_MISMATCH) {
                            q.ptr_src[i * (cpgp_nm + 1) + 1] = 0;
                        }
                    }
                }

                size_t sz = q.lnm * (cpgp_nm + 1) * sizeof(q.ptr_src[0]);
                // copy result to output buffer
                memcpy(out_ptr + q.pos * (cpgp_nm + 1), q.ptr_src + 1, sz);
                // set status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request from queue
                q1_ping.pop();
#ifdef RE_PERF_PROFILE
                gettimeofday(&tv_end, 0);
                double tvtime = details::tvdiff(tv_start, tv_end);
                double memcpy_size = (double)sz / 1024 / 1024;
                std::cout << "Output sec: " << q.sec << ", memcpy out, size: " << memcpy_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throghput: " << memcpy_size / 1024 / (double)tvtime / 1000000 << " GB/s"
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
#ifdef RE_PERF_PROFILE
                timeval tv_start, tv_end;
                gettimeofday(&tv_start, 0);
#endif
                unsigned char* max_str = (unsigned char*)malloc(65536);
                for (unsigned int i = 0; i < q.lnm; ++i) {
                    uint8_t result = q.ptr_src[i * (cpgp_nm + 1) + 1];
                    // step 1: stack overflow or large message
                    if (result == 2 || result == 3) {
                        // step 2: find the position and length message
                        size_t msg_pos = offt_ptr[q.pos + i];
                        const uint64_t* msg = &msg_ptr[msg_pos];
                        uint16_t msg_len = len_ptr[q.pos + i];
                        memcpy(max_str, msg, msg_len);
                        max_str[msg_len] = '\0';
                        UChar* str = (UChar*)max_str;
                        unsigned char* end = str + strlen((char*)str);
                        unsigned char* start = str;
                        unsigned char* range = end;
                        int r = onig_search(reg, str, end, start, range, region, ONIG_OPTION_NONE);
                        // printf("[DEBUG], post_proc: %d, r: %d\n", *(q.start_pos) + i, r);
                        // step 4: insert the result back to out_ptr
                        if (r == 0) {
                            q.ptr_src[i * (cpgp_nm + 1) + 1] = 1;
                            for (int j = 0; j < cpgp_nm; ++j) {
                                uint32_t out = region->end[j] * 65536 + region->beg[j];
                                q.ptr_src[i * (cpgp_nm + 1) + 2 + j] = out;
                            }
                        } else if (r == ONIG_MISMATCH) {
                            q.ptr_src[i * (cpgp_nm + 1) + 1] = 0;
                        }
                    }
                }

                size_t sz = q.lnm * (cpgp_nm + 1) * sizeof(q.ptr_src[0]);
                // copy result to output buffer
                memcpy(out_ptr + (size_t)(q.pos * (cpgp_nm + 1)), q.ptr_src + 1, sz);
                // set status of event
                clSetUserEventStatus(q.event[0], CL_COMPLETE);
                // remove the request from queue
                q1_pong.pop();
#ifdef RE_PERF_PROFILE
                gettimeofday(&tv_end, 0);
                double tvtime = details::tvdiff(tv_start, tv_end);
                double memcpy_size = (double)sz / 1024 / 1024;
                std::cout << "Output sec: " << q.sec << ", memcpy out, size: " << memcpy_size
                          << " MB, time: " << tvtime / 1000
                          << " ms, throghput: " << memcpy_size / 1024 / (double)tvtime / 1000000 << " GB/s"
                          << std::endl;
#endif
            }
        }
    }
}; // end of class threading_pool
ErrCode RegexEngine::match_all(const uint64_t* msg_buff,
                               uint32_t* offt_buff,
                               uint16_t* len_buff,
                               uint32_t* out_buff,
                               const uint64_t* re_cfg,
                               uint32_t total_lnm) {
    uint32_t cpgp_nm = reCfg.getCpgpNm();
    details::MM mm;

    // calculate the section number
    timeval tv_start, tv_end;
    gettimeofday(&tv_start, 0);
    uint32_t max_slice_lnm = 0;

    uint16_t* lnm_per_sec = mm.aligned_alloc<uint16_t>(kMaxSliceNum);
    uint32_t* pos_per_sec = mm.aligned_alloc<uint32_t>(kMaxSliceNum);

    uint32_t sec_num = findSecNum(len_buff, total_lnm, &max_slice_lnm, lnm_per_sec, pos_per_sec);

    gettimeofday(&tv_end, 0);
    double tvtime = details::tvdiff(tv_start, tv_end);
    fprintf(stdout, "The log file is partition into %d section with max_slice_lnm %d and  takes %f ms.\n", sec_num,
            max_slice_lnm, tvtime / 1000);

    // start threading pool threads
    threading_pool pool(kMsgSize, kMaxSliceSize, kMaxSliceNum);

    pool.init(msg_buff, offt_buff, len_buff, out_buff, total_lnm, cpgp_nm, reCfg.pattern);
    // Aussuming the input log is very large,  it is divided into several sections to improve throughput by overlap data
    // transfer and kernel exectuion
    // define memcpy in user events
    std::vector<std::vector<cl_event> > evt_memcpy_in(sec_num);
    for (unsigned int i = 0; i < sec_num; ++i) {
        evt_memcpy_in[i].resize(1);
        evt_memcpy_in[i][0] = clCreateUserEvent(ctx, &err);
    }
    // define memcpy out user events
    std::vector<std::vector<cl_event> > evt_memcpy_out(sec_num);
    for (unsigned int i = 0; i < sec_num; ++i) {
        evt_memcpy_out[i].resize(1);
        evt_memcpy_out[i][0] = clCreateUserEvent(ctx, &err);
    }
    // get kernel number
    std::string krnl_name = kernel_name;
    cl_uint cu_num;
    {
        cl_kernel k = clCreateKernel(prg, krnl_name.c_str(), &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create kernel.\n");
            return DEV_ERR;
        }
        clGetKernelInfo(k, CL_KERNEL_COMPUTE_UNIT_COUNT, sizeof(cu_num), &cu_num, nullptr);
        std::cout << "DEBUG: " << krnl_name << " has " << cu_num << " CU(s)" << std::endl;
        clReleaseKernel(k);
    }
    // host side pinned buffers
    std::vector<std::vector<uint64_t*> > msg_in_slice(2);
    std::vector<std::vector<uint16_t*> > len_in_slice(2);
    std::vector<std::vector<uint32_t*> > out_slice(2);
    for (int k = 0; k < 2; ++k) {
        msg_in_slice[k].resize(cu_num);
        len_in_slice[k].resize(cu_num);
        out_slice[k].resize(cu_num);
        for (cl_uint c = 0; c < cu_num; ++c) {
            msg_in_slice[k][c] = mm.aligned_alloc<uint64_t>((kMaxSliceSize / 8) + 1);
            len_in_slice[k][c] = mm.aligned_alloc<uint16_t>(max_slice_lnm + 2);
            out_slice[k][c] = mm.aligned_alloc<uint32_t>(max_slice_lnm * (cpgp_nm + 1) + 1);
        }
    }
    // create kernel
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    std::vector<std::vector<cl_kernel> > krnls(2);
    for (int i = 0; i < 2; ++i) {
        krnls[i].resize(cu_num);
        for (cl_uint c = 0; c < cu_num; ++c) {
            std::string krnl_full_name = krnl_name + ":{" + krnl_name + "_" + std::to_string(c + 1) + "}";
            krnls[i][c] = clCreateKernel(prg, krnl_full_name.c_str(), &err);
            logger.logCreateKernel(err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create kernel[%d].\n", c + 1);
                return DEV_ERR;
            }
        }
    }
    // create CL ping-pong buffer
    std::vector<cl_mem_ext_ptr_t> mext_cfg(cu_num);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_msg(2);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_len(2);
    std::vector<std::vector<cl_mem_ext_ptr_t> > mext_out(2);
    for (cl_uint c = 0; c < cu_num; ++c) {
        mext_cfg[c] = {0, (void*)re_cfg, krnls[0][c]};
    }
    for (int k = 0; k < 2; ++k) {
        mext_msg[k].resize(cu_num);
        mext_len[k].resize(cu_num);
        mext_out[k].resize(cu_num);
        for (cl_uint c = 0; c < cu_num; ++c) {
            mext_msg[k][c] = {1, msg_in_slice[k][c], krnls[k][c]};
            mext_len[k][c] = {2, len_in_slice[k][c], krnls[k][c]};
            mext_out[k][c] = {3, out_slice[k][c], krnls[k][c]};
        }
    }
    // device buffer
    std::vector<cl_mem> inCfgBuff(cu_num);
    std::vector<std::vector<cl_mem> > inMsgBuff(2);
    std::vector<std::vector<cl_mem> > inLenBuff(2);
    std::vector<std::vector<cl_mem> > outBuff(2);
    for (int k = 0; k < 2; ++k) {
        inMsgBuff[k].resize(cu_num);
        inLenBuff[k].resize(cu_num);
        outBuff[k].resize(cu_num);
        for (cl_uint c = 0; c < cu_num; ++c) {
            inMsgBuff[k][c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             sizeof(uint64_t) * ((kMaxSliceSize / 8) + 1), &mext_msg[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create msg buffer\n");
                return MEM_ERR;
            }
        }
        for (cl_uint c = 0; c < cu_num; ++c) {
            inLenBuff[k][c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                             sizeof(uint16_t) * (max_slice_lnm + 2), &mext_len[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create len buffer\n");
                return MEM_ERR;
            }
        }
        for (cl_uint c = 0; c < cu_num; ++c) {
            outBuff[k][c] =
                clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE,
                               sizeof(uint32_t) * ((cpgp_nm + 1) * max_slice_lnm + 1), &mext_out[k][c], &err);
            if (err != CL_SUCCESS) {
                fprintf(stderr, "ERROR: failed to create out buffer\n");
                return MEM_ERR;
            }
        }
    }
    for (cl_uint c = 0; c < cu_num; c++) {
        inCfgBuff[c] = clCreateBuffer(ctx, CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY,
                                      sizeof(uint64_t) * (kInstrDepth + kCharClassNum * 4 + 2), &mext_cfg[c], &err);
        if (err != CL_SUCCESS) {
            fprintf(stderr, "ERROR: failed to create cfg buffer\n");
            return MEM_ERR;
        }
    }
    // make sure all buffers are resident on device
    std::vector<cl_mem> tot_in_bufs[2];
    for (int k = 0; k < 2; k++) {
        for (cl_uint c = 0; c < cu_num; ++c) {
            tot_in_bufs[k].push_back(inMsgBuff[k][c]);
            tot_in_bufs[k].push_back(inLenBuff[k][c]);
        }
    }
    std::vector<cl_mem> tot_out_bufs[2];
    for (int k = 0; k < 2; k++) {
        for (cl_uint c = 0; c < cu_num; ++c) {
            tot_out_bufs[k].push_back(outBuff[k][c]);
        }
    }
    clEnqueueMigrateMemObjects(cq, tot_in_bufs[0].size(), tot_in_bufs[0].data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_in_bufs[1].size(), tot_in_bufs[1].data(),
                               CL_MIGRATE_MEM_OBJECT_CONTENT_UNDEFINED, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_out_bufs[0].size(), tot_out_bufs[0].data(), 0, 0, nullptr, nullptr);
    clEnqueueMigrateMemObjects(cq, tot_out_bufs[1].size(), tot_out_bufs[1].data(), 0, 0, nullptr, nullptr);
    // set kernel's arguements
    for (cl_uint c = 0; c < cu_num; ++c) {
        for (int k = 0; k < 2; ++k) {
            clSetKernelArg(krnls[k][c], 0, sizeof(cl_mem), &inCfgBuff[c]);
            clSetKernelArg(krnls[k][c], 1, sizeof(cl_mem), &inMsgBuff[k][c]);
            clSetKernelArg(krnls[k][c], 2, sizeof(cl_mem), &inLenBuff[k][c]);
            clSetKernelArg(krnls[k][c], 3, sizeof(cl_mem), &outBuff[k][c]);
        }
    }
    queue_struct mcpy_in_q[sec_num];
    queue_struct prc_q[sec_num];
    queue_struct mcpy_out_q[sec_num];
    std::vector<std::vector<cl_event> > evt_h2d_vec(sec_num);
    std::vector<std::vector<cl_event> > evt_h2d(sec_num);
    for (unsigned int sec = 0; sec < sec_num; sec++) {
        if (sec >= cu_num * 2)
            evt_h2d_vec[sec].resize(2);
        else
            evt_h2d_vec[sec].resize(1);
        evt_h2d[sec].resize(1);
    }
    std::vector<std::vector<cl_event> > evt_krnl_vec(sec_num);
    std::vector<std::vector<cl_event> > evt_krnl(sec_num);
    for (unsigned int sec = 0; sec < sec_num; sec++) {
        if (sec >= cu_num * 2)
            evt_krnl_vec[sec].resize(3);
        else if (sec >= cu_num)
            evt_krnl_vec[sec].resize(2);
        else
            evt_krnl_vec[sec].resize(1);
        evt_krnl[sec].resize(1);
    }

    std::vector<std::vector<cl_event> > evt_d2h_vec(sec_num);
    std::vector<std::vector<cl_event> > evt_d2h(sec_num);
    for (unsigned int sec = 0; sec < sec_num; sec++) {
        if (sec >= cu_num * 2)
            evt_d2h_vec[sec].resize(2);
        else
            evt_d2h_vec[sec].resize(1);
        evt_d2h[sec].resize(1);
    }

    timeval re_start, re_end;
    // cfg buffer migiration, only do this once
    std::vector<cl_mem> in_cfg_vec;
    for (cl_uint c = 0; c < cu_num; ++c) {
        in_cfg_vec.push_back(inCfgBuff[c]);
    }
    clEnqueueMigrateMemObjects(cq, in_cfg_vec.size(), in_cfg_vec.data(), 0, 0, nullptr, nullptr);
    clFinish(cq);
    gettimeofday(&re_start, 0);
    for (unsigned int sec = 0; sec < sec_num; ++sec) {
        int kid = (sec / cu_num) % 2;
        int cu_id = sec % cu_num;
        int mcpy_kid = sec % 2;
        // 1) memcpy_in
        mcpy_in_q[sec].sec = sec;
        mcpy_in_q[sec].msg_ptr_dst = msg_in_slice[kid][cu_id];
        mcpy_in_q[sec].len_ptr_dst = len_in_slice[kid][cu_id];
        mcpy_in_q[sec].lnm = lnm_per_sec[sec];
        mcpy_in_q[sec].pos = pos_per_sec[sec];
        if (sec >= cu_num * 2) {
            mcpy_in_q[sec].num_event_wait_list = evt_memcpy_out[sec - cu_num * 2].size();
            mcpy_in_q[sec].event_wait_list = evt_memcpy_out[sec - cu_num * 2].data();
        } else {
            mcpy_in_q[sec].num_event_wait_list = 0;
            mcpy_in_q[sec].event_wait_list = nullptr;
        }
        mcpy_in_q[sec].event = &evt_memcpy_in[sec][0];
        if (mcpy_kid == 0)
            pool.q0_ping.push(mcpy_in_q[sec]);
        else
            pool.q0_pong.push(mcpy_in_q[sec]);

        // clWaitForEvents(evt_memcpy_in[sec].size(), evt_memcpy_in[sec].data());
        // 2) H2D Migrate
        std::vector<cl_mem> in_vec;
        in_vec.push_back(inMsgBuff[kid][cu_id]);
        in_vec.push_back(inLenBuff[kid][cu_id]);

        evt_h2d_vec[sec][0] = evt_memcpy_in[sec][0];
        if (sec >= cu_num * 2) {
            evt_h2d_vec[sec][1] = evt_krnl[sec - 2 * cu_num][0];
        }
        clEnqueueMigrateMemObjects(cq, in_vec.size(), in_vec.data(), 0, evt_h2d_vec[sec].size(),
                                   evt_h2d_vec[sec].data(), &evt_h2d[sec][0]);
        // clFinish(cq);
        // printf("H2D transfer done\n");
        // 3) kernel launch
        evt_krnl_vec[sec][0] = evt_h2d[sec][0];
        if (sec >= cu_num) {
            evt_krnl_vec[sec][1] = evt_krnl[sec - cu_num][0];
        }
        if (sec >= 2 * cu_num) {
            evt_krnl_vec[sec][2] = evt_d2h[sec - 2 * cu_num][0];
        }
        clEnqueueTask(cq, krnls[kid][cu_id], evt_krnl_vec[sec].size(), evt_krnl_vec[sec].data(), &evt_krnl[sec][0]);
        // clFinish(cq);
        // printf("Kernel execution done\n");
        // 4) d2h, transfer partiion result back
        evt_d2h_vec[sec][0] = evt_krnl[sec][0];
        if (sec >= 2 * cu_num) {
            evt_d2h_vec[sec][1] = evt_memcpy_out[sec - 2 * cu_num][0];
        }
        std::vector<cl_mem> out_vec;
        out_vec.push_back(outBuff[kid][cu_id]);
        clEnqueueMigrateMemObjects(cq, out_vec.size(), out_vec.data(), CL_MIGRATE_MEM_OBJECT_HOST,
                                   evt_d2h_vec[sec].size(), evt_d2h_vec[sec].data(), &evt_d2h[sec][0]);
        // clFinish(cq);

        // 6)memcpy out and post-process
        mcpy_out_q[sec].sec = sec;
        mcpy_out_q[sec].event = &evt_memcpy_out[sec][0];
        mcpy_out_q[sec].ptr_src = out_slice[kid][cu_id];
        mcpy_out_q[sec].num_event_wait_list = evt_d2h[sec].size();
        mcpy_out_q[sec].event_wait_list = evt_d2h[sec].data();
        mcpy_out_q[sec].lnm = lnm_per_sec[sec];
        mcpy_out_q[sec].pos = pos_per_sec[sec];
        if (mcpy_kid == 0)
            pool.q1_ping.push(mcpy_out_q[sec]);
        else
            pool.q1_pong.push(mcpy_out_q[sec]);
        // clWaitForEvents(evt_memcpy_out[sec].size(), evt_memcpy_out[sec].data());
    }
    clWaitForEvents(evt_memcpy_out[sec_num - 1].size(), evt_memcpy_out[sec_num - 1].data());
    if (sec_num > 1) clWaitForEvents(evt_memcpy_out[sec_num - 2].size(), evt_memcpy_out[sec_num - 2].data());
    ;
    // clWaitForEvents(evt_memcpy_out[0].size(), evt_memcpy_out[0].data());
    // for(cl_uint c = 0; c < cu_num; ++c) {
    //    if(sec_num > c) clWaitForEvents(evt_memcpy_out[sec_num - c - 1].size(), evt_memcpy_out[sec_num-c-1].data());
    //    if(sec_num > cu_num && sec_num - cu_num > c) clWaitForEvents(evt_memcpy_out[sec_num - cu_num - c - 1].size(),
    //    evt_memcpy_out[sec_num - cu_num - c - 1].data());
    // }
    // stop all the threads
    pool.q0_ping_run = 0;
    pool.q1_ping_run = 0;
    pool.q0_pong_run = 0;
    pool.q1_pong_run = 0;
    gettimeofday(&re_end, 0);
#ifdef RE_PERF_PROFILE
    cl_ulong start, end;
    long evt_ns;
    for (int sec = 0; sec < sec_num; ++sec) {
        double input_memcpy_size = (double)(kMaxSliceSize / 8 + max_slice_lnm * 2) / 1024 / 1024;
        clGetEventProfilingInfo(evt_h2d[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_h2d[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "sec: " << sec << ", h2d, size: " << input_memcpy_size << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_memcpy_size / 1024 / ((double)evt_ns / 1000000) << " GB/s"
                  << std::endl;

        double input_msg_size = (double)kMaxSliceSize / 1024 / 1024;
        clGetEventProfilingInfo(evt_krnl[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_krnl[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        std::cout << "sec: " << sec << ", krnl, size: " << input_msg_size << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << input_msg_size / 1024 / ((double)evt_ns / 1000000) << " GB/s" << std::endl;

        clGetEventProfilingInfo(evt_d2h[sec][0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(evt_d2h[sec][0], CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        evt_ns = end - start;
        double output_memcpy_size = (double)(max_slice_lnm * (1 + cpgp_nm) * 4) / 1024 / 1024;
        std::cout << "sec: " << sec << ", d2h, size: " << output_memcpy_size
                  << " MB, time: " << double(evt_ns) / 1000000
                  << " ms, throughput: " << output_memcpy_size / 1024 / ((double)evt_ns / 1000000) << " GB/s"
                  << std::endl;
    }
#endif

    double re_tvtime = details::tvdiff(re_start, re_end);
    double total_log_size = (double)offt_buff[total_lnm - 1] * 8 / 1024 / 1024;
    std::cout << "regex pipelined, time: " << (double)re_tvtime / 1000 << " ms, size: " << total_log_size
              << " MB, throughput: " << total_log_size / 1024 / ((double)re_tvtime / 1000000) << " GB/s" << std::endl;
    std::cout
        << "-----------------------------Finished regex pipelined test----------------------------------------------"
        << std::endl
        << std::endl;
    return SUCCESS;
}
ErrCode RegexEngine::compile(std::string pattern) {
    return reCfg.compile(pattern);
}
ErrCode RegexEngine::match(
    uint32_t total_lnm, const uint64_t* msg_buff, uint32_t* offt_buff, uint16_t* len_buff, uint32_t* out_buff) {
    const uint64_t* cfg_buff = reCfg.getConfigBits();
    // match
    return match_all(msg_buff, offt_buff, len_buff, out_buff, cfg_buff, total_lnm);
}
uint32_t RegexEngine::getCpgpNm() const {
    return reCfg.getCpgpNm();
}
uint32_t RegexEngine::findSecNum(
    uint16_t* len_buff, uint32_t lnm, uint32_t* slice_lnm, uint16_t* lnm_per_sec, uint32_t* pos_per_sec) {
    uint32_t sec_sz = 0;
    uint32_t sec_nm = 0;
    uint32_t start_lnm = 0;
    uint32_t end_lnm = 0;
    uint32_t tmp_slice_nm = 0;
    for (unsigned int i = 0; i < lnm; ++i) {
        if (len_buff[i] < kMsgSize) {
            sec_sz += (len_buff[i] + 7) / 8;
            if (sec_sz > kMaxSliceSize / 8) {
                start_lnm = end_lnm;
                end_lnm = i;
                if (end_lnm - start_lnm > tmp_slice_nm) tmp_slice_nm = end_lnm - start_lnm;
                lnm_per_sec[sec_nm] = end_lnm - start_lnm;
                pos_per_sec[sec_nm] = start_lnm;
                sec_nm++;
                sec_sz = (len_buff[i] + 7) / 8;
            } else if (i == lnm - 1) {
                start_lnm = end_lnm;
                end_lnm = lnm;
                lnm_per_sec[sec_nm] = end_lnm - start_lnm;
                pos_per_sec[sec_nm] = start_lnm;
                sec_nm++;
                if (end_lnm - start_lnm > tmp_slice_nm) tmp_slice_nm = end_lnm - start_lnm;
            }
        }
    }
    *slice_lnm = tmp_slice_nm + 2;
    return sec_nm;
}

} // namesapce re
} // namespace text
} // namespace data_analytics
} // namespace xf
