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
#ifndef _DB_INTPAIR_SORT_HEADER_
#define _DB_INTPAIR_SORT_HEADER_
#include <stdlib.h>
#include <iostream>
#include <random>
// number of items to be sorted NUM_ITEMS(M)
#define NUM_ITEMS 64
#include <ap_int.h>
#include <string.h>
#include <time.h>
#include <algorithm>
#include <chrono>
#include <future>
#include <iterator>
#include <vector>
#define XCL_BANK(n) (((unsigned int)(n)) | XCL_MEM_TOPOLOGY)

namespace xf {
namespace database {
namespace intpair_sort {
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
        // printf("align_alloc %lu Bytes\n", sz);
        return reinterpret_cast<T*>(ptr);
    }
};
enum ErrCode { SUCCESS = 0, ERROR = 1 };
enum DATASIZE {
    sz_1k = 1024,
    sz_128k = 128 * 1024,
    sz_1m = 1024 * 1024,
    sz_8m = 8 * 1024 * 1024,
    sz_64m = 64 * 1024 * 1024,
    sz_128m = 128 * 1024 * 1024,
    sz_256m = 256 * 1024 * 1024,
    sz_512m = 512 * 1024 * 1024,
    sz_1g = 1024 * 1024 * 1024,
    sz_test = sz_256m,
    sz_test_ch = sz_test / sz_64m
};
class sortAPI {
   private:
    cl_int err;
    cl_context ctx;
    cl_device_id dev_id;
    cl_command_queue cq;
    cl_program prg;

    cl_mem buf_in[2];
    cl_mem buf_k0[2];
    cl_mem buf_k1[2];
    cl_mem buf_k2[2];
    cl_mem buf_k3;
    cl_mem buf_k3_subs[2];
    cl_mem buf_k4;
    ap_uint<64>* user_buf_tmp;
    int merge_num[4] = {8, 8, 8, 16};

    int counter = 0;
    std::vector<std::vector<cl_event> > evs_write;
    std::vector<std::vector<cl_event> > evs_insert;
    std::vector<std::vector<cl_event> > evs_m0;
    std::vector<std::vector<cl_event> > evs_m1;
    std::vector<std::vector<cl_event> > evs_m2;
    std::vector<std::vector<cl_event> > evs_read1;
    std::vector<std::vector<cl_event> > evs_write1;
    std::vector<cl_event> evs_m3;
    std::vector<cl_event> evs_read;

    std::string xclbin_path; // eg. q5kernel_VCU1525_hw.xclbin
    MM mm;
    void setProm(cl_event ev, std::promise<ErrCode>&& prom) {
        clWaitForEvents(1, &ev);
        prom.set_value(SUCCESS);
    }

    std::vector<int> get_merge_loop_num(size_t allsize);

   public:
    void init(std::string xclbin_path, int device_id, bool user_setting = false);

    // Sort function
    // std::future<ErrCode> sort(ap_uint<64>* user_in, ap_uint<64>* user_out, size_t
    // allsize, int order = 1) {
    std::future<ErrCode> sort_server(std::vector<const uint64_t*> user_in,
                                     std::vector<uint64_t*> user_out,
                                     size_t allsize,
                                     int order = 1);
    // Sort function
    std::future<ErrCode> sort(uint64_t* user_in, uint64_t* user_out, size_t allsize, int order = 1);
    void sort_64m_once(ap_uint<64>* user_in, ap_uint<64>* user_out, size_t size, int order = 1, int ppid = 0);
};
} // intpair_sort
} // database
} // xf
#ifdef L3_WITH_SERVER
namespace arrow {
namespace flight {

ARROW_FLIGHT_EXPORT

std::unique_ptr<FlightServerBase> IntSortServerInst(std::string xclbin_path);
}
}
#endif

#endif
