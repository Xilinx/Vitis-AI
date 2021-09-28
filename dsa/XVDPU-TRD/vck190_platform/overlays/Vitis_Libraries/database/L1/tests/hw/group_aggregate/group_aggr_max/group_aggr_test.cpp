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

#include <vector> // std::vector
#include <iostream>
#include <stdlib.h>

#define AP_INT_MAX_W 4096

#include "xf_database/group_aggregate.hpp"
#include "hls_stream.h"

#define GROUP_NUM 50
#define INT_RAND_MIN -100
#define INT_RAND_MAX 100
#define DOUBLE_RAND_MIN -2.5
#define DOUBLE_RAND_MAX -10.8
#define DATA_TYPE int // int||double
#define KEY_TYPE int
#define GAOP xf::database::AOP_MAX
#define FUN max

template <typename T, typename KEY_T>
struct row_msg {
    T data;
    KEY_T key_value;
};

// template <typename T, typename KEY_T>
void hls_db_group_aggr(hls::stream<DATA_TYPE>& in_strm,
                       hls::stream<bool>& e_in_strm,
                       hls::stream<DATA_TYPE>& out_strm,
                       hls::stream<bool>& e_out_strm,
                       hls::stream<KEY_TYPE>& kin_strm,
                       hls::stream<KEY_TYPE>& kout_strm) {
    xf::database::groupAggregate<GAOP>(in_strm, e_in_strm, out_strm, e_out_strm, kin_strm, kout_strm);
}

#ifndef __SYNTHESIS__
// generate a random integer sequence between specified limits a and b (a<b);
int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}
// generate a random double sequence between specified limits a and b (a<b);
double rand_double(double a, double b) {
    return (double)rand() / RAND_MAX * (b - a) + a;
}

int n_per_group[GROUP_NUM];

template <typename T, typename KEY_T>
int generate_test_data(uint64_t len, std::vector<row_msg<T, KEY_T> >& testvector) {
    int totalN = 0;
    for (int i = 0; i < len; i++) {
        int randnum = rand_int(1, 10); // number of data for each group
        n_per_group[i] = randnum;
        //      std::cout <<"data num of this group " << randnum << std::endl;
        for (int j = 0; j < randnum; j++) {
            testvector.push_back({rand_int(INT_RAND_MIN, INT_RAND_MAX), i});
            //            testvector.push_back({rand_double(DOUBLE_RAND_MIN, DOUBLE_RAND_MAX),i});// generate random int
            //            value for data
            totalN++;
        }
    }
    std::cout << " random test data generated! " << std::endl;
    return totalN;
}

namespace ref_group_aggr {
template <typename T, typename KEY_T>
void max(hls::stream<T>& din_strm,
         hls::stream<T>& dout_strm,
         hls::stream<KEY_T>& kin_strm,
         hls::stream<KEY_T>& kout_strm) {
    for (int i = 0; i < GROUP_NUM; i++) {
        T ret = din_strm.read();
        KEY_T rkey = kin_strm.read();
        for (int j = 1; j < n_per_group[i]; j++) {
            ret = std::max(ret, (din_strm.read()));
            rkey = kin_strm.read();
        }
        dout_strm.write(ret);
        kout_strm.write(rkey);
        std::cout << "rkey and ret is " << rkey << ", " << ret << std::endl;
    }
}
template <typename T, typename KEY_T>
void min(hls::stream<T>& din_strm,
         hls::stream<T>& dout_strm,
         hls::stream<KEY_T>& kin_strm,
         hls::stream<KEY_T>& kout_strm) {
    for (int i = 0; i < GROUP_NUM; i++) {
        T ret = din_strm.read();
        KEY_T rkey = kin_strm.read();
        for (int j = 1; j < n_per_group[i]; j++) {
            ret = std::min(ret, (din_strm.read()));
            rkey = kin_strm.read();
        }
        dout_strm.write(ret);
        kout_strm.write(rkey);
        std::cout << "rkey and ret is " << rkey << ", " << ret << std::endl;
    }
}
template <typename T, typename KEY_T>
void sum(hls::stream<T>& din_strm,
         hls::stream<T>& dout_strm,
         hls::stream<KEY_T>& kin_strm,
         hls::stream<KEY_T>& kout_strm) {
    for (int i = 0; i < GROUP_NUM; i++) {
        T ret = 0;
        KEY_T rkey;
        for (int j = 0; j < n_per_group[i]; j++) {
            rkey = kin_strm.read();
            ret += din_strm.read();
        }
        dout_strm.write(ret);
        kout_strm.write(rkey);
        std::cout << "rkey and ret is " << rkey << ", " << ret << std::endl;
    }
}
template <typename T, typename KEY_T>
void count(hls::stream<T>& din_strm,
           hls::stream<uint64_t>& out_strm,
           hls::stream<KEY_T>& kin_strm,
           hls::stream<KEY_T>& kout_strm) {
    for (int i = 0; i < GROUP_NUM; i++) {
        uint64_t ret = 0;
        KEY_T rkey;
        for (int j = 0; j < n_per_group[i]; j++) {
            rkey = kin_strm.read();
            din_strm.read();
            ret++;
        }
        out_strm.write(ret);
        kout_strm.write(rkey);
        std::cout << "rkey and ret is " << rkey << ", " << ret << std::endl;
    }
}
template <typename T, typename KEY_T>
void numNonZeros(hls::stream<T>& din_strm,
                 hls::stream<uint64_t>& out_strm,
                 hls::stream<KEY_T>& kin_strm,
                 hls::stream<KEY_T>& kout_strm) {
    for (int i = 0; i < GROUP_NUM; i++) {
        uint64_t ret = 0;
        KEY_T rkey;
        for (int j = 0; j < n_per_group[i]; j++) {
            rkey = kin_strm.read();
            if (din_strm.read() != 0) {
                ret++;
            }
        }
        out_strm.write(ret);
        kout_strm.write(rkey);
        std::cout << "rkey and ret is " << rkey << ", " << ret << std::endl;
    }
}

} // namespace ref_group_aggr

int main() {
    std::vector<row_msg<DATA_TYPE, KEY_TYPE> > testVector;
    hls::stream<KEY_TYPE> kin_strm("inkey_strm");
    hls::stream<DATA_TYPE> din_strm("din_strm");
    hls::stream<bool> e_in_strm("e_in_strm");
    hls::stream<DATA_TYPE> dout_strm("dout_strm");
    hls::stream<KEY_TYPE> kout_strm("outkey_strm");
    hls::stream<bool> e_out_strm("e_dout_strm");

    hls::stream<DATA_TYPE> ref_din_strm("ref_din_strm");
    hls::stream<KEY_TYPE> ref_kin_strm("ref_kin_strm");
    hls::stream<DATA_TYPE> ref_dout_strm("ref_dout_strm");
    hls::stream<KEY_TYPE> ref_kout_strm("ref_kout_strm");

    // generate test data
    int totalN = generate_test_data<DATA_TYPE, KEY_TYPE>(GROUP_NUM, testVector);

    int nerror = 0;

    // prepare input data
    std::cout << "test data: (indexing key, data)" << std::endl;
    for (std::string::size_type i = 0; i < totalN; i++) {
        // print vector value
        std::cout << testVector[i].key_value << " ," << testVector[i].data << std::endl;
        // wirte data to stream
        kin_strm.write(testVector[i].key_value);
        din_strm.write(testVector[i].data);
        e_in_strm.write(0);
        ref_kin_strm.write(testVector[i].key_value);
        ref_din_strm.write(testVector[i].data);
    }
    e_in_strm.write(1);
    std::cout << std::endl;

    // call group aggregate function
    hls_db_group_aggr(din_strm, e_in_strm, dout_strm, e_out_strm, kin_strm, kout_strm);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < GROUP_NUM; i++) {
        bool e = e_out_strm.read();
        if (e) {
            nerror++;
        }
    }
    // read out the last flag that e should =1
    bool e = e_out_strm.read();
    if (!e) {
        nerror++;
    }

    //===== run reference function to get the result ======
    ref_group_aggr::FUN<DATA_TYPE, KEY_TYPE>(ref_din_strm, ref_dout_strm, ref_kin_strm, ref_kout_strm);
    //===== compare the ref fun and hls func result
    for (int i = 0; i < GROUP_NUM; i++) {
        //  bool cmp_dout = (dout_cnt_strm.read() == ref_dout_cnt_strm.read()) ? 1 : 0;
        bool cmp_dout = (dout_strm.read() == ref_dout_strm.read()) ? 1 : 0;
        if (!cmp_dout) {
            nerror++;
            std::cout << "the agg data is incorrect" << std::endl;
        }
        // compare the key
        bool cmp_key = (kout_strm.read() == ref_kout_strm.read()) ? 1 : 0;
        if (!cmp_key) {
            nerror++;
            std::cout << "the agg key is incorrect" << std::endl;
        }
    }

    if (nerror) {
        std::cout << "\nFAIL: nerror= " << nerror << " errors found.\n";
    } else {
        std::cout << "\nPASS: no error found.\n";
    }
    return nerror;
}

#endif
