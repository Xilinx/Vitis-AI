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

#define AP_INT_MAX_W (4096)

#include "xf_database/direct_group_aggregate.hpp"
#include "hls_stream.h"

#define TEST_PROGRAMMABLE_AGGR 1
#define DIRECTW_TEST (10)
#define CNT_AGG_TEST (1 << DIRECTW_TEST)
#define DATINW_TEST (32)
#define DATOUTW_TEST (64)
#define direct_NUM (200)
#define INT_RAND_MIN (-100)
#define INT_RAND_MAX (100)
#define KEY_RAND_MIN (-100)
#define KEY_RAND_MAX (100)

/*
 * select the active internal of top function
 *   AOP_MAX      when TESTCASE == 1
 *   AOP_MIN      when TESTCASE == 2
 *   AOP_SUM      when TESTCASE == 3
 *   AOP_COUNT    when TESTCASE == 4
 *   AOP_MEAN     when TESTCASE == 8
 *   AOP_VARIANCE when TESTCASE == 9
 *   AOP_NORML1   when TESTCASE == 10
 *   AOP_NORML2   when TESTCASE == 11
 */
#define TESTCASE (4)
#if (TESTCASE == 1)
#define DAOP xf::database::AOP_MAX
#define FUN max
#elif (TESTCASE == 2)
#define DAOP xf::database::AOP_MIN
#define FUN min
#elif (TESTCASE == 3)
#define DAOP xf::database::AOP_SUM
#define FUN sum
#elif (TESTCASE == 4)
#define DAOP xf::database::AOP_COUNT
#define FUN countone
#elif (TESTCASE == 8)
#define DAOP xf::database::AOP_MEAN
#define FUN avg
#elif (TESTCASE == 9)
#define DAOP xf::database::AOP_VARIANCE
#define FUN variance
#elif (TESTCASE == 10)
#define DAOP xf::database::AOP_NORML1
#define FUN norml1
#elif (TESTCASE == 11)
#define DAOP xf::database::AOP_NORML2
#define FUN norml2
#endif

// For debug
#include <cstdio>
#define XF_DATABASE_VOID_CAST static_cast<void>
// XXX toggle here to debug this file
#if 1
#define XF_DATABASE_PRINT(msg...) \
    do {                          \
        printf(msg);              \
    } while (0)
#else
#define XF_DATABASE_PRINT(msg...) (XF_DATABASE_VOID_CAST(0))
#endif

struct row_msg {
    ap_int<DIRECTW_TEST> data;
    ap_uint<DIRECTW_TEST> key;
};

int distinct_key = 0;

// top-function
void hls_db_direct_aggr(int op,
                        hls::stream<ap_uint<DATINW_TEST> >& in_strm,
                        hls::stream<bool>& e_in_strm,
                        hls::stream<ap_uint<DATOUTW_TEST> >& out_strm,
                        hls::stream<bool>& e_out_strm,
                        hls::stream<ap_uint<DIRECTW_TEST> >& kin_strm,
                        hls::stream<ap_uint<DIRECTW_TEST> >& kout_strm) {
#pragma HLS inline off

    if (TEST_PROGRAMMABLE_AGGR)
        xf::database::directGroupAggregate<DATINW_TEST, DATOUTW_TEST, DIRECTW_TEST>(op, in_strm, e_in_strm, out_strm,
                                                                                    e_out_strm, kin_strm, kout_strm);
    else
        xf::database::directGroupAggregate<DAOP, DATINW_TEST, DATOUTW_TEST, DIRECTW_TEST>(
            in_strm, e_in_strm, out_strm, e_out_strm, kin_strm, kout_strm);
}

#ifndef __SYNTHESIS__
// generate a random integer sequence between speified limits a and b (a<b);
int rand_int(int a, int b) {
    return rand() % (b - a + 1) + a;
}
int n_per_direct[direct_NUM];
int loopn = 2 * direct_NUM;

template <int DIRECTW>
int generate_test_data(uint64_t len, std::vector<row_msg>& testvector) {
    int totalN = 0;
    std::cout << "data num of this direct " << loopn << std::endl;
    for (int i = 0; i < loopn; i++) {
        int randnum = rand_int(KEY_RAND_MIN, KEY_RAND_MAX);
        testvector.push_back({rand_int(INT_RAND_MIN, INT_RAND_MAX),
                              randnum}); // generate random int value for data with the random key// rand_int(1,10)
        totalN++;
    }
    std::cout << " random test data generated! " << std::endl;
    return totalN;
}

namespace ref_direct_aggr {

template <int DATINW, int DATOUTW, int DIRECTW>
void sum(hls::stream<ap_uint<DATINW> >& din_strm,
         hls::stream<ap_uint<DATOUTW> >& out_strm,
         hls::stream<ap_uint<DIRECTW> >& kin_strm,
         hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int sum[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        sum[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                sum[i] += ret;
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(sum[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and sum is " << i << ", " << sum[i] << std::endl;
        }
    }
}

template <int DATINW, int DATOUTW, int DIRECTW>
void max(hls::stream<ap_uint<DATINW> >& din_strm,
         hls::stream<ap_uint<DATOUTW> >& out_strm,
         hls::stream<ap_uint<DIRECTW> >& kin_strm,
         hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int max[CNT_AGG_TEST];
    int ret64;
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;

        max[i] = 0x80000000;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        out_key = rkey; // direct_key;
        ret64 = ret;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                max[i] = ((max[i] > ret64) ? max[i] : ret64);
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(max[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and max is " << i << ", " << max[i] << std::endl;
        }
    }
}
// min
template <int DATINW, int DATOUTW, int DIRECTW>
void min(hls::stream<ap_uint<DATINW> >& din_strm,
         hls::stream<ap_uint<DATOUTW> >& out_strm,
         hls::stream<ap_uint<DIRECTW> >& kin_strm,
         hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int min[CNT_AGG_TEST];
    int ret64;
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;

        min[i] = 0x3fffffff;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        out_key = rkey; // direct_key;
        ret64 = ret;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                min[i] = ((min[i] < ret64) ? min[i] : ret64);
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(min[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and min is " << i << ", " << min[i] << std::endl;
        }
    }
}
// countone
template <int DATINW, int DATOUTW, int DIRECTW>
void countone(hls::stream<ap_uint<DATINW> >& din_strm,
              hls::stream<ap_uint<DATOUTW> >& out_strm,
              hls::stream<ap_uint<DIRECTW> >& kin_strm,
              hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int cnt[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        cnt[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                cnt[i]++;
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(cnt[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and countone is " << i << ", " << cnt[i] << std::endl;
        }
    }
}
// countdistinct
template <int DATINW, int DATOUTW, int DIRECTW>
void countdistinct(hls::stream<ap_uint<DATINW> >& din_strm,
                   hls::stream<ap_uint<DATOUTW> >& out_strm,
                   hls::stream<ap_uint<DIRECTW> >& kin_strm,
                   hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int sum[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> direct_key;
    ap_uint<DIRECTW> out_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                // if( ret )//TODO
                // sum[i] ++;
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(sum[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and countone is " << i << ", " << sum[i] << std::endl;
        }
    }
}

// variance
template <int DATINW, int DATOUTW, int DIRECTW>
void variance(hls::stream<ap_uint<DATINW> >& din_strm,
              hls::stream<ap_uint<DATOUTW> >& out_strm,
              hls::stream<ap_uint<DIRECTW> >& kin_strm,
              hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int cnt[CNT_AGG_TEST];
    int sum[CNT_AGG_TEST];
    int tow[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> direct_key;
    ap_uint<DIRECTW> out_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        cnt[i] = 0;
        sum[i] = 0;
        tow[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                cnt[i]++;
                sum[i] = sum[i] + ret;
                tow[i] = tow[i] + ret * ret;
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write((tow[i] - sum[i] * sum[i] / cnt[i]) / cnt[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and variance is " << i << ", " << ((tow[i] - sum[i] * sum[i] / cnt[i]) / cnt[i])
                      << std::endl;
        }
    }
}
// avg
template <int DATINW, int DATOUTW, int DIRECTW>
void avg(hls::stream<ap_uint<DATINW> >& din_strm,
         hls::stream<ap_uint<DATOUTW> >& out_strm,
         hls::stream<ap_uint<DIRECTW> >& kin_strm,
         hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int cnt[CNT_AGG_TEST];
    int sum[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> direct_key;
    ap_uint<DIRECTW> out_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        cnt[i] = 0;
        sum[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                cnt[i]++;
                sum[i] = sum[i] + ret;
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(sum[i] / cnt[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and avg is " << i << ", " << sum[i] / cnt[i] << std::endl;
        }
    }
}
// norml1
template <int DATINW, int DATOUTW, int DIRECTW>
void norml1(hls::stream<ap_uint<DATINW> >& din_strm,
            hls::stream<ap_uint<DATOUTW> >& out_strm,
            hls::stream<ap_uint<DIRECTW> >& kin_strm,
            hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    ap_int<DATINW> ret = 0;
    ap_int<DATOUTW> sum[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        sum[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        XF_DATABASE_PRINT(" ret =%s \n", ret.to_string().c_str());
        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                sum[i] += ((ret > 0) ? ret : (ap_int<DATINW>)-ret);
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(sum[i]);
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and norml1 is " << i << ", " << sum[i] << std::endl;
        }
    }
}
// norml2
template <int DATINW, int DATOUTW, int DIRECTW>
void norml2(hls::stream<ap_uint<DATINW> >& din_strm,
            hls::stream<ap_uint<DATOUTW> >& out_strm,
            hls::stream<ap_uint<DIRECTW> >& kin_strm,
            hls::stream<ap_uint<DIRECTW> >& kout_strm) {
    // dat
    int ret = 0;
    int sum[CNT_AGG_TEST];
    // key
    ap_uint<DIRECTW> rkey;
    ap_uint<DIRECTW> o_rkey[CNT_AGG_TEST];
    ap_uint<32> in_key, direct_key;
    ap_uint<DIRECTW> out_key;
    ap_uint<DIRECTW> tmp_key;
    // flag
    int flag[CNT_AGG_TEST];

    // initial flag
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        flag[i] = 0;
        sum[i] = 0;
    }
    // aggregate
    for (int j = 0; j < loopn; j++) {
        ret = din_strm.read();
        rkey = kin_strm.read();
        in_key = rkey;
        tmp_key = rkey;

        //			xf::database::details::directlookup3_core(rkey,direct_key);

        out_key = rkey; // direct_key;

        for (int i = 0; i < CNT_AGG_TEST; i++) {
            if (!(out_key - i)) {
                o_rkey[i] = out_key;
                sum[i] += (ret * ret);
                flag[i] = 1;
            }
        }
    }
    // output stream
    for (int i = 0; i < CNT_AGG_TEST; i++) {
        if (flag[i]) {
            distinct_key++;
            out_strm.write(hls::sqrt(sum[i]));
            kout_strm.write(o_rkey[i]);
            std::cout << "rkey and norml2 is " << i << ", " << hls::sqrt(sum[i]) << std::endl;
        }
    }
}

} // namespace ref_direct_norml2

int main() {
    std::vector<row_msg> testVector;
    hls::stream<ap_uint<DIRECTW_TEST> > kin_strm("inkey_strm");
    hls::stream<ap_uint<DATINW_TEST> > din_strm("din_strm");
    hls::stream<bool> e_in_strm("e_in_strm");

    hls::stream<ap_uint<DATOUTW_TEST> > dout_strm("dout_strm");
    hls::stream<ap_uint<DIRECTW_TEST> > kout_strm("outkey_strm");
    hls::stream<bool> e_out_strm("e_dout_strm");

    hls::stream<ap_uint<DATINW_TEST> > ref_din_strm("ref_din_strm");
    hls::stream<ap_uint<DIRECTW_TEST> > ref_kin_strm("ref_kin_strm");
    hls::stream<ap_uint<DATOUTW_TEST> > ref_dout_strm("ref_dout_strm");
    hls::stream<ap_uint<DIRECTW_TEST> > ref_kout_strm("ref_kout_strm");

    // generate test data
    int totalN = generate_test_data<DIRECTW_TEST>(direct_NUM, testVector);

    int nerror = 0;

    // prepare input data
    std::cout << "test data: (indexing key, data)" << std::endl;
    for (std::string::size_type i = 0; i < totalN; i++) {
        // print vector value
        std::cout << testVector[i].key << " ," << testVector[i].data << std::endl;
        // wirte data to stream
        kin_strm.write(testVector[i].key);
        din_strm.write(testVector[i].data);
        e_in_strm.write(0);
        ref_kin_strm.write(testVector[i].key);
        ref_din_strm.write(testVector[i].data);
        //        XF_DATABASE_PRINT(" data =%s \n" , testVector[i].data.to_string().c_str());
    }
    e_in_strm.write(1);
    std::cout << std::endl;

    // call direct aggregate function
    hls_db_direct_aggr(DAOP, din_strm, e_in_strm, dout_strm, e_out_strm, kin_strm, kout_strm);

    //===== run reference function to get the result ======
    ref_direct_aggr::FUN(ref_din_strm, ref_dout_strm, ref_kin_strm, ref_kout_strm);

    //===== check if the output flag e_out_strm is correct or not =====
    for (int i = 0; i < distinct_key; i++) {
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

    //===== compare the ref fun and hls func result
    ap_int<DATOUTW_TEST> dout, ref_dout;
    ap_uint<DIRECTW_TEST> kout, ref_kout;
    for (int i = 0; i < distinct_key; i++) {
        // compare
        kout = kout_strm.read();
        ref_kout = ref_kout_strm.read();
        dout = dout_strm.read();
        ref_dout = ref_dout_strm.read();
        std::cout << "key=" << kout << " data=" << dout << std::endl;

        bool cmp_key = (kout == ref_kout) ? 1 : 0;
        if (!cmp_key) {
            nerror++;
            std::cout << " the aggr key is incorrect, ref=" << ref_kout << std::endl;
        }
        bool cmp_dout = (dout == ref_dout) ? 1 : 0;
        if (!cmp_dout) {
            nerror++;
            std::cout << " the aggr data is incorrect, ref=" << ref_dout << std::endl;
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
