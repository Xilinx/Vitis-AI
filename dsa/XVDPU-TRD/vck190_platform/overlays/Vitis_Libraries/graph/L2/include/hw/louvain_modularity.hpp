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

#ifndef _XF_GRAPH_LOUVAIN_MODULARITY_HPP_
#define _XF_GRAPH_LOUVAIN_MODULARITY_HPP_

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include "stdio.h"
#include "ap_int.h"
#include <hls_math.h>
#include <hls_stream.h>

//#define DEBUG_INIT_COMM true
typedef int DVERTEX;
typedef double DWEIGHT;
typedef int DF_V_T;
typedef int DF_D_T;
typedef double DF_W_T;
typedef int SMALL_T;
typedef long long DF_WI_T;

#define FLAGW (8)

#define NUM_SMALL (8 * 8)
#define NUM_SMALL_LOG (6)

namespace xf {
namespace graph {

template <typename MType>
union f_cast;

template <>
union f_cast<ap_uint<32> > {
    float f;
    uint32_t i;
};

template <>
union f_cast<ap_uint<64> > {
    double f;
    uint64_t i;
};

template <>
union f_cast<double> {
    double f;
    uint64_t i;
};

template <>
union f_cast<float> {
    float f;
    uint32_t i;
};

template <>
union f_cast<int> {
    int f;
    uint32_t i;
};

template <>
union f_cast<long long> {
    long long f;
    uint64_t i;
};

template <typename MType, typename DType>
union f_cast_;

template <typename DT>
union f_cast_<DT, ap_uint<32> > {
    DT f;
    uint32_t i;
};

template <typename DT>
union f_cast_<DT, ap_uint<64> > {
    DT f;
    uint64_t i;
};

template <typename DT, int W, int BIT_E = sizeof(DT) * 8>
class AxiMap {
   public:
    const int N_E = W / BIT_E;

    ap_uint<W>* data;
    AxiMap(ap_uint<W>* port) { data = port; }

    DT rdi(int addr) {
#pragma HLS INLINE
        int div;
        int dec;
        if (N_E == 1) {
            return data[addr];
        } else if (N_E == 2) {
            dec = addr & 0x1;
            div = (addr & 0xfffffffe) >> 1; // 1110
            return data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
        } else if (N_E == 4) {
            dec = addr & 0x3;
            div = (addr & 0xfffffffc) >> 2; // 1100
            return data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
        } else if (N_E == 8) {
            dec = addr & 0x7;
            div = (addr & 0xfffffff8) >> 3; // 1000
            return data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
        } else if (N_E == 16) {
            dec = addr & 0xf;
            div = (addr & 0xfffffff0) >> 4; // 0000
            return data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
        } else { // 32
            dec = addr & 0x1f;
            div = (addr & 0xffffffe0) >> 5; // 00000
            return data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
        }
    }

    DT rdf(int addr) {
#pragma HLS INLINE
        int div;
        int dec;
        f_cast<ap_uint<BIT_E> > tmp;
        if (N_E == 1) {
            tmp.i = data[addr];
            return tmp.f;
        } else if (N_E == 2) {
            dec = addr & 0x1;
            div = (addr & 0xfffffffe) >> 1; // 1110
            tmp.i = data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
            return tmp.f;
        } else if (N_E == 4) {
            dec = addr & 0x3;
            div = (addr & 0xfffffffc) >> 2; // 1100
            tmp.i = data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
            return tmp.f;
        } else if (N_E == 8) {
            dec = addr & 0x7;
            div = (addr & 0xfffffff8) >> 3; // 1000
            tmp.i = data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
            return tmp.f;
        } else if (N_E == 16) {
            dec = addr & 0xf;
            div = (addr & 0xfffffff0) >> 4; // 0000
            tmp.i = data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
            return tmp.f;
        } else { // 32
            dec = addr & 0x1f;
            div = (addr & 0xffffffe0) >> 5; // 00000
            tmp.i = data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E);
            return tmp.f;
        }
    }

    void wr(int addr, DT dat) {
#pragma HLS INLINE
        int div;
        int dec;
        if (N_E == 1) {
            f_cast<DT> tmp;
            tmp.f = dat;
            data[addr] = tmp.i;
        } else if (N_E == 2) {
            dec = addr & 0x1;
            div = (addr & 0xfffffffe) >> 1; // 1110
            f_cast<DT> tmp;
            tmp.f = dat;
            data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E) = tmp.i;
        } else if (N_E == 4) {
            dec = addr & 0x3;
            div = (addr & 0xfffffffc) >> 2; // 1100
            f_cast<DT> tmp;
            tmp.f = dat;
            data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E) = tmp.i;
        } else if (N_E == 8) {
            dec = addr & 0x7;
            div = (addr & 0xfffffff8) >> 3; // 1000
            f_cast<DT> tmp;
            tmp.f = dat;
            data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E) = tmp.i;
        } else if (N_E == 16) {
            dec = addr & 0xf;
            div = (addr & 0xfffffff0) >> 4; // 0000
            f_cast<DT> tmp;
            tmp.f = dat;
            data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E) = tmp.i;
        } else {
            dec = addr & 0x1f;
            div = (addr & 0xfffffff0) >> 5; // 0000
            f_cast<DT> tmp;
            tmp.f = dat;
            data[div].range((dec + 1) * BIT_E - 1, dec * BIT_E) = tmp.i;
        }
    }

    ap_uint<W> add(const ap_uint<W>& a1, const ap_uint<W>& a2) {
#pragma HLS INLINE
        ap_uint<W> sum;
        ap_uint<W> vUp = a1;
        ap_uint<W> vPr = a2;

        for (int i = 0; i < N_E; i++) {
#pragma HLS UNROLL
            int width = i * BIT_E;
            f_cast_<DT, ap_uint<BIT_E> > v_a1, v_a2, v_sum;
            v_a1.i = vUp.range(width + BIT_E - 1, width);
            v_a2.i = vPr.range(width + BIT_E - 1, width);
            v_sum.f = v_a1.f + v_a2.f;
            sum.range(width + BIT_E - 1, width) = v_sum.i;
        }
        return sum;
    }

    ap_uint<W> sub(const ap_uint<W>& a1, const ap_uint<W>& a2) {
#pragma HLS INLINE
        ap_uint<W> sub;
        ap_uint<W> eUp = a1;
        ap_uint<W> ePr = a2;
        for (int i = 0; i < N_E; i++) {
            int width = i * BIT_E;
            f_cast_<DT, ap_uint<BIT_E> > v_a1, v_a2, v_sub;
            v_a1.i = eUp.range(width + BIT_E - 1, width);
            v_a2.i = ePr.range(width + BIT_E - 1, width);
            v_sub.f = v_a1.f - v_a2.f;
            sub.range(width + BIT_E - 1, width) = v_sub.i;
        }
        return sub;
    }

    void wr(ap_uint<W>& a, DT dat) {
#pragma HLS INLINE
        ap_uint<W> tmp;
        for (int i = 0; i < N_E; i++) {
            int width = i * BIT_E;
            f_cast<ap_uint<BIT_E> > v_a1;
            v_a1.f = dat;
            tmp.range(width + BIT_E - 1, width) = v_a1.i;
        }
        a = tmp;
    }
};

static DF_W_T ToDouble(long long v, short scl) {
#pragma HLS INLINE
    return (DF_W_T)((v + (1 << (scl - 1))) >> scl);
}
static DF_W_T ToDouble(DF_W_T v, short scl) {
#pragma HLS INLINE
    return v;
}

static DF_WI_T ToInt(DF_W_T val, short scl) {
#pragma HLS INLINE
    return (DF_WI_T)val * (1L << scl);
}

struct unitVD {
    DF_V_T v;
    DF_D_T degree;
    void set(DF_V_T in_v, DF_D_T in_d) {
        v = in_v;
        degree = in_d;
    }
    void get(DF_V_T& o_v, DF_D_T& o_dgr) {
        o_v = v;
        o_dgr = degree;
    }
    void set(ap_uint<64> uint) {
        v = uint(31, 0);
        degree = uint(63, 32);
    }
    void get(ap_uint<64>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = degree;
    }
    void set(ap_uint<96> uint) {
        v = uint(31, 0);
        degree = uint(63, 32);
    }
    void get(ap_uint<96>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = degree;
    }
    void set(ap_uint<64>& uint, DF_V_T in_v, DF_D_T in_d) {
        v = in_v;
        degree = in_d;
        get(uint);
    }
    void get(ap_uint<64> uint, DF_V_T& o_v, DF_D_T& o_dgr) {
        set(uint);
        o_v = v;
        o_dgr = degree;
    }
    void set(ap_uint<96>& uint, DF_V_T in_v, DF_D_T in_d) {
        v = in_v;
        degree = in_d;
        get(uint);
    }
    void get(ap_uint<96> uint, DF_V_T& o_v, DF_D_T& o_dgr) {
        set(uint);
        o_v = v;
        o_dgr = degree;
    }
};

struct unitEW {
    DF_V_T edge;
    DF_W_T wght;
    void set(DF_V_T in_e, DF_W_T in_w) {
        edge = in_e;
        wght = in_w;
    }
    void get(DF_V_T& o_e, DF_W_T& o_w) {
        o_e = edge;
        o_w = wght;
    }
    void set(ap_uint<96> uint) {
        edge = uint(31, 0);
        wght = uint(95, 32);
    }
    void get(ap_uint<96>& uint) {
        uint(31, 0) = edge;
        uint(95, 32) = wght;
    }
    void set(ap_uint<96>& uint, DF_V_T in_e, DF_W_T in_w) {
        edge = in_e;
        wght = in_w;
        get(uint);
    }
    void get(ap_uint<96> uint, DF_V_T& o_e, DF_W_T& o_w) {
        set(uint);
        o_e = edge;
        o_w = wght;
    }
};
union DoubleUnit64 {
    double d;
    long long u;
};

static double Uint2Double(ap_uint<64> uin) {
#pragma HLS INLINE
    DoubleUnit64 uni;
    uni.u = uin;
    return uni.d;
}
static ap_uint<64> Double2Unit(double d) {
#pragma HLS INLINE
    DoubleUnit64 uni;
    uni.d = d;
    return uni.u;
}
template <class TG>
struct unitCidGain {
    DF_V_T cid;
    TG gain;
    void set(DF_V_T in_e, TG in_w) {
        cid = in_e;
        gain = in_w;
    }
    void get(DF_V_T& o_e, TG& o_w) {
        o_e = cid;
        o_w = gain;
    }
};
struct unitCidGain_d {
    DF_V_T cid;
    double gain;
    void set(DF_V_T in_e, double in_w) {
        cid = in_e;
        gain = in_w;
    }
    void get(DF_V_T& o_e, double& o_w) {
        o_e = cid;
        o_w = gain;
    }
    void set(ap_uint<32 + 64> uint) {
        cid = uint(31, 0);
        gain = Uint2Double(uint(95, 32));
    }
    void get(ap_uint<32 + 64>& uint) {
        uint(31, 0) = cid;
        uint(95, 32) = Double2Unit(gain);
    }
};

struct unitCidGain_ll {
    DF_V_T cid;
    long long gain;
    void set(DF_V_T in_e, long long in_w) {
        cid = in_e;
        gain = in_w;
    }
    void get(DF_V_T& o_e, long long& o_w) {
        o_e = cid;
        o_w = gain;
    }
    void set(ap_uint<32 + 64> uint) {
        cid = uint(31, 0);
        gain = uint(95, 32);
    }
    void get(ap_uint<32 + 64>& uint) {
        uint(31, 0) = cid;
        uint(95, 32) = gain;
    }
    void set(ap_uint<160> uint) {
        cid = uint(31, 0);
        gain = uint(95, 32);
    }
    void get(ap_uint<160>& uint) {
        uint(31, 0) = cid;
        uint(95, 32) = gain;
        uint(159, 96) = 0;
    }
};

struct unitVCDN {
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T degree;
    DF_D_T numComm;
    void set(DF_V_T in_v, DF_V_T in_c, DF_D_T in_d, DF_D_T in_n) {
        v = in_v;
        vCid = in_c;
        degree = in_d;
        numComm = in_n;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_dgr, DF_D_T& o_n) {
        o_v = v;
        o_c = vCid;
        o_dgr = degree;
        o_n = numComm;
    }
    void set(ap_uint<128> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
        degree = uint(95, 64);
        numComm = uint(127, 96);
    }
    void get(ap_uint<128>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
        uint(95, 64) = degree;
        uint(127, 96) = numComm;
    }
};
struct unitVCD {
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T degree;
    void set(DF_V_T in_v, DF_V_T in_c, DF_D_T in_d) {
        v = in_v;
        vCid = in_c;
        degree = in_d;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_dgr) {
        o_v = v;
        o_c = vCid;
        o_dgr = degree;
    }
    void set(ap_uint<96> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
        degree = uint(95, 64);
    }
    void get(ap_uint<96>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
        uint(95, 64) = degree;
    }
    void set(ap_uint<96>& uint, DF_V_T in_v, DF_V_T in_c, DF_D_T in_d) {
        v = in_v;
        vCid = in_c;
        degree = in_d;
        this->get(uint);
    }
    void get(ap_uint<96> uint, DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_dgr) {
        set(uint);
        o_v = v;
        o_c = vCid;
        o_dgr = degree;
    }
};
struct unitVCDe {
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T degree;
    bool e_v;
    void set(DF_V_T in_v, DF_V_T in_c, DF_D_T in_d, bool in_e_v) {
        v = in_v;
        vCid = in_c;
        degree = in_d;
        e_v = in_e_v;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_dgr, bool& o_e_v) {
        o_v = v;
        o_c = vCid;
        o_dgr = degree;
        o_e_v = e_v;
    }
    void set(ap_uint<128> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
        degree = uint(95, 64);
        e_v = uint(127, 96);
    }
    void get(ap_uint<128>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
        uint(95, 64) = degree;
        uint(127, 96) = e_v;
    }
    void set(ap_uint<128>& uint, DF_V_T in_v, DF_V_T in_c, DF_D_T in_d, bool in_e_v) {
        v = in_v;
        vCid = in_c;
        degree = in_d;
        e_v = in_e_v;
        get(uint);
    }
    void get(ap_uint<128> uint, DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_dgr, bool& o_e_v) {
        set(uint);
        o_v = v;
        o_c = vCid;
        o_dgr = degree;
        o_e_v = e_v;
    }
};

struct unitVCN {
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T numComm;
    void set(DF_V_T in_v, DF_V_T in_c, DF_D_T in_n) {
        v = in_v;
        vCid = in_c;
        numComm = in_n;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_n) {
        o_v = v;
        o_c = vCid;
        o_n = numComm;
    }
    void set(ap_uint<96> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
        numComm = uint(95, 64);
    }
    void get(ap_uint<96>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
        uint(95, 64) = numComm;
    }
    void set(ap_uint<96>& uint, DF_V_T in_v, DF_V_T in_c, DF_D_T in_n) {
        v = in_v;
        vCid = in_c;
        numComm = in_n;
        get(uint);
    }
    void get(ap_uint<96> uint, DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_n) {
        set(uint);
        o_v = v;
        o_c = vCid;
        o_n = numComm;
    }
};
union uint2int32 {
    int int32;
    unsigned int uint;
};
struct unitVCNKi {
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T numComm;
    DF_W_T ki;
    void set(DF_V_T in_v, DF_V_T in_c, DF_D_T in_n, DF_W_T in_ki) {
        v = in_v;
        vCid = in_c;
        numComm = in_n;
        ki = in_ki;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_n, DF_W_T& o_ki) {
        o_v = v;
        o_c = vCid;
        o_n = numComm;
        o_ki = ki;
    }
    void set(ap_uint<96 + 64> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
        uint2int32 t;
        // t.uint = uint( 95, 64);
        numComm = uint(95, 64);
        ki = Uint2Double(uint(159, 96));
    }
    void get(ap_uint<96 + 64>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
        uint(95, 64) = numComm;
        uint(159, 96) = Double2Unit(ki);
    }
    void set(ap_uint<96 + 64>& uint, DF_V_T in_v, DF_V_T in_c, DF_D_T in_n, DF_W_T in_ki) {
        v = in_v;
        vCid = in_c;
        numComm = in_n;
        ki = in_ki;
        get(uint);
    }
    void get(ap_uint<96 + 64> uint, DF_V_T& o_v, DF_V_T& o_c, DF_D_T& o_n, DF_W_T& o_ki) {
        set(uint);
        o_v = v;
        o_c = vCid;
        o_n = numComm;
        o_ki = ki;
    }
};
struct unitECW {
    DF_V_T edge;
    DF_V_T eCid;
    DF_W_T wght;
    void set(DF_V_T in_e, DF_V_T in_c, DF_W_T in_w) {
        edge = in_e;
        wght = in_w;
        eCid = in_c;
    }
    void get(DF_V_T& o_e, DF_V_T& o_c, DF_W_T& o_w) {
        o_e = edge;
        o_w = wght;
        o_c = eCid;
    }
    void set(ap_uint<32 + 32 + 64> uint) {
        edge = uint(31, 0);
        eCid = uint(63, 32);
        wght = Uint2Double(uint(127, 64));
    }
    void get(ap_uint<32 + 32 + 64>& uint) {
        uint(31, 0) = edge;
        uint(63, 32) = eCid;
        uint(127, 64) = Double2Unit(wght);
    }
    void set(ap_uint<32 + 32 + 64>& uint, DF_V_T in_e, DF_V_T in_c, DF_W_T in_w) {
        edge = in_e;
        wght = in_w;
        eCid = in_c;
        get(uint);
    }
    void get(ap_uint<32 + 32 + 64> uint, DF_V_T& o_e, DF_V_T& o_c, DF_W_T& o_w) {
        set(uint);
        o_e = edge;
        o_w = wght;
        o_c = eCid;
    }
};
struct unitKiSelf {
    DF_W_T ki;
    DF_W_T self;
    void set(DF_W_T in_k, DF_W_T in_s) {
        ki = in_k;
        self = in_s;
    }
    void get(DF_W_T& o_k, DF_W_T& o_s) {
        o_k = ki;
        o_s = self;
    }
    void set(ap_uint<64 + 64> uint) {
        ki = Uint2Double(uint(63, 0));
        self = Uint2Double(uint(127, 64));
    }
    void get(ap_uint<64 + 64>& uint) {
        uint(63, 0) = Double2Unit(ki);
        uint(127, 64) = Double2Unit(self);
    }
    void set(ap_uint<64 + 64>& uint, DF_W_T in_k, DF_W_T in_s) {
        ki = in_k;
        self = in_s;
        get(uint);
    }
    void get(ap_uint<64 + 64> uint, DF_W_T& o_k, DF_W_T& o_s) {
        set(uint);
        o_k = ki;
        o_s = self;
    }
};

struct unitNCkKin {
    DF_D_T num;
    DF_V_T ck;
    DF_W_T kin;
    void set(DF_D_T in_num, DF_V_T in_ck, DF_W_T in_kin) {
        num = in_num;
        ck = in_ck;
        kin = in_kin;
    }
    void get(DF_D_T& o_num, DF_V_T& o_ck, DF_W_T& o_kin) {
        o_num = num;
        o_ck = ck;
        o_kin = kin;
    }
    void set(ap_uint<32 + 32 + 64> uint) {
        num = uint(31, 0);
        ck = uint(63, 32);
        kin = Uint2Double(uint(127, 64));
    }
    void get(ap_uint<32 + 32 + 64>& uint) {
        uint(31, 0) = num;
        uint(63, 32) = ck;
        uint(127, 64) = Double2Unit(kin);
    }
    void set(ap_uint<32 + 32 + 64>& uint, DF_D_T in_num, DF_V_T in_ck, DF_W_T in_kin) {
        num = in_num;
        ck = in_ck;
        kin = in_kin;
        get(uint);
    }
    void get(ap_uint<32 + 32 + 64> uint, DF_D_T& o_num, DF_V_T& o_ck, DF_W_T& o_kin) {
        set(uint);
        o_num = num;
        o_ck = ck;
        o_kin = kin;
    }
};

struct unitCidTarKin {
    DF_V_T cid;
    DF_V_T tar;
    DF_W_T kin;
    void set(DF_V_T in_cid, DF_V_T in_tar, DF_W_T in_kin) {
        cid = in_cid;
        tar = in_tar;
        kin = in_kin;
    }
    void get(DF_V_T& o_cid, DF_V_T& o_tar, DF_W_T& o_kin) {
        o_cid = cid;
        o_tar = tar;
        o_kin = kin;
    }
    void set(ap_uint<32 + 32 + 64> uint) {
        cid = uint(31, 0);
        tar = uint(63, 32);
        kin = Uint2Double(uint(127, 64));
    }
    void get(ap_uint<32 + 32 + 64>& uint) {
        uint(31, 0) = cid;
        uint(63, 32) = tar;
        uint(127, 64) = Double2Unit(kin);
    }
    void set(ap_uint<32 + 32 + 64>& uint, DF_V_T in_cid, DF_V_T in_tar, DF_W_T in_kin) {
        cid = in_cid;
        tar = in_tar;
        kin = in_kin;
        get(uint);
    }
    void get(ap_uint<32 + 32 + 64> uint, DF_V_T& o_cid, DF_V_T& o_tar, DF_W_T& o_kin) {
        set(uint);
        o_cid = cid;
        o_tar = tar;
        o_kin = kin;
    }
};
struct unitCkKin {
    DF_V_T ck;
    DF_W_T kin;
    void set(DF_V_T in_ck, DF_W_T in_kin) {
        ck = in_ck;
        kin = in_kin;
    }
    void get(DF_V_T& o_ck, DF_W_T& o_kin) {
        o_ck = ck;
        o_kin = kin;
    }
    void set(ap_uint<32 + 64> uint) {
        ck = uint(31, 0);
        kin = Uint2Double(uint(95, 32));
    }
    void get(ap_uint<32 + 64>& uint) {
        uint(31, 0) = ck;
        uint(95, 32) = Double2Unit(kin);
    }
    void set(ap_uint<32 + 64>& uint, DF_V_T in_ck, DF_W_T in_kin) {
        ck = in_ck;
        kin = in_kin;
        get(uint);
    }
    void get(ap_uint<32 + 64> uint, DF_V_T& o_ck, DF_W_T& o_kin) {
        set(uint);
        o_ck = ck;
        o_kin = kin;
    }
};

struct GetVout {
    DF_V_T v;
    DF_V_T off;
    DF_D_T degree;
    void set(DF_V_T in1, DF_V_T in2, DF_D_T in3) {
        v = in1;
        off = in2;
        degree = in3;
    }
    void get(DF_V_T& o_v, DF_V_T& o_off, DF_D_T& o_dgr) {
        o_v = v;
        o_off = off;
        o_dgr = degree;
    }
    void set(ap_uint<96> uint) {
        v = uint(31, 0);
        off = uint(63, 32);
        degree = uint(95, 64);
    }
    void get(ap_uint<96>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = off;
        uint(95, 64) = degree;
    }
    void set(ap_uint<96>& uint, DF_V_T in1, DF_V_T in2, DF_D_T in3) {
        v = in1;
        off = in2;
        degree = in3;
        get(uint);
    }
    void get(ap_uint<96> uint, DF_V_T& o_v, DF_V_T& o_off, DF_D_T& o_dgr) {
        set(uint);
        o_v = v;
        o_off = off;
        o_dgr = degree;
    }
};
union GetEout {
    unitVD vd;
    unitEW ew;
};
union GetAggout {
    unitVCDe vcde;
    unitKiSelf kisf;
    unitNCkKin n_ckk;
};
union GetCout {
    unitVCDe vcde;
    unitECW ecw;
};
struct unitVC {
    DF_V_T v;
    DF_V_T vCid;
    void set(DF_V_T in_v, DF_V_T in_c) {
        v = in_v;
        vCid = in_c;
    }
    void get(DF_V_T& o_v, DF_V_T& o_c) {
        o_v = v;
        o_c = vCid;
    }
    void set(ap_uint<128> uint) {
        v = uint(31, 0);
        vCid = uint(63, 32);
    }
    void get(ap_uint<128>& uint) {
        uint(31, 0) = v;
        uint(63, 32) = vCid;
    }
};

struct unitVF {
    DF_V_T v;
    float f;
    void set(DF_V_T in_v, float in_f) {
        v = in_v;
        f = in_f;
    }
    void get(DF_V_T& o_v, float& o_f) {
        o_v = v;
        o_f = f;
    }
    void set(ap_uint<128> uint) {
        v = uint(31, 0);
        f_cast<float> tmp;
        tmp.i = uint(63, 32);
        f = tmp.f;
    }
    void get(ap_uint<128>& uint) {
        uint(31, 0) = v;
        f_cast<float> tmp;
        tmp.f = f;
        uint(63, 32) = tmp.i;
    }
};
union StrmBus_S {
    unitVD vd;
    unitVC vc;
    unitVF vf;
};
union StrmBus_M {
    unitVD vd;
    GetVout vod;
    unitVCD vcd;
    unitEW ew;
    unitCidGain_d cg;
    unitVCN vcn;
    double ki;
};
union StrmBus_L {
    unitVCDN vcdn;
    unitVCDe vcde;
    unitECW ecw;
    unitKiSelf kisf;
    unitNCkKin n_ckk;
    unitCkKin ckkin;
    unitCidTarKin ctk;
};

union StrmBus_XL {
    unitVCNKi vcnk;
    unitCidGain_d cg;
    unitCidGain_ll cgi;
};

template <class T_V, class T_W>
struct CkKins {
    T_V Ck;
    T_W kinCk;
};
template <class T_V>
struct ValAddr {
    T_V val;
    int addr;
};

template <class T_MEM, int W_ADDR, int SIZE>
class AggRAM {
   public:
    T_MEM mem[SIZE];
    ap_uint<W_ADDR> addr_pre_1;
    T_MEM val_pre_1;
    AggRAM() {
#pragma HLS INLINE
    INITAggRAM:
        for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE
            mem[i] = 0;
        }
    }
    void Aggregate(ap_uint<W_ADDR> addr, T_MEM w, bool isFirst) {
#pragma HLS INLINE
        T_MEM val_new;
        if (addr == addr_pre_1 && !isFirst)
            val_new = val_pre_1 + w;
        else
            val_new = mem[addr] + w;
        mem[addr] = val_new;
        addr_pre_1 = addr;
        val_pre_1 = val_new;
    }
};

static ap_uint<NUM_SMALL_LOG> MemAgg_GetAddr(ap_uint<NUM_SMALL> Hit) {
#pragma HLS INLINE
    ap_uint<NUM_SMALL_LOG + 1> ret = 0;
    for (int i = 0; i < NUM_SMALL; i++)
#pragma HLS UNROLL
        if (Hit[i] == 1) {
            ret = i;
            break;
        }
    return ret;
}

static ap_uint<NUM_SMALL_LOG> MemAgg_GetAddr(SMALL_T& num_cid_small, ap_uint<NUM_SMALL> Hit) {
#pragma HLS INLINE
    if (Hit == 0)
        return num_cid_small++;
    else
        return MemAgg_GetAddr(Hit);
}

static ap_uint<NUM_SMALL_LOG> MemAgg_core_key(DF_V_T key, SMALL_T& num_cid_small, DF_V_T mem_key[NUM_SMALL]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = mem_key complete dim = 1
    ap_uint<NUM_SMALL> Hit = 0;
    for (int i = 0; i < NUM_SMALL; i++) {
#pragma HLS UNROLL
        if (i >= num_cid_small) {
            mem_key[i] = key;
            Hit[i] = 0;
        } else if (mem_key[i] == key)
            Hit[i] = 1;
    }
    return MemAgg_GetAddr(num_cid_small, Hit);
}

static void UpdateRamAgg(ap_uint<NUM_SMALL_LOG> addr, DF_WI_T w, DF_WI_T mem[NUM_SMALL], bool isFirst) {
#pragma HLS INLINE
    static SMALL_T addr_pre_1;
    static DF_WI_T agg_pre_1;
    DF_WI_T agg_new;

    if (addr == addr_pre_1 && !isFirst)
        agg_new = agg_pre_1 + w;
    else
        agg_new = mem[addr] + w;
    mem[addr] = agg_new;
    addr_pre_1 = addr;
    agg_pre_1 = agg_new;
}

template <int WA, int WD>
ap_uint<(1 << WD)> AxiRead(ap_uint<(1 << WA)>* axi, int addr) {
#pragma HLS INLINE
    const int B_ADDR = 64;
    const int BITS = 1 << WD; // 1<<5;
    ap_uint<B_ADDR> addr_u = addr;
    ap_uint<1 << WA> tmp = axi[addr_u(B_ADDR - 1, (WA - WD))];
    ap_uint<WA - WD> off = addr_u((WA - WD) - 1, 0);
    return tmp(BITS * off + BITS - 1, BITS * off);
}

template <class T_MEM, int W_ADDR, int SIZE>
struct AggRAM_base {
    T_MEM mem[SIZE];
    ap_uint<W_ADDR> addr_pre_1;
    T_MEM val_pre_1;
};

template <class T_MEM, int W_ADDR, int SIZE>
void AggRAM_base_init(AggRAM_base<T_MEM, W_ADDR, SIZE>* pthis) {
#pragma HLS INLINE
INITAggRAM:
    for (int i = 0; i < SIZE; i++) {
#pragma HLS PIPELINE
        pthis->mem[i] = 0;
    }
}
template <class T_MEM, int W_ADDR, int SIZE>
void AggRAM_base_Aggregate(AggRAM_base<T_MEM, W_ADDR, SIZE>* pthis, ap_uint<W_ADDR> addr, T_MEM w, bool isFirst) {
#pragma HLS INLINE
    T_MEM val_new;
    if (addr == pthis->addr_pre_1 && !isFirst)
        val_new = pthis->val_pre_1 + w;
    else
        val_new = pthis->mem[addr] + w;

    pthis->mem[addr] = val_new;
    pthis->addr_pre_1 = addr;
    pthis->val_pre_1 = val_new;
}

template <int NUM_LOG, int NUM>
ap_uint<NUM_LOG> MemAgg_GetAddr_T(ap_uint<NUM> Hit) {
#pragma HLS INLINE
    ap_uint<NUM_LOG + 1> ret = 0;
    for (int i = 0; i < NUM; i++)
#pragma HLS UNROLL
        if (Hit[i] == 1) {
            ret = i;
            break;
        }
    return ret;
}

template <int NUM_LOG, int NUM>
ap_uint<NUM_LOG> MemAgg_core_key_T(DF_V_T key, SMALL_T& num_cid_small, DF_V_T mem_key[NUM]) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = mem_key complete dim = 1
    ap_uint<NUM> Hit = 0;
    for (int i = 0; i < NUM; i++) {
#pragma HLS UNROLL
        if (i >= num_cid_small) {
            mem_key[i] = key;
        } else if (mem_key[i] == key) {
            Hit[i] = 1;
        }
    }
    if (Hit == 0)
        return num_cid_small++;
    else
        return MemAgg_GetAddr_T<NUM_LOG, NUM>(Hit);
}

template <int DWIDTH, int BATCH, int BATCHLOG>
void AggBatch2Mem(short scl,
                  ap_uint<DWIDTH>* cUpdateSize,
                  ap_uint<DWIDTH>* totUpdate,
                  DF_V_T* mem_key,
                  AggRAM_base<DF_WI_T, BATCHLOG, BATCH>& mem_agg,
                  AggRAM_base<DF_D_T, BATCHLOG, BATCH>& mem_cnt,
                  DF_D_T& num_cid_small,
                  bool& isFirstAddr) {
#pragma HLS INLINE

    AxiMap<DF_V_T, DWIDTH> axi_cUpdateSize(cUpdateSize);
    AxiMap<float, DWIDTH> axi_totUpdate(totUpdate);

BATCH2MEM:
    for (int i = 0; i < num_cid_small; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totUpdate inter false
#pragma HLS DEPENDENCE variable = cUpdateSize inter false

        DF_V_T addr_target = mem_key[i];
        DF_W_T ki;
        ki = ToDouble(mem_agg.mem[i], scl);
        float tmptot = axi_totUpdate.rdf(addr_target);
        tmptot += ki;

        axi_totUpdate.wr(addr_target, tmptot);

        int tmpcSize = axi_cUpdateSize.rdi(addr_target);
        tmpcSize += mem_cnt.mem[i];
        axi_cUpdateSize.wr(addr_target, tmpcSize);

        mem_agg.mem[i] = 0;
        mem_cnt.mem[i] = 0;
    }
    num_cid_small = 0;
    isFirstAddr = true;
}
}
}

#endif
