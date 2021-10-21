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

#ifndef _LOUVAIN_SAMECOLOR_HASHAGG_H_
#define _LOUVAIN_SAMECOLOR_HASHAGG_H_

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

namespace xf {
namespace graph {

////////////////////////////////////////////////////
// big-hash

template <class T_V, class T_W, int W_HASH, int W_MEM>
class HashAgg {
   public:
    int cnt_agg;
    CkKins<T_V, T_W>* ckkins;
    //
    ValAddr<T_V>* valAdd;
    ap_uint<1 << W_MEM>* mem_used;

    HashAgg(CkKins<T_V, T_W>* p_ckkins, ValAddr<T_V>* p_valAdd, ap_uint<1 << W_MEM>* p_mem_used) {
        ckkins = p_ckkins;
        valAdd = p_valAdd;
        mem_used = p_mem_used;
        Init();
    }

    void Init() {
        cnt_agg = 0;
    FLAG_RESET:
        for (int i = 0; i < 1 << (W_HASH - W_MEM); i++) {
#pragma HLS PIPELINE II = 1
            mem_used[i] = 0; // not used
        }
    }

    ap_uint<1> Get_mem_used(T_V h) {
#pragma HLS INLINE
        ap_uint<1 << W_MEM> v = mem_used[h >> W_MEM];
        return v[h & ((1 << W_MEM) - 1)];
    }

    void Set1_mem_used(T_V h) {
#pragma HLS INLINE
        ap_uint<1 << W_MEM> v = mem_used[h >> W_MEM];
        v[h & ((1 << W_MEM) - 1)] = 1;
        mem_used[h >> W_MEM] = v;
    }

    int myHash(T_V v) {
#pragma HLS INLINE
        ap_uint<32> val = v;
        return val(W_HASH - 1, 0); // eg. val(16:0);
    }

    int GetAddr(T_V cid) {
#pragma HLS INLINE
        int h = myHash(cid);
        ap_uint<1> isUsed = Get_mem_used(h);
        int cid_hash = valAdd[h].val;
        int a_hash = valAdd[h].addr;
        if (isUsed == 0) {
            Set1_mem_used(h); // mem_used[h] =  true;
            valAdd[h].val = cid;
            valAdd[h].addr = cnt_agg;
            return cnt_agg;
        } else {
            if (cid == cid_hash) {
                return a_hash;
            } else {
                return -1;
            }
        }
    }

    void SetvCid(int cid) {
        ckkins[0].kinCk = 0.0;
        ckkins[0].Ck = cid;
        cnt_agg = 1;
    }

    bool TryToSet(T_V cid, T_W w) {
#pragma HLS INLINE
        int add = GetAddr(cid);
        if (add < 0)
            return false;
        else if (cnt_agg == add) {
            ckkins[add].kinCk = w;
            ckkins[add].Ck = cid;
            cnt_agg++;
        } else
            ckkins[add].kinCk += w;
        return true;
    }
    ///////////////////////////////////////////
    void Output(hls::stream<ap_uint<128> >& str_Aggout, int cpy_cnt_agg) {
#pragma HLS DATAFLOW
        for (int i = 0; i < cpy_cnt_agg; i++) {
            StrmBus_L out_agg_nkk;
            ap_uint<128> dinout;
            CkKins<T_V, T_W> ckk = ckkins[i];
            out_agg_nkk.n_ckk.set(dinout, i, ckk.Ck, ckk.kinCk);
            str_Aggout.write(dinout);
        }
        Init();
    }
    void Output(short scl, hls::stream<ap_uint<128> >& str_Aggout, int cpy_cnt_agg) {
#pragma HLS DATAFLOW
        for (int i = 0; i < cpy_cnt_agg; i++) {
            StrmBus_L out_agg_nkk;
            ap_uint<128> dinout;
            CkKins<T_V, T_W> ckk = ckkins[i];
            out_agg_nkk.n_ckk.set(dinout, i, ckk.Ck, ToDouble(ckk.kinCk, scl));
            str_Aggout.write(dinout);
        }
        Init();
    }
};

template <class T_V, class T_W, int W_HASH, int W_MEM>
void HashAgg_Output(int cnt_agg,
                    CkKins<T_V, T_W> ckkins[1 << W_HASH],
                    ValAddr<T_V> valAdd[1 << W_HASH],
                    ap_uint<1 << W_MEM> mem_used[1 << (W_HASH - W_MEM)],
                    hls::stream<DF_V_T>& str_GetKins_Ck,
                    hls::stream<DF_W_T>& str_GetKins_KinCk) {
#pragma HLS DATAFLOW
    for (int i = 0; i < cnt_agg; i++) {
        CkKins<T_V, T_W> ckk = ckkins[i];
        str_GetKins_Ck.write(ckk.Ck);
        str_GetKins_KinCk.write((DF_W_T)ckk.kinCk);
    }
    for (int i = 0; i < 1 << (W_HASH - W_MEM); i++) {
#pragma HLS PIPELINE II = 1
        mem_used[i] = 0; // not used
    }
}
template <class T_V, class T_W, int SIZE_REM>
class ScanAgg {
   public:
    int cnt_agg;
    CkKins<T_V, T_W>* ckkins;
    ScanAgg(CkKins<T_V, T_W>* p_mem) {
        ckkins = p_mem;
        Init();
    }
    void Init() { cnt_agg = 0; }

    void AggWeight_f(T_V cid, float w) {
        int trip = cnt_agg + 1;
        for (int i = 0; i < trip; i++) {
#pragma HLS PIPELINE II = 1
            if (i == cnt_agg) {
                ckkins[i].Ck = cid;
                ckkins[i].kinCk = w;
                cnt_agg++;
            } else if (ckkins[i].Ck == cid) {
                ckkins[i].kinCk += w;
                break;
            }
        } //	for
    }

    void AggWeight(T_V cid, long long w) {
        int trip = cnt_agg + 1;
    AGGSCAN:
        for (int i = 0; i < trip; i++) {
#pragma HLS PIPELINE II = 1
            if (i == cnt_agg) {
                ckkins[i].Ck = cid;
                ckkins[i].kinCk = w;
                cnt_agg++;
            } else if (ckkins[i].Ck == cid) {
                ckkins[i].kinCk += w;
                break;
            }
        } //	for
    }

    ///////////////////////////////////////////

    void Output(hls::stream<ap_uint<128> >& str_Aggout) {
        for (int i = 0; i < cnt_agg; i++) {
#pragma HLS PIPELINE II = 1
            StrmBus_L out_agg_nkk;
            ap_uint<128> dinout;
            CkKins<T_V, T_W> ckk = ckkins[i];
            out_agg_nkk.n_ckk.set(dinout, i, ckk.Ck, ckk.kinCk);
            str_Aggout.write(dinout);
        }
        Init();
    }
    void Output(short scl, hls::stream<ap_uint<128> >& str_Aggout) {
        for (int i = 0; i < cnt_agg; i++) {
#pragma HLS PIPELINE II = 1
            StrmBus_L out_agg_nkk;
            ap_uint<128> dinout;
            CkKins<T_V, T_W> ckk = ckkins[i];
            out_agg_nkk.n_ckk.set(dinout, i, ckk.Ck, ToDouble(ckk.kinCk, scl));
            str_Aggout.write(dinout);
        }
        Init();
    }

}; // class ScanAgg;

template <class T_V, class T_W, int LG2_W_HASHADDR, int LG2_W_MEMPORT>
void HashAgg_dataflow_core_p1(short scl,
                              int v,
                              int degree,
                              HashAgg<T_V, T_W, LG2_W_HASHADDR, LG2_W_MEMPORT>& myHash,
                              hls::stream<DF_W_T>& str_GetC_wght,
                              hls::stream<DF_V_T>& str_GetC_edge,
                              hls::stream<DF_V_T>& str_GetC_eCid,
                              DF_W_T& selfloop,
                              DF_W_T& ki,
                              hls::stream<T_V>& str_agg_cid,
                              hls::stream<T_W>& str_agg_w,
                              hls::stream<bool>& str_agg_e) {
    DF_WI_T selfloop_i, ki_i, deg_w_i;
    selfloop_i = 0;
    ki_i = 0;
SCAN_E:
    for (int j = 0; j < degree; j++) {
#pragma HLS PIPELINE II = 1
        DF_W_T deg_w = str_GetC_wght.read();
        DF_V_T edge = str_GetC_edge.read();
        DF_V_T eCid = str_GetC_eCid.read();
        deg_w_i = ToInt(deg_w, scl);
        ki_i += deg_w_i;
        if (v == edge) selfloop_i += deg_w_i;
        if (false == myHash.TryToSet(eCid, deg_w)) {
            str_agg_cid.write(eCid);
            str_agg_w.write(deg_w);
            str_agg_e.write(false);
        }
    } // j
    ki = ToDouble(ki_i, scl);
    selfloop = ToDouble(selfloop_i, scl);
    str_agg_e.write(true);
}
template <class T_V, class T_W, int LOG2_HASH>
void HashAgg_dataflow_core_p2(ScanAgg<T_V, T_W, 1 << LOG2_HASH>& myScan,
                              hls::stream<T_V>& str_agg_cid,
                              hls::stream<T_W>& str_agg_w,
                              hls::stream<bool>& str_agg_e) {
    bool e_agg = str_agg_e.read();
    while (e_agg == false) {
        T_V eCid = str_agg_cid.read();
        T_W deg_w = str_agg_w.read();
        e_agg = str_agg_e.read();
        myScan.AggWeight_f(eCid, deg_w);
    }
}

template <class T_V, class T_W, int LG2_W_HASHADDR, int LG2_W_MEMPORT>
void HashAgg_dataflow_core_p1_bus(short scl,
                                  DF_V_T v,
                                  DF_D_T degree,
                                  HashAgg<T_V, T_W, LG2_W_HASHADDR, LG2_W_MEMPORT>& myHash,
                                  hls::stream<ap_uint<128> >& str_GetCout,
                                  DF_W_T& selfloop,
                                  DF_W_T& ki,
                                  hls::stream<T_V>& str_agg_cid,
                                  hls::stream<T_W>& str_agg_w,
                                  hls::stream<bool>& str_agg_e) {
    StrmBus_L out_c;
    DF_WI_T selfloop_i, ki_i, deg_w_i;
    DF_W_T deg_w;
    DF_V_T edge;
    DF_V_T eCid;
    selfloop_i = 0;
    ki_i = 0;
    ap_uint<128> dinout;
SCAN_E:
    for (int j = 0; j < degree; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 64
        str_GetCout.read(dinout);
        out_c.ecw.get(dinout, edge, eCid, deg_w);
        deg_w_i = ToInt(deg_w, scl);
        ki_i += deg_w_i;
        if (v == edge) selfloop_i += deg_w_i;
        if (false == myHash.TryToSet(eCid, deg_w)) {
            str_agg_cid.write(eCid);
            str_agg_w.write(deg_w);
            str_agg_e.write(false);
        }
    } // j
    ki = ToDouble(ki_i, scl);
    selfloop = ToDouble(selfloop_i, scl);
    str_agg_e.write(true);
}

template <class T_V, class T_W, int LG2_W_HASH, int LG2_W_MEMPORT>
void HashAgg_dataflow_core_bus(short scl,
                               DF_V_T v,
                               DF_D_T degree,
                               HashAgg<T_V, T_W, LG2_W_HASH, LG2_W_MEMPORT>& myHash,
                               ScanAgg<T_V, T_W, 1 << LG2_W_HASH>& myScan,
                               hls::stream<ap_uint<128> >& str_GetCout,
                               DF_W_T& selfloop,
                               DF_W_T& ki) {
#pragma HLS DATAFLOW

    hls::stream<T_V> str_agg_cid("str_agg_cid");
#pragma HLS RESOURCE variable = str_agg_cid core = FIFO_SRL
#pragma HLS STREAM variable = str_agg_cid depth = 256
    hls::stream<T_W> str_agg_w("str_agg_w");
#pragma HLS RESOURCE variable = str_agg_w core = FIFO_SRL
#pragma HLS STREAM variable = str_agg_w depth = 256
    hls::stream<bool> str_agg_e("str_agg_e");
#pragma HLS RESOURCE variable = str_agg_e core = FIFO_SRL
#pragma HLS STREAM variable = str_agg_e depth = 256

    HashAgg_dataflow_core_p1_bus<T_V, T_W, LG2_W_HASH, LG2_W_MEMPORT>(scl, v, degree, myHash, str_GetCout, selfloop, ki,
                                                                      str_agg_cid, str_agg_w, str_agg_e);

    HashAgg_dataflow_core_p2<T_V, T_W, LG2_W_HASH>(myScan, str_agg_cid, str_agg_w, str_agg_e);
}

template <class T_V, class T_W, int LG2_W_HASH, int LG2_W_MEMPORT>
void HashAgg_dataflow_bus3(short scl,
                           int v,
                           int vCid,
                           int degree,
                           HashAgg<T_V, T_W, LG2_W_HASH, LG2_W_MEMPORT>& myHash,
                           ScanAgg<T_V, T_W, 1 << LG2_W_HASH>& myScan,
                           hls::stream<ap_uint<128> >& str_GetCout,
                           hls::stream<ap_uint<128> >& str_Aggout,
                           StrmBus_L out_agg_vcde) {
    DF_W_T selfloop = 0;
    DF_W_T ki = 0;
    StrmBus_L out_agg_kisf, out_agg_nkk;
    StrmBus_L out_c;
    ap_uint<128> dinout1;
    ap_uint<128> dinout2;
    ap_uint<128> dinout3;

    if (degree > 0) {
        myHash.TryToSet(vCid, 0);
        HashAgg_dataflow_core_bus<T_V, T_W, LG2_W_HASH, LG2_W_MEMPORT>(scl, v, degree, myHash, myScan, str_GetCout,
                                                                       selfloop, ki);

        int numComm = myHash.cnt_agg + myScan.cnt_agg;
        out_agg_vcde.vcdn.numComm = numComm;
        out_agg_vcde.vcdn.get(dinout1);
        out_agg_kisf.kisf.set(dinout2, ki, selfloop);

        str_Aggout.write(dinout1);
        str_Aggout.write(dinout2);
        myHash.Output(str_Aggout, myHash.cnt_agg);
        myScan.Output(str_Aggout);
    } else {
        // str_Aggout.write(out_agg_vcde);
        out_agg_vcde.vcdn.get(dinout3);
        str_Aggout.write(dinout3);
    }
}

template <class T_V, class T_W, int LG2_W_HASHADDR, int LG2_W_MEMPORT>
void SameColor_GetKins_HashAgg_big(short scl,
                                   hls::stream<ap_uint<128> >& str_GetCout,
                                   hls::stream<ap_uint<128> >& str_Aggout) {
    CkKins<T_V, T_W> ckkins[1 << LG2_W_HASHADDR];
#pragma HLS RESOURCE variable = ckkins core = RAM_T2P_URAM
    CkKins<T_V, T_W> ckkins_s[1 << LG2_W_HASHADDR];
#pragma HLS RESOURCE variable = ckkins_s core = RAM_T2P_URAM
    ValAddr<T_V> valAdd[1 << LG2_W_HASHADDR];
#pragma HLS RESOURCE variable = valAdd core = RAM_T2P_URAM
    ap_uint<1 << LG2_W_MEMPORT> mem_used[1 << (LG2_W_HASHADDR - LG2_W_MEMPORT)];
#pragma HLS RESOURCE variable = mem_used core = RAM_T2P_URAM

    HashAgg<T_V, T_W, LG2_W_HASHADDR, LG2_W_MEMPORT> myHash(ckkins, valAdd, mem_used);
    ScanAgg<T_V, T_W, 1 << LG2_W_HASHADDR> myScan(ckkins_s);
    bool e_GetC = false;
    ap_uint<128> dinout;
    StrmBus_L out_c, out_agg_vcde, out_agg_kisf, out_agg_nkk;
GET_KIN_HASH:
    while (e_GetC == false) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 10000000
        int v, vCid, degree;
        str_GetCout.read(dinout);
        out_c.vcde.get(dinout, v, vCid, degree, e_GetC);
        if (e_GetC == true) degree = -1;
        out_agg_vcde.vcde.set(dinout, v, vCid, degree, e_GetC);

        if (e_GetC == false) {
            HashAgg_dataflow_bus3<T_V, T_W, LG2_W_HASHADDR, LG2_W_MEMPORT>(scl, v, vCid, degree, myHash, myScan,
                                                                           str_GetCout, str_Aggout, out_agg_vcde);
        } else
            str_Aggout.write(dinout);
    }
};
// big-hash
////////////////////////////////////////////////////
////////////////////////////////////////////////////
// small-hash

static void SameColor_GetKins_HashAgg_small(short scl,
                                            hls::stream<ap_uint<128> >& str_GetCout,
                                            hls::stream<ap_uint<128> >& str_Aggout) {
    DF_D_T num_cid_small;
    DF_V_T mem_key[NUM_SMALL];
#pragma HLS ARRAY_PARTITION variable = mem_key dim = 1
    AggRAM<DF_WI_T, NUM_SMALL_LOG, NUM_SMALL> mem_agg;

    StrmBus_L out_c, out_agg_vcde, out_agg_kisf, out_agg_nkk;
    bool e_GetC = false;
    DF_D_T degree, j;
    DF_V_T v, vCid;
    long long ki_i, self_i;

    DF_D_T dgr;
    DF_V_T edge;
    DF_V_T key;
    DF_W_T deg_w;
    DF_WI_T val;

    ap_uint<128> dinout;
    ap_uint<128> dout_out_agg_vcde, dout_out_agg_kisf, dinout_vcde, dout_out_agg_nkk;

    do {
        str_GetCout.read(dinout);
        out_c.vcde.get(dinout, v, vCid, degree, e_GetC);
        if (e_GetC == true) degree = -1;
        out_agg_vcde.vcde.set(v, vCid, degree, e_GetC);
        {
            num_cid_small = 0;
            dgr = degree > 0 ? degree + 1 : 0;
            if (dgr > 0) {
            SAGG_WRAM:
                for (int j = 0; j < dgr; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = mem_agg inter false
                    if (j == 0) {
                        key = vCid;
                        val = 0;
                        ki_i = 0;
                        self_i = 0;
                    } else {
                        str_GetCout.read(dinout);
                        out_c.ecw.get(dinout, edge, key, deg_w);
                        val = ToInt(deg_w, scl);
                        ki_i += val;
                        if (v == edge) self_i += val;
                    }
                    ap_uint<NUM_SMALL_LOG> addr = MemAgg_core_key(key, num_cid_small, mem_key);
                    mem_agg.Aggregate(addr, val, j == 0);
                }

            SAGG_SEND:
                DF_W_T selfloop = ToDouble(self_i, scl);
                DF_W_T ki = ToDouble(ki_i, scl);
                out_agg_kisf.kisf.set(dout_out_agg_kisf, ki, selfloop);
                out_agg_nkk.n_ckk.set(num_cid_small, 0, 0);
                out_agg_vcde.vcdn.set(v, vCid, degree, num_cid_small);
                out_agg_vcde.vcdn.get(dout_out_agg_vcde);
                str_Aggout.write(dout_out_agg_vcde);
                str_Aggout.write(dout_out_agg_kisf);
#ifdef _DEBUG_SMALL
                printf("SMALL: v=%d,\t vCid=%d\t  degree=%d\t num=%d\t ki=%f\t self=%f\n", v, vCid, degree,
                       num_cid_small, ki, selfloop);
#endif
                for (int i = 0; i < num_cid_small; i++) {
#pragma HLS PIPELINE II = 1
                    DF_W_T tmp_db = ToDouble(mem_agg.mem[i], scl);
                    mem_agg.mem[i] = 0;
                    out_agg_nkk.n_ckk.set(dout_out_agg_nkk, i, mem_key[i], tmp_db);
                    str_Aggout.write(dout_out_agg_nkk);
#ifdef _DEBUG_SMALL
                    printf("SMALL:i=%d \t ck=%d\t kin=%f\n", i, mem_key[i], tmp_db);
#endif
                }
            } else {
                out_agg_vcde.vcde.get(dinout_vcde);
                str_Aggout.write(dinout_vcde);
            }
        }
    } while (e_GetC == false);
}
// small-hash
/////////////////////////////////////////////////////////////////////

static void dispatch_send_ewc(DF_D_T degree,
                              hls::stream<ap_uint<128> >& str_GetCout_in,
                              hls::stream<ap_uint<128> >& str_GetCout_out) {
#pragma HLS INLINE
    StrmBus_L cout;
    ap_uint<128> dinout;
    if (degree > 0) {
    SCAN_E:
        for (int j = 0; j < degree; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 64
            str_GetCout_in.read(dinout);
            str_GetCout_out.write(dinout);
        }
    }
}

static void SameColor_GetKins_hash_dispatch(int th_2,
                                            // input
                                            hls::stream<ap_uint<128> >& str_GetCout,
                                            // output
                                            hls::stream<ap_uint<128> >& str_GetCout0,
                                            hls::stream<ap_uint<128> >& str_GetCout1,
                                            hls::stream<ap_uint<128> >& str_GetCout2) {
    ap_uint<1> select_0 = 0;
    short pre_degree_0;
    short pre_degree_1;
    bool e_GetC = false;
    DF_V_T v;
    DF_V_T vCid;
    DF_D_T degree;
    DF_W_T deg_w;
    DF_V_T edge;
    DF_V_T eCid;
    StrmBus_L cout;
    ap_uint<128> dinout;

GET_KIN_DISPATCH:
    while (e_GetC == false) {
        str_GetCout.read(dinout);
        cout.vcde.get(dinout, v, vCid, degree, e_GetC);
        if (e_GetC == false) {
            if (degree < th_2) {
                if (select_0 == 0) {
                    str_GetCout0.write(dinout);
                    dispatch_send_ewc(degree, str_GetCout, str_GetCout0);
                    select_0 = 1;
                    pre_degree_0 = degree;
                } else {
                    str_GetCout1.write(dinout);
                    dispatch_send_ewc(degree, str_GetCout, str_GetCout1);
                    select_0 = 0;
                    pre_degree_1 = degree;
                }
            } else {
                str_GetCout2.write(dinout);
                dispatch_send_ewc(degree, str_GetCout, str_GetCout2);
            }
        } else {
            cout.vcde.set(0, 0, 0, true);
            str_GetCout0.write(dinout);
            str_GetCout1.write(dinout);
            str_GetCout2.write(dinout);
        }
    }
}

static bool Collector_bus_vcde_ewc(hls::stream<ap_uint<128> >& str_Aggout1, hls::stream<ap_uint<128> >& str_Aggout) {
#pragma HLS INLINE
    StrmBus_L vcdn;
    ap_uint<128> dinout1;
    bool e_1;
    str_Aggout1.read(dinout1);
    vcdn.vcdn.set(dinout1);
    e_1 = vcdn.vcdn.degree < 0;
    if (e_1 == false) {
        str_Aggout.write(dinout1);
        int degree = vcdn.vcdn.degree;
        int numComm = vcdn.vcdn.numComm;
        if (degree > 0) {
            str_Aggout1.read(dinout1);
            str_Aggout.write(dinout1);
            for (int i = 0; i < numComm; i++) {
#pragma HLS PIPELINE II = 1
                str_Aggout1.read(dinout1);
                str_Aggout.write(dinout1);
            }
        }
    }
    return e_1;
}

static void SameColor_GetKins_hash_Collector(
    // input
    hls::stream<ap_uint<128> >& str_Aggout0,
    hls::stream<ap_uint<128> >& str_Aggout1,
    hls::stream<ap_uint<128> >& str_Aggout2,
    // output
    hls::stream<ap_uint<128> >& str_Aggout) {
    ap_uint<128> dinout;
    ap_uint<2> selected = 0;
    bool e_0 = false;
    bool e_1 = false;
    bool e_2 = false;
    bool e012 = e_0 & e_1 & e_2;
    while (e012 == false) {
        if (selected == 0) {
            if (!str_Aggout0.empty()) e_0 = Collector_bus_vcde_ewc(str_Aggout0, str_Aggout);
            if (!str_Aggout1.empty()) e_1 = Collector_bus_vcde_ewc(str_Aggout1, str_Aggout);
            if (!str_Aggout2.empty()) e_2 = Collector_bus_vcde_ewc(str_Aggout2, str_Aggout);
            selected = 1;
        } else if (selected == 1) {
            if (!str_Aggout1.empty()) e_1 = Collector_bus_vcde_ewc(str_Aggout1, str_Aggout);
            if (!str_Aggout2.empty()) e_2 = Collector_bus_vcde_ewc(str_Aggout2, str_Aggout);
            if (!str_Aggout0.empty()) e_0 = Collector_bus_vcde_ewc(str_Aggout0, str_Aggout);
            selected = 2;
        } else {
            if (!str_Aggout2.empty()) e_2 = Collector_bus_vcde_ewc(str_Aggout2, str_Aggout);
            if (!str_Aggout0.empty()) e_0 = Collector_bus_vcde_ewc(str_Aggout0, str_Aggout);
            if (!str_Aggout1.empty()) e_1 = Collector_bus_vcde_ewc(str_Aggout1, str_Aggout);
            selected = 0;
        }
        e012 = e_0 & e_1 & e_2;
    }
    StrmBus_L bus_out;
    bus_out.vcdn.set(0, 0, -2, -2);
    bus_out.vcdn.get(dinout);
    str_Aggout.write(dinout);
}

static void SameColor_GetKins_hash_top(short scl,
                                       // input
                                       hls::stream<ap_uint<128> >& str_GetCout,
                                       hls::stream<ap_uint<128> >& str_Aggout) {
#pragma HLS DATAFLOW
    //
    hls::stream<ap_uint<128> > str_GetCout0("str_GetCout0");
#pragma HLS RESOURCE variable = str_GetCout0 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetCout0 depth = 128
    hls::stream<ap_uint<128> > str_GetCout1("str_GetCout1");
#pragma HLS RESOURCE variable = str_GetCout1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetCout1 depth = 128
    hls::stream<ap_uint<128> > str_GetCout2("str_GetCout2");
#pragma HLS RESOURCE variable = str_GetCout2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetCout2 depth = 128
    SameColor_GetKins_hash_dispatch(63,
                                    // input
                                    str_GetCout, str_GetCout0, str_GetCout1, str_GetCout2);

    hls::stream<ap_uint<128> > str_GetAggout0("str_GetAggout0");
#pragma HLS RESOURCE variable = str_GetAggout0 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetAggout0 depth = 128
    SameColor_GetKins_HashAgg_small(scl, str_GetCout0, str_GetAggout0);

    hls::stream<ap_uint<128> > str_GetAggout1("str_GetAggout1");
#pragma HLS RESOURCE variable = str_GetAggout1 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetAggout1 depth = 128

    SameColor_GetKins_HashAgg_small(scl, str_GetCout1, str_GetAggout1);

    hls::stream<ap_uint<128> > str_GetAggout2("str_GetAggout2");
#pragma HLS RESOURCE variable = str_GetAggout2 core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetAggout2 depth = 128
    const int LG2_NUM_HASH = 17;
    const int LG2_BITS_MEM = 10;
    SameColor_GetKins_HashAgg_big<DF_V_T, DF_W_T, LG2_NUM_HASH, LG2_BITS_MEM>(scl, str_GetCout2, str_GetAggout2);

    SameColor_GetKins_hash_Collector(str_GetAggout0, str_GetAggout1, str_GetAggout2, str_Aggout);
}

} // graph
} // xf
#endif
