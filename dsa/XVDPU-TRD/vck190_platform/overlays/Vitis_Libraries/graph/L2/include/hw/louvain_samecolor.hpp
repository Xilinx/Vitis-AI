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

#ifndef _LOUVAIN_SAMECOLOR_H_
#define _LOUVAIN_SAMECOLOR_H_

#include "louvain_modularity.hpp"
#include "louvain_samecolor_hash_agg.hpp"
#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_math.h>
#include <hls_stream.h>

//#define _DEBUG_SAMECOLOR
//#define _DEBUG_GetV 1
//#define _DEBUG_PUSHV 1
int pushcnt = 0;
#define USE_PUSH (1)

#ifdef _DEBUG_SAMECOLOR
#define _DEBUG_GetV
#define _DEBUG_GetE
#define _DEBUG_GetC
#define _DEBUG_SMALL
#define _DEBUG_CID1
#define _DEBUG_CID2
#define _DEBUG_GAIN
#endif
namespace xf {
namespace graph {

template <int CSRWIDTH, int COLORWIDTH>
void _new_bus(bool use_push_flag,
              ap_uint<FLAGW>* flag,
              int coloradj1,
              int coloradj2,
              ap_uint<COLORWIDTH>* colorInx,
              ap_uint<CSRWIDTH>* offset,
              hls::stream<ap_uint<96> >& str_GetVout) {
    StrmBus_M out_v;
    ap_uint<96> dinout;

    AxiMap<int, COLORWIDTH> axi_colorInx(colorInx);
    AxiMap<int, CSRWIDTH> axi_offset(offset);
GET_V:
    for (int k = coloradj1; k < coloradj2; k++) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 10000000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = colorInx inter false
#pragma HLS DEPENDENCE variable = offset inter false
        DF_V_T v = axi_colorInx.rdi(k);

        if (use_push_flag) {
            ap_uint<FLAGW> push_flag = flag[v];

            if (push_flag) {
                int adj1 = AxiRead<8, 5>(offset, v); // axi_offset.rdi(v);

                if (adj1 < 0) continue;

                adj1 = 0x7fffffff & adj1;
                int adj2 = 0x7fffffff & AxiRead<8, 5>(offset, v + 1);
                DF_D_T dgr = adj2 - adj1;
                out_v.vcd.set(v, adj1, dgr);
                out_v.vcd.get(dinout);
                str_GetVout.write(dinout);

#ifndef __SYNTHESIS__
#ifdef _DEBUG_PUSHV
                printf("pushv = %d\n", v);
#endif
#ifdef _DEBUG_GetV
                printf("Get_pushV push_flag=%d: k=%d\t v=%d\t adj1=%d\t adj1=%d\t dgr=%d\t \n", (int)push_flag, k, v,
                       adj1, adj2, dgr);
#endif
#endif

            } // push_flag == true

        } else {
            int adj1 = AxiRead<8, 5>(offset, v); // axi_offset.rdi(v);
            if (adj1 < 0) continue;

            adj1 = 0x7fffffff & adj1;
            int adj2 = 0x7fffffff & AxiRead<8, 5>(offset, v + 1);
            DF_D_T dgr = adj2 - adj1;
            out_v.vcd.set(v, adj1, dgr);
            out_v.vcd.get(dinout);
            str_GetVout.write(dinout);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GetV
            printf("GetV: k=%d\t v=%d\t adj1=%d\t adj1=%d\t dgr=%d\t \n", k, v, adj1, adj2, dgr);
#endif
#endif
        } // push
    }
    out_v.vcd.set(dinout, 0, 0, -1);
    str_GetVout.write(dinout);
}

template <int CSRWIDTH>
void SameColor_GetFlag2(hls::stream<int>& str_pushV,
                        ap_uint<CSRWIDTH>* offset,
                        ap_uint<CSRWIDTH>* index,
                        hls::stream<int>& str_flag) {
    int pushV = 0;
    DF_D_T d, e_dgr;
    e_dgr = 0, d = 0;
    AxiMap<int, CSRWIDTH> axi_index(index);
    int adj1, adj2;

    while (pushV >= 0) {
#pragma HLS PIPELINE II = 1
        if (d == e_dgr) {
            str_pushV.read(pushV);
            if (pushV >= 0) {
                adj1 = AxiRead<8, 5>(offset, pushV); // axi_offset.rdi(v);
                adj2 = AxiRead<8, 5>(offset, pushV + 1);
                e_dgr = adj2 - adj1;
            }
            d = 0;
        } else {
            DF_V_T edge = axi_index.rdi(adj1 + d);
            str_flag.write(edge);
            d++;
        }
    }

    str_flag.write(-1);
}

template <int CSRWIDTH>
void SameColor_GetFlag3(hls::stream<int>& str_pushV, ap_uint<CSRWIDTH>* offset, hls::stream<ap_uint<96> >& str_od) {
    int pushV = 0;
    DF_D_T e_dgr;
    int adj1, adj2;
    StrmBus_M out_v;
    ap_uint<96> dinout;
    int cnt = 0;

GetFlag3:
    while (pushV >= 0) {
#pragma HLS PIPELINE II = 1
        str_pushV.read(pushV);
        if (pushV >= 0) {
            adj1 = AxiRead<8, 5>(offset, pushV); // axi_offset.rdi(v);
            adj2 = AxiRead<8, 5>(offset, pushV + 1);
            e_dgr = adj2 - adj1;
            out_v.vod.set(pushV, adj1, e_dgr);
            out_v.vod.get(dinout);
            str_od.write(dinout);
#ifndef __SYNTHESIS__
            cnt++;
#ifdef _DEBUG_PUSHV
            printf("PUSHV : v=%d\t adj1=%d\t adj2=%d\t e_dgr=%d\t \n", pushV, adj1, adj2, e_dgr);
#endif
#endif
        } else {
            out_v.vod.set(dinout, 0, 0, -1);
            str_od.write(dinout);
#ifndef __SYNTHESIS__
            pushcnt += cnt;
// printf("PUSHV : cnt=%d\n", cnt);
#endif
        }
    }
}

template <int CSRWIDTH>
void SameColor_GetFlag4(bool wirte_push_flag,
                        hls::stream<ap_uint<96> >& str_od,
                        ap_uint<CSRWIDTH>* index,
                        hls::stream<int>& str_flag) {
    int pushV = 0;
    DF_D_T d, e_dgr;
    e_dgr = 0, d = 0;
    AxiMap<int, CSRWIDTH> axi_index(index);
    int adj1, adj2;
    StrmBus_M out_v;
    ap_uint<96> dinout;

GetFlag4:
    while (e_dgr >= 0) {
#pragma HLS PIPELINE II = 1
        if (d == e_dgr) {
            str_od.read(dinout);
            out_v.vod.get(dinout, pushV, adj1, e_dgr);
            d = 0;
        } else {
            DF_V_T edge = axi_index.rdi(adj1 + d);
            if (wirte_push_flag) str_flag.write(edge);
            d++;
        }
    }

    str_flag.write(-1);
}

template <int CSRWIDTH>
void SameColor_GetE(ap_uint<CSRWIDTH>* index,
                    ap_uint<CSRWIDTH>* weight,
                    hls::stream<ap_uint<96> >& str_GetVout,
                    hls::stream<ap_uint<96> >& str_GetEout) {
    DF_D_T e_dgr, d;
    DF_V_T v, off;
    StrmBus_M u_vod;
    StrmBus_M u_vd_ew;
    ap_uint<96> dinout;
    e_dgr = 0, d = 0;

    AxiMap<int, CSRWIDTH> axi_index(index);
    AxiMap<float, CSRWIDTH> axi_weight(weight);

GET_E:
    while (e_dgr >= 0) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 10000000
#pragma HLS pipeline II = 1
        if ((d + 0) == e_dgr) {
            str_GetVout.read(dinout);
            u_vod.vod.get(dinout, v, off, e_dgr);
            u_vd_ew.vd.set(dinout, v, e_dgr);
            str_GetEout.write(dinout);
            d = 0;
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GetE
            printf("GetE: d=%d\t v=%d,\t off=%d\t e_dgr=%d\t \n", d, v, off, e_dgr);
#endif
#endif
        } else {
            DF_V_T edge = axi_index.rdi(off + d);
            DF_W_T wght = axi_weight.rdf(off + d);
            u_vd_ew.ew.set(dinout, edge, wght);
            str_GetEout.write(dinout);
            d++;
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GetE
            printf("GetE: d=%d\t v=%d,\t off=%d\t edge=%d\t wght=%f\n", d, v, off, edge, wght);
#endif
#endif
        }
    }
}

template <int DWIDTH>
void SameColor_GetC(ap_uint<DWIDTH>* cidPrev,
                    hls::stream<ap_uint<96> >& str_GetEout,
                    hls::stream<ap_uint<128> >& str_GetCout) {
    DF_D_T e_dgr, d;
    DF_V_T v, vCid;
    DF_V_T edge, eCid;
    DF_W_T deg_w;
    StrmBus_M u_e;
    StrmBus_L u_c;
    ap_uint<96> din96;
    ap_uint<128> dout128;

    AxiMap<int, DWIDTH> axi_cidPrev(cidPrev);
    e_dgr = 0;
    d = 0;
GET_C:
    while (e_dgr >= 0) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 10000000
        if ((d + 0) == e_dgr) {
            str_GetEout.read(din96);
            u_e.vd.get(din96, v, e_dgr);
            if (e_dgr >= 0) {
                vCid = axi_cidPrev.rdi(v);
                u_c.vcde.set(dout128, v, vCid, e_dgr, false);
                str_GetCout.write(dout128);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GetC
                printf("GetC: d=%d\t v=%d,\t e_dgr=%d\t\n", d, v, e_dgr);
#endif
#endif
            }
            d = 0;
        } else {
            str_GetEout.read(din96);
            u_e.ew.get(din96, edge, deg_w);
            eCid = axi_cidPrev.rdi(edge);
            u_c.ecw.set(dout128, edge, eCid, deg_w);
            str_GetCout.write(dout128);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GetC
            printf("GetC: d=%d\t v=%d,\t e_dgr=%d\t edge=%d\t eCid=%d\n", d, v, e_dgr, edge, eCid);
#endif
#endif
            d++;
        }
    }
    u_c.vcde.set(dout128, 0, 0, 0, true);
    str_GetCout.write(dout128);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <int DWIDTH>
void SameColor_GetBest_Update_Gain_bus3(short scl,
                                        hls::stream<ap_uint<128> >& str_Aggout,
                                        DF_W_T constant_recip,
                                        ap_uint<DWIDTH>* totPrev,
                                        ap_uint<DWIDTH>* commWeight,
                                        // output
                                        hls::stream<ap_uint<160> >& str_Gainout) {
    double constant = (1.0 / constant_recip);
    long long constant_i = (long long)constant;
    int scl_2 = 10;

    AxiMap<float, DWIDTH> axi_totPrev(totPrev);
    AxiMap<float, DWIDTH> axi_commWeight(commWeight);

    DF_WI_T constant_i_scl = ToInt(constant, 0);
    DF_V_T v, vCid, target, best_comm;
    DF_W_T selfloop, ki, ei_u, a_u, best_gain;
    DF_WI_T selfloop_i, ki_i, ei_u_i, a_u_i;
    DF_WI_T best_gain_i;
    DF_D_T degree, numComm, d;
    DF_W_T kinCk_0_;
    DF_V_T Ck_0;
    bool GetKins_e;
    StrmBus_L vcdn, kisf, ckkin;
    StrmBus_M vcn, m_ki;
    StrmBus_XL vcnki, cgain, vcnki_tmp;
    ap_uint<128> dinout;
    ap_uint<160> dout;

    GetKins_e = false;
    d = 0;
    numComm = 0;
GET_BEST:
    while (GetKins_e == false) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 1000000

#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totPrev inter false
#pragma HLS DEPENDENCE variable = commWeight inter false
        if (d == numComm) {
            str_Aggout.read(dinout);
            vcdn.vcdn.set(dinout);
            vcdn.vcdn.get(v, vCid, degree, numComm);
            if (degree < 0) numComm = -1;
            vcn.vcn.set(v, vCid, numComm);
            vcnki.vcnk.set(v, vCid, numComm, 0.0);
            GetKins_e = degree < 0;
            if (GetKins_e == false) {
                str_Aggout.read(dinout);
                kisf.kisf.set(dinout);
                selfloop = degree > 0 ? kisf.kisf.self : 0;
                ki = degree > 0 ? kisf.kisf.ki : 0;
                vcnki.vcnk.ki = degree > 0 ? kisf.kisf.ki : 0;
                d = 0;
                ki_i = ToInt(ki, scl_2);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                printf("GAIN: v=%d\t, vCid=%d\t, degree=%d\t, numComm=%d\t, ki=%f\t, selfloop=%f\t \n", v, vCid, degree,
                       numComm, ki, selfloop);
#endif
#endif
            }
            vcnki.vcnk.get(dout);
            vcnki_tmp.vcnk.set(dout);
            str_Gainout.write(dout);
        } else if (GetKins_e == false) {
            if (degree > 0) {
                if (d == 0) {
                    str_Aggout.read(dinout);
                    ckkin.n_ckk.set(dinout);
                    kinCk_0_ = ckkin.n_ckk.kin;
                    Ck_0 = ckkin.n_ckk.ck;
                    best_gain = 0;
                    best_comm = vCid;
                    // commWeight[v]    = kinCk_0_;
                    axi_commWeight.wr(v, (float)kinCk_0_);
                    ei_u = kinCk_0_ - selfloop;
                    // a_u              = totPrev[vCid]-ki;
                    a_u = axi_totPrev.rdf(vCid) - ki;
                    best_gain_i = 0;
                    ei_u_i = ToInt(ei_u, scl_2 * 2);
                    a_u_i = ToInt(a_u, scl_2);
                    d = 1;
                    cgain.cgi.set(vCid, best_gain_i);
                    cgain.cgi.get(dout);
                    str_Gainout.write(dout);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                    printf("GAIN: vCid=%d\t, best_gain_i%x\t \n", vCid, best_gain_i);
#endif
#endif
                } else if (d < numComm) {
                    str_Aggout.read(dinout);
                    ckkin.n_ckk.set(dinout);
                    DF_V_T eCid = ckkin.n_ckk.ck;
                    DF_W_T ei_v = ckkin.n_ckk.kin;
                    // DF_W_T a_v              = totPrev[eCid];
                    DF_W_T a_v = axi_totPrev.rdf(eCid);
                    ;
                    DF_W_T diff_e = (ei_v - ei_u);
                    DF_W_T diff_a = (a_v - a_u);
                    DF_W_T v_pos = diff_e * constant;
                    DF_W_T v_neg = diff_a * ki;
                    DF_W_T curr_gain_org = v_pos - v_neg;
                    DF_WI_T ei_v_i = ToInt(ei_v, scl_2 * 2);
                    DF_WI_T a_v_i = ToInt(a_v, scl_2);
                    DF_WI_T diff_e_i = (ei_v_i - ei_u_i);
                    DF_WI_T diff_a_i = (a_v_i - a_u_i);
                    DF_WI_T v_pos_i = diff_e_i * constant_i_scl;
                    DF_WI_T v_neg_i = diff_a_i * ki_i;
                    DF_WI_T curr_gain_i = v_pos_i - v_neg_i;
                    DF_W_T curr_gain = (curr_gain_i >> (scl_2 * 2));
                    cgain.cgi.set(eCid, curr_gain_i);
                    cgain.cgi.get(dout);
                    str_Gainout.write(dout);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                    printf("GAIN: d=%d\t, eCid=%d\t, curr_gain_i%x\t \n", d, eCid, curr_gain_i);
#endif
#endif
                    d++;
                }
            }
        }
    }
}

template <int DWIDTH>
void SameColor_GetBest_Update_Gain_step2(ap_uint<DWIDTH>* cidCcommWeighturr, hls::stream<StrmBus_S>& str_GainUpdate) {
    SameColor_GetBest_Update_CID_step2(cidCcommWeighturr, str_GainUpdate);
}
template <int DWIDTH>
void SameColor_GetBest_Update_Gain_step1(short scl,
                                         hls::stream<ap_uint<128> >& str_Aggout,
                                         DF_W_T constant_recip,
                                         ap_uint<DWIDTH>* totPrev,
                                         // output
                                         hls::stream<StrmBus_S>& str_GainUpdate,
                                         hls::stream<ap_uint<160> >& str_Gainout) {
    double constant = (1.0 / constant_recip);
    long long constant_i = (long long)constant;
    int scl_2 = 10;

    AxiMap<float, DWIDTH> axi_totPrev(totPrev);
    // AxiMap<float, DWIDTH> axi_commWeight(commWeight);
    StrmBus_S vf;

    DF_WI_T constant_i_scl = ToInt(constant, 0);
    DF_V_T v, vCid, target, best_comm;
    DF_W_T selfloop, ki, ei_u, a_u, best_gain;
    DF_WI_T selfloop_i, ki_i, ei_u_i, a_u_i;
    DF_WI_T best_gain_i;
    DF_D_T degree, numComm, d;
    DF_W_T kinCk_0_;
    DF_V_T Ck_0;
    bool GetKins_e;
    StrmBus_L vcdn, kisf, ckkin;
    StrmBus_M vcn, m_ki;
    StrmBus_XL vcnki, cgain, vcnki_tmp;
    ap_uint<128> dinout;
    ap_uint<160> dout;

    GetKins_e = false;
    d = 0;
    numComm = 0;
GET_BEST:
    while (GetKins_e == false) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 1000000

#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totPrev inter false
        if (d == numComm) {
            str_Aggout.read(dinout);
            vcdn.vcdn.set(dinout);
            vcdn.vcdn.get(v, vCid, degree, numComm);
            if (degree < 0) numComm = -1;
            vcn.vcn.set(v, vCid, numComm);
            vcnki.vcnk.set(v, vCid, numComm, 0.0);
            GetKins_e = degree < 0;
            if (GetKins_e == false) {
                str_Aggout.read(dinout);
                kisf.kisf.set(dinout);
                selfloop = degree > 0 ? kisf.kisf.self : 0;
                ki = degree > 0 ? kisf.kisf.ki : 0;
                vcnki.vcnk.ki = degree > 0 ? kisf.kisf.ki : 0;
                d = 0;
                ki_i = ToInt(ki, scl_2);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                printf("GAIN: v=%d\t, vCid=%d\t, degree=%d\t, numComm=%d\t, ki=%f\t, selfloop=%f\t \n", v, vCid, degree,
                       numComm, ki, selfloop);
#endif
#endif
            }
            vcnki.vcnk.get(dout);
            vcnki_tmp.vcnk.set(dout);
            str_Gainout.write(dout);
        }

        if (GetKins_e == false)
            if (degree > 0) {
                if (d == 0) {
                    str_Aggout.read(dinout);
                    ckkin.n_ckk.set(dinout);
                    kinCk_0_ = ckkin.n_ckk.kin;
                    Ck_0 = ckkin.n_ckk.ck;
                    best_gain = 0;
                    best_comm = vCid;
                    // commWeight[v]    = kinCk_0_;
                    vf.vf.set(v, (float)kinCk_0_);
                    str_GainUpdate.write(vf);
                    // axi_commWeight.wr(v, (float)kinCk_0_);
                    ei_u = kinCk_0_ - selfloop;
                    // a_u              = totPrev[vCid]-ki;
                    a_u = axi_totPrev.rdf(vCid) - ki;
                    best_gain_i = 0;
                    ei_u_i = ToInt(ei_u, scl_2 * 2);
                    a_u_i = ToInt(a_u, scl_2);
                    d = 1;
                    cgain.cgi.set(vCid, best_gain_i);
                    cgain.cgi.get(dout);
                    str_Gainout.write(dout);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                    printf("GAIN: vCid=%d\t, best_gain_i%x\t \n", vCid, best_gain_i);
#endif
#endif
                } else if (d < numComm) {
                    str_Aggout.read(dinout);
                    ckkin.n_ckk.set(dinout);
                    DF_V_T eCid = ckkin.n_ckk.ck;
                    DF_W_T ei_v = ckkin.n_ckk.kin;
                    // DF_W_T a_v              = totPrev[eCid];
                    DF_W_T a_v = axi_totPrev.rdf(eCid);
                    ;
                    DF_W_T diff_e = (ei_v - ei_u);
                    DF_W_T diff_a = (a_v - a_u);
                    DF_W_T v_pos = diff_e * constant;
                    DF_W_T v_neg = diff_a * ki;
                    DF_W_T curr_gain_org = v_pos - v_neg;
                    DF_WI_T ei_v_i = ToInt(ei_v, scl_2 * 2);
                    DF_WI_T a_v_i = ToInt(a_v, scl_2);
                    DF_WI_T diff_e_i = (ei_v_i - ei_u_i);
                    DF_WI_T diff_a_i = (a_v_i - a_u_i);
                    DF_WI_T v_pos_i = diff_e_i * constant_i_scl;
                    DF_WI_T v_neg_i = diff_a_i * ki_i;
                    DF_WI_T curr_gain_i = v_pos_i - v_neg_i;
                    DF_W_T curr_gain = (curr_gain_i >> (scl_2 * 2));
                    cgain.cgi.set(eCid, curr_gain_i);
                    cgain.cgi.get(dout);
                    str_Gainout.write(dout);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_GAIN
                    printf("GAIN: d=%d\t, eCid=%d\t, curr_gain_i%x\t \n", d, eCid, curr_gain_i);
#endif
#endif
                    d++;
                }
            }
    }
    vf.vf.set(-1, -1);
    str_GainUpdate.write(vf);
}

template <int DWIDTH>
void SameColor_GetBest_Update_TOT_Csize_hash3(short scl,
                                              int& moves,
                                              ap_uint<DWIDTH>* cUpdateSize,
                                              ap_uint<DWIDTH>* totUpdate,
                                              hls::stream<StrmBus_L>& str_Cidout) {
    DF_V_T target, vCid;
    DF_W_T ki;
    StrmBus_L ctk;
    target = 0;
    DF_V_T mem_key[NUM_SMALL];
#pragma HLS ARRAY_PARTITION variable = mem_key dim = 1
    AggRAM<DF_WI_T, NUM_SMALL_LOG, NUM_SMALL> mem_agg;
    AggRAM<DF_D_T, NUM_SMALL_LOG, NUM_SMALL> mem_cnt;
    DF_D_T num_cid_small = 0;
    bool isFirstAddr = true;

    AxiMap<DF_V_T, DWIDTH> axi_cUpdateSize(cUpdateSize);
    AxiMap<float, DWIDTH> axi_totUpdate(totUpdate);

MDF_TOT_SIZE:
    while (target != -2) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 1000000
#pragma HLS PIPELINE II = 1
    AGG_BATCH:
        do {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totUpdate inter false
#pragma HLS DEPENDENCE variable = cUpdateSize inter false
#pragma HLS DEPENDENCE variable = mem_agg.mem inter false
#pragma HLS DEPENDENCE variable = mem_cnt.mem inter false
            str_Cidout.read(ctk);
            ctk.ctk.get(vCid, target, ki);
            if (target != -2) {
                if (target != vCid && target != -1) {
                    moves++;

                    DF_WI_T val = ToInt(ki, scl);
                    ap_uint<NUM_SMALL_LOG> addr;

                    addr = MemAgg_core_key(target, num_cid_small, mem_key);
                    mem_agg.Aggregate(addr, val, isFirstAddr);
                    mem_cnt.Aggregate(addr, 1, isFirstAddr);

                    float tmptot = axi_totUpdate.rdf(vCid); //(float)totUpdate[vCid];
                    tmptot -= ki;
                    // totUpdate  [vCid] = tmptot;
                    axi_totUpdate.wr(vCid, tmptot);

                    int tmpcSize = axi_cUpdateSize.rdi(vCid); // cUpdateSize[vCid];
                    tmpcSize--;
                    // cUpdateSize[vCid] = tmpcSize;
                    axi_cUpdateSize.wr(vCid, tmpcSize);
                }
            }
        } while (!(num_cid_small == 64 || target == -2));

        if (num_cid_small == 64 || (num_cid_small > 0 && target == -2)) {
        AGG_BATCH2MEM:
            for (int i = 0; i < num_cid_small; i++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totUpdate inter false
#pragma HLS DEPENDENCE variable = cUpdateSize inter false

                DF_V_T addr_target = mem_key[i];

                ki = ToDouble(mem_agg.mem[i], scl);
                float tmptot = axi_totUpdate.rdf(addr_target); //(float)totUpdate[addr_target];
                tmptot += ki;
                // totUpdate  [addr_target] = tmptot;
                axi_totUpdate.wr(addr_target, tmptot);

                int tmpcSize = axi_cUpdateSize.rdi(addr_target); // cUpdateSize[addr_target];
                tmpcSize += mem_cnt.mem[i];
                // cUpdateSize[addr_target] = tmpcSize;
                axi_cUpdateSize.wr(addr_target, tmpcSize);

                mem_agg.mem[i] = 0;
                mem_cnt.mem[i] = 0;
            }
            num_cid_small = 0;
            isFirstAddr = true;
        }
    }
}

template <int DWIDTH>
void SameColor_GetBest_Update_CID_step1(ap_uint<DWIDTH>* cidSize,
                                        hls::stream<ap_uint<160> >& str_Gainout,
                                        // output
                                        hls::stream<StrmBus_L>& str_Cidout,
                                        hls::stream<int>& str_pushV,
                                        hls::stream<StrmBus_S>& str_Cid_update) {
    AxiMap<int, DWIDTH> axi_cidSize(cidSize);

    DF_V_T v, vCid, eCid, best_comm, target;
    DF_D_T numCandi, d;
    DF_WI_T best_gain_i, curr_gain_i;
    DF_W_T ki;
    StrmBus_XL vcnki, cgain;
    StrmBus_L ctk;
    StrmBus_S vc;
    ap_uint<160> dinout;
    str_Gainout.read(dinout);
    vcnki.vcnk.get(dinout, v, vCid, numCandi, ki);
    d = 0;
    best_gain_i = 0;
UPDATE_CID:
    while (numCandi >= 0) {
#pragma HLS PIPELINE II = 1
        if (numCandi == 0) {
            best_comm = -1;
        } else { // numCandi
            str_Gainout.read(dinout);
            cgain.cgi.set(dinout);
            cgain.cgi.get(eCid, curr_gain_i);
            if ((d == 0) || (curr_gain_i > best_gain_i) ||
                ((curr_gain_i == best_gain_i) && (curr_gain_i > 0) && (eCid < best_comm))) {
                best_gain_i = curr_gain_i;
                best_comm = eCid;
            }
        }
        if (numCandi == 0) {
            vc.vc.set(v, -1);
            str_Cid_update.write(vc);
        } else if (d == numCandi - 1) {
            int size_best = axi_cidSize.rdi(best_comm);
            int size_vCid = axi_cidSize.rdi(vCid);
            if (size_best == 1 && size_vCid == 1 && best_comm > vCid)
                target = vCid;
            else
                target = best_comm;
#ifndef __SYNTHESIS__
#ifdef _DEBUG_CID1
            printf("CID1: size_best=%d\t, s_v=%d\t, best=%d\t, vCid=%d\t target=%d \n", size_best, size_vCid, best_comm,
                   vCid, target);
#endif
#endif

            if (target != vCid) {
                vc.vc.set(v, target);
                str_Cid_update.write(vc);
                ctk.ctk.set(vCid, target, ki);
                str_Cidout.write(ctk);
                str_pushV.write(v);
            }
            str_Gainout.read(dinout);
            vcnki.vcnk.set(dinout);
            vcnki.vcnk.get(v, vCid, numCandi, ki);
            d = 0;
            best_gain_i = 0;
        } else
            d++;

    } // while
    ctk.ctk.set(vCid, -2, ki);
    str_Cidout.write(ctk);
    str_pushV.write(-1);
    vc.vc.set(-1, 0);
    str_Cid_update.write(vc);
}

template <int DWIDTH>
void SameColor_GetBest_Update_CID_step2(ap_uint<DWIDTH>* cidCurr, hls::stream<StrmBus_S>& str_Cid_update) {
    AxiMap<int, DWIDTH> axi_cidCurr(cidCurr);
    const int bits_unit = 32;
    const int size_all = DWIDTH;
    const int num_unit = size_all / bits_unit;  // 8
    const int size_batch = size_all / num_unit; // 32

    int num_cid[num_unit];
    for (int i = 0; i < num_unit; i++) {
        num_cid[i] = 0;
    }

    int cnt_all = 0;
    StrmBus_S mem_vc[size_all];
    StrmBus_S vc;
    bool isEnd = false;
    bool isOneFull = false;
    while (!isEnd) {
    UPDATE_CID_REC:
        while (!isEnd && !isOneFull) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = mem_vc inter false
            str_Cid_update.read(vc);
            isEnd = vc.vc.v < 0;
            if (!isEnd) {
                int off = vc.vc.v & (num_unit - 1);
                int addr = off * size_batch + num_cid[off];
                mem_vc[addr].vc.set(vc.vc.v, vc.vc.vCid);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_CID2
                printf("CID2: cnt_all = %d\t%d\t%d\n", cnt_all, vc.vc.v, vc.vc.vCid);
#endif
#endif
                num_cid[off]++;
                if (num_cid[off] == size_batch) isOneFull = true;
                cnt_all++;
            }
        }

    UPDATE_CID_WR:
        for (int off = 0; off < num_unit; off++) {
        UPDATE_CID_WR_BATCH:
            for (int j = 0; j < num_cid[off]; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = cidCurr inter false
                int addr = off * size_batch + j; // num_cid[off];
                vc.vc.set(mem_vc[addr].vc.v, mem_vc[addr].vc.vCid);
                axi_cidCurr.wr(vc.vc.v, vc.vc.vCid);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_CID2
                printf("CID2: off=%d\t j=%d\t %d\t%d\n", off, j, vc.vc.v, vc.vc.vCid);
#endif
#endif
            }
            num_cid[off] = 0;
        }
        isOneFull = false;
        cnt_all = 0;
    }
}

template <int DWIDTH>
void SameColor_GetBest_Update_CID_step2_org(ap_uint<DWIDTH>* cidCurr, hls::stream<StrmBus_S>& str_Cid_update) {
    AxiMap<int, DWIDTH> axi_cidCurr(cidCurr);
    const int num_unit = 8;
    const int size_all = 256;
    const int size_batch = size_all / num_unit; // 32
    int num_cid[num_unit];
    for (int i = 0; i < num_unit; i++) {
        num_cid[i] = 0;
    }
    int cnt_all = 0;
    StrmBus_S mem_vc[size_all];
    StrmBus_S vc;
    bool isEnd = false;
    bool isOneFull = false;
    while (!isEnd) {
    UPDATE_CID_REC:
        while (!isEnd && !isOneFull) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = mem_vc inter false
            str_Cid_update.read(vc);
            isEnd = vc.vc.v < 0;
            if (!isEnd) {
                int off = vc.vc.v & (num_unit - 1);
                int addr = off * size_batch + num_cid[off];
                mem_vc[addr].vc.set(vc.vc.v, vc.vc.vCid);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_CID2
                printf("CID2: cnt_all = %d\t%d\t%d\n", cnt_all, vc.vc.v, vc.vc.vCid);
#endif
#endif
                num_cid[off]++;
                if (num_cid[off] == size_batch) isOneFull = true;
                cnt_all++;
            }
        }
        int i = 0;
    UPDATE_CID_WR:
        for (int off = 0; off < num_unit; off++) {
        UPDATE_CID_WR_BATCH:
            for (int j = 0; j < num_cid[off]; j++) {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = cidCurr inter false
                int addr = off * size_batch + j; // num_cid[off];
                vc.vc.set(mem_vc[addr].vc.v, mem_vc[addr].vc.vCid);
                axi_cidCurr.wr(vc.vc.v, vc.vc.vCid);
#ifndef __SYNTHESIS__
#ifdef _DEBUG_CID2
                printf("CID2: off=%d\t j=%d\t %d\t%d\n", off, j, vc.vc.v, vc.vc.vCid);
#endif
#endif
                i++;
                ;
            }
            num_cid[off] = 0;
        }
        isOneFull = false;
        cnt_all = 0;
    }
}

#define WIDTH_UNIT_LOG (5)
#define NUM_UNIT (8)     //(WIDTH_AXI/WIDTH_UNIT )   //8
#define NUM_UNIT_LOG (3) //(WIDTH_AXI_LOG - WIDTH_UNIT_LOG)
#define BATCH_UPDT (32)
#define BATCH_UPDT_LOG (5)
template <int DWIDTH>
void SameColor_GetBest_Update_TOT_Csize_hash3_new(short scl,
                                                  int& moves,
                                                  ap_uint<DWIDTH>* cUpdateSize,
                                                  ap_uint<DWIDTH>* totUpdate,
                                                  hls::stream<StrmBus_L>& str_Cidout) {
    DF_V_T target, vCid;
    DF_W_T ki;
    StrmBus_L ctk;
    target = 0;

    DF_V_T mem_key[NUM_UNIT][BATCH_UPDT];
#pragma HLS ARRAY_PARTITION variable = mem_key dim = 2
    AggRAM_base<DF_WI_T, BATCH_UPDT_LOG, BATCH_UPDT> mem_agg[NUM_UNIT];
#pragma HLS ARRAY_PARTITION variable = mem_agg complete
    AggRAM_base<DF_D_T, BATCH_UPDT_LOG, BATCH_UPDT> mem_cnt[NUM_UNIT];
#pragma HLS ARRAY_PARTITION variable = mem_cnt complete
    DF_D_T num_cid_small[NUM_UNIT];
#pragma HLS ARRAY_PARTITION variable = num_cid_small complete
    bool isFirstAddr[NUM_UNIT];
#pragma HLS ARRAY_PARTITION variable = isFirstAddr complete

    for (int i = 0; i < NUM_UNIT; i++) {
        num_cid_small[i] = 0;
        isFirstAddr[i] = true;
        AggRAM_base_init(&mem_agg[i]);
        AggRAM_base_init(&mem_cnt[i]);
    }
    bool isMoved = false;
    bool isOneFull = false;

    AxiMap<DF_V_T, DWIDTH> axi_cUpdateSize(cUpdateSize);
    AxiMap<float, DWIDTH> axi_totUpdate(totUpdate);

MDF_TOT_SIZE:
    while (target != -2) {
#pragma HLS LOOP_TRIPCOUNT MIN = 1 MAX = 1000000
#pragma HLS PIPELINE off
    AGG_BATCH:
        do {
#pragma HLS PIPELINE II = 1
#pragma HLS DEPENDENCE variable = totUpdate inter false
#pragma HLS DEPENDENCE variable = cUpdateSize inter false
            str_Cidout.read(ctk);
            ctk.ctk.get(vCid, target, ki);
            if (target != -2) {
                if (target != vCid && target != -1) {
                    moves++;

                    DF_WI_T val = ToInt(ki, scl);
                    ap_uint<NUM_SMALL_LOG> addr;
                    ap_uint<NUM_UNIT_LOG> off;

                    off = target % NUM_UNIT;
                    addr = MemAgg_core_key_T<BATCH_UPDT_LOG, BATCH_UPDT>(target, num_cid_small[off], mem_key[off]);
                    AggRAM_base_Aggregate<DF_WI_T, BATCH_UPDT_LOG, BATCH_UPDT>(&mem_agg[off], addr, val,
                                                                               isFirstAddr[off]);
                    AggRAM_base_Aggregate<DF_D_T, BATCH_UPDT_LOG, BATCH_UPDT>(&mem_cnt[off], addr, 1, isFirstAddr[off]);
                    if (num_cid_small[off] > (BATCH_UPDT - 2)) isOneFull = true;

                    off = vCid % NUM_UNIT;
                    addr = MemAgg_core_key_T<BATCH_UPDT_LOG, BATCH_UPDT>(vCid, num_cid_small[off], mem_key[off]);
                    AggRAM_base_Aggregate<DF_WI_T, BATCH_UPDT_LOG, BATCH_UPDT>(&mem_agg[off], addr, -val,
                                                                               isFirstAddr[off]);
                    AggRAM_base_Aggregate<DF_D_T, BATCH_UPDT_LOG, BATCH_UPDT>(&mem_cnt[off], addr, -1,
                                                                              isFirstAddr[off]);
                    isMoved = true;
                    if (num_cid_small[off] > (BATCH_UPDT - 2)) isOneFull = true;
                }
            }
        } while (!(isOneFull || target == -2));

        if (isMoved) {
        AGG_BATCH2MEM:
            for (int off = 0; off < NUM_UNIT; off++) {
#pragma HLS PIPELINE off

                AggBatch2Mem<DWIDTH, BATCH_UPDT, BATCH_UPDT_LOG>(scl, cUpdateSize, totUpdate, mem_key[off],
                                                                 mem_agg[off], mem_cnt[off], num_cid_small[off],
                                                                 isFirstAddr[off]);
            }
            isMoved = false;
            isOneFull = false;
        }
    }
}

template <int DWIDTH>
void SameColor_GetFlag(hls::stream<int>& str_pushV, ap_uint<8>* flagUpdate) {
    int pushV = 0;
    str_pushV.read(pushV);

GetFlag_writeout:
    while (pushV >= 0) {
#pragma HLS PIPELINE II = 1
        flagUpdate[pushV] = 1;
        str_pushV.read(pushV);
    }
}

template <int DWIDTH, int CSRWIDTH, int COLORWIDTH>
void SameColor_dataflow(int numVertex,
                        int coloradj1,
                        int coloradj2,
                        int& moves,
                        short scl,
                        long long total_w_i,
                        bool use_push_flag,
                        bool wirte_push_flag,
                        DF_W_T& constant_recip,
                        ap_uint<DWIDTH>* offset,
                        ap_uint<DWIDTH>* index,
                        ap_uint<DWIDTH>* weight,
                        ap_uint<COLORWIDTH>* colorInx,
                        ap_uint<DWIDTH>* cidPrev,
                        ap_uint<DWIDTH>* cidSize,
                        ap_uint<DWIDTH>* totPrev,
                        ap_uint<DWIDTH>* cidCurr,
                        ap_uint<DWIDTH>* cUpdateSize,
                        ap_uint<DWIDTH>* totUpdate,
                        ap_uint<DWIDTH>* commWeight,
                        ap_uint<CSRWIDTH>* offsetDup,
                        ap_uint<CSRWIDTH>* indexDup,
                        ap_uint<8>* flag,
                        ap_uint<8>* flagUpdate) {
#pragma HLS dataflow

    hls::stream<ap_uint<96> > str_GetVout("str_GetVout");
#pragma HLS RESOURCE variable = str_GetVout core = FIFO_SRL
#pragma HLS STREAM variable = str_GetVout depth = 256
    _new_bus<CSRWIDTH, COLORWIDTH>(use_push_flag, flag, coloradj1, coloradj2, colorInx, offset, str_GetVout);

    hls::stream<ap_uint<96> > str_GetEout("str_GetEout");
#pragma HLS RESOURCE variable = str_GetEout core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetEout depth = 256
    SameColor_GetE<CSRWIDTH>(index, weight, str_GetVout, str_GetEout);

    hls::stream<ap_uint<128> > str_GetCout("str_GetCout");
#pragma HLS RESOURCE variable = str_GetCout core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_GetCout depth = 256
    SameColor_GetC<DWIDTH>(cidPrev, str_GetEout, str_GetCout);

    hls::stream<ap_uint<128> > str_Aggout("str_Aggout");
#pragma HLS RESOURCE variable = str_Aggout core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_Aggout depth = 256
    SameColor_GetKins_hash_top(scl, str_GetCout, str_Aggout);

    hls::stream<ap_uint<160> > str_Gainout("str_Gainout");
#pragma HLS RESOURCE variable = str_Gainout core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_Gainout depth = 256
    hls::stream<StrmBus_S> str_Gainout_update("str_Gainout_update");
#pragma HLS RESOURCE variable = str_Gainout_update core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_Gainout_update depth = 256
    // SameColor_GetBest_Update_Gain_bus3<DWIDTH>(scl, str_Aggout, constant_recip, totPrev, commWeight, str_Gainout);
    SameColor_GetBest_Update_Gain_step1<DWIDTH>(scl, str_Aggout, constant_recip, totPrev, str_Gainout_update,
                                                str_Gainout);
    SameColor_GetBest_Update_Gain_step2<DWIDTH>(commWeight, str_Gainout_update);

    hls::stream<StrmBus_L> str_Cidout("str_Cidout");
#pragma HLS RESOURCE variable = str_Cidout core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_Cidout depth = 256
    hls::stream<StrmBus_S> str_Cid_update("str_Cid_update");
#pragma HLS RESOURCE variable = str_Cid_update core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_Cid_update depth = 256
    hls::stream<int> str_pushV("str_pushV");
#pragma HLS RESOURCE variable = str_pushV core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_pushV depth = 256
    SameColor_GetBest_Update_CID_step1<DWIDTH>(cidSize, str_Gainout, str_Cidout, str_pushV, str_Cid_update);
    SameColor_GetBest_Update_CID_step2<DWIDTH>(cidCurr, str_Cid_update);
    // SameColor_GetBest_Update_TOT_Csize_org<DWIDTH>(
    SameColor_GetBest_Update_TOT_Csize_hash3_new<DWIDTH>(scl, moves, cUpdateSize, totUpdate, str_Cidout);

    hls::stream<ap_uint<96> > str_od("str_od");
#pragma HLS RESOURCE variable = str_od core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_od depth = 256
    hls::stream<int> str_flag("str_flag");
#pragma HLS RESOURCE variable = str_flag core = FIFO_LUTRAM
#pragma HLS STREAM variable = str_flag depth = 256
    // SameColor_GetFlag2<CSRWIDTH>(str_pushV, offset, index, str_flag);
    SameColor_GetFlag3<CSRWIDTH>(str_pushV, offsetDup, str_od);
    SameColor_GetFlag4<CSRWIDTH>(wirte_push_flag, str_od, indexDup, str_flag);
    SameColor_GetFlag<DWIDTH>(str_flag, flagUpdate);

} // SameColor

} // graph
} // xf
#endif
