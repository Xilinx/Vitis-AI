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
#include "xf_data_analytics/regression/decision_tree_L2.hpp"
#include "xf_data_analytics/common/utils.hpp"
#include "xf_data_analytics/classification/decision_tree_train.hpp"

#ifdef __DT_DEBUG__
#include <stdio.h>
#endif

namespace xf {
namespace data_analytics {
namespace regression {
namespace decisiontree {

template <typename MType, int WD>
inline ap_uint<WD> addII2I(ap_uint<WD> a, ap_uint<WD> b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.i = b;
    cc2.f = cc0.f + cc1.f;
    return cc2.i;
}

template <typename MType, int WD>
inline MType addII2F(ap_uint<WD> a, ap_uint<WD> b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.i = b;
    cc2.f = cc0.f + cc1.f;
    return cc2.f;
}

template <typename MType, int WD>
inline ap_uint<WD> addFI2I(MType a, ap_uint<WD> b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.f = a;
    cc1.i = b;
    cc2.f = cc0.f + cc1.f;
    return cc2.i;
}

template <typename MType, int WD>
inline ap_uint<WD> minusIF2I(ap_uint<WD> a, MType b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.f = b;
    cc2.f = cc0.f - cc1.f;
    return cc2.i;
}

template <typename MType, int WD>
inline MType minusIF2F(ap_uint<WD> a, MType b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.f = b;
    cc2.f = cc0.f - cc1.f;
    return cc2.f;
}

template <typename MType, int WD>
inline MType mulII2F(ap_uint<WD> a, ap_uint<WD> b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.i = b;
    cc2.f = cc0.f * cc1.f;
    return cc2.f;
}

template <typename MType, int WD>
inline MType divII2F(ap_uint<WD> a, ap_uint<WD> b) {
#pragma HLS inline
    f_cast<MType> cc0, cc1, cc2;
    cc0.i = a;
    cc1.f = (MType)b;
    cc2.f = cc0.f / cc1.f;
    return cc2.f;
}

template <typename MType, unsigned WD, unsigned MAX_FEAS, unsigned MAX_TREE_DEPTH, unsigned dupnum>
void predict(ap_uint<WD> onesample[dupnum][MAX_FEAS],
             struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
             ap_uint<MAX_TREE_DEPTH> s_nodeid,
             ap_uint<MAX_TREE_DEPTH> e_nodeid,
             unsigned tree_dp,
             ap_uint<MAX_TREE_DEPTH>& node_id) {
#pragma HLS inline
    int count_layer = 0;
statics_predict_loop:
    for (unsigned i = 0; i < MAX_TREE_DEPTH; i++) {
        ap_uint<72> nodeInfo = nodes[i][node_id].nodeInfo;
        if (i < tree_dp && nodeInfo.range(71, 32) != INVALID_NODEID) {
            count_layer++;
            ap_uint<8> feature_id = nodeInfo.range(8 + 15, 16);
            f_cast<MType> feature_val_;
            feature_val_.i = onesample[i >> 1][feature_id];
            MType feature_val = feature_val_.f;
            MType threshold = nodes[i][node_id].threshold;
            if (feature_val <= threshold) { // go left
                node_id = nodeInfo.range(71, 32);
            } else { // go right
                node_id = nodeInfo.range(71, 32) + 1;
            }
        }
    }
    if (count_layer != tree_dp || node_id < s_nodeid || node_id >= e_nodeid) node_id = INVALID_NODEID;
}

template <typename MType, int _WAxi, unsigned int WD, unsigned MAX_FEAS, unsigned MAX_TREE_DEPTH>
void filterByPredict(hls::stream<ap_uint<WD> > dstrm_batch[_WAxi / WD],
                     hls::stream<bool>& estrm_batch,
                     struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                     ap_uint<MAX_TREE_DEPTH> s_nodeid,
                     ap_uint<MAX_TREE_DEPTH> e_nodeid,
                     unsigned tree_dp,
                     ap_uint<8> features_num,
                     ap_uint<30> samples_num,
                     hls::stream<ap_uint<WD> > dstrm_batch_disp[_WAxi / WD],
                     hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm_disp) {
    const int dupnum = (MAX_TREE_DEPTH + 1) / 2;
    const int CHN_B = _WAxi / WD;
    const int MAX_BN = MAX_FEAS / CHN_B;
    const int onebn = (features_num + CHN_B) / CHN_B;
    int samples_read_count = samples_num * onebn;
    ap_uint<MAX_TREE_DEPTH> node_id = 0;
    bool e = estrm_batch.read();
    ap_uint<WD> onesample_all_batch_dups[dupnum][MAX_FEAS];
#pragma HLS array_partition variable = onesample_all_batch_dups dim = 0
    ap_uint<WD> onesample_one_batch[CHN_B];
#pragma HLS array_partition variable = onesample_one_batch dim = 0
    ap_uint<WD> onesample_all_batch[MAX_FEAS];
#pragma HLS array_partition variable = onesample_all_batch dim = 0 complete
    for (int bn = 0; bn < samples_read_count; bn++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
        int to_bn = (bn % onebn);
        int offset = to_bn * CHN_B;
        for (int i = 0; i < CHN_B; i++) {
            onesample_one_batch[i] = dstrm_batch[i].read();
            dstrm_batch_disp[i].write(onesample_one_batch[i]);
            onesample_all_batch[offset + i] = onesample_one_batch[i];
        }
        for (int i = 0; i < dupnum; i++) { // duplicate this sample
            for (int j = 0; j < MAX_FEAS; j++) {
                onesample_all_batch_dups[i][j] = onesample_all_batch[j];
            }
        }
        if (to_bn == onebn - 1) { // one sample end
            node_id = 0;
            predict<MType, WD, MAX_FEAS, MAX_TREE_DEPTH, dupnum>(onesample_all_batch_dups, nodes, s_nodeid, e_nodeid,
                                                                 tree_dp, node_id);
            nstrm_disp.write(node_id);
            e = estrm_batch.read();
        }
    }
}

template <typename MType,
          int _WAxi,
          unsigned int WD,
          unsigned MAX_FEAS,
          unsigned MAX_SPLITS,
          unsigned MAX_SPLITS_PARA,
          unsigned MAX_TREE_DEPTH>
void dispatchSplit(hls::stream<ap_uint<WD> > dstrm_batch_disp[_WAxi / WD],
                   hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm_disp,
                   int features_ids[MAX_SPLITS],
                   ap_uint<8> features_num,
                   ap_uint<30> samples_num,
                   ap_uint<MAX_TREE_DEPTH> s_nodeid,
                   unsigned cur_split_offset,
                   unsigned cur_split_num,
                   hls::stream<ap_uint<WD> > dstrm[MAX_SPLITS_PARA + 1],
                   hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm,
                   hls::stream<bool>& estrm) {
    const int CHN_B = _WAxi / WD;
    const int MAX_BN = MAX_FEAS / CHN_B;
    const int onebn = (features_num + CHN_B) / CHN_B;
    const int label_index = features_num % CHN_B;
    int samples_read_count = samples_num * onebn;
    ap_uint<MAX_TREE_DEPTH> node_id = 0;
    ap_uint<WD> onesample_one_batch[CHN_B];
    ap_uint<WD> onesplit[MAX_SPLITS_PARA + 1];
#pragma HLS array_partition variable = onesample_one_batch dim = 0
#pragma HLS array_partition variable = onesplit dim = 0
    for (int bn = 0; bn < samples_read_count; bn++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
        int to_bn = (bn % onebn);
        int offset = to_bn * CHN_B;
        for (int i = 0; i < CHN_B; i++) {
            onesample_one_batch[i] = dstrm_batch_disp[i].read();
        }

        int nxt_offset = offset + CHN_B;
        for (int i = 0; i < MAX_SPLITS_PARA; i++) {
            int feature_id = features_ids[cur_split_offset + i]; // get the feature id for each split
            if (feature_id >= offset && feature_id < nxt_offset && i < cur_split_num) {
                onesplit[i] =
                    onesample_one_batch[feature_id - offset]; // replicate the feature value depend on its split number
            }
        }

        if (to_bn == onebn - 1) { // one sample finish
            onesplit[MAX_SPLITS_PARA] = onesample_one_batch[label_index];
            node_id = nstrm_disp.read();
            if (node_id != INVALID_NODEID) {
                ap_uint<MAX_TREE_DEPTH> map_id = node_id - s_nodeid;
                estrm.write(true);
                for (int i = 0; i <= MAX_SPLITS_PARA; i++) {
                    dstrm[i].write(onesplit[i]);
                }
                nstrm.write(map_id);
            }
        }
    }
    estrm.write(false);
}

template <typename MType, unsigned MAX_TREE_DEPTH>
void genBitsFromTree(struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                     hls::stream<ap_uint<512> >& axiStream,
                     int nodes_num,
                     ap_uint<MAX_TREE_DEPTH> layer_nodes_num[MAX_TREE_DEPTH + 1]) {
    int elem_per_line = 2;
    ap_uint<512> tmp;
    tmp.range(31, 0) = nodes_num;
    axiStream.write(tmp);

    int accum_layer_num[MAX_TREE_DEPTH] = {0};
#pragma HLS array_partition variable = accum_layer_num dim = 0
    accum_layer_num[0] = layer_nodes_num[0];
    for (int i = 1; i < MAX_TREE_DEPTH; i++) {
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
#pragma HLS pipeline
        accum_layer_num[i] = accum_layer_num[i - 1] + layer_nodes_num[i];
    }
    int layer_id = 0;
    int layer_count = 0;
    ;
    int node_count = 0;
    int read = 0;
    while (node_count < nodes_num) {
#pragma HLS loop_tripcount min = 1024 max = 1024 avg = 1024
#pragma HLS pipeline
        int i = layer_id;
        int j = layer_count;
        ap_uint<72> tmp = nodes[i][j].nodeInfo;
        tmp.range(71, 32) = tmp.range(71, 32) + accum_layer_num[i];
        nodes[0][node_count].nodeInfo = tmp;
        nodes[0][node_count].regValue = nodes[i][j].regValue; // add regression value
        nodes[0][node_count].threshold = nodes[i][j].threshold;
        node_count++;
        layer_count++;
        if (layer_count == layer_nodes_num[i]) {
            layer_id++;
            layer_count = 0;
        }
    }
    for (unsigned i = 0; i < nodes_num; i += 2) {
#pragma HLS loop_tripcount min = 512 max = 512 avg = 512
#pragma HLS pipeline
        for (int j = 0; j < 2; j++) {
            int index = i + j;
            int offset = 256 * j;
            tmp.range(offset + 71, offset) = nodes[0][index].nodeInfo;
            f_cast<MType> cc0, cc1;
            cc0.f = nodes[0][index].regValue;
            cc1.f = nodes[0][index].threshold;
            tmp.range(offset + 135, offset + 72) = cc0.i;
            tmp.range(offset + 255, offset + 192) = cc1.i;
        }
        axiStream.write(tmp);
    }
}

void writeDDR(hls::stream<ap_uint<512> >& axiStream, ap_uint<512> tree[TREE_SIZE], int nodes_num) {
    int axiLen = ((nodes_num + 1) >> 1) + 1;
    for (int i = 0; i < axiLen; i++) {
#pragma HLS loop_tripcount min = 512 max = 512 avg = 512
#pragma HLS pipeline
        tree[i] = axiStream.read();
    }
}

template <typename MType, unsigned MAX_TREE_DEPTH>
void writeOut(struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
              ap_uint<512> tree[TREE_SIZE],
              int nodes_num,
              ap_uint<MAX_TREE_DEPTH> layer_nodes_num[MAX_TREE_DEPTH + 1]) {
    hls::stream<ap_uint<512> > axiStream;
#pragma HLS stream variable = axiStream depth = 8
#pragma HLS dataflow
    genBitsFromTree<MType, MAX_TREE_DEPTH>(nodes, axiStream, nodes_num, layer_nodes_num);
    writeDDR(axiStream, tree, nodes_num);
}

template <typename MType, unsigned WD, unsigned MAX_FEAS, unsigned MAX_SPLITS>
void readConfig(ap_uint<512>* configs,
                ap_uint<30>& samples_num,
                ap_uint<8>& features_num,
                int& para_splits,
                Paras& paras,
                ap_uint<8> numSplits[MAX_FEAS],
                DataType splits[MAX_SPLITS],
                int features_ids[MAX_SPLITS]) {
    const unsigned split_num_per_line = 512 / WD;
    ap_uint<512> configs_in[30];
    for (int i = 0; i < 30; i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS pipeline
        configs_in[i] = configs[i];
    }

    ap_uint<512> onerow = configs_in[0];
    samples_num = onerow.range(30 - 1, 0);
    para_splits = onerow.range(63, 32);
    features_num = onerow.range(63 + 8, 64);

    onerow = configs_in[1];
    paras.cretiea = onerow.range(31, 0);
    paras.maxBins = onerow.range(63, 32);
    paras.max_tree_depth = onerow.range(95, 64);
    paras.min_leaf_size = onerow.range(127, 96);
    unsigned min_info_gain_ = onerow.range(159, 128);
    f_cast<float> tmp;
    tmp.i = min_info_gain_;
    paras.min_info_gain = tmp.f;
    unsigned max_leaf_cat_per_ = onerow.range(191, 160);
    tmp.i = max_leaf_cat_per_;
    paras.max_leaf_cat_per = tmp.f;

    int r = 2;
    for (int i = 0; i < features_num; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
        int i_r = i & 0x3f;
        if (i_r == 0) {
            onerow = configs_in[r++];
        }
        int index = i_r * 8;
        numSplits[i] = onerow.range(index + 8 - 1, index);
    }

    for (int i = 0; i < MAX_SPLITS; i++) {
#pragma HLS unroll
        features_ids[i] = features_num;
    }
    int c = 0;
    int c_ = 0;
    f_cast<MType> tmp1;
    onerow = configs_in[r];
    for (int i = 0; i < features_num; i++) {
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
        for (int j = 0; j < numSplits[i]; j++) {
#pragma HLS loop_tripcount min = 1 max = MAX_SPLITS avg = 8
#pragma HLS dependence variable = splits inter false
            features_ids[c_] = i;
            ap_uint<WD> thres = onerow.range(c * WD + WD - 1, c * WD);
            tmp1.i = thres;
            splits[c_] = tmp1.f;
            c++;
            c_++;
            if (c == split_num_per_line) {
                r++;
                c = 0;
                onerow = configs_in[r];
            }
        }
    }
}

template <typename MType,
          unsigned int WD,
          unsigned MAX_SPLITS,
          unsigned MAX_SPLITS_PARA,
          unsigned PARA_NUM,
          unsigned LATENCY,
          unsigned MAX_TREE_DEPTH>
void statisticAndCompute(hls::stream<ap_uint<WD> > dstrm[MAX_SPLITS_PARA + 1],
                         hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm,
                         hls::stream<bool>& estrm,
                         MType splits[MAX_SPLITS],
                         int features_ids[MAX_SPLITS],
                         int para_splits,
                         unsigned tree_dp,
                         ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num,
                         ap_uint<8> features_num,
                         Paras paras,
                         unsigned cur_split_offset,
                         unsigned cur_split_num,
                         MType max_gainratio_value[PARA_NUM],
                         bool ifstop[PARA_NUM],
                         MType max_classes[PARA_NUM],
                         int max_feature_ids[PARA_NUM],
                         MType max_split_values[PARA_NUM]) {
    const int MAX_SPLITS_1 = MAX_SPLITS_PARA + 1;
    const int MAX_PARAS = PARA_NUM;
#ifndef __SYNTHESIS__
    ap_uint<WD * 4 + 32>* num_in_cur_nfs_cat[MAX_SPLITS_1];
    for (int i = 0; i < MAX_SPLITS_1; i++) {
        num_in_cur_nfs_cat[i] = (ap_uint<WD * 4 + 32>*)malloc(MAX_PARAS * LATENCY * sizeof(ap_uint<WD * 4 + 32>));
    }
#else
    ap_uint<WD * 4 + 32> num_in_cur_nfs_cat[MAX_SPLITS_1]
                                           [MAX_PARAS * LATENCY]; // right(var, avg), left(var, avg), count
#pragma HLS array_partition variable = num_in_cur_nfs_cat dim = 1
#pragma HLS bind_storage variable = num_in_cur_nfs_cat type = ram_2p impl = uram
#endif
init_num_in_cur_nfs_cat:
    for (int i = 0; i < MAX_PARAS * LATENCY; i++) {
#pragma HLS loop_tripcount min = MAX_PARAS max = MAX_PARAS avg = MAX_PARAS
#pragma HLS pipeline
        for (int j = 0; j < MAX_SPLITS_1; j++) {
            num_in_cur_nfs_cat[j][i] = 0;
        }
    }
    bool e = estrm.read();
    ap_uint<WD> onesample[MAX_SPLITS_1];
#pragma HLS array_partition variable = onesample dim = 0
    ap_uint<MAX_TREE_DEPTH> node_mid;
    ap_uint<32> cnt[PARA_NUM] = {0};
#pragma HLS array_partition variable = cnt dim = 0
    for (int i = 0; i < PARA_NUM; i++) {
#pragma HLS pipeline
        cnt[i] = 0;
    }

main_statistic:
    while (e) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
#pragma HLS dependence variable = num_in_cur_nfs_cat inter false
        for (int i = 0; i < MAX_SPLITS_1; i++) {
#pragma HLS unroll
            onesample[i] = dstrm[i].read();
        }
        node_mid = nstrm.read();
        e = estrm.read();
        f_cast<MType> label_;
        label_.i = onesample[MAX_SPLITS_PARA]; // the last one is class id
        MType label = label_.f;
        MType label_square = label * label;
        ap_uint<MAX_TREE_DEPTH + 4> n_cid = node_mid * LATENCY + cnt[node_mid] % LATENCY;

        num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(31, 0) =
            num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(31, 0) + 1; // count value
        num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(WD + 31, 32) =
            addFI2I<MType, WD>(label, num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(WD + 31, 32)); // avg
        num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(WD * 2 + 31, WD + 32) =
            addFI2I<MType, WD>(label_square, num_in_cur_nfs_cat[MAX_SPLITS_PARA][n_cid].range(WD * 2 + 31, WD + 32));

        for (int j = 0; j < MAX_SPLITS_PARA; j++) {
            f_cast<MType> feature_val_;
            feature_val_.i = onesample[j];
            MType features_val = feature_val_.f;
            MType thre = splits[cur_split_offset + j];
            if (features_val <= thre) { // belong to the left children node
                num_in_cur_nfs_cat[j][n_cid].range(31, 0) = num_in_cur_nfs_cat[j][n_cid].range(31, 0) + 1;
                num_in_cur_nfs_cat[j][n_cid].range(WD + 31, 32) =
                    addFI2I<MType, WD>(label, num_in_cur_nfs_cat[j][n_cid].range(WD + 31, 32));
                num_in_cur_nfs_cat[j][n_cid].range(WD * 2 + 31, WD + 32) =
                    addFI2I<MType, WD>(label_square, num_in_cur_nfs_cat[j][n_cid].range(WD * 2 + 31, WD + 32));
                num_in_cur_nfs_cat[j][n_cid].range(WD * 3 + 31, WD * 2 + 32) =
                    num_in_cur_nfs_cat[j][n_cid].range(WD * 3 + 31, WD * 2 + 32);
                num_in_cur_nfs_cat[j][n_cid].range(WD * 4 + 31, WD * 3 + 32) =
                    num_in_cur_nfs_cat[j][n_cid].range(WD * 4 + 31, WD * 3 + 32);
            } else { // right children node
                num_in_cur_nfs_cat[j][n_cid].range(31, 0) = num_in_cur_nfs_cat[j][n_cid].range(31, 0);
                num_in_cur_nfs_cat[j][n_cid].range(WD + 31, 32) = num_in_cur_nfs_cat[j][n_cid].range(WD + 31, 32);
                num_in_cur_nfs_cat[j][n_cid].range(WD * 2 + 31, WD + 32) =
                    num_in_cur_nfs_cat[j][n_cid].range(WD * 2 + 31, WD + 32);
                num_in_cur_nfs_cat[j][n_cid].range(WD * 3 + 31, WD * 2 + 32) =
                    addFI2I<MType, WD>(label, num_in_cur_nfs_cat[j][n_cid].range(WD * 3 + 31, WD * 2 + 32));
                num_in_cur_nfs_cat[j][n_cid].range(WD * 4 + 31, WD * 3 + 32) =
                    addFI2I<MType, WD>(label_square, num_in_cur_nfs_cat[j][n_cid].range(WD * 4 + 31, WD * 3 + 32));
            }
        }

        cnt[node_mid]++;
    }

    ap_uint<WD * 4 + 32> num_in_cur_nfs_cat_merge[MAX_PARAS];
#pragma HLS bind_storage variable = num_in_cur_nfs_cat_merge type = ram_2p impl = bram
MERGE_SUM_LOOP:
    for (int j = 0; j < MAX_SPLITS_1; j++) {
    COMPUTE_ONE_SPLITS_LOOP:
        for (int k = 0; k < LATENCY; k++) {
            for (int i = 0; i < PARA_NUM; i++) {
#pragma HLS pipeline
                ap_uint<WD * 4 + 32> tmp;
                if (k != 0) {
                    tmp = num_in_cur_nfs_cat_merge[i];
                    ap_uint<WD* 4 + 32> data = num_in_cur_nfs_cat[j][i * LATENCY + k];

                    tmp.range(31, 0) = tmp.range(31, 0) + data.range(31, 0);
                    tmp.range(WD + 31, 32) = addII2I<MType, WD>(tmp.range(WD + 31, 32), data.range(WD + 31, 32));
                    tmp.range(WD * 2 + 31, WD + 32) =
                        addII2I<MType, WD>(tmp.range(WD * 2 + 31, WD + 32), data.range(WD * 2 + 31, WD + 32));
                    tmp.range(WD * 3 + 31, WD * 2 + 32) =
                        addII2I<MType, WD>(tmp.range(WD * 3 + 31, WD * 2 + 32), data.range(WD * 3 + 31, WD * 2 + 32));
                    tmp.range(WD * 4 + 31, WD * 3 + 32) =
                        addII2I<MType, WD>(tmp.range(WD * 4 + 31, WD * 3 + 32), data.range(WD * 4 + 31, WD * 3 + 32));
                } else {
                    tmp = num_in_cur_nfs_cat[j][i * LATENCY];
                }
                num_in_cur_nfs_cat_merge[i] = tmp;
            }
        }

    MERGE_STORE_ONE_SPLITS_LOOP:
        for (int i = 0; i < MAX_PARAS; i++) {
#pragma HLS pipeline
            num_in_cur_nfs_cat[j][i * LATENCY] = num_in_cur_nfs_cat_merge[i];
        }
    }

    ap_uint<64> clklogclk[PARA_NUM][MAX_SPLITS_PARA];
    ap_uint<64> crklogcrk[PARA_NUM][MAX_SPLITS_PARA];
    MType num_in_avg[PARA_NUM];
    ap_uint<64> num_in_cur_nfs_cat_sum[PARA_NUM][MAX_SPLITS_PARA];
#pragma HLS array_partition variable = clklogclk dim = 2 cyclic factor = 2
#pragma HLS array_partition variable = crklogcrk dim = 2 cyclic factor = 2
#pragma HLS array_partition variable = num_in_cur_nfs_cat_sum dim = 2 cyclic factor = 2
#pragma HLS bind_storage variable = clklogclk type = ram_2p impl = uram
#pragma HLS bind_storage variable = crklogcrk type = ram_2p impl = uram
#pragma HLS bind_storage variable = num_in_cur_nfs_cat_sum type = ram_2p impl = uram
compute_gain_loop_init:
    for (int i = 0; i < PARA_NUM; i++) {
#pragma HLS loop_tripcount min = PARA_NUM max = PARA_NUM avg = PARA_NUM
        for (int j = 0; j < MAX_SPLITS_PARA; j++) {
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
            clklogclk[i][j] = 0;
            crklogcrk[i][j] = 0;
            num_in_avg[i] = 0;
            num_in_cur_nfs_cat_sum[i][j] = 0;
        }
    }
compute_gain_loop_gainratio:
    for (int l = 0; l < PARA_NUM; l++) {
#pragma HLS loop_tripcount min = PARA_NUM max = PARA_NUM avg = PARA_NUM
        ap_uint<WD* 4 + 32> sample_num_ = num_in_cur_nfs_cat[MAX_SPLITS_PARA][l * LATENCY];
        num_in_avg[l] =
            mulII2F<MType, WD>(sample_num_(WD + 31, 32), sample_num_(WD + 31, 32)) / (MType)sample_num_(31, 0);
        for (int i = 0; i < MAX_SPLITS_PARA; i += 2) { // Bubble exists
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
            for (int j = 0; j < 2; j++) {
                ap_uint<WD* 4 + 32> nfs_value = num_in_cur_nfs_cat[i + j][l * LATENCY];
                ap_uint<32> l_cat_k = nfs_value(31, 0);             // left count value
                ap_uint<32> r_cat_k = sample_num_(31, 0) - l_cat_k; // inference to right

                MType tmp1 = mulII2F<MType, WD>(nfs_value(WD + 31, 32), nfs_value(WD + 31, 32)) / (MType)l_cat_k;
                clklogclk[l][i + j] = minusIF2I<MType, WD>(nfs_value(WD * 2 + 31, WD + 32), tmp1);
                MType tmp2 =
                    mulII2F<MType, WD>(nfs_value(WD * 3 + 31, WD * 2 + 32), nfs_value(WD * 3 + 31, WD * 2 + 32)) /
                    (MType)r_cat_k;
                crklogcrk[l][i + j] = minusIF2I<MType, WD>(nfs_value(WD * 4 + 31, WD * 3 + 32), tmp2);
                num_in_cur_nfs_cat_sum[l][i + j] =
                    minusIF2I<MType, WD>(sample_num_(WD * 2 + 31, WD + 32), num_in_avg[l]);
            }
        }
    }
main_compute_gain_loop:
    for (int l = 0; l < cur_layer_nodes_num; l++) {
#pragma HLS loop_tripcount min = 1 max = PARA_NUM avg = PARA_NUM
        int num_in_cur_node_cat_sum = num_in_cur_nfs_cat[MAX_SPLITS_PARA][l * LATENCY].range(31, 0);
        int maxPerNums = paras.max_leaf_cat_per * num_in_cur_node_cat_sum;
        max_classes[l] =
            divII2F<MType, WD>(num_in_cur_nfs_cat[MAX_SPLITS_PARA][l * LATENCY](WD + 31, 32), num_in_cur_node_cat_sum);

        MType max_gainratio = max_gainratio_value[l];
        MType cretia = 1 << 30;
        int max_split_id = 0;
        MType max_split_value = splits[cur_split_offset];
        bool is_update = false;
        for (int i = 0; i < cur_split_num; i++) { // choose candidate split
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
            MType tmp1 = addII2F<MType, WD>(clklogclk[l][i], crklogcrk[l][i]);
            cretia = minusIF2F<MType, WD>(num_in_cur_nfs_cat_sum[l][i], tmp1);
            if (cretia > max_gainratio) { // find the largest one
                is_update = true;
                max_gainratio = cretia;
                max_split_id = i;                               // split id
                max_split_value = splits[cur_split_offset + i]; // threshold
            }
        }

        bool is_zero = (max_gainratio <= 0.0);
        if (is_update || is_zero) {
            max_gainratio_value[l] = max_gainratio;
            max_feature_ids[l] = features_ids[cur_split_offset + max_split_id];
            max_split_values[l] = max_split_value; // threshold value

            ap_uint<32> left_count = num_in_cur_nfs_cat[max_split_id][l * LATENCY](31, 0);
            if (num_in_cur_node_cat_sum <= paras.min_leaf_size || left_count > maxPerNums ||
                (num_in_cur_node_cat_sum - left_count) > maxPerNums || is_zero || tree_dp >= paras.max_tree_depth) {
                ifstop[l] = true;
            } else {
                ifstop[l] = false;
            }
        }
    }
}

template <typename MType, unsigned WD, unsigned MAX_TREE_DEPTH, unsigned MAX_SPLITS, unsigned PARA_NUM>
void updateTree(ap_uint<MAX_TREE_DEPTH> s_nodeid,
                ap_uint<MAX_TREE_DEPTH> e_nodeid,
                unsigned tree_dp,
                ap_uint<MAX_TREE_DEPTH>& next_layer_nodes_num,
                MType splits[MAX_SPLITS],
                struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                bool ifstop[PARA_NUM],
                MType max_classes[PARA_NUM],
                int max_feature_ids[PARA_NUM],
                MType max_split_values[PARA_NUM]) {
    unsigned cur_layer_nodes_num = e_nodeid - s_nodeid;
update_layer:
    for (int j = 0; j < cur_layer_nodes_num; j++) {
#pragma HLS loop_tripcount min = PARA_NUM max = PARA_NUM avg = PARA_NUM
#pragma HLS pipeline
        int mid = max_feature_ids[j];
        MType mspl = max_split_values[j];
        unsigned cur_nodeid = j + s_nodeid;
        if (ifstop[j]) {
            nodes[tree_dp][cur_nodeid].nodeInfo.range(0, 0) = true;
            nodes[tree_dp][cur_nodeid].regValue = max_classes[j]; // fill the regression value
        } else {
            ap_uint<56> nodeId_l = next_layer_nodes_num++;
            ap_uint<MAX_TREE_DEPTH> nodeId_r = next_layer_nodes_num++;
            ap_uint<56> adder = mid + (nodeId_l << 16);
            nodes[tree_dp][cur_nodeid].nodeInfo.range(71, 16) = adder;
            nodes[tree_dp][cur_nodeid].threshold = mspl;
        }
    }
}

template <typename MType,
          int _WAxi,
          unsigned int WD,
          unsigned MAX_FEAS,
          unsigned MAX_SPLITS,
          unsigned MAX_SPLITS_PARA,
          unsigned PARA_NUM,
          unsigned MAX_TREE_DEPTH>
void decisionTreeFlow(ap_uint<512> data[DATASIZE],
                      ap_uint<8> features_num,
                      unsigned samples_num,
                      unsigned tree_dp,
                      ap_uint<MAX_TREE_DEPTH> s_nodeid,
                      ap_uint<MAX_TREE_DEPTH> e_nodeid,
                      ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num,
                      unsigned cur_split_offset,
                      unsigned cur_split_num,
                      Paras paras,
                      MType splits[MAX_SPLITS],
                      struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                      int features_ids[MAX_SPLITS],
                      int para_splits,
                      MType max_gainratio_value[PARA_NUM],
                      int max_feature_ids[PARA_NUM],
                      MType max_split_values[PARA_NUM],
                      bool ifstop[PARA_NUM],
                      MType max_classes[PARA_NUM]) {
#pragma HLS dataflow
    hls::stream<ap_uint<WD> > dstrm_batch[_WAxi / WD];
#pragma HLS stream variable = dstrm_batch depth = 128
#pragma HLS bind_storage variable = dstrm_batch type = fifo impl = lutram
    hls::stream<bool> estrm_batch;
#pragma HLS stream variable = estrm_batch depth = 128
#pragma HLS bind_storage variable = estrm_batch type = fifo impl = lutram

    hls::stream<ap_uint<WD> > dstrm_batch_disp[_WAxi / WD];
#pragma HLS stream variable = dstrm_batch_disp depth = 128
#pragma HLS bind_storage variable = dstrm_batch_disp type = fifo impl = lutram
    hls::stream<ap_uint<MAX_TREE_DEPTH> > nstrm_disp;
#pragma HLS stream variable = nstrm_disp depth = 128
#pragma HLS bind_storage variable = nstrm_disp type = fifo impl = lutram

    hls::stream<ap_uint<WD> > dstrm[MAX_SPLITS_PARA + 1];
#pragma HLS stream variable = dstrm depth = 128
#pragma HLS bind_storage variable = dstrm type = fifo impl = lutram
    hls::stream<ap_uint<MAX_TREE_DEPTH> > nstrm;
#pragma HLS stream variable = nstrm depth = 128
#pragma HLS bind_storage variable = nstrm type = fifo impl = lutram
    hls::stream<bool> estrm;
#pragma HLS stream variable = estrm depth = 128
#pragma HLS bind_storage variable = estrm type = fifo impl = lutram

Scan:
    xf::data_analytics::classification::axiVarColToStreams<64, _WAxi, WD>(data, 1, samples_num, features_num + 1,
                                                                          dstrm_batch, estrm_batch);
FilterByPredict:
    filterByPredict<MType, _WAxi, WD, MAX_FEAS, MAX_TREE_DEPTH>(dstrm_batch, estrm_batch, nodes, s_nodeid, e_nodeid,
                                                                tree_dp, features_num, samples_num, dstrm_batch_disp,
                                                                nstrm_disp);
DispatchSplit:
    dispatchSplit<MType, _WAxi, WD, MAX_FEAS, MAX_SPLITS, MAX_SPLITS_PARA, MAX_TREE_DEPTH>(
        dstrm_batch_disp, nstrm_disp, features_ids, features_num, samples_num, s_nodeid, cur_split_offset,
        cur_split_num, dstrm, nstrm, estrm);
Count:
    statisticAndCompute<MType, WD, MAX_SPLITS, MAX_SPLITS_PARA, PARA_NUM, 9, MAX_TREE_DEPTH>(
        dstrm, nstrm, estrm, splits, features_ids, para_splits, tree_dp, cur_layer_nodes_num, features_num, paras,
        cur_split_offset, cur_split_num, max_gainratio_value, ifstop, max_classes, max_feature_ids, max_split_values);
}

template <typename MType,
          int _WAxi,
          unsigned int WD,
          unsigned MAX_FEAS,
          unsigned MAX_SPLITS,
          unsigned MAX_SPLITS_PARA,
          unsigned PARA_NUM,
          unsigned MAX_TREE_DEPTH>
void decisionTreeFun(ap_uint<512> data[DATASIZE],
                     MType splits[MAX_SPLITS],
                     int features_ids[MAX_SPLITS],
                     int para_splits,
                     ap_uint<8> features_num,
                     unsigned samples_num,
                     unsigned tree_dp,
                     ap_uint<MAX_TREE_DEPTH> s_nodeid,
                     ap_uint<MAX_TREE_DEPTH> e_nodeid,
                     ap_uint<MAX_TREE_DEPTH>& next_layer_nodes_num,
                     Paras paras,
                     struct NodeR<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM]) {
    const int loop_num = (para_splits + MAX_SPLITS_PARA - 1) / MAX_SPLITS_PARA;
    ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num = e_nodeid - s_nodeid;

    int max_feature_ids[PARA_NUM];
    MType max_split_values[PARA_NUM];
    MType max_gainratio_value[PARA_NUM];
    bool ifstop[PARA_NUM];
    MType max_classes[PARA_NUM];

    for (int i = 0; i < PARA_NUM; i++) {
#pragma HLS pipeline
        max_gainratio_value[i] = 0;
    }

    for (int i = 0; i < loop_num; i++) { // loop for computing all split
        unsigned cur_split_offset = i * MAX_SPLITS_PARA;
        unsigned cur_split_num =
            (cur_split_offset + MAX_SPLITS_PARA < para_splits) ? MAX_SPLITS_PARA : (para_splits - cur_split_offset);
        decisionTreeFlow<MType, _WAxi, WD, MAX_FEAS, MAX_SPLITS, MAX_SPLITS_PARA, PARA_NUM, MAX_TREE_DEPTH>(
            data, features_num, samples_num, tree_dp, s_nodeid, e_nodeid, cur_layer_nodes_num, cur_split_offset,
            cur_split_num, paras, splits, nodes, features_ids, para_splits, max_gainratio_value, max_feature_ids,
            max_split_values, ifstop, max_classes);
    }
    updateTree<MType, WD, MAX_TREE_DEPTH, MAX_SPLITS, PARA_NUM>(s_nodeid, e_nodeid, tree_dp, next_layer_nodes_num,
                                                                splits, nodes, ifstop, max_classes, max_feature_ids,
                                                                max_split_values);
}

} // namespace decisiontree
} // namespace regression
} // namespace data_analytics
} // namespace xf

extern "C" void DecisionTree(ap_uint<512> data[DATASIZE], ap_uint<512> configs[30], ap_uint<512> tree[TREE_SIZE]) {
// clang-format off
#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_0 port = data

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_1 port = configs

#pragma HLS INTERFACE m_axi offset = slave latency = 64 \
	num_write_outstanding = 16 num_read_outstanding = 16 \
	max_write_burst_length = 64 max_read_burst_length = 64 \
	bundle = gmem0_2 port = tree

#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = configs bundle = control
#pragma HLS INTERFACE s_axilite port = tree bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<30> samples_num;
    ap_uint<8> features_num;
    Paras paras;
    ap_uint<8> numSplits[MAX_FEAS_];
    DataType splits[MAX_SPLITS_];
    int features_ids[MAX_SPLITS_];
    int para_splits = 0;
#pragma HLS array_partition variable = splits dim = 0
#pragma HLS array_partition variable = features_ids dim = 0 complete
init_features_splits_ids:
    xf::data_analytics::regression::decisiontree::readConfig<DataType, 64, MAX_FEAS_, MAX_SPLITS_>(
        configs, samples_num, features_num, para_splits, paras, numSplits, splits, features_ids);

#ifdef __DT_DEBUG__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n\n", features_num.to_int());
    printf("para_splits: %d\n\n", para_splits);

    printf("paras.cretiea: %d\n", paras.cretiea);
    printf("paras.maxBins: %d\n", paras.maxBins);
    printf("paras.max_tree_depth: %d\n", paras.max_tree_depth);
    printf("paras.min_leaf_size: %d\n", paras.min_leaf_size);
    printf("paras.max_leaf_cat_per: %f\n", paras.max_leaf_cat_per);
    printf("paras.min_info_gain: %f\n\n", paras.min_info_gain);

    for (int i = 0; i < features_num; i++) {
        printf("numsplits of %d:%d\n", i, numSplits[i].to_int());
    }
    int cnt = 0;
    for (int i = 0; i < features_num; i++) {
        for (int j = 0; j < numSplits[i]; j++) {
            printf("%d,%d: %lf\n", i, j, splits[cnt++]);
        }
    }
    for (int i = 0; i < MAX_SPLITS_; i++) printf("%d ,", features_ids[i]);
#endif
    // 2. decision tree process

    struct NodeR<DataType> nodes[MAX_TREE_DEPTH_][MAX_NODES_NUM];
#pragma HLS array_partition variable = nodes dim = 1
#pragma HLS bind_storage variable = nodes type = ram_2p impl = uram

    ap_uint<MAX_TREE_DEPTH_> layer_nodes_num[MAX_TREE_DEPTH_ + 1];
    ap_uint<MAX_TREE_DEPTH_> nodes_num = 0;
#pragma HLS array_partition variable = layer_nodes_num dim = 0 complete
    for (int i = 0; i < MAX_TREE_DEPTH_ + 1; i++) {
#pragma HLS unroll
        layer_nodes_num[i] = 0;
    }
    for (int i = 0; i < MAX_NODES_NUM; i++) {
#pragma HLS loop_tripcount min = 1024 max = 1024 avg = 1024
#pragma HLS pipeline
        for (int j = 0; j < MAX_TREE_DEPTH_; j++) {
            nodes[j][i].nodeInfo = 0;
            nodes[j][i].nodeInfo.range(71, 32) = INVALID_NODEID;
            nodes[j][i].regValue = 0;
            nodes[j][i].threshold = 0;
        }
    }
    layer_nodes_num[0] = 1;
    int tree_dp = 0;
mainloop:
    while (layer_nodes_num[tree_dp] > 0) {
        int nxt_tree_dp = tree_dp + 1;
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
        unsigned s_nodeid = 0;
        nodes_num += layer_nodes_num[tree_dp];
        unsigned e_nodeid_ = layer_nodes_num[tree_dp];
#ifdef __DT_DEBUG__
        printf("\nStart Layer_Iter:%d,start from %d, end in %d\n,Valid nodes:", tree_dp, s_nodeid, e_nodeid_);
#endif
        for (int j = 0; j < layer_nodes_num[tree_dp]; j += PARA_NUM_) {
#pragma HLS loop_tripcount min = 1 max = 3 avg = 2
            unsigned e_nodeid = s_nodeid + PARA_NUM_;
            e_nodeid = (e_nodeid_ < e_nodeid) ? e_nodeid_ : e_nodeid;
            xf::data_analytics::regression::decisiontree::decisionTreeFun<DataType, 512, 64, MAX_FEAS_, MAX_SPLITS_,
                                                                          MAX_SPLITS_PARA_, PARA_NUM_, MAX_TREE_DEPTH_>(
                data, splits, features_ids, para_splits, features_num, samples_num, tree_dp, s_nodeid, e_nodeid,
                layer_nodes_num[nxt_tree_dp], paras, nodes);
            s_nodeid = s_nodeid + PARA_NUM_;
        }
        tree_dp++;
    }
    // 3. warp and write out nodes
    xf::data_analytics::regression::decisiontree::writeOut<DataType, MAX_TREE_DEPTH_>(nodes, tree, nodes_num,
                                                                                      layer_nodes_num);

#ifdef __DT_DEBUG__
    printf("nodes_num:%d\n", nodes_num.to_int());
#endif
}
