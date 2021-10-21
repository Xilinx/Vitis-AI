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
#include "xf_data_analytics/classification/decision_tree_L2.hpp"
#include "xf_data_analytics/classification/decision_tree_train.hpp"
#include "xf_data_analytics/common/utils.hpp"
#ifdef __DT_DEBUG__
#include <stdio.h>
#endif

namespace xf {
namespace data_analytics {
namespace classification {
namespace decisiontree {

template <typename MType, unsigned WD, unsigned MAX_FEAS, unsigned MAX_TREE_DEPTH, unsigned dupnum>
void predict(ap_uint<WD> onesample[MAX_FEAS],
             struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
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
            feature_val_.i = onesample[feature_id];
            MType feature_val = feature_val_.f;
            MType threshold = nodes[i][node_id].threshold;
            if (feature_val <= threshold) {
                node_id = nodeInfo.range(71, 32);
            } else {
                node_id = nodeInfo.range(71, 32) + 1;
            }
        }
    }
    if (count_layer != tree_dp || node_id < s_nodeid || node_id >= e_nodeid) node_id = INVALID_NODEID;
}

template <typename MType, int _WAxi, unsigned int WD, unsigned MAX_FEAS, unsigned MAX_TREE_DEPTH>
void filterByPredict(hls::stream<ap_uint<WD> > dstrm_batch[_WAxi / WD],
                     hls::stream<bool>& estrm_batch,
                     struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
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

        if (to_bn == onebn - 1) {
            node_id = 0;
            predict<MType, WD, MAX_FEAS, MAX_TREE_DEPTH, dupnum>(onesample_all_batch, nodes, s_nodeid, e_nodeid,
                                                                 tree_dp, node_id);
            nstrm_disp.write(node_id);
            e = estrm_batch.read();
        }
    }
}

template <typename MType, int _WAxi, unsigned int WD, unsigned MAX_FEAS, unsigned MAX_SPLITS, unsigned MAX_TREE_DEPTH>
void dispatchSplit(hls::stream<ap_uint<WD> > dstrm_batch_disp[_WAxi / WD],
                   hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm_disp,
                   int features_ids[MAX_SPLITS],
                   ap_uint<8> features_num,
                   ap_uint<30> samples_num,
                   ap_uint<MAX_TREE_DEPTH> s_nodeid,
                   hls::stream<ap_uint<WD> > dstrm[MAX_SPLITS],
                   hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm,
                   hls::stream<bool>& estrm) {
    const int CHN_B = _WAxi / WD;
    const int MAX_BN = MAX_FEAS / CHN_B;
    const int onebn = (features_num + CHN_B) / CHN_B;
    int samples_read_count = samples_num * onebn;
    ap_uint<MAX_TREE_DEPTH> node_id = 0;
    ap_uint<WD> onesample_one_batch[CHN_B];
    ap_uint<WD> onesplit[MAX_SPLITS];
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
        for (int i = 0; i < MAX_SPLITS; i++) {
            int feature_id = features_ids[i];
            if (feature_id >= offset && feature_id < nxt_offset) {
                onesplit[i] = onesample_one_batch[feature_id - offset];
            }
        }
        if (to_bn == onebn - 1) {
            node_id = nstrm_disp.read();
            if (node_id != INVALID_NODEID) {
                ap_uint<MAX_TREE_DEPTH> map_id = node_id - s_nodeid;
                estrm.write(true);
                for (int i = 0; i < MAX_SPLITS; i++) {
                    dstrm[i].write(onesplit[i]);
                }
                nstrm.write(map_id);
            }
        }
    }
    estrm.write(false);
}

template <typename MType, unsigned MAX_TREE_DEPTH>
void genBitsFromTree(struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                     ap_uint<512> tree[TREE_SIZE],
                     int nodes_num,
                     ap_uint<MAX_TREE_DEPTH> layer_nodes_num[MAX_TREE_DEPTH + 1]) {
    int elem_per_line = 2;
    ap_uint<512> tmp;
    tmp.range(31, 0) = nodes_num;

    int axiCounter = 0;
    tree[axiCounter++] = tmp;

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
            f_cast<MType> thre;
            thre.f = nodes[0][index].threshold;
            tmp.range(offset + 255, offset + 192) = thre.i;
        }
        tree[axiCounter++] = tmp;
    }
}

template <typename MType, unsigned MAX_TREE_DEPTH>
void writeOut(struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
              ap_uint<512> tree[TREE_SIZE],
              int nodes_num,
              ap_uint<MAX_TREE_DEPTH> layer_nodes_num[MAX_TREE_DEPTH + 1]) {
    genBitsFromTree<MType, MAX_TREE_DEPTH>(nodes, tree, nodes_num, layer_nodes_num);
}

template <typename MType, unsigned WD, unsigned MAX_FEAS, unsigned MAX_SPLITS>
void readConfig(ap_uint<512>* configs,
                ap_uint<30>& samples_num,
                ap_uint<8>& features_num,
                ap_uint<8>& numClass,
                int& para_splits,
                Paras& paras,
                ap_uint<8> numSplits[MAX_FEAS],
                DataType splits[MAX_SPLITS],
                int features_ids[MAX_SPLITS]) {
    const unsigned split_num_per_line = 512 / WD;

    ap_uint<512> onerow = configs[0];
    samples_num = onerow.range(30 - 1, 0);
    para_splits = onerow.range(63, 32);
    features_num = onerow.range(63 + 8, 64);
    numClass = onerow.range(95 + 8, 96);

    onerow = configs[1];
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
            onerow = configs[r++];
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
    onerow = configs[r];
    for (int i = 0; i < features_num; i++) {
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
        for (int j = 0; j < numSplits[i]; j++) {
#pragma HLS loop_tripcount min = 1 max = MAX_SPLITS
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
                onerow = configs[r];
            }
        }
    }
}

template <typename MType,
          unsigned int _WAxi,
          unsigned int WD,
          unsigned MAX_CAT_NUM,
          unsigned MAX_FEAS,
          unsigned MAX_SPLITS,
          unsigned PARA_NUM,
          unsigned MAX_TREE_DEPTH>
void statisticAndCompute(hls::stream<ap_uint<WD> > dstrm_batch_disp[_WAxi / WD],
                         hls::stream<ap_uint<MAX_TREE_DEPTH> >& nstrm_disp,
                         MType splits[MAX_SPLITS],
                         int features_ids[MAX_SPLITS],
                         int para_splits,
                         unsigned tree_dp,
                         ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num,
                         ap_uint<8> features_num,
                         ap_uint<30> samples_num,
                         ap_uint<8> numClass,
                         ap_uint<MAX_TREE_DEPTH> s_nodeid,
                         Paras paras,
                         bool ifstop[PARA_NUM],
                         int max_classes[PARA_NUM],
                         int max_feature_ids[PARA_NUM],
                         MType max_split_values[PARA_NUM]) {
    const int MAX_SPLITS_1 = MAX_SPLITS + 1;
    const int MAX_PARAS = (PARA_NUM) * (MAX_CAT_NUM);
    int num_in_cur_nfs_cat[MAX_SPLITS_1][MAX_PARAS];
#pragma HLS array_partition variable = num_in_cur_nfs_cat dim = 1
#pragma HLS bind_storage variable = num_in_cur_nfs_cat type = ram_2p impl = uram
init_num_in_cur_nfs_cat: // splits are are reuse by nodes, so clear statics in (node,split)
    for (int i = 0; i < MAX_PARAS; i++) {
#pragma HLS loop_tripcount min = MAX_PARAS max = MAX_PARAS avg = MAX_PARAS
#pragma HLS pipeline
        for (int j = 0; j < MAX_SPLITS_1; j++) {
            num_in_cur_nfs_cat[j][i] = 0;
        }
    }
    ap_uint<WD> onesample[MAX_SPLITS];
#pragma HLS array_partition variable = onesample dim = 0
    ap_uint<MAX_TREE_DEPTH> node_mid;
    ap_uint<MAX_TREE_DEPTH + MAX_CAT_NUM> cache_nid_cid[8] = {-1, -1, -1, -1, -1, -1, -1, -1};
    unsigned cache_elem[8][MAX_SPLITS_1];

    ap_uint<MAX_TREE_DEPTH + MAX_CAT_NUM> nid_cid = 0;
    unsigned elem[MAX_SPLITS_1];
    unsigned nxt_elem[MAX_SPLITS_1];
#pragma HLS array_partition variable = elem dim = 0
#pragma HLS array_partition variable = nxt_elem dim = 0
#pragma HLS array_partition variable = cache_elem dim = 0
    ap_uint<4> in_cache_id = 8;

    const int CHN_B = _WAxi / WD;
    const int MAX_BN = MAX_FEAS / CHN_B;
    const int onebn = (features_num + CHN_B) / CHN_B;
    int samples_read_count = samples_num * onebn;
    ap_uint<MAX_TREE_DEPTH> node_id = 0;
    ap_uint<WD> onesample_one_batch[CHN_B];
#pragma HLS array_partition variable = onesample_one_batch dim = 0

main_statistic:
    for (int bn = 0; bn < samples_read_count; bn++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 1000000 max = 1000000 avg = 1000000
#pragma HLS dependence variable = num_in_cur_nfs_cat inter false
        /*
                for (int i = 0; i < MAX_SPLITS; i++) {
        #pragma HLS unroll
                    onesample[i] = dstrm[i].read();
                }
                node_mid = nstrm.read();
                e = estrm.read();
        */
        int to_bn = (bn % onebn);
        int offset = to_bn * CHN_B;
        for (int i = 0; i < CHN_B; i++) {
            onesample_one_batch[i] = dstrm_batch_disp[i].read();
        }

        int nxt_offset = offset + CHN_B;
        for (int i = 0; i < MAX_SPLITS; i++) {
            int feature_id = features_ids[i];
            if (feature_id >= offset && feature_id < nxt_offset) {
                onesample[i] = onesample_one_batch[feature_id - offset];
            }
        }

        if (to_bn == onebn - 1) {
            node_mid = nstrm_disp.read();
            if (node_mid != INVALID_NODEID) {
                node_mid -= s_nodeid;

                f_cast<MType> classId_;
                classId_.i = onesample[MAX_SPLITS - 1];
                ap_uint<WD> classId = classId_.f;
                ap_uint<MAX_TREE_DEPTH + MAX_CAT_NUM> n_cid = node_mid * MAX_CAT_NUM + classId;
                nid_cid.range(MAX_TREE_DEPTH - 1, 0) = node_mid;
                nid_cid.range(MAX_CAT_NUM + MAX_TREE_DEPTH - 1, MAX_TREE_DEPTH) = classId;

                if (nid_cid == cache_nid_cid[0]) {
                    in_cache_id = 0;
                } else if (nid_cid == cache_nid_cid[1]) {
                    in_cache_id = 1;
                } else if (nid_cid == cache_nid_cid[2]) {
                    in_cache_id = 2;
                } else if (nid_cid == cache_nid_cid[3]) {
                    in_cache_id = 3;
                } else if (nid_cid == cache_nid_cid[4]) {
                    in_cache_id = 4;
                } else if (nid_cid == cache_nid_cid[5]) {
                    in_cache_id = 5;
                } else if (nid_cid == cache_nid_cid[6]) {
                    in_cache_id = 6;
                } else if (nid_cid == cache_nid_cid[7]) {
                    in_cache_id = 7;
                } else {
                    in_cache_id = 8;
                }
                if (in_cache_id != 8) {
                    for (int j = 0; j < MAX_SPLITS_1; j++) {
                        elem[j] = cache_elem[in_cache_id][j];
                    }
                } else {
                    for (int j = 0; j < MAX_SPLITS_1; j++) {
                        elem[j] = num_in_cur_nfs_cat[j][n_cid];
                    }
                }
                nxt_elem[MAX_SPLITS] = elem[MAX_SPLITS] + 1;
                for (int j = 0; j < MAX_SPLITS; j++) {
                    f_cast<MType> feature_val_;
                    feature_val_.i = onesample[j];
                    MType features_val = feature_val_.f;
                    MType thre = splits[j];
                    if (features_val <= thre) {
                        nxt_elem[j] = elem[j] + 1;
                    } else {
                        nxt_elem[j] = elem[j];
                    }
                }
                // write back to uram
                for (int j = 0; j < MAX_SPLITS_1; j++) {
                    num_in_cur_nfs_cat[j][n_cid] = nxt_elem[j];
                }
                // shift left
                for (int i = 7; i > 0; i--) {
                    cache_nid_cid[i] = cache_nid_cid[i - 1];
                    for (int j = 0; j < MAX_SPLITS_1; j++) {
                        cache_elem[i][j] = cache_elem[i - 1][j];
                    }
                }
                cache_nid_cid[0] = nid_cid;
                for (int j = 0; j < MAX_SPLITS_1; j++) {
                    cache_elem[0][j] = nxt_elem[j];
                }
            }
        }
    }

    ap_uint<64> clklogclk[PARA_NUM][MAX_SPLITS];
    ap_uint<64> crklogcrk[PARA_NUM][MAX_SPLITS];
    int num_in_cur_nfs_cat_sum[PARA_NUM][MAX_SPLITS];
#pragma HLS array_partition variable = clklogclk dim = 2 cyclic factor = 2
#pragma HLS array_partition variable = crklogcrk dim = 2 cyclic factor = 2
#pragma HLS array_partition variable = num_in_cur_nfs_cat_sum dim = 2 cyclic factor = 2
#pragma HLS bind_storage variable = clklogclk type = ram_2p impl = uram
#pragma HLS bind_storage variable = crklogcrk type = ram_2p impl = uram
#pragma HLS bind_storage variable = num_in_cur_nfs_cat_sum type = ram_2p impl = uram
compute_gain_loop_init:
    for (int i = 0; i < PARA_NUM; i++) {
#pragma HLS loop_tripcount min = PARA_NUM max = PARA_NUM avg = PARA_NUM
        for (int j = 0; j < MAX_SPLITS; j++) {
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
            clklogclk[i][j] = 0;
            crklogcrk[i][j] = 0;
            num_in_cur_nfs_cat_sum[i][j] = 0;
        }
    }
compute_gain_loop_gainratio:
    for (int k = 0; k < numClass; k++) {
#pragma HLS loop_tripcount min = 2 max = 16 avg = 8
        for (int l = 0; l < PARA_NUM; l++) {
#pragma HLS loop_tripcount min = PARA_NUM max = PARA_NUM avg = PARA_NUM
            for (int i = 0; i < MAX_SPLITS; i += 2) {
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
                for (int j = 0; j < 2; j++) {
                    int offset = l * MAX_CAT_NUM;
                    ap_uint<30> l_cat_k = num_in_cur_nfs_cat[i + j][offset + k];
                    ap_uint<30> r_cat_k = num_in_cur_nfs_cat[MAX_SPLITS][offset + k] - l_cat_k;

                    clklogclk[l][i + j] += l_cat_k * l_cat_k;
                    crklogcrk[l][i + j] += r_cat_k * r_cat_k;
                    num_in_cur_nfs_cat_sum[l][i + j] += l_cat_k;
                }
            }
        }
    }
    int num_in_cur_node_cat_sum = 0;
main_compute_gain_loop:
    for (int l = 0; l < cur_layer_nodes_num; l++) {
#pragma HLS loop_tripcount min = 1 max = PARA_NUM avg = PARA_NUM
        int offset = l * MAX_CAT_NUM;
        int maxClass_count = 0;
        int maxClass = 0;
        num_in_cur_node_cat_sum = 0;
        for (ap_uint<8> k = 0; k < numClass; k++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 2 max = 32 avg = 8
            int num_in_cur_node_cat_k = num_in_cur_nfs_cat[MAX_SPLITS][offset + k];
            num_in_cur_node_cat_sum += num_in_cur_node_cat_k;
            if (num_in_cur_node_cat_k > maxClass_count) {
                maxClass_count = num_in_cur_node_cat_k;
                maxClass = k;
            }
        }
        max_classes[l] = maxClass;
        int maxPerNums = paras.max_leaf_cat_per * num_in_cur_node_cat_sum;
        MType max_gainratio = 1 << 30;
        MType cretia = 1 << 30;
        int max_split_id = 0;
        MType max_split_value = splits[0];
        for (int i = 0; i < para_splits; i++) {
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#pragma HLS pipeline
            ap_uint<30> clsum = num_in_cur_nfs_cat_sum[l][i];
            ap_uint<30> crsum = num_in_cur_node_cat_sum - num_in_cur_nfs_cat_sum[l][i];
            if (clsum != 0 && crsum != 0) {
                cretia = num_in_cur_node_cat_sum - (clklogclk[l][i] / (MType)clsum + crklogcrk[l][i] / (MType)crsum);
            }
            if (cretia < max_gainratio) {
                max_gainratio = cretia;
                max_split_id = i;
                max_split_value = splits[i];
            }
        }
        max_feature_ids[l] = features_ids[max_split_id];
        max_split_values[l] = max_split_value;

        if (num_in_cur_node_cat_sum <= paras.min_leaf_size || maxClass_count > maxPerNums ||
            max_gainratio >= (1 << 30) || tree_dp >= paras.max_tree_depth) {
            ifstop[l] = true;
        } else {
            ifstop[l] = false;
        }
    }
}

template <typename MType, unsigned WD, unsigned MAX_TREE_DEPTH, unsigned MAX_SPLITS, unsigned PARA_NUM>
void updateTree(ap_uint<8> numClass,
                ap_uint<MAX_TREE_DEPTH> s_nodeid,
                ap_uint<MAX_TREE_DEPTH> e_nodeid,
                unsigned tree_dp,
                ap_uint<MAX_TREE_DEPTH>& next_layer_nodes_num,
                MType splits[MAX_SPLITS],
                struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                bool ifstop[PARA_NUM],
                int max_classes[PARA_NUM],
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
            nodes[tree_dp][cur_nodeid].nodeInfo.range(15, 8) = max_classes[j];
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
          unsigned MAX_CAT_NUM,
          unsigned PARA_NUM,
          unsigned MAX_TREE_DEPTH>
void decisionTreeFlow(ap_uint<512> data[DATASIZE],
                      ap_uint<8> features_num,
                      ap_uint<8> numClass,
                      unsigned samples_num,
                      unsigned tree_dp,
                      ap_uint<MAX_TREE_DEPTH> s_nodeid,
                      ap_uint<MAX_TREE_DEPTH> e_nodeid,
                      ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num,
                      Paras paras,
                      MType splits[MAX_SPLITS],
                      struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM],
                      int features_ids[MAX_SPLITS],
                      int para_splits,
                      int max_feature_ids[PARA_NUM],
                      MType max_split_values[PARA_NUM],
                      bool ifstop[PARA_NUM],
                      int max_classes[PARA_NUM]) {
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

    hls::stream<ap_uint<WD> > dstrm[MAX_SPLITS];
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
Count:
    statisticAndCompute<MType, _WAxi, WD, MAX_FEAS, MAX_CAT_NUM, MAX_SPLITS, PARA_NUM, MAX_TREE_DEPTH>(
        dstrm_batch_disp, nstrm_disp, splits, features_ids, para_splits, tree_dp, cur_layer_nodes_num, features_num,
        samples_num, numClass, s_nodeid, paras, ifstop, max_classes, max_feature_ids, max_split_values);
}

template <typename MType,
          int _WAxi,
          unsigned int WD,
          unsigned MAX_FEAS,
          unsigned MAX_SPLITS,
          unsigned MAX_CAT_NUM,
          unsigned PARA_NUM,
          unsigned MAX_TREE_DEPTH>
void decisionTreeFun(ap_uint<512> data[DATASIZE],
                     MType splits[MAX_SPLITS],
                     int features_ids[MAX_SPLITS],
                     int para_splits,
                     ap_uint<8> numClass,
                     ap_uint<8> features_num,
                     unsigned samples_num,
                     unsigned tree_dp,
                     ap_uint<MAX_TREE_DEPTH> s_nodeid,
                     ap_uint<MAX_TREE_DEPTH> e_nodeid,
                     ap_uint<MAX_TREE_DEPTH>& next_layer_nodes_num,
                     Paras paras,
                     struct Node<MType> nodes[MAX_TREE_DEPTH][MAX_NODES_NUM]) {
    int max_feature_ids[PARA_NUM];
    MType max_split_values[PARA_NUM];
    bool ifstop[PARA_NUM];
    int max_classes[PARA_NUM];
    ap_uint<MAX_TREE_DEPTH> cur_layer_nodes_num = e_nodeid - s_nodeid;

    decisionTreeFlow<MType, _WAxi, WD, MAX_FEAS, MAX_SPLITS, MAX_CAT_NUM, PARA_NUM, MAX_TREE_DEPTH>(
        data, features_num, numClass, samples_num, tree_dp, s_nodeid, e_nodeid, cur_layer_nodes_num, paras, splits,
        nodes, features_ids, para_splits, max_feature_ids, max_split_values, ifstop, max_classes);
    updateTree<MType, WD, MAX_TREE_DEPTH, MAX_SPLITS, PARA_NUM>(numClass, s_nodeid, e_nodeid, tree_dp,
                                                                next_layer_nodes_num, splits, nodes, ifstop,
                                                                max_classes, max_feature_ids, max_split_values);
}

} // namespace decisiontree
} // namespace classification
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
	bundle = gmem0_0 port = configs

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
    ap_uint<8> numClass;
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
    xf::data_analytics::classification::decisiontree::readConfig<DataType, 64, MAX_FEAS_, MAX_SPLITS_>(
        configs, samples_num, features_num, numClass, para_splits, paras, numSplits, splits, features_ids);

#ifdef __DT_DEBUG__
    printf("numClass: %d\n", numClass.to_int());
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

    struct Node<DataType> nodes[MAX_TREE_DEPTH_][MAX_NODES_NUM];
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
            xf::data_analytics::classification::decisiontree::decisionTreeFun<DataType, 512, 64, MAX_FEAS_, MAX_SPLITS_,
                                                                              MAX_CAT_NUM_, PARA_NUM_, MAX_TREE_DEPTH_>(
                data, splits, features_ids, para_splits, numClass, features_num, samples_num, tree_dp, s_nodeid,
                e_nodeid, layer_nodes_num[nxt_tree_dp], paras, nodes);
            s_nodeid = s_nodeid + PARA_NUM_;
        }
        tree_dp++;
    }
    // 3. warp and write out nodes
    xf::data_analytics::classification::decisiontree::writeOut<DataType, MAX_TREE_DEPTH_>(nodes, tree, nodes_num,
                                                                                          layer_nodes_num);

#ifdef __DT_DEBUG__
    printf("nodes_num:%d\n", nodes_num.to_int());
#endif
}
