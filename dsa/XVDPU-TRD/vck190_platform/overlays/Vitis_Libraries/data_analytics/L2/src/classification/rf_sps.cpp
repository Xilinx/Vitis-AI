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

#include "xf_data_analytics/common/table_sample.hpp"
#include "xf_data_analytics/common/utils.hpp"
#ifdef USE_RF_TREE_QT
#include "xf_data_analytics/classification/decision_tree_quantize.hpp"
#else
#include "xf_data_analytics/classification/decision_tree_L2.hpp"
#endif
#include "xf_utils_hw/stream_to_axi.hpp"
#include "xf_data_analytics/common/math_helper.hpp"
#define MAX_OUTER_FEAS MAX_FEAS_
#define MAX_INNER_FEAS MAX_FEAS_
#define MAX_SPLITS MAX_SPLITS_
const ap_uint<32> data_out_header_len = 1024;
const int WA = 8;

namespace xf {
namespace data_analytics {
namespace classification {
namespace internal {

template <int _WAxi, int _WData, int _BurstLen, typename MType, typename FType>
class RandForestImp {
   private:
    xf::data_analytics::common::internal::MT19937 rng;

   public:
    RandForestImp(){
#pragma HLS inline
    };

    void loadNumSplits(ap_uint<_WAxi>* configs,
                       const ap_uint<32> numsplit_start,
                       const ap_uint<32> numsplit_len,
                       const ap_uint<32> features_num,
                       ap_uint<8> numSplits[MAX_OUTER_FEAS]) {
        ap_uint<_WAxi> configs_in[20];
        ap_uint<_WAxi> onerow;
        for (int i = 0; i < numsplit_len; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
            configs_in[i] = configs[numsplit_start + i];
        }
        int r = 0;
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
    }

    void readSplitsiDDR(ap_uint<_WAxi>* configs,
                        const ap_uint<32> splits_start,
                        const ap_uint<32> splits_len,
                        hls::stream<ap_uint<_WAxi> >& vecStrm) {
        for (int i = 0; i < splits_len; i++) {
#pragma HLS loop_tripcount min = 10 max = 10 avg = 10
#pragma HLS pipeline
            vecStrm.write(configs[splits_start + i]);
        }
    }

    void Convert2DataStrm(hls::stream<ap_uint<_WAxi> >& vecStrm,
                          // ap_uint<32> splits_elem_num,
                          int splits_elem_num,
                          hls::stream<ap_uint<_WData> >& dataStrm) {
        int full_batch = _WAxi / _WData;
        ap_uint<_WAxi> tmp;
        for (int i = 0; i < splits_elem_num; i += full_batch) {
            tmp = vecStrm.read();
            int write_batch = ((i + full_batch) < splits_elem_num) ? full_batch : (splits_elem_num - i);
            for (int j = 0; j < write_batch; j++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
                int off = j * _WData;
                dataStrm.write(tmp.range(off + _WData - 1, off));
            }
        }
    }

    void Convert2DataArray(hls::stream<ap_uint<_WData> >& dataStrm,
                           ap_uint<8> numSplits[MAX_OUTER_FEAS],
                           const ap_uint<32> cols, // features_num+1
                           float fraction,
                           ap_uint<32>& cols_sp,
                           int featuresIndex[MAX_INNER_FEAS],
                           ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                           FType splits[MAX_SPLITS]) {
        ap_uint<MAX_OUTER_FEAS> fea_if_sampling = 0;
        cols_sp = (cols - 1) * fraction + 1;
        const int MAX_FEA_NUM_SMPLING = MAX_OUTER_FEAS;
        int j = 0;
        int j_tmp = 0;
        for (int i = 0; i < cols - 1; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = MAX_FEA_NUM_SMPLING max = MAX_FEA_NUM_SMPLING avg = MAX_FEA_NUM_SMPLING
            float tmpNext = rng.next();
            if (tmpNext < fraction && j < cols_sp - 1) {
                featuresIndex[j] = i;
                numSplits_in[j] = numSplits[i];
                fea_if_sampling.range(i, i) = true;
                j++;
            }
#ifndef __SYNTHESIS__
//          printf("%d,numSplits::%d\n", i,numSplits[i].to_int());
#endif
        }
        featuresIndex[j] = cols - 1;
        cols_sp = j + 1;
        for (int i = cols_sp; i < MAX_INNER_FEAS; i++) {
            featuresIndex[i] = MAX_INNER_FEAS;
        }
        // feaNumIn.write(cols_sp);
        int fea_ind = 0;
        ap_uint<8> n = numSplits[fea_ind];
        ap_uint<32> counter = 0;
        ap_uint<32> fea_counter = 0;
        while (n) {
            bool if_w = (fea_if_sampling >> fea_ind) & 0x01;
            if (if_w) numSplits_in[fea_counter++] = n;
#ifndef __SYNTHESIS__
            if (if_w)
                printf("if_w:%d,fea_ind:%d->fea_counter:%d,numSplits:%d\n", if_w, fea_ind, fea_counter.to_int(),
                       n.to_int());
#endif
            for (int i = 0; i < n; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
                f_cast<FType> d;
                d.i = dataStrm.read();
                if (if_w) {
                    splits[counter] = d.f;
                    counter++;
#ifndef __SYNTHESIS__
                    printf("d.f:%f\n", d.f);
#endif
                }
            }
            if (fea_ind == cols - 2) {
                n = 0;
            } else {
                n = numSplits[++fea_ind];
            }
        }
    }

    void readConfigFlow(ap_uint<_WAxi>* configs,
                        const ap_uint<32> splits_start,
                        const ap_uint<32> splits_len,
                        const ap_uint<32> splits_elem_num,
                        const ap_uint<32> cols, // features_num + 1
                        float fraction,
                        ap_uint<8> numSplits[MAX_OUTER_FEAS],
                        ap_uint<32>& cols_sp,
                        int featuresIndex[MAX_INNER_FEAS],
                        ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                        FType splits[MAX_SPLITS]) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<_WData> > dataStrm;

#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = dataStrm type = fifo impl = lutram
#pragma HLS stream variable = dataStrm depth = fifo_depth

#pragma HLS dataflow
        readSplitsiDDR(configs, splits_start, splits_len, vecStrm);
        Convert2DataStrm(vecStrm, splits_elem_num, dataStrm);
        Convert2DataArray(dataStrm, numSplits, cols, fraction, cols_sp, featuresIndex, numSplits_in, splits);
    }

    void genFeatureSubsets(const ap_uint<32> features_num_in,
                           const float fraction,
                           hls::stream<ap_uint<MAX_INNER_FEAS> >& featureSubStrm,
                           hls::stream<bool>& eFeatureSubStrm) {
        ap_uint<MAX_INNER_FEAS> featureSubsets[MAX_NODES_NUM];
        int k = features_num_in * fraction;
        const int maxnodes = MAX_NODES_NUM;
        for (int i = 0; i < MAX_NODES_NUM; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = maxnodes max = maxnodes avg = maxnodes
            featureSubsets[i] = -1;
            featureSubsets[i].range(MAX_INNER_FEAS - 1, k) = 0;
        }
        for (int j = k; j < features_num_in; j++) {
#pragma HLS loop_tripcount min = maxnodes max = maxnodes avg = maxnodes
            for (int i = 0; i < MAX_NODES_NUM; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = maxnodes max = maxnodes avg = maxnodes
                // double randd = rand.nextDouble();
                float randd = rng.next();
                int rep = randd * j;
                if (rep < k) {
                    featureSubsets[i].range(rep, rep) = 0;
                    featureSubsets[i].range(k, k) = 1;
                }
            }
        }
        for (int i = 0; i < MAX_NODES_NUM; i++) {
            featureSubStrm.write(featureSubsets[i]);
            eFeatureSubStrm.write(false);
        }
        eFeatureSubStrm.write(true);
    }

    void writeFeatureSubsets2DDR(hls::stream<ap_uint<MAX_INNER_FEAS> >& featureSubStrm,
                                 hls::stream<bool>& eFeatureSubStrm,
                                 const ap_uint<32> offset,
                                 ap_uint<_WAxi>* config) {
        xf::common::utils_hw::streamToAxi<_BurstLen, _WAxi, MAX_INNER_FEAS>(config + offset, featureSubStrm,
                                                                            eFeatureSubStrm);
    }

    // cols is the all features before spread into different trees, cols* fraction = features_num in genFeatureSubsets
    void featureSampling(hls::stream<MType> dstrmIn[_WAxi / _WData],
                         hls::stream<bool>& estrmIn,
                         ap_uint<32> cols,
                         ap_uint<32> cols_sp,
                         int featuresIndex[MAX_INNER_FEAS],
                         hls::stream<MType> dstrmOut[_WAxi / _WData],
                         hls::stream<bool>& estrmOut) {
        const int MAX_FEA_NUM_SMPLING = MAX_INNER_FEAS;
#ifndef __SYNTHESIS__
        printf("featuresIndex: ");
        for (int i = 0; i < cols; i++) {
            printf("%d ", featuresIndex[i]);
        }
        printf("Final cols_sp:%d\n", cols_sp.to_int());
        printf("\n");
#endif
        const int full_batch = _WAxi / _WData;
        const int full_batch_2 = full_batch + full_batch;
        int featuresIndex_r[MAX_FEA_NUM_SMPLING] = {0};
#pragma HLS array_partition variable = featuresIndex_r dim = 0 complete
        // sampling feature col index in batch_num streams
        int featuresIndex_c[MAX_FEA_NUM_SMPLING] = {0};
#pragma HLS array_partition variable = featuresIndex_c dim = 0 complete
    main_loop_1:
        for (int i = 0; i < MAX_FEA_NUM_SMPLING; i++) {
#pragma HLS loop_tripcount min = MAX_FEA_NUM_SMPLING max = MAX_FEA_NUM_SMPLING avg = MAX_FEA_NUM_SMPLING
#pragma HLS pipeline
            featuresIndex_r[i] = featuresIndex[i] / full_batch;
            featuresIndex_c[i] = featuresIndex[i] % full_batch;
#ifndef __SYNTHESIS__
//    printf("r:%d,c:%d\n", featuresIndex_r[i], featuresIndex_c[i]);
#endif
        }
        // summery: in one all_sample batch_num * dstrmIn[_WAxi/_WData]
        // the i'th sampling feature is (featuresIndex_r[i],dstrmIn[featuresIndex_r[i]]) in batch_num streams
        //  int cols_sp = cols * fraction;
        int batch_num = (cols + full_batch - 1) / full_batch;
        int batch_num_sampling = (cols_sp + full_batch - 1) / full_batch;
        const int tmp_tail_batch = cols_sp % full_batch;
        const int tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        bool bn_con = (batch_num_sampling == 1) ? true : false;
        bool e = estrmIn.read();
        MType tmps_origin[full_batch];
#pragma HLS array_partition variable = tmps_origin dim = 0 complete
        MType tmps_sp[full_batch_2] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
#pragma HLS array_partition variable = tmps_sp dim = 0 complete
        ap_uint<2 * _WAxi> cage = 0;
        int bn_sp = 0;
        int bn = 0;
        int batch_valid_num[8]; // = {5,4,5};
#pragma HLS array_partition variable = batch_valid_num dim = 0 complete
        for (int i = 0; i < 8; i++) {
            batch_valid_num[i] = 0;
        }
        for (int i = 0; i < MAX_FEA_NUM_SMPLING; i++) {
            int index = featuresIndex_r[i];
            batch_valid_num[index]++;
        }
        int bc = batch_valid_num[0];
        bool branch = false;
        int branch_index = featuresIndex_r[(batch_num_sampling - 1) * full_batch - 1];
        if (branch_index == batch_num - 1) branch = true;

        int debug = 0;
    main_loop:
        while (!e) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
            if (bn == batch_num && branch) {
                for (int i = 0; i < full_batch; i++) {
                    tmps_sp[i] = cage.range((i + 1) * _WData - 1, i * _WData);
                    dstrmOut[i].write(tmps_sp[i]);
                }
#ifndef __SYNTHESIS__
#endif
                estrmOut.write(false);
                cage = 0;
                bn = 0;
                bn_sp = 0;
            } else {
                for (int i = 0; i < full_batch; i++) {
                    tmps_origin[i] = dstrmIn[i].read();
                    tmps_sp[i] = cage.range((i + 1) * _WData - 1, i * _WData);
                }

#ifndef __SYNTHESIS__
                debug++;
                if (debug < 10) {
                    for (int i = 0; i < full_batch; i++) {
                        f_cast<double> tmp;
                        tmp.i = tmps_origin[i];
                        printf("%lf ", tmp.f);
                    }
                    printf("\n");
                }
#endif

                for (int i = 0; i < full_batch_2; i++) {
                    int off = bn_sp * full_batch;
                    int c = featuresIndex_c[off + i];
                    int r = featuresIndex_r[off + i];
                    if (r == bn) {
                        tmps_sp[i] = tmps_origin[c];
#ifndef __SYNTHESIS__
                        if (debug < 10) printf("!c:%d,r:%d,bn:%d,bc:%d\n", c, r, bn, bc);
#endif
                    }
#ifndef __SYNTHESIS__
//                    printf("c:%d,r:%d,bn:%d,bc:%d,value:%ld\n", c, r, bn, bc, tmps_origin[c].to_int());
#endif
                }

                for (int i = 0; i < full_batch_2; i++) {
                    cage.range((i + 1) * _WData - 1, i * _WData) = tmps_sp[i];
                }

                bn++;
                int bn_nxt = bn;
                int tmp_bc = bc;
                // if (bn_con && tmp_bc == cols_sp) bc = 0;
                if (bn == batch_num) {
                    if (!branch) {
                        bn = 0;
                    }
                    bn_nxt = 0;
                    bc = 0;
                }

                if (bc < full_batch) {
                    bc += batch_valid_num[bn_nxt];
                } else {
                    bc += (batch_valid_num[bn_nxt] - full_batch);
                }

                if (tmp_bc >= full_batch || (bn_sp == batch_num_sampling - 1 && tmp_bc == tail_batch)) {
                    for (int i = 0; i < full_batch; i++) {
                        dstrmOut[i].write(tmps_sp[i]);
                    }
                    estrmOut.write(false);
#ifndef __SYNTHESIS__
                    if (debug < 10) {
                        for (int i = 0; i < full_batch; i++) {
                            f_cast<double> tmp;
                            tmp.i = tmps_sp[i];
                            printf(" writestrm : %lf ", tmp.f);
                        }
                        printf("\n");
                    }
#endif
                    bn_sp++;
                    if (bn_sp == batch_num_sampling) {
                        bn_sp = 0;
                        cage = 0;
                    }
                    cage >>= _WAxi;
                }
                e = estrmIn.read();
            }
        }
        if (branch) {
            for (int i = 0; i < full_batch; i++) {
                tmps_sp[i] = cage.range((i + 1) * _WData - 1, i * _WData);
                dstrmOut[i].write(tmps_sp[i]);
            }
            estrmOut.write(false);
        }
        estrmOut.write(true);
    }

    void readRaw(ap_uint<_WAxi>* ddr,
                 const ap_uint<32> offset,
                 const ap_uint<32> start,
                 const ap_uint<32> rows,
                 const ap_uint<32> cols,
                 hls::stream<ap_uint<_WAxi> >& vecStrm,
                 hls::stream<ap_uint<32> >& firstShiftStrm) {
        const ap_uint<64> passedElements = start * cols;
        const ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
        const ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
        const ap_uint<64> finishElements = (start + rows) * cols;
        const ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
        const ap_uint<64> nread = finishWAxi - passedWAxi;
        const ap_uint<64> realOffset = offset + passedWAxi;

        firstShiftStrm.write(passedTail);
    READ_RAW:
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            vecStrm.write(ddr[realOffset + i]);
        }
    }

    void varSplit(hls::stream<ap_uint<_WAxi> >& vecStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  hls::stream<MType> data[_WAxi / _WData],
                  hls::stream<bool>& eData) {
        const ap_uint<32> full_batch = (_WAxi / _WData);
        const ap_uint<32> tmp_tail_batch = cols % full_batch;
        const ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        const ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
        const ap_uint<64> n_op = rows * batch_num;

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        ap_uint<32> firstShift = firstShiftStrm.read();
        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vecStrm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }
            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vecStrm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                if (k < output) {
                    tmp[k] = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp[k] = 0;
                }
                f_cast<MType> cc;
                cc.i = tmp[k];
                data[k].write(cc.f);
            }
            eData.write(false);
        }
        eData.write(true);
    }

    void cage_shift_right(ap_uint<_WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
        if (s >= 0 && s <= _WAxi / _WData) {
            source >>= s * _WData;
        }
    }
    /*698~956 lines are extracted from table_sample.hpp:tableRandomLoader*/
    void read_raw(ap_uint<_WAxi>* ddr,
                  const ap_uint<64> nread,
                  const ap_uint<64> realOffset,
                  hls::stream<ap_uint<_WAxi> >& vec_strm) {
        for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            vec_strm.write(ddr[realOffset + i]);
        }
    }
    void varSplit(hls::stream<ap_uint<_WAxi> >& vec_strm,
                  const ap_uint<32> firstShift,
                  const ap_uint<32> tail_batch,
                  const ap_uint<32> batch_num,
                  const ap_uint<64> n_op,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        const ap_uint<32> full_batch = (_WAxi / _WData);

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;

        if (firstShift != 0) {
            reserve = (_WAxi / _WData) - firstShift;
            inventory = vec_strm.read();
            inventory >>= firstShift * _WData;
        }
        for (ap_uint<64> i = 0; i < n_op; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            ap_uint<32> output;
            if (batch_counter == batch_num - 1) {
                output = tail_batch;
            } else {
                output = full_batch;
            }
            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }

            ap_uint<_WAxi> new_come;
            ap_uint<32> tmp_reserve = reserve;
            if (reserve < output) {
                new_come = vec_strm.read();
                reserve += (full_batch - output);
            } else {
                new_come = 0;
                reserve -= output;
            }

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

            cage_shift_right(cage, output);
            inventory = cage.range(_WAxi - 1, 0);

            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                ap_uint<_WData> tmp;
                if (k < output) {
                    tmp = pre_local_output.range((k + 1) * _WData - 1, k * _WData);
                } else {
                    tmp = 0;
                }
                dataStrm[k].write(tmp);
            }
            eDataStrm.write(false);
        }
    }

    void axiVarColToStreams(ap_uint<_WAxi>* ddr,
                            const ap_uint<64> nread,
                            const ap_uint<64> realOffset,
                            const ap_uint<32> firstShift,
                            const ap_uint<32> tail_batch,
                            const ap_uint<32> batch_num,
                            const ap_uint<64> n_op,
                            hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                            hls::stream<bool>& eDataStrm) {
        const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vec_strm;
#pragma HLS bind_storage variable = vec_strm type = fifo impl = lutram
#pragma HLS stream variable = vec_strm depth = fifo_depth
#pragma HLS dataflow
        read_raw(ddr, nread, realOffset, vec_strm);
        varSplit(vec_strm, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
    }

    void genRandom(const ap_uint<32> rows,
                   const ap_uint<32> cols,
                   const ap_uint<32> offset,
                   const ap_uint<32> bucketSize,
                   bool ifJump,
                   float fraction,
                   hls::stream<ap_uint<64> >& nreadStrm,
                   hls::stream<ap_uint<64> >& realOffsetStrm,
                   hls::stream<ap_uint<32> >& firstShiftStrm,
                   hls::stream<ap_uint<32> >& tailBatchStrm,
                   hls::stream<ap_uint<32> >& batchNumStrm,
                   hls::stream<ap_uint<64> >& nOpStrm,
                   hls::stream<bool>& eAxiCfgStrm,
                   hls::stream<bool>& ifDropStrm) {
        if (ifJump) {
            float log1MF = xf::data_analytics::internal::m::log(1 - fraction);

            bool e = false;
            ap_uint<32> start = 0;
            while (!e) {
#pragma HLS pipeline
#pragma HLS loop_tripcount max = 3 min = 3 avg = 3
                ap_ufixed<33, 0> ftmp = rng.next();
                ftmp[0] = 1;
                float tmp = ftmp;
                int jump = xf::data_analytics::internal::m::log(tmp) / log1MF;
                ap_uint<32> nextstart = start + jump * bucketSize;
                ap_uint<32> nextrows;

                if (nextstart < rows) {
                    if ((nextstart + bucketSize) > rows) {
                        nextrows = rows - nextstart;
                        e = true;
                    } else {
                        nextrows = bucketSize;
                    }

                    ap_uint<64> passedElements = nextstart * cols;
                    ap_uint<64> passedWAxi = passedElements / (_WAxi / _WData);
                    ap_uint<32> passedTail = passedElements % (_WAxi / _WData);
                    ap_uint<64> finishElements = (nextstart + nextrows) * cols;
                    ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
                    ap_uint<64> nread = finishWAxi - passedWAxi;
                    ap_uint<64> realOffset = offset + passedWAxi;

                    ap_uint<32> full_batch = (_WAxi / _WData);
                    ap_uint<32> tmp_tail_batch = cols % full_batch;
                    ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? (full_batch) : tmp_tail_batch;
                    ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
                    ap_uint<64> n_op = nextrows * batch_num;

                    nreadStrm.write(nread);
                    realOffsetStrm.write(realOffset);
                    firstShiftStrm.write(passedTail);
                    tailBatchStrm.write(tail_batch);
                    batchNumStrm.write(batch_num);
                    nOpStrm.write(n_op);
                    eAxiCfgStrm.write(false);
                } else {
                    e = true;
                }
                start += (1 + jump) * bucketSize;
            }
            eAxiCfgStrm.write(true);
        } else {
            ap_uint<64> finishElements = rows * cols;
            ap_uint<64> finishWAxi = (finishElements + ((_WAxi / _WData) - 1)) / (_WAxi / _WData);
            ap_uint<64> nread = finishWAxi;
            ap_uint<64> realOffset = offset;

            ap_uint<32> full_batch = (_WAxi / _WData);
            ap_uint<32> tmp_tail_batch = cols % full_batch;
            ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
            ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;
            ap_uint<64> n_op = rows * batch_num;

            nreadStrm.write(nread);
            realOffsetStrm.write(realOffset);
            firstShiftStrm.write(0);
            tailBatchStrm.write(tail_batch);
            batchNumStrm.write(batch_num);
            nOpStrm.write(n_op);

            eAxiCfgStrm.write(false);
            eAxiCfgStrm.write(true);
            for (ap_uint<32> i = 0; i < rows; i++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 100 min = 100 avg = 100
                float tmpNext = rng.next();
                if (tmpNext < fraction) {
                    ifDropStrm.write(false);
                } else {
                    ifDropStrm.write(true);
                }
            }
        }
    }

    void jumpScan(ap_uint<_WAxi>* ddr,
                  hls::stream<ap_uint<64> >& nreadStrm,
                  hls::stream<ap_uint<64> >& realOffsetStrm,
                  hls::stream<ap_uint<32> >& firstShiftStrm,
                  hls::stream<ap_uint<32> >& tailBatchStrm,
                  hls::stream<ap_uint<32> >& batchNumStrm,
                  hls::stream<ap_uint<64> >& nOpStrm,
                  hls::stream<bool>& eAxiCfgStrm,
                  hls::stream<ap_uint<_WData> > dataStrm[_WAxi / _WData],
                  hls::stream<bool>& eDataStrm) {
        while (!eAxiCfgStrm.read()) {
#pragma HLS loop_tripcount max = 1 min = 1 avg = 1
            const ap_uint<64> nread = nreadStrm.read();
            const ap_uint<64> realOffset = realOffsetStrm.read();
            const ap_uint<32> firstShift = firstShiftStrm.read();
            const ap_uint<32> tail_batch = tailBatchStrm.read();
            const ap_uint<32> batch_num = batchNumStrm.read();
            const ap_uint<64> n_op = nOpStrm.read();
            axiVarColToStreams(ddr, nread, realOffset, firstShift, tail_batch, batch_num, n_op, dataStrm, eDataStrm);
        }
        eDataStrm.write(true);
    }

    void dropControl(const bool ifJump,
                     const ap_uint<32> cols,
                     hls::stream<bool>& ifDropStrm,
                     hls::stream<ap_uint<_WData> > inDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eInDataStrm,
                     hls::stream<MType> outDataStrm[_WAxi / _WData],
                     hls::stream<bool>& eOutDataStrm) {
        ap_uint<32> colCounter = 0;
        bool drop = false;

        while (!eInDataStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount max = 300 min = 300 avg = 300
            if (!ifJump) {
                if (colCounter == 0) {
                    drop = ifDropStrm.read();
                }
            }

            ap_uint<_WData> tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                tmp[i] = inDataStrm[i].read();
            }
            if (ifJump || !drop) {
                for (int i = 0; i < (_WAxi / _WData); i++) {
#pragma HLS unroll
                    f_cast<MType> cc;
                    cc.i = tmp[i];
                    outDataStrm[i].write(cc.f);
                }
                eOutDataStrm.write(false);
            }

            colCounter += (_WAxi / _WData);
            if (colCounter >= cols) {
                colCounter = 0;
            }
        }
        eOutDataStrm.write(true);
    }
    /*698~956 lines are extracted from table_sample.hpp:tableRandomLoader*/

    void store_1(hls::stream<MType> data[_WAxi / _WData],
                 hls::stream<bool>& eData,
                 const ap_uint<32> cols,
                 ap_uint<_WAxi>* ddr) {
        bool e = eData.read();
        int bn = 0;
        while (!e) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = 64 max = 64 avg = 64
            for (int k = 0; k < 8; k++) {
                int off = k * _WData;
                MType tmp = data[k].read();
                ddr[bn].range(off + 63, off) = tmp;
            }
            bn++;
            if (bn == (cols + 7) / 8) {
                bn = 0;
                e = eData.read();
            }
        }
    }
    void burstWrite(ap_uint<_WAxi>* wbuf,
                    const ap_uint<32> offset,
                    hls::stream<ap_uint<_WAxi> >& axi_strm,
                    hls::stream<ap_uint<8> >& nb_strm) {
        // write each burst to axi
        ap_uint<_WAxi> tmp;
        int n = nb_strm.read();
        int total = 0;
        int total_axi = n;
    doing_burst:
        while (n) {
        doing_one_burst:
            for (int i = 0; i < n; i++) {
#pragma HLS pipeline II = 1
                tmp = axi_strm.read();
                wbuf[total * _BurstLen + i + offset] = tmp;
            }
            total++;
            n = nb_strm.read();
            total_axi += n;
        }
        wbuf[0].range(511, 480) = total_axi;
#ifndef __SYNTHESIS__
        printf("total:%d\n", total_axi);
#endif
    }

    void varMerge(hls::stream<MType> data[_WAxi / _WData],
                  hls::stream<bool>& eData,
                  const ap_uint<32> cols,
                  hls::stream<ap_uint<_WAxi> >& vecStrm,
                  hls::stream<ap_uint<8> >& nb_strm) {
        bool e = eData.read();
        const ap_uint<32> full_batch = (_WAxi / _WData);
        const ap_uint<32> tmp_tail_batch = cols % full_batch;
        const ap_uint<32> tail_batch = (tmp_tail_batch == 0) ? full_batch : tmp_tail_batch;
        const ap_uint<32> batch_num = (cols + full_batch - 1) / full_batch;

        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter = 0;
        ap_uint<32> burst_counter = 0;
        ap_uint<_WAxi> new_come;

        while (!e) {
#pragma HLS loop_tripcount min = 300 max = 300 avg = 300
#pragma HLS pipeline II = 1

            e = eData.read();
            ap_uint<32> valid_num;
            if (batch_counter == batch_num - 1) {
                valid_num = tail_batch;
            } else {
                valid_num = full_batch;
            }

            batch_counter++;
            if (batch_counter == batch_num) {
                batch_counter = 0;
            }
            MType tmp[_WAxi / _WData];
#pragma HLS array_partition variable = tmp dim = 1 complete
            for (int k = 0; k < full_batch; k++) {
#pragma HLS unroll
                tmp[k] = data[k].read();
                f_cast<MType> cc;
                cc.f = tmp[k];
                new_come.range((k + 1) * _WData - 1, k * _WData) = cc.i;
            }

            ap_uint<32> tmp_reserve = reserve;
            reserve += valid_num;

            ap_uint<_WAxi* 2> cage = 0;
            cage.range(_WAxi * 2 - 1, _WAxi) = new_come;
            cage_shift_right(cage, full_batch - tmp_reserve);

            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            if (reserve >= full_batch) {
                reserve -= full_batch;
                ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);
                vecStrm.write(pre_local_output);
                burst_counter++;
                if (burst_counter == (_BurstLen)) {
                    nb_strm.write(_BurstLen);
                    burst_counter = 0;
                }
                // eStrm.write(false);
                cage >>= _WAxi;
                //  cage_shift_right(cage, full_batch);
            }
            inventory = cage.range(_WAxi - 1, 0);
        }
        if (reserve != 0) {
            burst_counter++;
            vecStrm.write(inventory);
        }
        if (burst_counter != 0) {
            nb_strm.write(burst_counter);
        }
        nb_strm.write(0);
    }
    void quantile2AXi(hls::stream<ap_uint<_WData> > dInStrm[_WAxi / _WData],
                      hls::stream<bool>& eInStrm,
                      ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                      FType splits[MAX_SPLITS],
                      ap_uint<32> cols_sp,
                      hls::stream<ap_uint<_WAxi> >& vecStrm,
                      hls::stream<ap_uint<8> >& nb_strm) {
        bool e = eInStrm.read();
        ap_uint<32> features_num_in = cols_sp - 1;
        const int full_batch_wd = _WAxi / _WData;
        const int full_batch_wa = _WAxi / WA;
        int tmp_tail_batch = cols_sp % full_batch_wd;
        int tail_batch = (tmp_tail_batch == 0) ? full_batch_wd : tmp_tail_batch;
        int batch_num_wd = (cols_sp + full_batch_wd - 1) / full_batch_wd;
        int batch_num_wa = (cols_sp + full_batch_wa - 1) / full_batch_wa;
        ap_uint<8> presum[MAX_INNER_FEAS + 1];
#pragma HLS array_partition variable = presum dim = 0 complete
        presum[0] = 0;
        for (int i = 0; i < MAX_INNER_FEAS; i++) {
            presum[i + 1] = presum[i] + numSplits_in[i];
#ifndef __SYNTHESIS__
            printf("presum:%d\n", presum[i + 1].to_int());
#endif
        }
#ifndef __SYNTHESIS__
        printf("quantile_cols_sp:%d,batch_num_wd:%d,batch_num_wa:%d\n", cols_sp.to_int(), batch_num_wd, batch_num_wa);

#endif
        FType splits_dups[MAX_INNER_FEAS][4][MAX_SPLITS];
#pragma HLS array_partition variable = splits_dups dim = 1 complete
#pragma HLS array_partition variable = splits_dups dim = 2 complete
#pragma HLS bind_storage variable = splits_dups type = ram_2p impl = uram
        for (ap_uint<8> i = 0; i < features_num_in; i++) {
            for (ap_uint<8> j = 0; j < numSplits_in[i]; j++) {
                for (ap_uint<8> k = 0; k < 4; k++) {
#pragma HLS pipeline
                    splits_dups[i][k][j] = splits[j + presum[i]];
                }
            }
        }
        ap_uint<32> reserve = 0;
        ap_uint<_WAxi> inventory = 0;
        ap_uint<32> batch_counter_wd = 0;
        ap_uint<32> burst_counter = 0;

        FType tmps_wd[full_batch_wd];
#pragma HLS array_partition variable = tmps_wd dim = 0 complete
        ap_uint<WA> tmps_wa[full_batch_wd];
#pragma HLS array_partition variable = tmps_wa dim = 0 complete
        ap_uint<32> valid_num = 0;
        ap_uint<32> featureBatchStart = 0;
        while (!e) {
#pragma HLS pipeline
            e = eInStrm.read();
            if (batch_counter_wd == batch_num_wd - 1) {
                valid_num = tail_batch;
            } else {
                valid_num = full_batch_wd;
            }
            ap_uint<full_batch_wd* WA> new_come = 0;
            for (int i = 0; i < full_batch_wd; i++) {
                ap_uint<32> featureInd = i + featureBatchStart;
                f_cast<FType> tmp;
                tmp.i = dInStrm[i].read();
                tmps_wd[i] = tmp.f;
                tmps_wa[i] = (uint8_t)tmp.f;
                if (featureInd < features_num_in) {
                    bool tag = false;
                    ap_uint<8> start = 0;
                    ap_uint<8> end = numSplits_in[featureInd];
                    int index = (start + end) >> 1;
                    for (int j = 0; j < 8; j++) {
                        FType v1 = splits_dups[i + batch_counter_wd * 8][j >> 1][index];
                        bool lg = tmps_wd[i] > v1 ? true : false;
                        if (end - start == 1 && !tag) {
                            tag = true;
                            if (!lg) {
                                tmps_wa[i] = start;
                            } else {
                                tmps_wa[i] = end;
                            }
                        }
                        if (!tag) {
                            if (!lg) {
                                end = index;
                            } else {
                                start = index;
                            }
                            index = (start + end) >> 1;
                        }
                    }
                }
                new_come.range((i + 1) * WA - 1, i * WA) = tmps_wa[i];
            }

            batch_counter_wd++;
            if (batch_counter_wd == batch_num_wd) {
                featureBatchStart = 0;
                batch_counter_wd = 0;
            } else {
                featureBatchStart += full_batch_wd;
            }

            ap_uint<32> tmp_reserve = reserve;
            reserve += valid_num;

            ap_uint<_WAxi + full_batch_wd* WA> cage = 0;
            cage.range(_WAxi + full_batch_wd * WA - 1, _WAxi) = new_come;
            if (full_batch_wa - tmp_reserve > 0) cage >>= (full_batch_wa - tmp_reserve) * WA;
            cage.range(_WAxi - 1, 0) = cage.range(_WAxi - 1, 0) ^ inventory.range(_WAxi - 1, 0);

            if (reserve >= full_batch_wa) {
                reserve -= full_batch_wa;
                ap_uint<_WAxi> pre_local_output = cage.range(_WAxi - 1, 0);

                vecStrm.write(pre_local_output);
                burst_counter++;
                if (burst_counter == (_BurstLen)) {
                    nb_strm.write(_BurstLen);
                    burst_counter = 0;
                }
                cage >>= _WAxi;
            }
            inventory = cage.range(_WAxi - 1, 0);
        }
        if (reserve != 0) {
            burst_counter++;
            vecStrm.write(inventory);
        }
        if (burst_counter != 0) {
            nb_strm.write(burst_counter);
        }
        nb_strm.write(0);
    }

    void seedInitialization(ap_uint<32> seed) { rng.seedInitialization(seed); }
    // instance sampling  + quantilize + writeout
    void sample_1(ap_uint<_WAxi>* ddr,
                  const ap_uint<32> offset,
                  const ap_uint<32> start,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  const float instance_fraction,
                  const ap_uint<32> cols_sp,
                  const ap_uint<32> data_out_header_len,
                  ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                  FType splits[MAX_SPLITS],
                  ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;

        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth

#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth

#pragma HLS dataflow

        genRandom(rows, cols, offset, 0, false, instance_fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(false, cols, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm_t, eDataStrm_t);

        quantile2AXi(dataStrm_t, eDataStrm_t, numSplits_in, splits, cols_sp, vecStrm_w, nb_strm_w);

        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }

    // instance sampling  + writeout
    void sample_2(ap_uint<_WAxi>* ddr,
                  const ap_uint<32> offset,
                  const ap_uint<32> start,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  const float instance_fraction,
                  const ap_uint<32> data_out_header_len,
                  ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;

        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth

#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth

#pragma HLS dataflow

        genRandom(rows, cols, offset, 0, false, instance_fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(false, cols, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm_t, eDataStrm_t);

        varMerge(dataStrm_t, eDataStrm_t, cols, vecStrm_w, nb_strm_w);

        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }
    // instance sp + feature sp + quantile + burstWrite
    void sample(ap_uint<_WAxi>* ddr,
                const ap_uint<32> offset,
                const ap_uint<32> start,
                const ap_uint<32> rows,
                const ap_uint<32> cols,
                const float instance_fraction,
                const ap_uint<32> cols_sp,
                const ap_uint<32> data_out_header_len,
                int featuresIndex[MAX_INNER_FEAS],
                ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                FType splits[MAX_SPLITS],
                ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;

        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<MType> dataStrm[_WAxi / _WData];
        hls::stream<bool> eDataStrm;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = dataStrm type = fifo impl = lutram
#pragma HLS stream variable = dataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth

#pragma HLS dataflow

        genRandom(rows, cols, offset, 0, false, instance_fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(false, cols, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm_t, eDataStrm_t);

        featureSampling(dataStrm_t, eDataStrm_t, cols, cols_sp, featuresIndex, dataStrm, eDataStrm);

        // varMerge(dataStrm, eDataStrm, cols_sp, vecStrm_w, nb_strm_w);
        // burstWrite(ddr_out,1, vecStrm_w, nb_strm_w);
        quantile2AXi(dataStrm, eDataStrm, numSplits_in, splits, cols_sp, vecStrm_w, nb_strm_w);
        // burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }

    // instance sampling + feature sampling + varMerge + burstWrite
    void sample_a(ap_uint<_WAxi>* ddr,
                  const ap_uint<32> offset,
                  const ap_uint<32> start,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  const float instance_fraction,
                  const ap_uint<32> cols_sp,
                  const ap_uint<32> data_out_header_len,
                  int featuresIndex[MAX_INNER_FEAS],
                  ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<64> > nreadStrm;
        hls::stream<ap_uint<64> > realOffsetStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<ap_uint<32> > tailBatchStrm;
        hls::stream<ap_uint<32> > batchNumStrm;
        hls::stream<ap_uint<64> > nOpStrm;
        hls::stream<bool> eAxiCfgStrm;
        hls::stream<bool> ifDropStrm;
        hls::stream<ap_uint<_WData> > interDataStrm[_WAxi / _WData];
        hls::stream<bool> eInterDataStrm;

        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<MType> dataStrm[_WAxi / _WData];
        hls::stream<bool> eDataStrm;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = nreadStrm type = fifo impl = lutram
#pragma HLS stream variable = nreadStrm depth = 4
#pragma HLS bind_storage variable = realOffsetStrm type = fifo impl = lutram
#pragma HLS stream variable = realOffsetStrm depth = 4
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 4
#pragma HLS bind_storage variable = tailBatchStrm type = fifo impl = lutram
#pragma HLS stream variable = tailBatchStrm depth = 4
#pragma HLS bind_storage variable = batchNumStrm type = fifo impl = lutram
#pragma HLS stream variable = batchNumStrm depth = 4
#pragma HLS bind_storage variable = nOpStrm type = fifo impl = lutram
#pragma HLS stream variable = nOpStrm depth = 4
#pragma HLS bind_storage variable = eAxiCfgStrm type = fifo impl = lutram
#pragma HLS stream variable = eAxiCfgStrm depth = 4

#pragma HLS bind_storage variable = ifDropStrm type = fifo impl = lutram
#pragma HLS stream variable = ifDropStrm depth = fifo_depth
#pragma HLS bind_storage variable = interDataStrm type = fifo impl = lutram
#pragma HLS stream variable = interDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eInterDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eInterDataStrm depth = fifo_depth

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = dataStrm type = fifo impl = lutram
#pragma HLS stream variable = dataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm depth = fifo_depth
#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth

#pragma HLS dataflow

        genRandom(rows, cols, offset, 3, false, instance_fraction, nreadStrm, realOffsetStrm, firstShiftStrm,
                  tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm, ifDropStrm);
        jumpScan(ddr, nreadStrm, realOffsetStrm, firstShiftStrm, tailBatchStrm, batchNumStrm, nOpStrm, eAxiCfgStrm,
                 interDataStrm, eInterDataStrm);
        dropControl(false, cols, ifDropStrm, interDataStrm, eInterDataStrm, dataStrm_t, eDataStrm_t);

        featureSampling(dataStrm_t, eDataStrm_t, cols, cols_sp, featuresIndex, dataStrm, eDataStrm);
        varMerge(dataStrm, eDataStrm, cols_sp, vecStrm_w, nb_strm_w);
        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }
    //
    void sample_b(ap_uint<_WAxi>* ddr,
                  const ap_uint<32> offset,
                  const ap_uint<32> start,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  const ap_uint<32> cols_sp,
                  const ap_uint<32> data_out_header_len,
                  int featuresIndex[MAX_INNER_FEAS],
                  ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                  FType splits[MAX_SPLITS],
                  ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<MType> dataStrm[_WAxi / _WData];
        hls::stream<bool> eDataStrm;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 2

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = dataStrm type = fifo impl = lutram
#pragma HLS stream variable = dataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm depth = fifo_depth

#pragma HLS dataflow
        readRaw(ddr, offset, start, rows, cols, vecStrm, firstShiftStrm);
        //        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm, eDataStrm);
        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm_t, eDataStrm_t);
        featureSampling(dataStrm_t, eDataStrm_t, cols, cols_sp, featuresIndex, dataStrm, eDataStrm);

        // varMerge(dataStrm, eDataStrm, cols_sp, vecStrm_w, nb_strm_w);
        // burstWrite(ddr_out,1, vecStrm_w, nb_strm_w);
        quantile2AXi(dataStrm, eDataStrm, numSplits_in, splits, cols_sp, vecStrm_w, nb_strm_w);
        // burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }

    void sample_c(ap_uint<_WAxi>* ddr,
                  const ap_uint<32> offset,
                  const ap_uint<32> start,
                  const ap_uint<32> rows,
                  const ap_uint<32> cols,
                  const ap_uint<32> cols_sp,
                  const ap_uint<32> data_out_header_len,
                  int featuresIndex[MAX_INNER_FEAS],
                  ap_uint<_WAxi>* ddr_out) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<_WAxi> > vecStrm;
        hls::stream<ap_uint<32> > firstShiftStrm;
        hls::stream<MType> dataStrm_t[_WAxi / _WData];
        hls::stream<bool> eDataStrm_t;
        hls::stream<MType> dataStrm[_WAxi / _WData];
        hls::stream<bool> eDataStrm;
        hls::stream<ap_uint<_WAxi> > vecStrm_w;
        hls::stream<ap_uint<8> > nb_strm_w;

#pragma HLS bind_storage variable = vecStrm_w type = fifo impl = lutram
#pragma HLS stream variable = vecStrm_w depth = fifo_depth
#pragma HLS bind_storage variable = nb_strm_w type = fifo impl = lutram
#pragma HLS stream variable = nb_strm_w depth = fifo_depth
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifo_depth
#pragma HLS bind_storage variable = firstShiftStrm type = fifo impl = lutram
#pragma HLS stream variable = firstShiftStrm depth = 2

#pragma HLS bind_storage variable = dataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = dataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm_t type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm_t depth = fifo_depth
#pragma HLS bind_storage variable = dataStrm type = fifo impl = lutram
#pragma HLS stream variable = dataStrm depth = fifo_depth
#pragma HLS bind_storage variable = eDataStrm type = fifo impl = lutram
#pragma HLS stream variable = eDataStrm depth = fifo_depth
#pragma HLS dataflow
        readRaw(ddr, offset, start, rows, cols, vecStrm, firstShiftStrm);
        varSplit(vecStrm, firstShiftStrm, rows, cols, dataStrm_t, eDataStrm_t);
        featureSampling(dataStrm_t, eDataStrm_t, cols, cols_sp, featuresIndex, dataStrm, eDataStrm);
        varMerge(dataStrm, eDataStrm, cols_sp, vecStrm_w, nb_strm_w);
        burstWrite(ddr_out, data_out_header_len, vecStrm_w, nb_strm_w);
    }

    void sample_d(const ap_uint<32> features_num,
                  const float fraction_1,
                  const ap_uint<32> offset, // 30
                  ap_uint<_WAxi>* config) {
        static const int fifo_depth = _BurstLen * 2;
        hls::stream<ap_uint<MAX_INNER_FEAS> > featureSubStrm;
        hls::stream<bool> eFeatureSubStrm;
#pragma HLS bind_storage variable = featureSubStrm type = fifo impl = lutram
#pragma HLS stream variable = featureSubStrm depth = fifo_depth
#pragma HLS bind_storage variable = eFeatureSubStrm type = fifo impl = lutram
#pragma HLS stream variable = eFeatureSubStrm depth = fifo_depth
#pragma HLS dataflow
        genFeatureSubsets(features_num, fraction_1, featureSubStrm, eFeatureSubStrm);
        writeFeatureSubsets2DDR(featureSubStrm, eFeatureSubStrm, offset, config);
    }

    void readConfig(ap_uint<_WAxi>* configs,
                    const ap_uint<32> numsplit_start,
                    const ap_uint<32> numsplit_len,
                    const ap_uint<32> splits_start,
                    const ap_uint<32> splits_len,
                    const ap_uint<32> splits_elem_num,
                    const ap_uint<32> cols, // features_num + 1
                    float fraction,
                    ap_uint<32>& cols_sp,
                    int featuresIndex[MAX_INNER_FEAS],
                    ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                    FType splits[MAX_SPLITS]) {
        ap_uint<32> features_num = cols - 1;
        ap_uint<8> numSplits[MAX_OUTER_FEAS];
        loadNumSplits(configs, numsplit_start, numsplit_len, features_num, numSplits);
        readConfigFlow(configs, splits_start, splits_len, splits_elem_num, cols, fraction, numSplits, cols_sp,
                       featuresIndex, numSplits_in, splits);
    }
    void writeConfig2DataHeader(ap_uint<_WAxi>* configs,
                                ap_uint<8> numSplits_in[MAX_INNER_FEAS],
                                FType splits[MAX_SPLITS],
                                ap_uint<32> samples_num_in,
                                ap_uint<32> features_num_in,
                                ap_uint<32> class_num,
                                ap_uint<_WAxi>* data_out) {
        //  data_out[0].range(511,482) stores axiLen
        data_out[0].range(31, 0) = samples_num_in;
        data_out[0].range(63, 32) = features_num_in;
        data_out[0].range(95, 64) = class_num;
        data_out[1] = configs[1];
        const int elems_per_line = _WAxi / _WData;

        int r = -1;
        int splitsnum = 0;
        ap_uint<_WAxi> tmp_conf[32];
        ap_uint<_WAxi> onerow;
        for (int i = 0; i < features_num_in; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
            int i_r = i & 0x3f;
            if (i_r == 0) {
                r++;
            }
            int index = i_r * 8;
            tmp_conf[r].range(index + 8 - 1, index) = numSplits_in[i];
            splitsnum += numSplits_in[i];
#ifndef __SYNTHESIS__
            printf("numSplits:%d\n", numSplits_in[i].to_int());
#endif
        }
        r++;
        f_cast<FType> cc;
        int c = 0;
        for (int i = 0; i < splitsnum; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 128 max = 128 avg = 128
#ifndef __SYNTHESIS__
            printf("Splits:%lf\n", splits[i]);
#endif
            cc.f = splits[i];
            tmp_conf[r].range(c * _WData + _WData - 1, c * _WData) = cc.i;
            c++;
            if (c == elems_per_line) {
                r++;
                c = 0;
            }
        }
        if (c != elems_per_line) r++;

        for (int i = 0; i < r; i++) {
#pragma HLS pipeline
#pragma HLS loop_tripcount min = 15 max = 15 avg = 15
            data_out[2 + i] = tmp_conf[i];
        }

        data_out[0].range(127, 96) = splitsnum;
    }
};

} // end of internal

extern "C" void rfSP(ap_uint<512> data[DATASIZE], ap_uint<512> configs[300], ap_uint<512> data_out[DATASIZE]) {
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
	bundle = gmem0_2 port = data_out

#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<32> samples_num;
    ap_uint<32> features_num;
    ap_uint<32> class_num;
    ap_uint<32> seed;
    float instance_fraction;
    float feature_fraction_0;
    float feature_fraction_1;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;

    ap_uint<512> header = data[0];
    samples_num = header.range(31, 0);
    features_num = header.range(63, 32);
    class_num = header.range(95, 64);
    instance_fraction_.i = header.range(127, 96);
    feature_fraction_0_.i = header.range(159, 128);
    feature_fraction_1_.i = header.range(191, 160);
    seed = header.range(223, 192);

    instance_fraction = instance_fraction_.f;
    feature_fraction_0 = feature_fraction_0_.f;
    feature_fraction_1 = feature_fraction_1_.f;

    header = configs[0];
    ap_uint<32> axiLen = header.range(31, 0);
    ap_uint<32> para_start = header.range(63, 32);
    ap_uint<32> numsplit_start = header.range(95, 64);
    ap_uint<32> splits_start = header.range(127, 96);
    ap_uint<32> splits_elem_num = header.range(159, 128);
    ap_uint<32> numsplit_len = splits_start - numsplit_start;
    ap_uint<32> splits_len = axiLen - splits_start;

    int featuresIndex[MAX_INNER_FEAS];
    ap_uint<8> numSplits_in[MAX_INNER_FEAS];
    double splits[MAX_SPLITS];
    ap_uint<32> cols_sp;
#pragma HLS array_partition variable = numSplits_in dim = 0 complete
#pragma HLS array_partition variable = splits dim = 0 complete
    ap_uint<32> data_out_header_len_in = 30;

#ifndef __SYNTHESIS__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n", features_num.to_int());
    printf("seed: %d\n", seed.to_int());
    printf("insance_fraction: %f\n", instance_fraction);
    printf("feature_fraction_0: %f\n", feature_fraction_0);
    printf("feature_fraction_1: %f\n", feature_fraction_1);
#endif
    ap_uint<32> cols = features_num + 1;

    xf::data_analytics::classification::internal::RandForestImp<512, 64, 64, TType, double> loader;

    loader.seedInitialization(seed);
    /*
     * load config
     * */

    loader.readConfig(configs, numsplit_start, numsplit_len, splits_start, splits_len, splits_elem_num, cols,
                      feature_fraction_0, cols_sp, featuresIndex, numSplits_in, splits);

    ap_uint<32> features_num_in = cols_sp - 1;
    // instance sp + fea sp + quantile + burstWrite
    loader.sample(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                  numSplits_in, splits, data_out);
    /*
        // instance sp + fea sp + varMerge + burstWrite
        loader.sample_a(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                        data_out);
        // full scan + fea sp + quantile + burstWrite
        loader.sample_b(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, numSplits_in,
       splits,
                        data_out);
        // full scan + fea sp + varMerge + burstWrite
        loader.sample_c(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, data_out);
    */
    loader.sample_d(features_num_in, feature_fraction_1, data_out_header_len_in, data_out);
    loader.writeConfig2DataHeader(configs, numSplits_in, splits, samples_num, features_num_in, class_num, data_out);
}
extern "C" void rfSP_noqt(ap_uint<512> data[DATASIZE], ap_uint<512> configs[300], ap_uint<512> data_out[DATASIZE]) {
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
	bundle = gmem0_2 port = data_out

#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<32> samples_num;
    ap_uint<32> features_num;
    ap_uint<32> class_num;
    ap_uint<32> seed;
    float instance_fraction;
    float feature_fraction_0;
    float feature_fraction_1;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;

    ap_uint<512> header = data[0];
    samples_num = header.range(31, 0);
    features_num = header.range(63, 32);
    class_num = header.range(95, 64);
    instance_fraction_.i = header.range(127, 96);
    feature_fraction_0_.i = header.range(159, 128);
    feature_fraction_1_.i = header.range(191, 160);
    seed = header.range(223, 192);

    instance_fraction = instance_fraction_.f;
    feature_fraction_0 = feature_fraction_0_.f;
    feature_fraction_1 = feature_fraction_1_.f;

    header = configs[0];
    ap_uint<32> axiLen = header.range(31, 0);
    ap_uint<32> para_start = header.range(63, 32);
    ap_uint<32> numsplit_start = header.range(95, 64);
    ap_uint<32> splits_start = header.range(127, 96);
    ap_uint<32> splits_elem_num = header.range(159, 128);
    ap_uint<32> numsplit_len = splits_start - numsplit_start;
    ap_uint<32> splits_len = axiLen - splits_start;

    int featuresIndex[MAX_INNER_FEAS];
    ap_uint<8> numSplits_in[MAX_INNER_FEAS];
    double splits[MAX_SPLITS];
    ap_uint<32> cols_sp;
#pragma HLS array_partition variable = numSplits_in dim = 0 complete
#pragma HLS array_partition variable = splits dim = 0 complete
    ap_uint<32> data_out_header_len_in = 30;

#ifndef __SYNTHESIS__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n", features_num.to_int());
    printf("seed: %d\n", seed.to_int());
    printf("insance_fraction: %f\n", instance_fraction);
    printf("feature_fraction_0: %f\n", feature_fraction_0);
    printf("feature_fraction_1: %f\n", feature_fraction_1);
#endif
    ap_uint<32> cols = features_num + 1;

    xf::data_analytics::classification::internal::RandForestImp<512, 64, 64, TType, double> loader;

    loader.seedInitialization(seed);
    /*
     * load config
     * */

    loader.readConfig(configs, numsplit_start, numsplit_len, splits_start, splits_len, splits_elem_num, cols,
                      feature_fraction_0, cols_sp, featuresIndex, numSplits_in, splits);

    ap_uint<32> features_num_in = cols_sp - 1;
    // instance sp + fea sp + quantile + burstWrite
    /*
    loader.sample(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                  numSplits_in, splits, data_out);
    */
    // instance sp + fea sp + varMerge + burstWrite
    loader.sample_a(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                    data_out);
    /*
    // full scan + fea sp + quantile + burstWrite
    loader.sample_b(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, numSplits_in, splits,
                    data_out);
    // full scan + fea sp + varMerge + burstWrite
    loader.sample_c(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, data_out);
    */
    loader.sample_d(features_num_in, feature_fraction_1, data_out_header_len_in, data_out);
    loader.writeConfig2DataHeader(configs, numSplits_in, splits, samples_num, features_num_in, class_num, data_out);
}
extern "C" void randomForestSP(const int seedid,
                               ap_uint<512> data[DATASIZE],
                               ap_uint<512> configs[300],
                               ap_uint<512> data_out[DATASIZE]) {
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
	bundle = gmem0_2 port = data_out

#pragma HLS INTERFACE s_axilite port = seedid bundle = control
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = configs bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<32> samples_num;
    ap_uint<32> features_num;
    ap_uint<32> class_num;
    ap_uint<32> seed;
    float instance_fraction;
    float feature_fraction_0 = 1;
    float feature_fraction_1;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;

    ap_uint<512> header = data[0];
    samples_num = header.range(31, 0);
    features_num = header.range(63, 32);
    class_num = header.range(95, 64);
    instance_fraction_.i = header.range(127, 96);
    //    feature_fraction_0_.i = header.range(159, 128);
    feature_fraction_1_.i = header.range(191, 160);
    seed = header.range(223 + seedid * 32, 192 + seedid * 32);

    instance_fraction = instance_fraction_.f;
    //    feature_fraction_0 = feature_fraction_0_.f;
    feature_fraction_1 = feature_fraction_1_.f;

    header = configs[0];
    ap_uint<32> axiLen = header.range(31, 0);
    ap_uint<32> para_start = header.range(63, 32);
    ap_uint<32> numsplit_start = header.range(95, 64);
    ap_uint<32> splits_start = header.range(127, 96);
    ap_uint<32> splits_elem_num = header.range(159, 128);
    ap_uint<32> numsplit_len = splits_start - numsplit_start;
    ap_uint<32> splits_len = axiLen - splits_start;

    int featuresIndex[MAX_INNER_FEAS];
    ap_uint<8> numSplits_in[MAX_INNER_FEAS];
    double splits[MAX_SPLITS];
    ap_uint<32> cols_sp;
#pragma HLS array_partition variable = numSplits_in dim = 0 complete
#pragma HLS array_partition variable = splits dim = 0 complete
    ap_uint<32> data_out_header_len_in = 30;

#ifndef __SYNTHESIS__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n", features_num.to_int());
    printf("seed: %d\n", seed.to_int());
    printf("insance_fraction: %f\n", instance_fraction);
    printf("feature_fraction_0: %f\n", feature_fraction_0);
    printf("feature_fraction_1: %f\n", feature_fraction_1);
#endif
    ap_uint<32> cols = features_num + 1;

    xf::data_analytics::classification::internal::RandForestImp<512, 64, 64, TType, double> loader;

    loader.seedInitialization(seed);
    /*
     * load config
     * */

    loader.readConfig(configs, numsplit_start, numsplit_len, splits_start, splits_len, splits_elem_num, cols,
                      feature_fraction_0, cols_sp, featuresIndex, numSplits_in, splits);

    ap_uint<32> features_num_in = cols_sp - 1;
    // instance sp + quantize + burstWrite
    loader.sample_1(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, numSplits_in,
                    splits, data_out);
    loader.sample_d(features_num_in, feature_fraction_1, data_out_header_len_in, data_out);
    loader.writeConfig2DataHeader(configs, numSplits_in, splits, samples_num, features_num_in, class_num, data_out);
    data[0].range(223 + seedid * 32, 192 + seedid * 32) = seed + 1;
}

extern "C" void randomForestSP_0(const int seedid,
                                 ap_uint<512> data[DATASIZE],
                                 ap_uint<512> configs[300],
                                 ap_uint<512> data_out[DATASIZE]) {
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
	bundle = gmem0_2 port = data_out

#pragma HLS INTERFACE s_axilite port = seedid bundle = control
#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = configs bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<32> samples_num;
    ap_uint<32> features_num;
    ap_uint<32> class_num;
    ap_uint<32> seed;
    float instance_fraction;
    float feature_fraction_0 = 1;
    float feature_fraction_1;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;

    ap_uint<512> header = data[0];
    samples_num = header.range(31, 0);
    features_num = header.range(63, 32);
    class_num = header.range(95, 64);
    instance_fraction_.i = header.range(127, 96);
    //    feature_fraction_0_.i = header.range(159, 128);
    feature_fraction_1_.i = header.range(191, 160);
    seed = header.range(223 + seedid * 32, 192 + seedid * 32);

    instance_fraction = instance_fraction_.f;
    //    feature_fraction_0 = feature_fraction_0_.f;
    feature_fraction_1 = feature_fraction_1_.f;

    header = configs[0];
    ap_uint<32> axiLen = header.range(31, 0);
    ap_uint<32> para_start = header.range(63, 32);
    ap_uint<32> numsplit_start = header.range(95, 64);
    ap_uint<32> splits_start = header.range(127, 96);
    ap_uint<32> splits_elem_num = header.range(159, 128);
    ap_uint<32> numsplit_len = splits_start - numsplit_start;
    ap_uint<32> splits_len = axiLen - splits_start;

    int featuresIndex[MAX_INNER_FEAS];
    ap_uint<8> numSplits_in[MAX_INNER_FEAS];
    double splits[MAX_SPLITS];
    ap_uint<32> cols_sp;
#pragma HLS array_partition variable = numSplits_in dim = 0 complete
#pragma HLS array_partition variable = splits dim = 0 complete
    ap_uint<32> data_out_header_len_in = 30;

#ifndef __SYNTHESIS__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n", features_num.to_int());
    printf("seed: %d\n", seed.to_int());
    printf("insance_fraction: %f\n", instance_fraction);
    printf("feature_fraction_0: %f\n", feature_fraction_0);
    printf("feature_fraction_1: %f\n", feature_fraction_1);
#endif
    ap_uint<32> cols = features_num + 1;

    xf::data_analytics::classification::internal::RandForestImp<512, 64, 64, TType, double> loader;

    loader.seedInitialization(seed);
    /*
     * load config
     * */

    loader.readConfig(configs, numsplit_start, numsplit_len, splits_start, splits_len, splits_elem_num, cols,
                      feature_fraction_0, cols_sp, featuresIndex, numSplits_in, splits);

    ap_uint<32> features_num_in = cols_sp - 1;
    // instance sp + quantize + burstWrite
    loader.sample_1(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, numSplits_in,
                    splits, data_out);
    loader.sample_d(features_num_in, feature_fraction_1, data_out_header_len_in, data_out);
    loader.writeConfig2DataHeader(configs, numSplits_in, splits, samples_num, features_num_in, class_num, data_out);
    data[0].range(223 + seedid * 32, 192 + seedid * 32) = seed + 1;
}

extern "C" void randomForestSP_noqt(ap_uint<512> data[DATASIZE],
                                    ap_uint<512> configs[300],
                                    ap_uint<512> data_out[DATASIZE]) {
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
	bundle = gmem0_2 port = data_out

#pragma HLS INTERFACE s_axilite port = data bundle = control
#pragma HLS INTERFACE s_axilite port = configs bundle = control
#pragma HLS INTERFACE s_axilite port = data_out bundle = control
#pragma HLS INTERFACE s_axilite port = return bundle = control
    // clang-format on
    // 1.read config
    ap_uint<32> samples_num;
    ap_uint<32> features_num;
    ap_uint<32> class_num;
    ap_uint<32> seed;
    float instance_fraction;
    float feature_fraction_0 = 1;
    float feature_fraction_1;

    f_cast<float> instance_fraction_;
    f_cast<float> feature_fraction_0_;
    f_cast<float> feature_fraction_1_;

    ap_uint<512> header = data[0];
    samples_num = header.range(31, 0);
    features_num = header.range(63, 32);
    class_num = header.range(95, 64);
    instance_fraction_.i = header.range(127, 96);
    //    feature_fraction_0_.i = header.range(159, 128);
    feature_fraction_1_.i = header.range(191, 160);
    seed = header.range(223, 192);

    instance_fraction = instance_fraction_.f;
    //    feature_fraction_0 = feature_fraction_0_.f;
    feature_fraction_1 = feature_fraction_1_.f;

    header = configs[0];
    ap_uint<32> axiLen = header.range(31, 0);
    ap_uint<32> para_start = header.range(63, 32);
    ap_uint<32> numsplit_start = header.range(95, 64);
    ap_uint<32> splits_start = header.range(127, 96);
    ap_uint<32> splits_elem_num = header.range(159, 128);
    ap_uint<32> numsplit_len = splits_start - numsplit_start;
    ap_uint<32> splits_len = axiLen - splits_start;

    int featuresIndex[MAX_INNER_FEAS];
    ap_uint<8> numSplits_in[MAX_INNER_FEAS];
    double splits[MAX_SPLITS];
    ap_uint<32> cols_sp;
#pragma HLS array_partition variable = numSplits_in dim = 0 complete
#pragma HLS array_partition variable = splits dim = 0 complete
    ap_uint<32> data_out_header_len_in = 30;

#ifndef __SYNTHESIS__
    printf("samples_num: %d\n", samples_num.to_int());
    printf("features_num: %d\n", features_num.to_int());
    printf("seed: %d\n", seed.to_int());
    printf("insance_fraction: %f\n", instance_fraction);
    printf("feature_fraction_0: %f\n", feature_fraction_0);
    printf("feature_fraction_1: %f\n", feature_fraction_1);
#endif
    ap_uint<32> cols = features_num + 1;

    xf::data_analytics::classification::internal::RandForestImp<512, 64, 64, TType, double> loader;

    loader.seedInitialization(seed);
    /*
     * load config
     * */

    loader.readConfig(configs, numsplit_start, numsplit_len, splits_start, splits_len, splits_elem_num, cols,
                      feature_fraction_0, cols_sp, featuresIndex, numSplits_in, splits);

    ap_uint<32> features_num_in = cols_sp - 1;
    // instance sp + fea sp + quantile + burstWrite
    /*
    loader.sample(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                  numSplits_in, splits, data_out);
    // instance sp + fea sp + varMerge + burstWrite
    loader.sample_a(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, featuresIndex,
                    data_out);
    // full scan + fea sp + quantile + burstWrite
    loader.sample_b(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, numSplits_in, splits,
                    data_out);
    // full scan + fea sp + varMerge + burstWrite
    loader.sample_c(data, 1, 0, samples_num, cols, cols_sp, data_out_header_len, featuresIndex, data_out);
    */
    // instance sp + quantize + burstWrite
    // loader.sample_1(data, 1, 0, samples_num, cols, instance_fraction, cols_sp, data_out_header_len, numSplits_in,
    // splits, data_out);
    // instance sp + quantize + burstWrite
    loader.sample_2(data, 1, 0, samples_num, cols, instance_fraction, data_out_header_len, data_out);
    loader.sample_d(features_num_in, feature_fraction_1, data_out_header_len_in, data_out);
    loader.writeConfig2DataHeader(configs, numSplits_in, splits, samples_num, features_num_in, class_num, data_out);
}
} // end of name space classification
} // end of name space data_analytics
} // end of name space xf
