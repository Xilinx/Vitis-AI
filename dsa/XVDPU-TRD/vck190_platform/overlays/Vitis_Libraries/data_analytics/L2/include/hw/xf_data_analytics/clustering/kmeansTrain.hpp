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

/**
 * @file kmeansTrain.hpp
 * @brief header file for kmeans training.
 * This file part of Vitis  Library.
 *
 */

#ifndef _XF_DATA_ANALYTICS_CLUSTERING_KMEANS_TRAIN_HPP_
#define _XF_DATA_ANALYTICS_CLUSTERING_KMEANS_TRAIN_HPP_

#include "config.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#include "xf_utils_hw/stream_n_to_one/round_robin.hpp"
#include "xf_utils_hw/stream_dup.hpp"
#include "xf_data_analytics/clustering/kmeansPredict.hpp"
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
#include <iostream>
#endif

namespace xf {
namespace data_analytics {
namespace clustering {

namespace internal {
/**
 * axiVarColToStreams is used for scanning data from DDR.
 *
 */

namespace kmeansScan {

template <int WAxi, int BurstLen, int WData>
void readRaw(ap_uint<WAxi>* ddr,
             const ap_uint<32> offset,
             const ap_uint<32> rows,
             const ap_uint<32> cols,
             hls::stream<ap_uint<WAxi> >& vecStrm) {
    const ap_uint<64> nread = (rows * cols + (WAxi / WData - 1)) / (WAxi / WData);
    const int NUMS = NDBATCH; // 1000 * 1000 * 3;
READ_RAW:
    for (ap_uint<64> i = 0; i < nread; i++) {
#pragma HLS loop_tripcount min = NUMS max = NUMS
#pragma HLS pipeline II = 1
        vecStrm.write(ddr[offset + i]);
    }
}

template <int WAxi, int WData>
void cageShitRight(ap_uint<WAxi * 2>& source, unsigned int s) {
#pragma HLS inline off
    if (s >= 0 && s <= WAxi / WData) {
        source >>= s * WData;
    }
}

template <int WAxi, int WData>
void varSplit(hls::stream<ap_uint<WAxi> >& vecStrm,
              const ap_uint<32> rows,
              const ap_uint<32> cols,
              hls::stream<ap_uint<WData> > data[WAxi / WData],
              hls::stream<bool>& eData) {
    const int fullBatch = (WAxi / WData);
    const int tmpTailBatch = cols % fullBatch;
    const int tailBatch = (tmpTailBatch == 0) ? fullBatch : tmpTailBatch;

    const int batchNum = (cols + fullBatch - 1) / fullBatch;

    int reserve = 0;
    ap_uint<WAxi> inventory = 0;
    const int NUMS = NUM; // 1000 * 1000;
    const int ubatch = UBATCH;
LOOP1:
    for (int i = 0; i < rows; i++) {
#pragma HLS loop_tripcount min = NUMS max = NUMS
    LOOP2:
        for (int j = 0; j < batchNum; j++) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = ubatch max = ubatch
            int output;
            if (j == batchNum - 1) {
                output = tailBatch;
            } else {
                output = fullBatch;
            }

            ap_uint<WAxi> newCome;
            int tmpReserve = reserve;
            if (reserve < output) {
                newCome = vecStrm.read();
                reserve += (fullBatch - output);
            } else {
                newCome = 0;
                reserve -= output;
            }

            ap_uint<WAxi* 2> cage = 0;
            cage.range(WAxi * 2 - 1, WAxi) = newCome;
            cageShitRight<WAxi, WData>(cage, fullBatch - tmpReserve);

            cage.range(WAxi - 1, 0) = cage.range(WAxi - 1, 0) ^ inventory.range(WAxi - 1, 0);

            ap_uint<WAxi> preLocalOutput = cage.range(WAxi - 1, 0);

            cageShitRight<WAxi, WData>(cage, output);
            inventory = cage.range(WAxi - 1, 0);

            for (int k = 0; k < fullBatch; k++) {
#pragma HLS unroll
                ap_uint<WData> tmp;
                if (k < output) {
                    tmp = preLocalOutput.range((k + 1) * WData - 1, k * WData);
                } else {
                    tmp = 0;
                }
                data[k].write(tmp);
            }
            // if (j == 0) {
            eData.write(false);
            // }
        }
    }
    eData.write(true);
}

} // namespace kmeansScan

template <int BurstLen, int WAxi, int WData>
void axiVarColToStreams(ap_uint<WAxi>* ddr,
                        const ap_uint<32> offset,
                        const ap_uint<32> rows,
                        const ap_uint<32> cols,
                        hls::stream<ap_uint<WData> > data[WAxi / WData],
                        hls::stream<bool>& eData) {
    static const int fifoDepth = BurstLen * 2;
    hls::stream<ap_uint<WAxi> > vecStrm;
#pragma HLS bind_storage variable = vecStrm type = fifo impl = lutram
#pragma HLS stream variable = vecStrm depth = fifoDepth
#pragma HLS dataflow
    kmeansScan::readRaw<WAxi, BurstLen, WData>(ddr, offset, rows, cols, vecStrm);
    kmeansScan::varSplit(vecStrm, rows, cols, data, eData);
}

//-----------------   scan old centers and writeout new centers -----------//

/**
 * @brief storeCenters2Array is used for storing centers to local memory.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions, dims<=Dim
 * @tparam Kcluster the maximum number of cluster,kcluster<=Kcluster
 * @tparam uramDepth the depth of uram
 * @tparam KU unroll factor of Kcluster
 * @tparam DV unroll factor of Dim
 *
 * @param centerStrm input centers streams
 * @param ecenterStrm the end flag of centerStrm
 * @param dims  the number of dimensions
 * @param centers local memory for storing centers
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void storeCenters2Array(hls::stream<ap_uint<sizeof(DT) * 8> > centerStrm[DV],
                        hls::stream<bool>& eCenterStrm,
                        const int dims,
                        ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth]) {
    const int sz = sizeof(DT) * 8;
    const int batch = (dims + DV - 1) / DV;
    int pos = 0;
    int base = 0;
    int c = 0;
    const int tc = KU * uramDepth;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "start to store centers" << std::endl;
#endif
    while (!eCenterStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tc max = tc
        ap_uint<sz * DV> uc;
        for (int j = 0; j < DV; ++j) {
            ap_uint<sz> ut = centerStrm[j].read();
            uc.range((j + 1) * sz - 1, j * sz) = ut;
        }
        centers[c][base + pos] = uc;
        if (pos + 1 == batch) {
            pos = 0;
            if (c + 1 == KU) {
                c = 0;
                base += batch;
            } else {
                c++;
            }
        } else {
            pos++;
        }
    } // while
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "finished storage centers" << std::endl;
#endif
}

/**
 * @brief scanAndStoreCenters is used for scanning initial centers from DDR and storing to local memory.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions, dims<=Dim
 * @tparam Kcluster the maximum number of cluster,kcluster<=Kcluster
 * @tparam uramDepth the depth of uram
 * @tparam KU unroll factor of Kcluster
 * @tparam DV unroll factor of Dim
 *
 * @param data initial centers on DDR
 * @param offset the start position of initial centers in data
 * @param dims  the number of dimensions
 * @param kcluster  the number of clusters
 * @param centers local memory for storing centers
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void scanAndStoreCenters(ap_uint<512>* data,
                         const ap_uint<32> offset,
                         const int dims,
                         const int kcluster,
                         ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth]) {
#pragma HLS dataflow
    const int DEPTH = 4;
    const int sz = sizeof(DT) * 8;
    const int sn = 512 / sz;
    hls::stream<ap_uint<sizeof(DT) * 8> > centerStrm[512 / 8 / sizeof(DT)];
#pragma HLS STREAM variable = centerStrm depth = DEPTH
#pragma HLS bind_storage variable = centerStrm type = fifo impl = lutram
    hls::stream<bool> endCenterStrm;
#pragma HLS STREAM variable = endCenterStrm depth = DEPTH
#pragma HLS bind_storage variable = endCenterStrm type = fifo impl = lutram

    hls::stream<ap_uint<sz> > dpStrm[sn];
#pragma HLS stream variable = dpStrm depth = 4
#pragma HLS bind_storage variable = dpStrm type = fifo impl = lutram

    hls::stream<bool> edpStrm[sn];
#pragma HLS stream variable = edpStrm depth = 4
#pragma HLS bind_storage variable = edpStrm type = fifo impl = lutram

    hls::stream<ap_uint<sz * DV> > vStrm;
#pragma HLS stream variable = vStrm depth = 4
#pragma HLS bind_storage variable = vStrm type = fifo impl = lutram

    hls::stream<bool> evStrm;
#pragma HLS stream variable = evStrm depth = 4
#pragma HLS bind_storage variable = evStrm type = fifo impl = lutram

    hls::stream<ap_uint<sz> > newStrm[DV];
#pragma HLS stream variable = newStrm depth = 4
#pragma HLS bind_storage variable = newStrm type = fifo impl = lutram

    hls::stream<bool> enewStrm;
#pragma HLS stream variable = enewStrm depth = 4
#pragma HLS bind_storage variable = enewStrm type = fifo impl = lutram

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "start to scan centers" << std::endl;
#endif
    axiVarColToStreams<32, 512, sizeof(DT) * 8>(data, (ap_uint<32>)offset, (ap_uint<32>)kcluster, (ap_uint<32>)dims,
                                                centerStrm, endCenterStrm);

    if (DV == sn) {
        storeCenters2Array<DT, Dim, Kcluster, uramDepth, KU, DV>(centerStrm, endCenterStrm, dims, centers);
    } else {
        dupStrm<sz, sn, (Dim + sn - 1) / sn>(centerStrm, endCenterStrm, dpStrm, edpStrm);
        xf::common::utils_hw::streamNToOne<sz, sz * DV, sn>(dpStrm, edpStrm, vStrm, evStrm,
                                                            xf::common::utils_hw::RoundRobinT());
        split<sz, DV, (Dim + DV - 1) / DV>(vStrm, evStrm, newStrm, enewStrm);
        storeCenters2Array<DT, Dim, Kcluster, uramDepth, KU, DV>(newStrm, enewStrm, dims, centers);
    }

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "end scanning centers" << std::endl;
#endif
}

/**
 * @brief writeCenters writes best centers to DDR.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions, dims<=Dim
 * @tparam Kcluster the maximum number of cluster,kcluster<=Kcluster
 * @tparam uramDepth the depth of uram
 * @tparam KU unroll factor of Kcluster
 * @tparam DV unroll factor of Dim
 *
 * @param centersArray best centers
 * @param dims  the number of dimensions
 * @param kcluster  the number of clusters
 * @param it  iterations
 * @param kcenters centers in DDR
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void writeCenters(ap_uint<sizeof(DT) * 8 * DV> centersArray[KU][uramDepth],
                  const int dims,
                  const int kcluster,
                  const int it,
                  ap_uint<512>* kcenters) {
#pragma HLS inline off
    const int sz = sizeof(DT) * 8;
    const int batch = (dims + DV - 1) / DV;
    const int maxBatch = (Dim + DV - 1) / DV;
    ap_uint<512> v = 0;
    int p = 0;
    int s = 0;
    const int MAXC = (Kcluster + KU - 1) / KU;
WRITE_CENTERS:
    for (int j = 0; j < (kcluster + KU - 1) / KU; ++j) {
#pragma HLS loop_tripcount min = MAXC max = MAXC
#pragma HLS pipeline off
    LOOP2:
        for (int k = 0; k < KU; ++k) {
#pragma HLS loop_tripcount min = KU max = KU
#pragma HLS pipeline off
        LOOP3:
            for (int r = 0; r < batch; ++r) {
#pragma HLS loop_tripcount min = maxBatch max = maxBatch
#pragma HLS pipeline off
                ap_uint<sz* DV> uc = centersArray[k][j * batch + r];

                for (int i = 0; i < DV; ++i) {
#pragma HLS pipeline // off
                    p++;
                    int q = p - 1;
                    v.range((q + 1) * sz - 1, q * sz) = uc.range((i + 1) * sz - 1, i * sz);
                    if (p * sz >= 512) {
                        kcenters[s++] = v;
                        v = 0;
                        p = 0;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
                        std::cout << "s=" << s - 1 << "  k=" << k << " j=" << j << "  i=" << i << "  batch=" << batch
                                  << std::endl;
#endif
                    } // if

                } // for i
            }     // for r
        }         // for k
    }             // for j
    v.range(32, 0) = it;
    kcenters[s++] = v;
}

// ---------------------- update centers -----------------//
/**
 * @brief updateC0 is used for updating centers
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions, dims<=Dim
 * @tparam Kcluster the maximum number of cluster,kcluster<=Kcluster
 * @tparam uramDepth the depth of uram
 * @tparam KU unroll factor of Kcluster
 * @tparam DV unroll factor of Dim
 *
 * @param sampleStrm input sample streams
 * @param endSampleStrm the end flag of input sample streams
 * @param tagStrm input tag streams,each tag is the id which  each sample belongs to cluster.
 * @param etagStrm the end flag of tag streams
 * @param dims  the number of dimensions.
 * @param betterCenters updated centers.
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void updateC0(hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[DV],
              hls::stream<bool>& endSampleStrm,
              hls::stream<ap_uint<32> >& tagStrm,
              hls::stream<bool>& etagStrm,
              const int dims,
              ap_uint<sizeof(DT) * 8 * DV> betterCenters[KU][uramDepth]) {
#pragma HLS inline off
    const int maxKU = (Kcluster + KU - 1) / KU;
    int cnt[KU][(Kcluster + KU - 1) / KU];
    const int sz = sizeof(DT) * 8;
    const int batch = (dims + DV - 1) / DV; // dynamic
    for (int k = 0; k < KU; ++k) {
        for (int d = 0; d < (Kcluster + KU - 1) / KU; ++d) {
#pragma HLS pipeline II = 1
            cnt[k][d] = 0;
        }
    }
    int p = 0;
    ap_uint<32> tk = 0;
    ap_uint<16> cr; // row in centers array
    ap_uint<16> cl; // col in centers array
    bool sw = false;
    const int tsmp = NUMS * ((Dim + DV - 1) / DV);
SUM_LOOP:
    while (!endSampleStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tsmp max = tsmp

        if (p == 0) {
            // each sample has a tag
            // it takes (dims+DV-1)/DV cycles to read each sample,so p is from 0 to (dims+DV-1)/DV-1.
            if (!etagStrm.read()) {
                // ap_uint<32> tk = minp * KU + mink;
                // tk.range(15, 0) = minp;
                // tk.range(31, 16) = mink;
                tk = tagStrm.read();
                cr = tk / KU;
                cl = tk % KU;
                cnt[cl][cr]++;
            }
        }
        ap_uint<sz* DV> uc = betterCenters[cl][cr * batch + p];
        ap_uint<sz * DV> nuc;
        for (int i = 0; i < DV; ++i) {
#pragma HLS unroll
            ap_uint<sz> ud = sampleStrm[i].read();
            DT smp = dataTypeConverter<DT>::getDT(ud);
            ap_uint<sz> ut = uc.range((i + 1) * sz - 1, i * sz);
            DT cter = dataTypeConverter<DT>::getDT(ut);
            DT tmp = smp + cter;
            ap_uint<sz> us = dataTypeConverter<DT>::getUT(tmp);
            nuc.range((i + 1) * sz - 1, i * sz) = us;
        }
        betterCenters[cl][cr * batch + p] = nuc;

        if (p + 1 < batch) {
            p++;
        } else {
            p = 0;
        }
    }
    etagStrm.read();
    const int maxBatch = (Dim + DV - 1) / DV;
AVEG_LOOP:
    for (int k = 0; k < KU; ++k) {
#pragma HLS pipeline off
    AVEG_K:
        for (int j = 0; j < (Kcluster + KU - 1) / KU; ++j) {
#pragma HLS pipeline off
            int c = cnt[k][j] > 0 ? cnt[k][j] : 1;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
            std::cout << "cnt[" << k << "][" << j << "]=" << cnt[k][j] << std::endl;
            std::cout << "c[" << j * KU + k << "]= (";
#endif
        AVEG_BATCH2:
            for (int r = 0; r < batch; ++r) {
#pragma HLS pipeline off
#pragma HLS loop_tripcount min = maxBatch max = maxBatch
                ap_uint<sz* DV> uc = betterCenters[k][j * batch + r];
                ap_uint<sz * DV> nuc;
            AVEG_DV2:
                for (int i = 0; i < DV; ++i) {
#pragma HLS unroll
                    ap_uint<sz> ut = uc.range((i + 1) * sz - 1, i * sz);
                    DT cter = dataTypeConverter<DT>::getDT(ut);
                    DT nc = (r * DV + i < dims) ? (cter / c) : 0;
                    ap_uint<sz> u = dataTypeConverter<DT>::getUT(nc);
                    nuc.range((i + 1) * sz - 1, i * sz) = u;
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
                    std::cout << ",  " << nc;
#endif
                }
                betterCenters[k][j * batch + r] = nuc;
            }
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
            std::cout << ") " << std::endl;
#endif
        }
    }
}
//----------------------- auxiliary functions -------//
/**
 * @brief convertPort converts sn streams to DV streams, in which the s1 and s2 are same.
 * @tparam sz the width input/output stream
 * @tparam sn the number of input streams
 * @tparam Dim the number of dimensions
 * @tparam DV the number of output streams
 *
 * @param sampleStrm input streams.
 * @param endSampleStrm the end flag of input streams
 * @param s1Strm duplicate stream.
 * @param es1Strm  the end flag of s1Strm.
 * @param s2Strm duplicate stream.
 * @param es2Strm  the end flag of s1Strm.
 *
 */
template <int sz, int sn, int Dim, int DV>
void convertPort(hls::stream<ap_uint<sz> > sampleStrm[sn],
                 hls::stream<bool>& endSampleStrm,
                 hls::stream<ap_uint<sz> > s1Strm[DV],
                 hls::stream<bool>& es1Strm,
                 hls::stream<ap_uint<sz> > s2Strm[DV],
                 hls::stream<bool>& es2Strm) {
#pragma HLS dataflow

    hls::stream<ap_uint<sz> > dpStrm[sn];
#pragma HLS stream variable = dpStrm depth = 4
#pragma HLS bind_storage variable = dpStrm type = fifo impl = lutram

    hls::stream<bool> edpStrm[sn];
#pragma HLS stream variable = edpStrm depth = 4
#pragma HLS bind_storage variable = edpStrm type = fifo impl = lutram

    hls::stream<ap_uint<sz * DV> > vStrm;
#pragma HLS stream variable = vStrm depth = 4
#pragma HLS bind_storage variable = vStrm type = fifo impl = lutram

    hls::stream<bool> evStrm;
#pragma HLS stream variable = evStrm depth = 4
#pragma HLS bind_storage variable = evStrm type = fifo impl = lutram

    hls::stream<ap_uint<sz * DV> > rvStrm[2];
#pragma HLS stream variable = rvStrm depth = 4
#pragma HLS bind_storage variable = rvStrm type = fifo impl = lutram

    hls::stream<bool> ervStrm[2];
#pragma HLS stream variable = ervStrm depth = 4
#pragma HLS bind_storage variable = ervStrm type = fifo impl = lutram

    if (DV == sn) {
        // duplicate
        dupStrm<sz, sn, (Dim + sn - 1) / sn>(sampleStrm, endSampleStrm, s1Strm, es1Strm, s2Strm, es2Strm);
    } else {
        // fill end flag stream
        dupStrm<sz, sn, (Dim + sn - 1) / sn>(sampleStrm, endSampleStrm, dpStrm, edpStrm);
        // merge to one stream
        xf::common::utils_hw::streamNToOne<sz, sz * DV, sn>(dpStrm, edpStrm, vStrm, evStrm,
                                                            xf::common::utils_hw::RoundRobinT());
        // duplicate
        xf::common::utils_hw::streamDup<ap_uint<sz * DV>, 2>(vStrm, evStrm, rvStrm, ervStrm);
        // split to DV streams
        split<sz, DV, (Dim + DV - 1) / DV>(rvStrm[0], ervStrm[0], s1Strm, es1Strm);
        split<sz, DV, (Dim + DV - 1) / DV>(rvStrm[1], ervStrm[1], s2Strm, es2Strm);
    }
}
/**
 * @brief isConverged is used for checking whether new centers is close to previous centers and copy betterCenters to
 * centers with clearing betterCenters in order to reduce resource.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceiling(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param centers previous centers.
 * @param betterCenters new centers
 * @param dims  the number of dimensions.
 * @param kcluster  the number of clusters.
 * @param eps  distance thredhold.
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
bool isConverged(ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                 ap_uint<sizeof(DT) * 8 * DV> betterCenters[KU][uramDepth],
                 const int dims,
                 const int kcluster,
                 const DT eps) {
#pragma HLS inline off
    const int sz = sizeof(DT) * 8;
    const int batch = (dims + DV - 1) / DV;
    const int maxBatch = (Dim + DV - 1) / DV;
    const int MAXC = (Kcluster + KU - 1) / KU;
    int count = 0;
    bool isCvg = true;
    DT dff[DV];
#pragma HLS array_partition variable = dff complete dim = 1
LOOP1:
    for (int j = 0; j < (kcluster + KU - 1) / KU; ++j) {
#pragma HLS pipeline off
#pragma HLS loop_tripcount min = MAXC max = MAXC
    LOOP2:
        for (int k = 0; k < KU; ++k) {
#pragma HLS pipeline off
#pragma HLS loop_tripcount min = KU max = KU
            DT err = 0;
        LOOP3:
            for (int r = 0; r < batch; ++r) {
#pragma HLS pipeline // off
#pragma HLS loop_tripcount min = maxBatch max = maxBatch

                ap_uint<sz* DV> uc1 = centers[k][j * batch + r];
                ap_uint<sz* DV> uc2 = betterCenters[k][j * batch + r];
                centers[k][j * batch + r] = uc2;
                betterCenters[k][j * batch + r] = 0;
                for (int i = 0; i < DV; ++i) {
                    //#pragma HLS pipeline
                    ap_uint<sz> ut1 = uc1.range((i + 1) * sz - 1, i * sz);
                    DT nc1 = dataTypeConverter<DT>::getDT(ut1);
                    ap_uint<sz> ut2 = uc2.range((i + 1) * sz - 1, i * sz);
                    DT nc2 = dataTypeConverter<DT>::getDT(ut2);
                    dff[i] = (nc1 - nc2) * (nc1 - nc2);
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
                    std::cout << "(, " << nc1 << "," << nc2 << ")";
#endif
                }
                err += addTree<DT, DV>(dff, 0);
                count++;
            } // for r
            isCvg &= (err <= eps);
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
            std::cout << ")" << std::endl;
            std::cout << "isCvg=" << isCvg << "   err=" << err << " eps=" << eps << std::endl;
#endif
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
            std::cout << "cunt=" << count << " batch" << batch << "  id" << (j * KU + k) * batch << std::endl;
#endif
        } // for k
    }     // for j
    return isCvg;
}
// ----------------- training -------------------------- //
/**
 * @brief kMeansTrainIter is used for one iteration of training: scan samples + predict + update centers.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceiling(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param centers input centers used for calculating distance.
 * @param nsample the number of input samples.
 * @param dims  the number of dimensions.
 * @param kcluster  the number of clusters.
 * @param offset  the start position of samples in data, offset = 1 + ceiling(kcluster*dims*8*sizeof(DT)/512).
 * @param betterCenters updated centers as better centers
 *
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void kMeansTrainIter(ap_uint<512>* data,
                     ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                     const int nsample,
                     const int dims,
                     const int kcluster,
                     const int offset,
                     ap_uint<sizeof(DT) * 8 * DV> betterCenters[KU][uramDepth]) {
    const int DEPTH = 64;
    const int DEPTH2 = 4 * ((Dim + DV - 1) / DV) + DEPTH;
    const int sz = sizeof(DT) * 8;
    const int sn = 512 / sz;
#pragma HLS dataflow
    hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[sn];
#pragma HLS STREAM variable = sampleStrm depth = DEPTH
#pragma HLS bind_storage variable = sampleStrm type = fifo impl = lutram
    hls::stream<bool> endSampleStrm;
#pragma HLS STREAM variable = endSampleStrm depth = DEPTH
#pragma HLS bind_storage variable = endSampleStrm type = fifo impl = lutram
    hls::stream<ap_uint<sizeof(DT) * 8> > s1Strm[DV];
#pragma HLS STREAM variable = s1Strm depth = DEPTH
#pragma HLS bind_storage variable = s1Strm type = fifo impl = lutram
    hls::stream<bool> es1Strm;
#pragma HLS STREAM variable = es1Strm depth = DEPTH
#pragma HLS bind_storage variable = es1Strm type = fifo impl = lutram

    hls::stream<ap_uint<sizeof(DT) * 8> > s2Strm[DV];
#pragma HLS STREAM variable = s2Strm depth = DEPTH2
#pragma HLS bind_storage variable = s2Strm type = fifo impl = lutram
    hls::stream<bool> es2Strm;
#pragma HLS STREAM variable = es2Strm depth = DEPTH2
#pragma HLS bind_storage variable = es2Strm type = fifo impl = lutram

    hls::stream<ap_uint<32> > tagStrm;
#pragma HLS STREAM variable = tagStrm depth = 4
#pragma HLS bind_storage variable = tagStrm type = fifo impl = lutram
    hls::stream<bool> endTagStrm;
#pragma HLS STREAM variable = endTagStrm depth = 4
#pragma HLS bind_storage variable = endTagStrm type = fifo impl = lutram
    // sample in DDR --> 8-double/16-float streams
    //                           |
    //                          \|/            s1Strm
    //                DV double/float streams---------> predict
    //                  |                                 |
    //                  | s2Strm                  tag     |
    //                  |-------------      --------------|
    //                               |     |
    //                              \|/   \|/
    //                          updating centers
    //
    // The depth of s2 is bigger than s1 because kMeansPredict has a ping-pong buffer to repeat output each sample
    // ceiling(kcluster/KU) times and each sample is used for updating centers until it is predicted. scan samples from
    // DDR to streams
    axiVarColToStreams<32, 512, sizeof(DT) * 8>(data, offset, nsample, dims, sampleStrm, endSampleStrm);
    // convert 16-float or 8-double streams to DV float/double streams
    convertPort<sz, sn, Dim, DV>(sampleStrm, endSampleStrm, s1Strm, es1Strm, s2Strm, es2Strm);
    // predict which cluster each sample is belonged to using existed centers
    kMeansPredict<DT, Dim, Kcluster, uramDepth, KU, DV>(s1Strm, es1Strm, centers, dims, kcluster, tagStrm, endTagStrm);
    // update centers
    updateC0<DT, Dim, Kcluster, uramDepth, KU, DV>(s2Strm, es2Strm, tagStrm, endTagStrm, dims, betterCenters);
}
template <typename DT>
void parser(ap_uint<512> config, int& nsample, int& dims, int& kcluster, int& maxIter, DT& eps) {
    kcluster = config.range(31, 0);
    dims = config.range(63, 32);
    nsample = config.range(95, 64);
    maxIter = config.range(127, 96);
    const int sz = sizeof(DT) * 8;
    ap_uint<sz> ut = config.range(sz + 127, 128);
    eps = dataTypeConverter<DT>::getDT(ut);
}
/**
 * @brief kMeansTrainImp is used for implementation k means training.
 * It parsers all dynamic configures from compressed input data at first, then stores the initial centers to local
 * memory. When iterations is up to maximum number or new centers are very close to previous centers, the last centers
 * as best centers are ouput.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param data input data from host
 * @param kcenters the output best centers
 *
 */
template <typename DT, int Dim, int Kcluster, int KU, int DV>
void kMeansTrainImp(ap_uint<512>* data, ap_uint<512>* kcenters) {
    const int sz = sizeof(DT) * 8;
    const int sn = 512 / sz;
    const int uramDepth = ((Dim + DV - 1) / DV) * ((Kcluster + KU - 1) / KU);
    int kcluster;
    int dims;
    int nsample;
    int maxIter;
    DT eps;
    ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth];
#pragma HLS array_partition variable = centers complete dim = 1
#pragma HLS bind_storage variable = centers type = ram_2p impl = uram // uram
    ap_uint<sizeof(DT) * 8 * DV> betterCenters[KU][uramDepth];
#pragma HLS bind_storage variable = betterCenters type = ram_2p impl = uram // uram
    // parser dynamic configures
    parser<DT>(data[0], nsample, dims, kcluster, maxIter, eps);
    // scan initial centers to local memory
    scanAndStoreCenters<DT, Dim, Kcluster, uramDepth, KU, DV>(data, 1, dims, kcluster, betterCenters);
    isConverged<DT, Dim, Kcluster, uramDepth, KU, DV>(centers, betterCenters, dims, kcluster, eps);
    const int dsz = dims * sz;
    const int offset = 1 + (kcluster * dsz + 511) / 512;
    DT e = 10;
    int it = 0;
    bool stop = false;

    while (!stop) {
#pragma HLS loop_tripcount min = 300 max = 300

        kMeansTrainIter<DT, Dim, Kcluster, uramDepth, KU, DV>(data, centers, nsample, dims, kcluster, offset,
                                                              betterCenters);

        it++;
        // converge + copy betterCenters to centers + clear betterCenters
        bool isCvg = isConverged<DT, Dim, Kcluster, uramDepth, KU, DV>(centers, betterCenters, dims, kcluster, eps);
        stop = (isCvg || it >= maxIter);
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
        std::cout << "it=" << it << " maxIter=" << maxIter << std::endl;
#endif
    }
    // oputput best centers
    writeCenters<DT, Dim, Kcluster, uramDepth, KU, DV>(centers, dims, kcluster, it, kcenters);
}
// -------------------------------------------------------//
} // end of namespace internal

/**
 * @brief k-means is a clustering algorithm that aims to partition n samples into k clusters in which each sample
 * belongs to the cluster with the nearest mean. The implementation is based on "native k-means"(also referred to as
 * Lloyd's algorithm). The implemenation aims to change computational complexity O(Nsample * Kcluster * Dim * maxIter)
 * to O(Nsample* (Kcluster/KU)*(Dim/DV)*maxIter) by accelerating calculating distances.Athough more speedup are achieved
 * to as KU*DV grows in theory,KU and DV should be configured properly because the both effect on storing centers on
 * chip. The input data contains : 1) dynamic configures in data[0],including the number of samples,the number of
 * dimensions,the number of clusters,the maximum number of iterations,the distance threshold used for determining
 * whether the iteration is converged. 2) initial centers, which are provided by host and compressed into many 512-bit
 * packages. 3) smaples used for training,which are also compressed. kcenters is used for output best centers only.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param data input data from host
 * @param kcenters the output best centers
 */

template <typename DT, int Dim, int Kcluster, int KU, int DV = 128 / KU>
void kMeansTrain(ap_uint<512>* data, ap_uint<512>* kcenters) {
    internal::kMeansTrainImp<DT, Dim, Kcluster, KU, DV>(data, kcenters);
}
} // end of namespace clustering
} // end of namespace data_analytics
} // end of namespace xf
#endif // _XF_DATA_ANALYTICS_KMEANS_TRAIN_HPP_
