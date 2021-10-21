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
 * @file kmeansPredict.hpp
 * @brief header file for kmeans.
 * This file part of Vitis  Library.
 *
 */

#ifndef _XF_DATA_ANALYTICS_L1_KMEANS_PREDICT_HPP_
#define _XF_DATA_ANALYTICS_L1_KMEANS_PREDICT_HPP_
//#include "config.hpp"
#include <ap_int.h>
#include <hls_stream.h>
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
#include <iostream>
#endif

namespace xf {
namespace data_analytics {
namespace clustering {

namespace internal {
const int NUMS = 10000; // only use it for trip_count
////////////////////////////////////////////////////////////////
// common
// -------------  max Value and min Value ------------------------------//
template <typename DT>
struct maxMinValue {};

template <>
struct maxMinValue<float> {
    static const float maxValue;
    static const float minValue;
};

const float maxMinValue<float>::maxValue = 3.40282e+38;
const float maxMinValue<float>::minValue = 1.17549e-38;
template <>
struct maxMinValue<double> {
    static const double maxValue;
    static const double minValue;
};
const double maxMinValue<double>::maxValue = 1.79769e+308;
const double maxMinValue<double>::minValue = 2.22507e-308;

// Recursive tree + inner backup
template <typename DT, int Dim>
DT minValueTree(DT* x, int& k, const int s) {
#pragma HLS inline
    const int dr = (Dim + 1) >> 1;
    int kl = 0;
    int kr = 0;
    DT left = minValueTree<DT, Dim / 2>(x, kl, s);
    DT right = minValueTree<DT, dr>(x, kr, s + Dim / 2);
    k = left <= right ? kl : kr + Dim / 2;
    return (left <= right) ? left : right;
}
// in the case of a function template, only full specialization is allowed by the C++ standard
template <>
double minValueTree<double, 2>(double* x, int& k, const int s) {
#pragma HLS inline
    k = x[0 + s] <= x[1 + s] ? 0 : 1;
    return x[s + 0] <= x[s + 1] ? x[s + 0] : x[s + 1];
}
template <>
double minValueTree<double, 1>(double* x, int& k, const int s) {
#pragma HLS inline
    k = 0;
    return x[0 + s];
}
template <>
float minValueTree<float, 2>(float* x, int& k, const int s) {
#pragma HLS inline
    k = x[0 + s] <= x[1 + s] ? 0 : 1;
    return x[s + 0] <= x[s + 1] ? x[s + 0] : x[s + 1];
}
template <>
float minValueTree<float, 1>(float* x, int& k, const int s) {
#pragma HLS inline
    k = 0;
    return x[0 + s];
}
// -------------  add tree ------------------------------//
template <typename DT, int Dim>
DT addTree(DT* x, const int s) {
#pragma HLS inline
    DT left = addTree<DT, (Dim >> 1)>(x, s);
    DT right = addTree<DT, ((Dim + 1) >> 1)>(x, s + (Dim >> 1));
    return (left + right);
}

template <>
float addTree<float, 2>(float* x, const int s) {
#pragma HLS inline
    return (x[0 + s] + x[1 + s]);
}
template <>
float addTree<float, 1>(float* x, const int s) {
#pragma HLS inline
    return x[0 + s];
}

template <>
double addTree<double, 2>(double* x, const int s) {
#pragma HLS inline
    return (x[0 + s] + x[1 + s]);
}
template <>
double addTree<double, 1>(double* x, const int s) {
#pragma HLS inline
    return x[0 + s];
}
// ---------------   convert DT to inner format ------//
template <typename DT>
union conv {};
template <>
union conv<float> {
    float dt;
    unsigned int ut;
};
template <>
union conv<double> {
    double dt;
    unsigned long long int ut;
};

template <typename DT>
struct dataTypeConverter {
    static DT getDT(ap_uint<sizeof(DT) * 8> ut) {
        conv<DT> temp;
        temp.ut = ut;
        return temp.dt;
    }
    static ap_uint<sizeof(DT) * 8> getUT(DT dt) {
        conv<DT> temp;
        temp.dt = dt;
        return temp.ut;
    }
};
////////////////////////////////////////////////

template <int IW, int Nstrm, int trip>
void split(hls::stream<ap_uint<IW * Nstrm> >& inStrm,
           hls::stream<bool>& eInStrm,
           hls::stream<ap_uint<IW> > outStrm[Nstrm],
           hls::stream<bool>& eOutStrm) {
    const int tm = NUMS * trip;
    while (!eInStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tm max = tm
        ap_uint<IW* Nstrm> data = inStrm.read();
        ;
        for (int i = 0; i < Nstrm; ++i) {
            ap_uint<IW> d = data.range((i + 1) * IW - 1, i * IW);
            outStrm[i].write(d);
        }
        eOutStrm.write(false);
    }
    eOutStrm.write(true);
}
// add end flags for each stream
template <int sz, int sn, int trip>
void dupStrm(hls::stream<ap_uint<sz> > inStrm[sn],
             hls::stream<bool>& eInStrm,
             hls::stream<ap_uint<sz> > s1Strm[sn],
             hls::stream<bool> es1Strm[sn]) {
    int c = 0;
    const unsigned long tcm = NUMS * trip;
    while (!eInStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tcm max = tcm
        for (int i = 0; i < sn; i++) {
            ap_uint<sz> t = inStrm[i].read();
            s1Strm[i].write(t);
            es1Strm[i].write(false);
            c++;
        }
    }
    for (int i = 0; i < sn; i++) {
        es1Strm[i].write(true);
    }
}
//  duplicate streams
template <int sz, int sn, int trip>
void dupStrm(hls::stream<ap_uint<sz> > inStrm[sn],
             hls::stream<bool>& eInStrm,
             hls::stream<ap_uint<sz> > s1Strm[sn],
             hls::stream<bool>& es1Strm,
             hls::stream<ap_uint<sz> > s2Strm[sn],
             hls::stream<bool>& es2Strm) {
    int c = 0;
    const unsigned long tcm = NUMS * trip;
    while (!eInStrm.read()) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tcm max = tcm
        for (int i = 0; i < sn; i++) {
            ap_uint<sz> t = inStrm[i].read();
            s1Strm[i].write(t);
            s2Strm[i].write(t);
            c++;
        }
        es1Strm.write(false);
        es2Strm.write(false);
    }
    es1Strm.write(true);
    es2Strm.write(true);
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "dup c=" << c << std::endl;
#endif
}
// buffer a sample then output it many times
template <int W, int Dim, int Kcluster, int KU, int DV>
void repeateData(hls::stream<ap_uint<W> > inStrm[DV],
                 hls::stream<bool>& eInStrm,
                 const int dims,
                 const int kcluster,
                 hls::stream<ap_uint<W> > outStrm[DV],
                 hls::stream<bool>& eOutStrm) {
    const int maxStep = (Dim + DV - 1) / DV;
    const int step = (dims + DV - 1) / DV;    // dynamic
    const int rep = (kcluster + KU - 1) / KU; // dynamic
    const int trip = ((Kcluster + KU - 1) / KU) * ((Dim + DV - 1) / DV);
    ap_uint<W> buff_a[DV][maxStep];
#pragma HLS array_partition variable = buff_a complete dim = 1
    ap_uint<W> buff_b[DV][maxStep];
#pragma HLS array_partition variable = buff_b complete dim = 1
    bool sw = true;
    bool last = eInStrm.read();
    int count = 0;
    int inc = 0;
    int r = 0;
    int p = 0;
    int kp = 0;
    bool end = false;
    const unsigned long tc = NUMS * trip;
    while (!end) {
#pragma HLS pipeline II = 1
#pragma HLS loop_tripcount min = tc max = tc
#pragma HLS DEPENDENCE variable = buff_a inter false
#pragma HLS DEPENDENCE variable = buff_b inter false
        if (r == 0 && !last) {
            for (int i = 0; i < DV; ++i) {
                ap_uint<W> data = inStrm[i].read();
                if (sw)
                    buff_a[i][p] = data;
                else
                    buff_b[i][p] = data;
            }
            inc++;
            last = eInStrm.read();
        }

        count++;
        if (count > step * rep) {
            for (int i = 0; i < DV; ++i) {
                ap_uint<W> data = sw ? buff_b[i][p] : buff_a[i][p];
                outStrm[i].write(data);
            }
            eOutStrm.write(false);
        }
        if (p + 1 < step) {
            p++;
        } else {
            p = 0;
            sw = r + 1 < rep ? sw : !sw;
            r = r + 1 < rep ? (r + 1) : 0;
            kp = last ? kp + 1 : kp;
            end = (kp == 2 * rep) && last;
        }
    }
#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "inc=" << inc << " step= " << step << "  rep=" << rep << std::endl;
    std::cout << "repeate samples " << count << " times" << std::endl;
#endif
    eOutStrm.write(true);
}
/**
 * @brief computingDistance calculates  distance of each sample and  all centers.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceil(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param sampleStrm input sample streams
 * @param endSampleStrm the end flag of sample stream
 * @param centers an array stored centers
 * @param dims  the number of dimensions
 * @param kcluster  the number of clusters
 * @param distStrm distance streams
 * @param eDistStrm the end flag of distance stream
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void computingDistance(hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[DV],
                       hls::stream<bool>& endSampleStrm,
                       ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                       const int dims,
                       const int kcluster,
                       hls::stream<DT> distStrm[KU],
                       hls::stream<bool>& eDistStrm) {
    const int sz = sizeof(DT) * 8;
    const DT maxDT = maxMinValue<DT>::maxValue;
    DT psum[KU];
#pragma HLS array_partition variable = psum complete dim = 1
    DT data[DV];
#pragma HLS array_partition variable = data complete dim = 1
    int p = 0;
    const int totalRows = ((kcluster + KU - 1) / KU) * ((dims + DV - 1) / DV);
    const int tmsp = NUMS * ((Dim + DV - 1) / DV) * ((Kcluster + KU - 1) / KU);
    int r = 0;
DISTANCE:
    while (!endSampleStrm.read()) {
#pragma HLS pipeline II = 1 // 6
#pragma HLS loop_tripcount min = tmsp max = tmsp
        int q = p;
        for (int i = 0; i < DV; ++i) {
            ap_uint<sz> ut = sampleStrm[i].read();
            DT dt = dataTypeConverter<DT>::getDT(ut);
            data[i] = dt;
        }
        bool ov = (p + DV >= dims) ? true : false;
        p = ov ? 0 : p + DV;
    SUB_DISTANCE:
        for (int k = 0; k < KU; ++k) {
#pragma HLS unroll
            DT dff[DV];
            ap_uint<sz* DV> uc = centers[k][r];
#pragma HLS array_partition variable = dff complete dim = 1
            for (int i = 0; i < DV; ++i) {
#pragma HLS unroll
                ap_uint<sz> ut = uc.range((i + 1) * sz - 1, i * sz);
                DT dt = dataTypeConverter<DT>::getDT(ut);
                DT dt2 = dt;
                DT pw = (data[i] - dt2) * (data[i] - dt2);
                dff[i] = q + i < dims ? pw : 0;
            } // for i<DV

            DT f = addTree<DT, DV>(dff, 0);
            DT ps = q == 0 ? 0 : psum[k];
            DT tmp = f + ps;
            psum[k] = tmp;

            if (ov) {
                DT t = tmp; // psum[k];
                distStrm[k].write(t);
            }
        } // for k

        if (ov) eDistStrm.write(false);

        r = r + 1 < totalRows ? r + 1 : 0;
    } // while
    eDistStrm.write(true);
}

/**
 * @brief minDist calculates the cluster index according to the minimum distance of each sample and  all centers.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceil(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param distStrm distance streams
 * @param eDistStrm the end flag of distance stream
 * @param kcluster  the number of clusters
 * @param tagStrm tag stream, label a cluster ID for each sample
 * @param eTagStrm  end flag of tag stream
 */
template <typename DT, int Dim, int Kcluster, int KU>
void minDist(hls::stream<DT> distStrm[KU],
             hls::stream<bool>& eDistStrm,
             const int kcluster,
             hls::stream<ap_uint<32> >& tagStrm,
             hls::stream<bool>& eTagStrm) {
    const DT maxDT = maxMinValue<DT>::maxValue;
    const int maxL = KU;
    const int maxl = (kcluster + KU - 1) / KU;
    DT ds[KU];
#pragma HLS array_partition variable = ds complete dim = 0
    DT minD;
    int mink;
    int minp;
    for (int k = 0; k < KU; ++k) {
#pragma HLS unroll
        ds[k] = maxDT;
    }
    minD = maxDT;
    int p = 0;
    int cck = 0;
    const int tc = NUMS * ((Kcluster + KU - 1) / KU);
    while (!eDistStrm.read()) {
#pragma HLS pipeline II = 2
#pragma HLS loop_tripcount min = tc max = tc
        for (int k = 0; k < KU; ++k) {
            DT dis = distStrm[k].read();
            ds[k] = cck + k < kcluster ? dis : maxDT;
        }
        int kp;
        cck += KU;
        DT m = minValueTree<DT, KU>(ds, kp, 0);
        if (minD > m) {
            mink = kp;
            minp = p;
            minD = m;
        }
        if (p + 1 >= maxl) {
            p = 0;
            cck = 0;
            ap_uint<32> tk = minp * KU + mink;
            // tk.range(15, 0) = minp;
            // tk.range(31, 16) = mink;
            tagStrm.write(tk);
            eTagStrm.write(false);
            minD = maxDT;
        } else
            p++;
    }
    eTagStrm.write(true);
}

/**
 * @brief closetCenters computes the cluster index according to the minimum distance of each sample and  all centers.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceil(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param sampleStrm input sample streams
 * @param endSampleStrm the end flag of sample stream
 * @param centers an array stored centers
 * @param dims  the number of dimensions
 * @param kcluster  the number of clusters
 * @param tagStrm tag stream, label a cluster ID for each sample
 * @param endTagStrm  end flag of tag stream
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void closestCenter(hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[DV],
                   hls::stream<bool>& endSampleStrm,
                   ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                   const int dims,
                   const int kcluster,
                   hls::stream<ap_uint<32> >& tagStrm,
                   hls::stream<bool>& eTagStrm) {
#pragma HLS dataflow
    const int DEPTH = 64;
    hls::stream<DT> psStrm[KU];
#pragma HLS STREAM variable = psStrm depth = DEPTH
#pragma HLS bind_storage variable = psStrm type = fifo impl = lutram

    hls::stream<bool> ePsStrm;
#pragma HLS STREAM variable = ePsStrm depth = DEPTH
#pragma HLS bind_storage variable = ePsStrm type = fifo impl = lutram

    hls::stream<DT> distStrm[KU];
#pragma HLS STREAM variable = distStrm depth = DEPTH
#pragma HLS bind_storage variable = distStrm type = fifo impl = lutram

    hls::stream<bool> eDistStrm;
#pragma HLS STREAM variable = eDistStrm depth = DEPTH
#pragma HLS bind_storage variable = eDistStrm type = fifo impl = lutram
    // distance
    computingDistance<DT, Dim, Kcluster, uramDepth, KU, DV>(sampleStrm, endSampleStrm, centers, dims, kcluster,
                                                            distStrm, eDistStrm);

    minDist<DT, Dim, Kcluster, KU>(distStrm, eDistStrm, kcluster, tagStrm, eTagStrm);
}

/**
 * @brief kMeansPredict predicts cluster index for each sample.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceil(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param sampleStrm input sample streams, a sample needs ceiling(dims/DV) times to read.
 * @param endSampleStrm the end flag of sample stream.
 * @param centers an array stored centers, user should partition dim=1 in its defination.
 * @param dims  the number of dimensions.
 * @param kcluster  the number of clusters.
 * @param tagStrm tag stream, label a cluster ID for each sample.
 * @param endTagStrm  end flag of tag stream.
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void kMeansPredictImp(hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[DV],
                      hls::stream<bool>& endSampleStrm,
                      ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                      const int dims,
                      const int kcluster,
                      hls::stream<ap_uint<32> >& tagStrm,
                      hls::stream<bool>& endTagStrm) {
#pragma HLS inline off
    const int DEPTH = 16;
    const int DEPTH2 = DEPTH * 2;
#pragma HLS dataflow
    hls::stream<ap_uint<sizeof(DT) * 8> > repSmpStrm[DV];
#pragma HLS STREAM variable = repSmpStrm depth = DEPTH
#pragma HLS bind_storage variable = repSmpStrm type = fifo impl = lutram
    hls::stream<bool> endRepSmpStrm;
#pragma HLS STREAM variable = endRepSmpStrm depth = DEPTH
#pragma HLS bind_storage variable = endRepSmpStrm type = fifo impl = lutram

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "starting cluster" << std::endl;
#endif
    // ouput each sample ceiling(kcluster/KU) times assume kcluster>KU and dims>DV.
    repeateData<sizeof(DT) * 8, Dim, Kcluster, KU, DV>(sampleStrm, endSampleStrm, dims, kcluster, repSmpStrm,
                                                       endRepSmpStrm);
    // compute the minimum distance of each sample and all centers
    closestCenter<DT, Dim, Kcluster, uramDepth, KU, DV>(repSmpStrm, endRepSmpStrm, centers, dims, kcluster, tagStrm,
                                                        endTagStrm);

#if !defined(__SYNTHESIS__) && XF_DATA_ANALYTICS_DEBUG == 1
    std::cout << "end cluster" << std::endl;
#endif
}

// -------------------------------------------------------//
} // end of namespace internal

/**
 * @brief kMeansPredict predicts cluster index for each sample.
 * In order to achive to acceleration, please make sure partition 1-dim  of centers.
 * @tparam DT data type, supporting float and double
 * @tparam Dim the maximum number of dimensions,dynamic number of dimension should be not greater than the maximum.
 * @tparam Kcluster the maximum number of cluster,dynamic number of cluster should be not greater than the maximum.
 * @tparam uramDepth the depth of uram where centers are stored. uramDepth should be not less than ceiling(Kcluster/KU)
 * * ceil(Dim/DV)
 * @tparam KU unroll factor of Kcluster, KU centers are took part in calculating distances concurrently with one sample.
 * After Kcluster/KU+1 times at most, ouput the minimum  distance of a sample and Kcluster centers.
 * @tparam DV unroll factor of Dim, DV elements in a center are took part in calculating distances concurrently with one
 * sample.
 *
 * @param sampleStrm input sample streams, a sample needs ceiling(dims/DV) times to read.
 * @param endSampleStrm the end flag of sample stream.
 * @param centers an array stored centers, user should partition dim=1 in its defination.
 * @param dims  the number of dimensions.
 * @param kcluster  the number of clusters.
 * @param tagStrm tag stream, label a cluster ID for each sample.
 * @param endTagStrm  end flag of tag stream.
 */
template <typename DT, int Dim, int Kcluster, int uramDepth, int KU, int DV>
void kMeansPredict(hls::stream<ap_uint<sizeof(DT) * 8> > sampleStrm[DV],
                   hls::stream<bool>& endSampleStrm,
                   ap_uint<sizeof(DT) * 8 * DV> centers[KU][uramDepth],
                   const int dims,
                   const int kcluster,
                   hls::stream<ap_uint<32> >& tagStrm,
                   hls::stream<bool>& endTagStrm) {
    internal::kMeansPredictImp<DT, Dim, Kcluster, uramDepth, KU, DV>(sampleStrm, endSampleStrm, centers, dims, kcluster,
                                                                     tagStrm, endTagStrm);
}
} // end of namespace clustering
} // end of namespace data_analytics
} // end of namespace xf
#endif // _XF_DATA_ANALYTICS_KMEANS_PREDICT_HPP_
