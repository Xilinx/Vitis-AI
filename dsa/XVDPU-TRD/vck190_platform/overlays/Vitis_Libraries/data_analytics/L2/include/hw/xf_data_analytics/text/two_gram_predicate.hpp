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
 * @file geoip.hpp
 * @brief This file include the class GeoIP
 *
 */
#ifndef XF_TEXT_TWO_GRAM_PREDICATE_HPP
#define XF_TEXT_TWO_GRAM_PREDICATE_HPP

#ifndef __SYNTHESIS__
#include <iostream>
#endif
#include <ap_int.h>
#include <hls_stream.h>
#include <hls_math.h>

namespace xf {
namespace data_analytics {
namespace text {

namespace internal {

union DTConvert64 {
    uint64_t dt0;
    double dt1;
};

// def Pair (ap_uint<128> ) <=> key (ap_uint<63>) + end (bool) + data (double)
template <typename KT, typename DT>
struct Pair : public ap_uint<8 * sizeof(DT) + 8 * sizeof(KT)> {
    enum { DW = 8 * sizeof(DT), KW = 8 * sizeof(KT) };
    void key(KT k) { this->range(KW - 2, 0) = k; }
    void data(DT d) {
        DTConvert64 dtc;
        dtc.dt1 = d;
        this->range(DW + KW - 1, KW) = dtc.dt0;
    }
    void end(bool e) { this->range(KW - 1, KW - 1) = e; }

    Pair() {}
    Pair(const Pair& o) { this->V = o.V; }
    Pair(KT k, DT d, bool e) {
        key(k);
        data(d);
        end(e);
    }

    KT key() { return this->range(KW - 2, 0); };
    DT data() {
        DTConvert64 dtc;
        dtc.dt0 = this->range(DW + KW - 1, KW);
        return dtc.dt1;
    };
    bool end() { return this->range(KW - 1, KW - 1); }
};

// if key1 = key2, data = data1 + data2
// else if key1 < key2, data = data1
// else data = data2
template <typename PT>
void mergeSum(hls::stream<PT>& in1Strm, hls::stream<PT>& in2Strm, hls::stream<PT>& outStrm) {
    const uint64_t MAX_ID = 0xFFFFFFFFFFFFFFFF;
    PT pt1 = in1Strm.read();
    PT pt2 = in2Strm.read();
    bool end1 = pt1.end();
    bool end2 = pt2.end();
    ap_uint<64> id1 = 0, id2 = 0;
    double tf1 = 0.0, tf2 = 0.0;
    while (!end1 || !end2) {
#pragma HLS loop_tripcount max = 20000 min = 20000
#pragma HLS pipeline ii = 1
        if (id1 == id2) {
            if (!end1) {
                tf1 = pt1.data();
                id1 = pt1.key();
                pt1 = in1Strm.read();
                end1 = pt1.end();
            } else {
                id1 = MAX_ID;
            }
            if (!end2) {
                tf2 = pt2.data();
                id2 = pt2.key();
                pt2 = in2Strm.read();
                end2 = pt2.end();
            } else {
                id2 = MAX_ID;
            }
        } else if (id1 > id2) {
            if (!end2) {
                tf2 = pt2.data();
                id2 = pt2.key();
                pt2 = in2Strm.read();
                end2 = pt2.end();
            } else {
                tf1 = pt1.data();
                id1 = pt1.key();
                id2 = MAX_ID;
                pt1 = in1Strm.read();
                end1 = pt1.end();
            }

        } else {
            if (!end1) {
                tf1 = pt1.data();
                id1 = pt1.key();
                pt1 = in1Strm.read();
                end1 = pt1.end();
            } else {
                tf2 = pt2.data();
                id1 = MAX_ID;
                id2 = pt2.key();
                pt2 = in2Strm.read();
                end2 = pt2.end();
            }
        }
        // std::cout << "mergeSum: id1=" << id1 << ", tf1=" << tf1 << ", id2=" << id2 << ", tf2=" << tf2 << std::endl;
        if (id1 == id2) {
            outStrm.write({id1, tf1 + tf2, 0});
        } else if (id1 < id2) {
            outStrm.write({id1, tf1, 0});

        } else {
            outStrm.write({id2, tf2, 0});
        }
    }
    outStrm.write({0.0, 0, 1});
}

// mergeSum Wrapper for dataflow
template <typename PT>
void mergeSumWrapper(int docSize, hls::stream<PT>& in1Strm, hls::stream<PT>& in2Strm, hls::stream<PT>& outStrm) {
    for (int j = 0; j < docSize; j++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
        mergeSum<PT>(in1Strm, in2Strm, outStrm);
    }
}

// mergeTree by recursively
template <typename PT, int N>
struct mergeTree {
    static void f(int docSize, hls::stream<PT> idTfStrm[N], hls::stream<PT>& idTfOutStrm) {
#pragma HLS dataflow
        hls::stream<PT> idTfStrm1;
#pragma HLS stream variable = idTfStrm1 depth = 4
#pragma HLS resource variable = idTfStrm1 core = FIFO_LUTRAM
        hls::stream<PT> idTfStrm2;
#pragma HLS stream variable = idTfStrm2 depth = 4
#pragma HLS resource variable = idTfStrm2 core = FIFO_LUTRAM

        mergeTree<PT, N / 2>::f(docSize, idTfStrm, idTfStrm1);
        mergeTree<PT, N / 2>::f(docSize, &idTfStrm[N / 2], idTfStrm2);
        mergeSumWrapper<PT>(docSize, idTfStrm1, idTfStrm2, idTfOutStrm);
    }
};

template <typename PT>
struct mergeTree<PT, 2> {
    static void f(int docSize, hls::stream<PT> idTfStrm[2], hls::stream<PT>& idTfOutStrm) {
        mergeSumWrapper<PT>(docSize, idTfStrm[0], idTfStrm[1], idTfOutStrm);
    }
};

// back-pressure control
template <typename PT, int BL>
void backPressure(int docSize, hls::stream<PT>& inStrm, hls::stream<bool>& bpStrm, hls::stream<PT>& outStrm) {
    for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
        PT pt = inStrm.read();
        int cnt = 0;
        while (!pt.end()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 20000 min = 20000
            outStrm.write(pt);
            if (cnt == BL - 1) {
                bpStrm.read();
                cnt = 0;
            } else {
                cnt++;
            }
            pt = inStrm.read();
        }
        outStrm.write(pt);
        if (cnt) bpStrm.read();
    }
}

// back-pressure wrapper
template <typename PT, int N, int BL>
void mBackPressure(int docSize, hls::stream<PT> inStrm[N], hls::stream<bool> bpStrm[N], hls::stream<PT> outStrm[N]) {
#pragma HLS dataflow
    for (int i = 0; i < N; i++) {
#pragma HLS unroll
        backPressure<PT, BL>(docSize, inStrm[i], bpStrm[i], outStrm[i]);
    }
}

// burst read Array to Stream
template <typename DT>
void readArr2Strm(int size, DT* arrIn, hls::stream<DT>& outStrm) {
loop_readArr2Strm:
    for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 100000 min = 100000
        outStrm.write(arrIn[i]);
    }
}

// burst write Stream to Array
template <typename DT>
void writeStrm2Arr(int size, hls::stream<DT>& inStrm, DT* arrOut) {
    for (int i = 0; i < size; i++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 100000 min = 100000
        arrOut[i] = inStrm.read();
    }
}

// delete special characters and use 6-bit encoding characters
// The encoding rules are: 0~9 <-> '0'~'9', 10~35 <-> 'a/A'~'z/Z', 36 -> ' ' & '\n'.
template <int CH>
void preProcess(int docSize,
                hls::stream<uint8_t>& fieldStrm,
                hls::stream<uint32_t>& offsetStrm,
                hls::stream<ap_uint<6> >& fieldCodeStrm,
                hls::stream<ap_uint<8> >& sizeStrm,
                hls::stream<bool>& fieldCodeEndStrm) {
    uint32_t begin = 0;
    for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
        ap_uint<6> last_code = 36;
        uint32_t offset = offsetStrm.read();
        ap_uint<8> cnt = 0;
        for (int j = begin; j < offset; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
            uint8_t ch = fieldStrm.read();
            ap_uint<6> ch_code;
            if (ch >= 65 && ch <= 90) ch += 32;
            if (ch >= 48 && ch <= 57)
                ch_code = ch - 48;
            else if (ch >= 97 && ch <= 122)
                ch_code = ch - 87;
            else if (ch == 32 || ch == 10) // space & \n
                ch_code = 36;
            else {
                // std::cout << std::endl << (int)ch << std::endl;
                continue;
            }
            if (!(ch_code == last_code && ch_code == 36)) { // Leave a space
                fieldCodeStrm.write(ch_code);
                fieldCodeEndStrm.write(false);
                cnt++;
                // std::cout << ch;
            }
            last_code = ch_code;
        }
        if (last_code == 36)
            sizeStrm.write(cnt - 1);
        else
            sizeStrm.write(cnt);
        fieldCodeEndStrm.write(true);
        // std::cout << std::endl;
        begin = offset;
    }
}

} // internal

/**
 * @brief TwoGramPredicate the 2-Gram Predicate is a search of the inverted index with a term of 2 characters.
 *
 * @tparam AXI_DT AXI data type
 * @tparam CH the channel of merge tree
 * @tparam BL the burst length
 *
 */
template <typename AXI_DT, int CH = 64, int BL = 512>
class TwoGramPredicate {
   private:
    static const int TN = 4096; // term number
    static const int SN = 4;    // split number
    typedef internal::Pair<ap_uint<64>, double> PT;
    int doc_size, field_size;
    ap_uint<128> idf_tfAddr[TN]; // 63~0: idf value, 127~64: tf address

    // split field to terms according to 2-Gram
    void twoGram(int docSize,
                 hls::stream<ap_uint<6> >& fieldCodeStrm,
                 hls::stream<ap_uint<8> >& fieldSizeStrm,
                 hls::stream<bool>& fieldCodeEndStrm,
                 hls::stream<ap_uint<12> >& termStrm,
                 hls::stream<ap_uint<8> >& chanNumStrm) {
        ap_uint<6> last_char, now_char;
        for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
            int cnt = 0;
            ap_uint<8> size = fieldSizeStrm.read();
            chanNumStrm.write(size - 1);
            while (!fieldCodeEndStrm.read()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
                now_char = fieldCodeStrm.read();
                if (cnt > 0 && cnt < size) {
                    ap_uint<12> term = now_char.concat(last_char); // now_char<<6+last_char
                    termStrm.write(term);
                }
                cnt++;
                last_char = now_char;
            }
        }
    }

    // sort term
    void insertSort(int docSize,
                    hls::stream<ap_uint<12> >& termStrm,
                    hls::stream<ap_uint<8> >& numStrm,
                    hls::stream<ap_uint<12> >& termOutStrm,
                    hls::stream<ap_uint<8> >& numOutStrm) {
        ap_uint<12> term_arr[CH * 2];
#pragma HLS ARRAY_PARTITION variable = term_arr complete
        for (int i = 0; i < docSize; i++) { // sort
#pragma HLS loop_tripcount max = 100000 min = 100000
            ap_uint<8> num = numStrm.read();
            numOutStrm.write(num);
            for (int j = 0; j < num; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
                ap_uint<12> term = termStrm.read();
                term_arr[j] = term;
                for (int k = CH * 2 - 2; k >= 0; k--) {
                    if (k < j) {
                        if (term < term_arr[k]) {
                            term_arr[k + 1] = term_arr[k];
                            term_arr[k] = term;
                        }
                    }
                }
            }

            for (int j = 0; j < num; j++) { // output
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
                termOutStrm.write(term_arr[j]);
            }
        }
    }

    // read memory to get idf value and tf address in DRAM
    void getIDFTFAddr(int docSize,
                      hls::stream<ap_uint<12> >& termStrm,
                      hls::stream<ap_uint<8> >& numStrm,
                      hls::stream<double>& thldStrm,
                      hls::stream<double>& idfStrm,
                      hls::stream<ap_uint<64> >& tfAddrStrm,
                      hls::stream<ap_uint<8> >& chanStrm) {
        for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
            ap_uint<8> num = numStrm.read();
            double threshold = 0.0;
            ap_uint<8> chan = 0;
            ap_uint<8> cnt = 1, cnt_last = 1;
            ap_uint<12> term, term_last;
            internal::DTConvert64 dtc;
            for (int j = 0; j < num + 1; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
                dtc.dt1 = 0.0;
                if (num > 0 && j < num) term = termStrm.read(); // if term exist, then read
                if (chan < CH) {                                // filter term for number more than CH
                    if (j == num) {
                        ap_uint<128> tmp = idf_tfAddr[term_last];
                        uint64_t idf = tmp(63, 0);
                        dtc.dt0 = idf;
                        idfStrm.write(dtc.dt1 * cnt);
                        tfAddrStrm.write(tmp(127, 64));
                        chan++;
                        cnt_last = cnt;
                    } else if (j > 0) {
                        if (term == term_last) { // merge term
                            cnt++;
                        } else {
                            ap_uint<128> tmp = idf_tfAddr[term_last];
                            uint64_t idf = tmp(63, 0);
                            dtc.dt0 = idf;
                            idfStrm.write(dtc.dt1 * cnt);
                            tfAddrStrm.write(tmp(127, 64));
                            ap_uint<32> begin = tmp(95, 64);
                            ap_uint<32> end = tmp(127, 96);
#ifndef __SYNTHESIS__
#ifdef DEBUGx
                            std::cout << "c=" << chan << ", begin=" << begin << ", end=" << end
                                      << ", delta=" << end - begin << std::endl;
#endif
#endif
                            chan++;
                            cnt_last = cnt;
                            cnt = 1;
                        }
                    }
                } else {
#ifndef __SYNTHESIS__
#ifdef DEBUG
                    std::cout << "[WARNING] " << num << " more than max channel number (" << CH << ") \n";
#endif
#endif
                }
#ifndef __SYNTHESIS__
#ifdef DEBUG
                std::cout << "threshold[" << i << "]=" << threshold << ", dtc.dt1=" << dtc.dt1
                          << ", cnt_last=" << cnt_last << ", term_last=" << term_last << std::endl;
#endif
#endif
                term_last = term;
                threshold += dtc.dt1 * dtc.dt1 * cnt_last;
            }
#ifndef __SYNTHESIS__
#ifdef DEBUG
            std::cout << "threshold=" << std::sqrt(threshold) * 0.8;
            std::cout << ", num=" << num << ", chan=" << chan << std::endl;
#endif
#endif
            thldStrm.write(threshold * 0.64);
            chanStrm.write(chan);
        }
    }

    // split channel
    void splitChan(int docSize,
                   hls::stream<ap_uint<8> >& chanSizeStrm,
                   hls::stream<double>& idfStrm,
                   hls::stream<ap_uint<64> >& tfAddrStrm,
                   hls::stream<ap_uint<8> > chanSizeSplitStrm[SN],
                   hls::stream<double> idfSplitStrm[SN],
                   hls::stream<ap_uint<64> > tfAddrSplitStrm[SN]) {
        for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
            ap_uint<8> chan_size = chanSizeStrm.read();
            for (int s = 0; s < SN; s++) chanSizeSplitStrm[s].write(chan_size);
            for (int c = 0; c < chan_size; c++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH min = CH
                double idf = idfStrm.read();
                idfSplitStrm[c % SN].write(idf);
                ap_uint<64> tf_addr = tfAddrStrm.read();
                tfAddrSplitStrm[c % SN].write(tf_addr);
            }
        }
    }

    // distribute data to CH / SN channel
    void distData(int docSize,
                  int chanNum,
                  hls::stream<ap_uint<8> >& chanSizeStrm,
                  hls::stream<double>& idfStrm,
                  hls::stream<ap_uint<64> >& tfAddrStrm,
                  AXI_DT* tfValue,
                  hls::stream<PT> idTfStrm[CH / SN],
                  hls::stream<bool> backPressStrm[CH / SN]) {
        for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
            ap_uint<8> chan_size = chanSizeStrm.read();
            ap_uint<64> tf_addr[CH / SN];
            double idf_value[CH / SN];
            ap_uint<CH / SN> chan_flag = 0;
            ap_uint<CH / SN> end_flag = 0;
            for (int c = 0; c < (chan_size + SN - 1 - chanNum) / SN; c++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH / SN min = CH / SN
                tf_addr[c] = tfAddrStrm.read();
                chan_flag[c] = 1;
                idf_value[c] = idfStrm.read();
            }
            if (!chan_flag) {
                for (int c = 0; c < CH / SN; c++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = CH / SN min = CH / SN
                    idTfStrm[c].write({0, 0, 1});
                }
            }
            while (chan_flag) {
#pragma HLS loop_tripcount max = CH / SN min = CH / SN
                for (int c = 0; c < CH / SN; c++) {
#pragma HLS loop_tripcount max = CH / SN min = CH / SN
                    ap_uint<64> addr = tf_addr[c];
                    ap_uint<32> begin = addr(31, 0);
                    ap_uint<32> end = addr(63, 32);
                    int delta = end - begin;
                    if (chan_flag[c]) {                 // implement back pressure
                        if (end == 0 && begin == end) { // no match term
                            chan_flag[c] = 0;
                        } else if (backPressStrm[c].write_nb(1)) { // implement back pressure
#ifndef __SYNTHESIS__
#ifdef DEBUGx
                            std::cout << "distData: c=" << c << ", begin=" << begin << ", end=" << end
                                      << ", delta=" << delta << std::endl;
#endif
#endif
                            if (delta > BL) {
                                tf_addr[c] = addr + BL;
                            } else {
                                chan_flag[c] = 0;
                            }
                        loop_distData_5:
                            for (int j = 0; j < BL; j++) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = BL min = BL
                                AXI_DT tf = tfValue[begin + j];
                                if (j < delta) {
                                    internal::DTConvert64 dtc;
                                    dtc.dt0 = tf(127, 64);
                                    idTfStrm[c].write({tf(62, 0), dtc.dt1 * idf_value[c], 0});
                                }
                            }
                            begin += BL;
                        }
                    }
                    if (!chan_flag[c] && !end_flag[c]) {
                        idTfStrm[c].write({0, 0, 1});
                        end_flag[c] = 1;
                    }
                }
            }
        }
    }

    // one is divided into two to improve the frequency
    void distDataWrapper(int docSize,
                         hls::stream<ap_uint<8> > chanSizeStrm[SN],
                         hls::stream<double> idfStrm[SN],
                         hls::stream<ap_uint<64> > tfAddrStrm[SN],
                         AXI_DT* tfValue,
                         AXI_DT* tfValue2,
                         AXI_DT* tfValue3,
                         AXI_DT* tfValue4,
                         hls::stream<PT> idTfStrm[CH],
                         hls::stream<bool> backPressStrm[CH]) {
#pragma HLS dataflow
#ifndef __SYNTHESIS__
#ifdef DEBUG
        std::cout << "distDataWrapper\n";
#endif
#endif
        distData(docSize, 0, chanSizeStrm[0], idfStrm[0], tfAddrStrm[0], tfValue, idTfStrm, backPressStrm);
        distData(docSize, 1, chanSizeStrm[1], idfStrm[1], tfAddrStrm[1], tfValue2, &idTfStrm[CH / SN],
                 &backPressStrm[CH / SN]);
        distData(docSize, 2, chanSizeStrm[2], idfStrm[2], tfAddrStrm[2], tfValue3, &idTfStrm[CH * 2 / SN],
                 &backPressStrm[CH * 2 / SN]);
        distData(docSize, 3, chanSizeStrm[3], idfStrm[3], tfAddrStrm[3], tfValue4, &idTfStrm[CH * 3 / SN],
                 &backPressStrm[CH * 3 / SN]);
    }

    // get first index id when tf > threshold
    void byValue(int docSize, hls::stream<double>& thldStrm, hls::stream<PT>& idTfStrm, hls::stream<uint32_t>& idStrm) {
        for (int i = 0; i < docSize; i++) {
#pragma HLS loop_tripcount max = 100000 min = 100000
            bool first_flag = true;
            double threshold = thldStrm.read();
            PT pt = idTfStrm.read();
            while (!pt.end()) {
#pragma HLS pipeline ii = 1
#pragma HLS loop_tripcount max = 20000 min = 20000
                uint32_t id = pt.key();
                double tf = pt.data();
                double tmp = tf * tf;
                if (first_flag && tmp > threshold) {
#ifndef __SYNTHESIS__
#ifdef DEBUG
                    std::cout << "i=" << i << ", id=" << id << ", threshold=" << std::sqrt(threshold) << ", tf=" << tf
                              << std::endl;
#endif
#endif
                    first_flag = false;
                    idStrm.write(id); // output the first value that meets the condition
                }
                pt = idTfStrm.read();
            }
            if (first_flag) idStrm.write(-1); // not meeting the condition
        }
    }

   public:
    /**
     * @brief TwoGramPredicate constructor
     */
    TwoGramPredicate() {
#pragma HLS inline
#pragma HLS resource variable = idf_tfAddr core = RAM_T2P_URAM
#ifndef __SYNTHESIS__
        std::cout << "TwoGramPredicate constructor\n";
#endif
    }

    /**
     * @brief init the parameter initial function
     * @param docSize the document size
     * @param fieldSize all field's string length
     * @param idfValue the idfValue buffer that stores idf value
     * @param tfAddr the tfAddr buffer that stores tf address
     */
    void init(int docSize, int fieldSize, double idfValue[TN], uint64_t tfAddr[TN]) {
        doc_size = docSize;
        field_size = fieldSize;
    loop_init:
        for (int i = 0; i < TN; i++) {
#pragma HLS pipeline ii = 1
            internal::DTConvert64 dtc;
            dtc.dt1 = idfValue[i];
            ap_uint<128> tmp;
            tmp(63, 0) = dtc.dt0;
            tmp(127, 64) = tfAddr[i];
            // store idf value and tf address into a URAM
            idf_tfAddr[i] = tmp;
            ap_uint<32> begin = tmp(95, 64);
            ap_uint<32> end = tmp(127, 96);
#ifndef __SYNTHESIS__
#ifdef DEBUG
            if (end - begin < 0)
                std::cout << "i=" << i << ", begin=" << begin << ", end=" << end << ", delta=" << end - begin
                          << std::endl;
#endif
#endif
        }
    }

    /**
     * @brief search Find the matching id in the inverted index.
     * @param field One column field
     * @param offset The offset address of field
     * @param tfValue The first pointer of term frequency values in memory
     * @param tfValue The second pointer of term frequency values in memory
     * @param the index id of return
     */
    void search(uint8_t* field,
                uint32_t* offset,
                AXI_DT* tfValue,
                AXI_DT* tfValue2,
                AXI_DT* tfValue3,
                AXI_DT* tfValue4,
                uint32_t* indexId) {
#pragma HLS dataflow
        hls::stream<uint8_t> fieldStrm;
#pragma HLS stream variable = fieldStrm depth = 4
#pragma HLS resource variable = fieldStrm core = FIFO_LUTRAM
        internal::readArr2Strm<uint8_t>(field_size, field, fieldStrm);

        hls::stream<uint32_t> offsetStrm;
#pragma HLS stream variable = offsetStrm depth = 4
#pragma HLS resource variable = offsetStrm core = FIFO_LUTRAM
        internal::readArr2Strm<uint32_t>(doc_size, offset, offsetStrm);

        hls::stream<ap_uint<6> > fieldCodeStrm;
#pragma HLS stream variable = fieldCodeStrm depth = 2048
#pragma HLS resource variable = fieldCodeStrm core = FIFO_BRAM
        hls::stream<ap_uint<8> > fieldSizeStrm;
#pragma HLS stream variable = fieldSizeStrm depth = 32
#pragma HLS resource variable = fieldSizeStrm core = FIFO_LUTRAM
        hls::stream<bool> fieldCodeEndStrm;
#pragma HLS stream variable = fieldCodeEndStrm depth = 2048
#pragma HLS resource variable = fieldCodeEndStrm core = FIFO_BRAM
        internal::preProcess<CH>(doc_size, fieldStrm, offsetStrm, fieldCodeStrm, fieldSizeStrm, fieldCodeEndStrm);

        hls::stream<ap_uint<12> > termStrm;
#pragma HLS stream variable = termStrm depth = 4
#pragma HLS resource variable = termStrm core = FIFO_LUTRAM
        hls::stream<ap_uint<8> > chanNumStrm;
#pragma HLS stream variable = chanNumStrm depth = 4
#pragma HLS resource variable = chanNumStrm core = FIFO_LUTRAM
        twoGram(doc_size, fieldCodeStrm, fieldSizeStrm, fieldCodeEndStrm, termStrm, chanNumStrm);

        hls::stream<ap_uint<12> > termSortStrm;
#pragma HLS stream variable = termSortStrm depth = 4
#pragma HLS resource variable = termSortStrm core = FIFO_LUTRAM
        hls::stream<ap_uint<8> > numSortStrm;
#pragma HLS stream variable = numSortStrm depth = 4
#pragma HLS resource variable = numSortStrm core = FIFO_LUTRAM
        insertSort(doc_size, termStrm, chanNumStrm, termSortStrm, numSortStrm);

        hls::stream<double> thldStrm;
#pragma HLS stream variable = thldStrm depth = 4
#pragma HLS resource variable = thldStrm = FIFO_LUTRAM
        hls::stream<double> idfStrm;
#pragma HLS stream variable = idfStrm depth = 128
#pragma HLS resource variable = idfStrm core = FIFO_LUTRAM
        hls::stream<ap_uint<64> > tfAddrStrm;
#pragma HLS stream variable = tfAddrStrm depth = 128
#pragma HLS resource variable = tfAddrStrm core = FIFO_LUTRAM
        hls::stream<ap_uint<8> > chanStrm;
#pragma HLS stream variable = chanStrm depth = 4
#pragma HLS resource variable = chanStrm core = FIFO_LUTRAM
        getIDFTFAddr(doc_size, termSortStrm, numSortStrm, thldStrm, idfStrm, tfAddrStrm, chanStrm);

// ==== weighted Union Module Start ======
#ifndef __SYNTHESIS__
        std::cout << "Weighted Union Module Start...\n";
#endif
        hls::stream<ap_uint<8> > chanNumSplitStrm[SN];
#pragma HLS stream variable = chanNumSplitStrm depth = 4
#pragma HLS resource variable = chanNumSplitStrm core = FIFO_LUTRAM
        hls::stream<double> idfSplitStrm[SN];
#pragma HLS stream variable = idfSplitStrm depth = 64
#pragma HLS resource variable = idfSplitStrm core = FIFO_LUTRAM
        hls::stream<ap_uint<64> > tfAddrSplitStrm[SN];
#pragma HLS stream variable = tfAddrSplitStrm depth = 64
#pragma HLS resource variable = tfAddrSplitStrm core = FIFO_LUTRAM
        splitChan(doc_size, chanStrm, idfStrm, tfAddrStrm, chanNumSplitStrm, idfSplitStrm, tfAddrSplitStrm);

        hls::stream<PT> idTfStrm[CH];
#pragma HLS stream variable = idTfStrm depth = 4 * BL
#pragma HLS resource variable = idTfStrm core = FIFO_URAM
        hls::stream<bool> backPressStrm[CH];
#pragma HLS stream variable = backPressStrm depth = 4
#pragma HLS resource variable = backPressStrm core = FIFO_LUTRAM

        distDataWrapper(doc_size, chanNumSplitStrm, idfSplitStrm, tfAddrSplitStrm, tfValue, tfValue2, tfValue3,
                        tfValue4, idTfStrm, backPressStrm);
#ifndef __SYNTHESIS__
        std::cout << "distDataWrapper End!\n";
#endif
        hls::stream<PT> idTfStrm2[CH];
#pragma HLS stream variable = idTfStrm2 depth = 4
#pragma HLS resource variable = idTfStrm2 core = FIFO_LUTRAM
        internal::mBackPressure<PT, CH, BL>(doc_size, idTfStrm, backPressStrm, idTfStrm2);
#ifndef __SYNTHESIS__
        std::cout << "mBackPressure End!\n";
#endif

        hls::stream<PT> idTfMergeStrm;
#pragma HLS stream variable = idTfMergeStrm depth = 4
#pragma HLS resource variable = idTfMergeStrm core = FIFO_LUTRAM

        internal::mergeTree<PT, CH>::f(doc_size, idTfStrm2, idTfMergeStrm);
#ifndef __SYNTHESIS__
        std::cout << "mergeTree End!\n";
#endif

        hls::stream<uint32_t> idStrm;
#pragma HLS stream variable = idStrm depth = 4
#pragma HLS resource variable = idStrm core = FIFO_LUTRAM
        byValue(doc_size, thldStrm, idTfMergeStrm, idStrm);
#ifndef __SYNTHESIS__
        std::cout << "Weighted Union Module End!\n";
#endif
        // ==== weighted Union Module end ======

        internal::writeStrm2Arr<uint32_t>(doc_size, idStrm, indexId);
    }
};

} // namespace text
} // namespace data_analytics
} // namespace xf
#endif
