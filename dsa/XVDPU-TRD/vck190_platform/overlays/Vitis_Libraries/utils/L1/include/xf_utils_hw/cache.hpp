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
#ifndef XF_UTILS_HW_CACHE_H
#define XF_UTILS_HW_CACHE_H

#include <hls_stream.h>
#include <ap_int.h>
#include <stdint.h>

namespace xf {
namespace common {
namespace utils_hw {

/**
 * @brief cache is a URAM design for caching Read-only DDR/HBM memory spaces
 *
 * This function stores history data recently loaded from DDR/HBM in the on-chip memory(URAM).
 * It aims to reduce DDR/HBM access when the memory is accessed randomly.
 *
 * @tparam T 			    The type of the actual data accessed. Float and double is not supported.
 * @tparam ramRow			The number of rows each on chip ram has
 * @tparam groupramPart		The number of on chip ram used in cache
 * @tparam dataOneLine		The number of actual data each 512 can contain
 * @tparam addrWidth		The width of the address to access the memory
 * @tparam validRamType     The ram type of the valid flag array. 0 for LUTRAM, 1 for BRAM, 2 for URAM
 * @tparam addrRamType      The ram type of the onchip addr array. 0 for LUTRAM, 1 for BRAM, 2 for URAM
 * @tparam dataRamType      The ram type of the onchip data array. 0 for LUTRAM, 1 for BRAM, 2 for URAM
 */

template <typename T,
          int ramRow,
          int groupRamPart,
          int dataOneLine,
          int addrWidth,
          int validRamType,
          int addrRamType,
          int dataRamType>
class cache {
   public:
    cache() {
#pragma HLS inline
#pragma HLS array_partition variable = valid block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipRam1 block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipRam0 block factor = 1 dim = 2
#pragma HLS array_partition variable = onChipAddr block factor = 1 dim = 2

        if (validRamType == 0) {
#pragma HLS resource variable = valid core = RAM_S2P_LUTRAM
        } else if (validRamType == 2) {
#pragma HLS resource variable = valid core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = valid core = RAM_S2P_URAM
        }

        if (addrRamType == 0) {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_LUTRAM
        } else if (addrRamType == 2) {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = onChipAddr core = RAM_S2P_URAM
        }

        if (dataRamType == 0) {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_LUTRAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_LUTRAM
        } else if (dataRamType == 2) {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_BRAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_BRAM
        } else {
#pragma HLS resource variable = onChipRam0 core = RAM_S2P_URAM
#pragma HLS resource variable = onChipRam1 core = RAM_S2P_URAM
        }

#ifndef __SYNTHESIS__
        valid = new ap_uint<dataOneLine>*[ramRow];
        onChipRam1 = new ap_uint<512>*[ramRow];
        onChipRam0 = new ap_uint<512>*[ramRow];
        onChipAddr = new ap_uint<addrWidth>*[ramRow];
        for (int i = 0; i < ramRow; ++i) {
            valid[i] = new ap_uint<dataOneLine>[ groupRamPart ];
            onChipRam0[i] = new ap_uint<512>[ groupRamPart ];
            onChipRam1[i] = new ap_uint<512>[ groupRamPart ];
            onChipAddr[i] = new ap_uint<addrWidth>[ groupRamPart ];
        }
#endif
    }

    ~cache() {
#ifndef __SYNTHESIS__
        for (int i = 0; i < ramRow; ++i) {
            delete[] valid[i];
            delete[] onChipRam0[i];
            delete[] onChipRam1[i];
            delete[] onChipAddr[i];
        }
        delete[] onChipAddr;
        delete[] onChipRam0;
        delete[] onChipRam1;
        delete[] valid;
#endif
    }

    /// @brief  Initialization of the on chip memory when controlling single off chip memory.
    void initSingleOffChip() {
    Loop_init_uram:
        for (int j = 0; j < ramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
            for (int i = 0; i < groupRamPart; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
                valid[j][i] = 0;
                onChipAddr[j][i] = 0;
                onChipRam0[j][i] = 0;
            }
        }
    }

    /// @brief  Initialization of the on chip memory when controlling dual off chip memory.
    void initDualOffChip() {
    Loop_init_uram:
        for (int j = 0; j < ramRow; ++j) {
#pragma HLS loop_tripcount min = 4096 avg = 4096 max = 4096
            for (int i = 0; i < groupRamPart; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
                valid[j][i] = 0;
                onChipAddr[j][i] = 0;
                onChipRam0[j][i] = 0;
                onChipRam1[j][i] = 0;
            }
        }
    }

    /// @brief readOnly function with end flags.
    /// @param ddrMem 		The pointer for the off chip memory
    /// @param addrStrm		The read address should be sent from this stream
    /// @param e_addrStrm	The end flag for the addrStrm
    /// @param dataStrm		The data loaded from off chip memory
    /// @param e_dataStrm	The end flag for the dataStrm
    void readOnly(ap_uint<512>* ddrMem,
                  hls::stream<ap_uint<32> >& addrStrm,
                  hls::stream<bool>& e_addrStrm,
                  hls::stream<T>& dataStrm,
                  hls::stream<bool>& e_dataStrm) {
#pragma HLS inline off

        ap_uint<dataOneLine> validCnt;
        for (int i = 0; i < dataOneLine; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
            validCnt[i] = 1;
        }
        const int size = sizeof(T) * 8;

        ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
        ap_uint<512> pingQue[4] = {0, 0, 0, 0};
        ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
        int validAddrQue[4] = {-1, -1, -1, -1};
        int addrAddrQue[4] = {-1, -1, -1, -1};
        int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

        bool e = e_addrStrm.read();

        while (!e) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipRam0 inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false

            e = e_addrStrm.read();
            int index = addrStrm.read();
            int k00 = index % dataOneLine;
            int k01 = index / dataOneLine;
            int k10 = k01 % groupRamPart;
            int k11 = k01 / groupRamPart;
            int k20 = k11 % ramRow;
            int k21 = k11 / ramRow;
            int k30 = k21;

            ap_uint<dataOneLine> validTmp;
            if (k01 == validAddrQue[0]) {
                validTmp = validQue[0];
            } else if (k01 == validAddrQue[1]) {
                validTmp = validQue[1];
            } else if (k01 == validAddrQue[2]) {
                validTmp = validQue[2];
            } else if (k01 == validAddrQue[3]) {
                validTmp = validQue[3];
            } else {
                validTmp = valid[k20][k10];
            }
            ap_uint<1> validBool = validTmp[k00]; // to check
            ap_uint<512> tmpV;
            ap_uint<addrWidth> address;
            ap_uint<addrWidth> addrTmp;
            if (k01 == addrAddrQue[0]) {
                addrTmp = addrQue[0];
            } else if (k01 == addrAddrQue[1]) {
                addrTmp = addrQue[1];
            } else if (k01 == addrAddrQue[2]) {
                addrTmp = addrQue[2];
            } else if (k01 == addrAddrQue[3]) {
                addrTmp = addrQue[3];
            } else {
                addrTmp = onChipAddr[k20][k10];
            }
            address = addrTmp;
            if ((validBool == 1) && (address == k30)) {
                if (k01 == ramAddrQue[0]) {
                    tmpV = pingQue[0];
                } else if (k01 == ramAddrQue[1]) {
                    tmpV = pingQue[1];
                } else if (k01 == ramAddrQue[2]) {
                    tmpV = pingQue[2];
                } else if (k01 == ramAddrQue[3]) {
                    tmpV = pingQue[3];
                } else {
                    tmpV = onChipRam0[k20][k10];
                }
            } else {
                tmpV = ddrMem[k01];
                onChipRam0[k20][k10] = tmpV;
                pingQue[3] = pingQue[2];
                pingQue[2] = pingQue[1];
                pingQue[1] = pingQue[0];
                pingQue[0] = tmpV;
                ramAddrQue[3] = ramAddrQue[2];
                ramAddrQue[2] = ramAddrQue[1];
                ramAddrQue[1] = ramAddrQue[0];
                ramAddrQue[0] = k01;
                ap_uint<addrWidth> tAddr = (ap_uint<addrWidth>)k30;
                addrTmp = tAddr;
                onChipAddr[k20][k10] = addrTmp;
                addrQue[3] = addrQue[2];
                addrQue[2] = addrQue[1];
                addrQue[1] = addrQue[0];
                addrQue[0] = addrTmp;
                addrAddrQue[3] = addrAddrQue[2];
                addrAddrQue[2] = addrAddrQue[1];
                addrAddrQue[1] = addrAddrQue[0];
                addrAddrQue[0] = k01;
            }
            if (validBool == 0) {
                validTmp = validCnt;
                valid[k20][k10] = validTmp;
                validQue[3] = validQue[2];
                validQue[2] = validQue[1];
                validQue[1] = validQue[0];
                validQue[0] = validTmp;
                validAddrQue[3] = validAddrQue[2];
                validAddrQue[2] = validAddrQue[1];
                validAddrQue[1] = validAddrQue[0];
                validAddrQue[0] = k01;
            }
            dataStrm.write(tmpV.range(size * (k00 + 1) - 1, size * k00));
            e_dataStrm.write(0);
        }
        e_dataStrm.write(1);
    }

    /// @brief readOnly function without end flags.
    /// @param cnt			The number of access to the memory
    /// @param ddrMem 		The pointer for the off chip memory
    /// @param addrStrm		The read address should be sent from this stream
    /// @param dataStrm		The data loaded from off chip memory
    void readOnly(int cnt, ap_uint<512>* ddrMem, hls::stream<ap_uint<32> >& addrStrm, hls::stream<T>& dataStrm) {
#pragma HLS inline off

        ap_uint<dataOneLine> validCnt;
        for (int i = 0; i < dataOneLine; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
            validCnt[i] = 1;
        }
        const int size = sizeof(T) * 8;

        ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
        ap_uint<512> pingQue[4] = {0, 0, 0, 0};
        ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
        int validAddrQue[4] = {-1, -1, -1, -1};
        int addrAddrQue[4] = {-1, -1, -1, -1};
        int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

        for (int i = 0; i < cnt; i++) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipRam0 inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false

            int index = addrStrm.read();
            int k00 = index % dataOneLine;
            int k01 = index / dataOneLine;
            int k10 = k01 % groupRamPart;
            int k11 = k01 / groupRamPart;
            int k20 = k11 % ramRow;
            int k21 = k11 / ramRow;
            int k30 = k21;

            ap_uint<dataOneLine> validTmp;
            if (k01 == validAddrQue[0]) {
                validTmp = validQue[0];
            } else if (k01 == validAddrQue[1]) {
                validTmp = validQue[1];
            } else if (k01 == validAddrQue[2]) {
                validTmp = validQue[2];
            } else if (k01 == validAddrQue[3]) {
                validTmp = validQue[3];
            } else {
                validTmp = valid[k20][k10];
            }
            ap_uint<1> validBool = validTmp[k00]; // to check
            ap_uint<512> tmpV;
            ap_uint<addrWidth> address;
            ap_uint<addrWidth> addrTmp;
            if (k01 == addrAddrQue[0]) {
                addrTmp = addrQue[0];
            } else if (k01 == addrAddrQue[1]) {
                addrTmp = addrQue[1];
            } else if (k01 == addrAddrQue[2]) {
                addrTmp = addrQue[2];
            } else if (k01 == addrAddrQue[3]) {
                addrTmp = addrQue[3];
            } else {
                addrTmp = onChipAddr[k20][k10];
            }
            address = addrTmp;
            if ((validBool == 1) && (address == k30)) {
                if (k01 == ramAddrQue[0]) {
                    tmpV = pingQue[0];
                } else if (k01 == ramAddrQue[1]) {
                    tmpV = pingQue[1];
                } else if (k01 == ramAddrQue[2]) {
                    tmpV = pingQue[2];
                } else if (k01 == ramAddrQue[3]) {
                    tmpV = pingQue[3];
                } else {
                    tmpV = onChipRam0[k20][k10];
                }
            } else {
                tmpV = ddrMem[k01];
                onChipRam0[k20][k10] = tmpV;
                pingQue[3] = pingQue[2];
                pingQue[2] = pingQue[1];
                pingQue[1] = pingQue[0];
                pingQue[0] = tmpV;
                ramAddrQue[3] = ramAddrQue[2];
                ramAddrQue[2] = ramAddrQue[1];
                ramAddrQue[1] = ramAddrQue[0];
                ramAddrQue[0] = k01;
                ap_uint<addrWidth> tAddr = (ap_uint<addrWidth>)k30;
                addrTmp = tAddr;
                onChipAddr[k20][k10] = addrTmp;
                addrQue[3] = addrQue[2];
                addrQue[2] = addrQue[1];
                addrQue[1] = addrQue[0];
                addrQue[0] = addrTmp;
                addrAddrQue[3] = addrAddrQue[2];
                addrAddrQue[2] = addrAddrQue[1];
                addrAddrQue[1] = addrAddrQue[0];
                addrAddrQue[0] = k01;
            }
            if (validBool == 0) {
                validTmp = validCnt;
                valid[k20][k10] = validTmp;
                validQue[3] = validQue[2];
                validQue[2] = validQue[1];
                validQue[1] = validQue[0];
                validQue[0] = validTmp;
                validAddrQue[3] = validAddrQue[2];
                validAddrQue[2] = validAddrQue[1];
                validAddrQue[1] = validAddrQue[0];
                validAddrQue[0] = k01;
            }
            dataStrm.write(tmpV.range(size * (k00 + 1) - 1, size * k00));
        }
    }

    /// @brief readOnly function that index two off-chip buffers without end flags. Both of the buffers should be
    /// indexed in exactly the same behaviour.
    /// @param cnt			The number of access to the memory
    /// @param ddrMem0		The pointer for the first off chip memory
    /// @param ddrMem1		The pointer for the second off chip memory
    /// @param addrStrm		The read address should be sent from this stream
    /// @param data0Strm	The data loaded from the first off chip memory
    /// @param data1Strm	The data loaded from the second off chip memory
    void readOnly(int cnt,
                  ap_uint<512>* ddrMem0,
                  ap_uint<512>* ddrMem1,
                  hls::stream<ap_uint<32> >& addrStrm,
                  hls::stream<T>& data0Strm,
                  hls::stream<T>& data1Strm) {
#pragma HLS inline off

        ap_uint<dataOneLine> validCnt;
        for (int i = 0; i < dataOneLine; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
            validCnt[i] = 1;
        }
        const int size = sizeof(T) * 8;

        ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
        ap_uint<512> pingQue[4] = {0, 0, 0, 0};
        ap_uint<512> cntQue[4] = {0, 0, 0, 0};
        ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
        int validAddrQue[4] = {-1, -1, -1, -1};
        int addrAddrQue[4] = {-1, -1, -1, -1};
        int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = cntQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

        for (int i = 0; i < cnt; ++i) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipRam0 inter false
#pragma HLS DEPENDENCE variable = onChipRam1 inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false
            int index = addrStrm.read();
            int k00 = index % dataOneLine;
            int k01 = index / dataOneLine;
            int k10 = k01 % groupRamPart;
            int k11 = k01 / groupRamPart;
            int k20 = k11 % ramRow;
            int k21 = k11 / ramRow;
            int k30 = k21;

            ap_uint<dataOneLine> validTmp;
            if (k01 == validAddrQue[0]) {
                validTmp = validQue[0];
            } else if (k01 == validAddrQue[1]) {
                validTmp = validQue[1];
            } else if (k01 == validAddrQue[2]) {
                validTmp = validQue[2];
            } else if (k01 == validAddrQue[3]) {
                validTmp = validQue[3];
            } else {
                validTmp = valid[k20][k10];
            }
            ap_uint<1> validBool = validTmp[k00]; // to check
            ap_uint<512> tmpV, tmpC;
            ap_uint<addrWidth> address;
            ap_uint<addrWidth> addrTmp;
            if (k01 == addrAddrQue[0]) {
                addrTmp = addrQue[0];
            } else if (k01 == addrAddrQue[1]) {
                addrTmp = addrQue[1];
            } else if (k01 == addrAddrQue[2]) {
                addrTmp = addrQue[2];
            } else if (k01 == addrAddrQue[3]) {
                addrTmp = addrQue[3];
            } else {
                addrTmp = onChipAddr[k20][k10];
            }
            address = addrTmp;
            if ((validBool == 1) && (address == k30)) {
                if (k01 == ramAddrQue[0]) {
                    tmpV = pingQue[0];
                    tmpC = cntQue[0];
                } else if (k01 == ramAddrQue[1]) {
                    tmpV = pingQue[1];
                    tmpC = cntQue[1];
                } else if (k01 == ramAddrQue[2]) {
                    tmpV = pingQue[2];
                    tmpC = cntQue[2];
                } else if (k01 == ramAddrQue[3]) {
                    tmpV = pingQue[3];
                    tmpC = cntQue[3];
                } else {
                    tmpV = onChipRam0[k20][k10];
                    tmpC = onChipRam1[k20][k10];
                }
            } else {
                tmpV = ddrMem0[k01];
                tmpC = ddrMem1[k01];
                onChipRam0[k20][k10] = tmpV;
                onChipRam1[k20][k10] = tmpC;
                pingQue[3] = pingQue[2];
                pingQue[2] = pingQue[1];
                pingQue[1] = pingQue[0];
                pingQue[0] = tmpV;
                cntQue[3] = cntQue[2];
                cntQue[2] = cntQue[1];
                cntQue[1] = cntQue[0];
                cntQue[0] = tmpC;
                ramAddrQue[3] = ramAddrQue[2];
                ramAddrQue[2] = ramAddrQue[1];
                ramAddrQue[1] = ramAddrQue[0];
                ramAddrQue[0] = k01;
                ap_uint<addrWidth> tAddr = (ap_uint<addrWidth>)k30;
                addrTmp = tAddr;
                onChipAddr[k20][k10] = addrTmp;
                addrQue[3] = addrQue[2];
                addrQue[2] = addrQue[1];
                addrQue[1] = addrQue[0];
                addrQue[0] = addrTmp;
                addrAddrQue[3] = addrAddrQue[2];
                addrAddrQue[2] = addrAddrQue[1];
                addrAddrQue[1] = addrAddrQue[0];
                addrAddrQue[0] = k01;
            }
            if (validBool == 0) {
                validTmp = validCnt;
                valid[k20][k10] = validTmp;
                validQue[3] = validQue[2];
                validQue[2] = validQue[1];
                validQue[1] = validQue[0];
                validQue[0] = validTmp;
                validAddrQue[3] = validAddrQue[2];
                validAddrQue[2] = validAddrQue[1];
                validAddrQue[1] = validAddrQue[0];
                validAddrQue[0] = k01;
            }
            data0Strm.write(tmpV.range(size * (k00 + 1) - 1, size * k00));
            data1Strm.write(tmpC.range(size * (k00 + 1) - 1, size * k00));
        }
    }

    /// @brief readOnly function that index two off-chip buffers without end flags. Both of the buffers should be
    /// indexed in exactly the same behaviour.
    /// @param ddrMem0		The pointer for the first off chip memory
    /// @param ddrMem1		The pointer for the second off chip memory
    /// @param addrStrm		The read address should be sent from this stream
    /// @param e_addrStrm   The end flag for the addrStrm
    /// @param data0Strm	The data loaded from the first off chip memory
    /// @param e_data0Strm  The end flag for the data0Strm
    /// @param data1Strm	The data loaded from the second off chip memory
    /// @param e_data1Strm  The end flag for the data1Strm
    void readOnly(ap_uint<512>* ddrMem0,
                  ap_uint<512>* ddrMem1,
                  hls::stream<ap_uint<32> >& addrStrm,
                  hls::stream<bool>& e_addrStrm,
                  hls::stream<T>& data0Strm,
                  hls::stream<bool>& e_data0Strm,
                  hls::stream<T>& data1Strm,
                  hls::stream<bool>& e_data1Strm

                  ) {
#pragma HLS inline off

        ap_uint<dataOneLine> validCnt;
        for (int i = 0; i < dataOneLine; ++i) {
#pragma HLS loop_tripcount min = 8 avg = 8 max = 8
#pragma HLS pipeline II = 1
            validCnt[i] = 1;
        }
        const int size = sizeof(T) * 8;

        ap_uint<addrWidth + 1> addrQue[4] = {-1, -1, -1, -1};
        ap_uint<512> pingQue[4] = {0, 0, 0, 0};
        ap_uint<512> cntQue[4] = {0, 0, 0, 0};
        ap_uint<dataOneLine> validQue[4] = {0, 0, 0, 0};
        int validAddrQue[4] = {-1, -1, -1, -1};
        int addrAddrQue[4] = {-1, -1, -1, -1};
        int ramAddrQue[4] = {-1, -1, -1, -1};
#pragma HLS array_partition variable = validQue complete dim = 1
#pragma HLS array_partition variable = validAddrQue complete dim = 1
#pragma HLS array_partition variable = cntQue complete dim = 1
#pragma HLS array_partition variable = ramAddrQue complete dim = 1
#pragma HLS array_partition variable = pingQue complete dim = 1
#pragma HLS array_partition variable = addrQue complete dim = 1
#pragma HLS array_partition variable = addrAddrQue complete dim = 1

        bool e = e_addrStrm.read();

        while (!e) {
#pragma HLS loop_tripcount min = 16500000 avg = 16500000 max = 16500000
#pragma HLS pipeline II = 1
#pragma HLS DEPENDENCE variable = valid inter false
#pragma HLS DEPENDENCE variable = onChipRam0 inter false
#pragma HLS DEPENDENCE variable = onChipRam1 inter false
#pragma HLS DEPENDENCE variable = onChipAddr inter false

            e = e_addrStrm.read();

            int index = addrStrm.read();
            int k00 = index % dataOneLine;
            int k01 = index / dataOneLine;
            int k10 = k01 % groupRamPart;
            int k11 = k01 / groupRamPart;
            int k20 = k11 % ramRow;
            int k21 = k11 / ramRow;
            int k30 = k21;

            ap_uint<dataOneLine> validTmp;
            if (k01 == validAddrQue[0]) {
                validTmp = validQue[0];
            } else if (k01 == validAddrQue[1]) {
                validTmp = validQue[1];
            } else if (k01 == validAddrQue[2]) {
                validTmp = validQue[2];
            } else if (k01 == validAddrQue[3]) {
                validTmp = validQue[3];
            } else {
                validTmp = valid[k20][k10];
            }
            ap_uint<1> validBool = validTmp[k00]; // to check
            ap_uint<512> tmpV, tmpC;
            ap_uint<addrWidth> address;
            ap_uint<addrWidth> addrTmp;
            if (k01 == addrAddrQue[0]) {
                addrTmp = addrQue[0];
            } else if (k01 == addrAddrQue[1]) {
                addrTmp = addrQue[1];
            } else if (k01 == addrAddrQue[2]) {
                addrTmp = addrQue[2];
            } else if (k01 == addrAddrQue[3]) {
                addrTmp = addrQue[3];
            } else {
                addrTmp = onChipAddr[k20][k10];
            }
            address = addrTmp;
            if ((validBool == 1) && (address == k30)) {
                if (k01 == ramAddrQue[0]) {
                    tmpV = pingQue[0];
                    tmpC = cntQue[0];
                } else if (k01 == ramAddrQue[1]) {
                    tmpV = pingQue[1];
                    tmpC = cntQue[1];
                } else if (k01 == ramAddrQue[2]) {
                    tmpV = pingQue[2];
                    tmpC = cntQue[2];
                } else if (k01 == ramAddrQue[3]) {
                    tmpV = pingQue[3];
                    tmpC = cntQue[3];
                } else {
                    tmpV = onChipRam0[k20][k10];
                    tmpC = onChipRam1[k20][k10];
                }
            } else {
                tmpV = ddrMem0[k01];
                tmpC = ddrMem1[k01];
                onChipRam0[k20][k10] = tmpV;
                onChipRam1[k20][k10] = tmpC;
                pingQue[3] = pingQue[2];
                pingQue[2] = pingQue[1];
                pingQue[1] = pingQue[0];
                pingQue[0] = tmpV;
                cntQue[3] = cntQue[2];
                cntQue[2] = cntQue[1];
                cntQue[1] = cntQue[0];
                cntQue[0] = tmpC;
                ramAddrQue[3] = ramAddrQue[2];
                ramAddrQue[2] = ramAddrQue[1];
                ramAddrQue[1] = ramAddrQue[0];
                ramAddrQue[0] = k01;
                ap_uint<addrWidth> tAddr = (ap_uint<addrWidth>)k30;
                addrTmp = tAddr;
                onChipAddr[k20][k10] = addrTmp;
                addrQue[3] = addrQue[2];
                addrQue[2] = addrQue[1];
                addrQue[1] = addrQue[0];
                addrQue[0] = addrTmp;
                addrAddrQue[3] = addrAddrQue[2];
                addrAddrQue[2] = addrAddrQue[1];
                addrAddrQue[1] = addrAddrQue[0];
                addrAddrQue[0] = k01;
            }
            if (validBool == 0) {
                validTmp = validCnt;
                valid[k20][k10] = validTmp;
                validQue[3] = validQue[2];
                validQue[2] = validQue[1];
                validQue[1] = validQue[0];
                validQue[0] = validTmp;
                validAddrQue[3] = validAddrQue[2];
                validAddrQue[2] = validAddrQue[1];
                validAddrQue[1] = validAddrQue[0];
                validAddrQue[0] = k01;
            }
            data0Strm.write(tmpV.range(size * (k00 + 1) - 1, size * k00));
            e_data0Strm.write(0);
            data1Strm.write(tmpC.range(size * (k00 + 1) - 1, size * k00));
            e_data1Strm.write(0);
        }
        e_data0Strm.write(1);
        e_data1Strm.write(1);
    }

   private:
#ifndef __SYNTHESIS__
    ap_uint<dataOneLine>** valid;
    ap_uint<512>** onChipRam0;
    ap_uint<512>** onChipRam1;
    ap_uint<addrWidth>** onChipAddr;
#else
    ap_uint<dataOneLine> valid[ramRow][groupRamPart];
    ap_uint<512> onChipRam0[ramRow][groupRamPart];
    ap_uint<512> onChipRam1[ramRow][groupRamPart];
    ap_uint<addrWidth> onChipAddr[ramRow][groupRamPart];
#endif

}; // cache class

} // namespace utils_hw
} // namespace common
} // namespace xf
#endif //#ifndef XF_UTILS_HW_CACHE_H
