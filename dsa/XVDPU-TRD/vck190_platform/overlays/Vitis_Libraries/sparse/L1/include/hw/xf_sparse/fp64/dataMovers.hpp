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
 * @file datamovers.hpp
 * @brief SPARSE Level 1 template function implementation for moving and buffering data
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_DATAMOVERS_HPP
#define XF_SPARSE_DATAMOVERS_HPP

#ifndef __cplusplus
#error "SPARSE Library only works with C++."
#endif

#include <cstdint>
#include "ap_int.h"
#include "hls_stream.h"
#include "xf_blas.hpp"
#ifndef __SYNTHESIS__
#include <iostream>
#endif

using namespace xf::blas;
namespace xf {
namespace sparse {

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits, unsigned int t_MemBits>
void loadNnz(ap_uint<t_MemBits>* p_nnzPtr, hls::stream<ap_uint<t_MemBits> >& p_outNnzStr) {
#ifndef __SYNTHESIS__
    assert(t_MemBits == t_DataBits * t_ParEntries);
#endif
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    ap_uint<t_MemBits>* l_nnzPtr = p_nnzPtr;

    WideType<uint32_t, t_ParamEntries> l_param = l_nnzPtr[0];
#pragma HLS ARRAY_PARTITION variable = l_param complete
    uint32_t l_nnzBlocks = l_param[0];
    l_nnzPtr++;

    for (unsigned int i = 0; i < l_nnzBlocks; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_MemBits> l_val = l_nnzPtr[i];
        p_outNnzStr.write(l_val);
    }
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_IndexBits, unsigned int t_MemBits>
void loadIdx(ap_uint<t_MemBits>* p_memPtr,
             hls::stream<uint32_t> p_paramStr[t_MemChannels],
             hls::stream<ap_uint<t_IndexBits> > p_multCompStr[t_MemChannels]) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    static const unsigned int t_ParamReads = t_MemChannels / t_ParamEntries;
#ifndef __SYNTHESIS__
    assert(t_IndexBits * t_MemChannels == t_MemBits);
    assert(t_MemChannels % t_ParamEntries == 0);
#endif

    ap_uint<t_MemBits>* l_memPtr = p_memPtr;
    WideType<uint32_t, t_MemChannels> l_numEntries;
#pragma HLS ARRAY_PARTITION variable = l_numEntries complete dim = 1

    for (unsigned int i = 0; i < t_ParamReads; ++i) {
#pragma HLS PIPELINE
        WideType<uint32_t, t_ParamEntries> l_param = l_memPtr[i];
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
        for (unsigned int j = 0; j < t_ParamEntries; j++) {
            l_numEntries[i * t_ParamEntries + j] = l_param[j];
            p_paramStr[i * t_ParamEntries + j].write(l_param[j]);
        }
    }
    l_memPtr += 2;

    WideType<uint32_t, t_MemChannels> l_numPaddedEntries;
#pragma HLS ARRAY_PARTITION variable = l_numPaddedEntries complete dim = 1
    for (unsigned int i = 0; i < t_ParamReads; ++i) {
#pragma HLS PIPELINE
        WideType<uint32_t, t_ParamEntries> l_param = l_memPtr[i];
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
        for (unsigned int j = 0; j < t_ParamEntries; j++) {
            l_numPaddedEntries[i * t_ParamEntries + j] = l_param[j];
            p_paramStr[i * t_ParamEntries + j].write(l_param[j]);
        }
    }
    l_memPtr += 2;

    for (unsigned int i = 0; i < l_numPaddedEntries[0]; ++i) {
#pragma HLS PIPELINE
        WideType<t_IndexType, t_MemChannels> l_val = l_memPtr[i];
#pragma HLS ARRAY_PARTITION variable = l_val complete dim = 1
        for (unsigned int j = 0; j < t_MemChannels; ++j) {
            p_multCompStr[j].write(l_val[j]);
        }
    }
}

template <unsigned int t_MemChannels, unsigned int t_IndexBits>
void fwdIdx(hls::stream<uint32_t>& p_paramStr,
            hls::stream<ap_uint<t_IndexBits> >& p_in4MultCompStr,
            hls::stream<ap_uint<t_IndexBits> >& p_out4MultCompStr) {
    uint32_t l_numEntries = p_paramStr.read();
    uint32_t l_numPaddedEntries = p_paramStr.read();
    for (unsigned int i = 0; i < l_numPaddedEntries; ++i) {
        ap_uint<t_IndexBits> l_val = p_in4MultCompStr.read();
        if (i < l_numEntries) {
            p_out4MultCompStr.write(l_val);
        }
    }
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_IndexBits, unsigned int t_MemBits>
void loadColParam(ap_uint<t_MemBits>* p_paramPtr,
                  hls::stream<ap_uint<t_MemBits> >& p_xParamStr,
                  hls::stream<uint32_t> p_compParamStr[t_MemChannels]) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    static const unsigned int t_ParamReads = t_MemChannels / t_ParamEntries;
#ifndef __SYNTHESIS__
    assert(t_MemChannels % t_ParamEntries == 0);
#endif
    WideType<uint32_t, t_ParamEntries> l_param = p_paramPtr[0];
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
    uint32_t l_totalPars = l_param[0];
    p_xParamStr.write(l_param);
    for (unsigned int i = 0; i < t_MemChannels; ++i) {
#pragma HLS UNROLL
        p_compParamStr[i].write(l_totalPars);
    }
    uint32_t l_totalReads = l_totalPars * 8;

    for (unsigned int i = 0; i < l_totalReads; ++i) {
#pragma HLS PIPELINE
        l_param = p_paramPtr[1 + i];
        if (i % 8 < 4) {
            for (unsigned int j = 0; j < t_ParamEntries; ++j) {
                p_compParamStr[(i % 2) * t_ParamEntries + j].write(l_param[j]);
            }
        } else if (i % 8 < 7) {
            p_xParamStr.write(l_param);
        }
    }
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_IndexBits, unsigned int t_MemBits>
void loadParParam(ap_uint<t_MemBits>* p_paramPtr,
                  hls::stream<ap_uint<t_MemBits> >& p_xParamStr,
                  hls::stream<ap_uint<32 * t_MemChannels> >& p_compParamStr) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    static const unsigned int t_ParamReads = t_MemChannels / t_ParamEntries;
#ifndef __SYNTHESIS__
    assert(t_MemChannels % t_ParamEntries == 0);
#endif
    WideType<uint32_t, t_ParamEntries> l_param = p_paramPtr[0];
    WideType<uint32_t, t_MemChannels> l_compParamVal;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_compParamVal complete dim = 1
    uint32_t l_totalPars = l_param[0];
    p_xParamStr.write(l_param);
    uint32_t l_totalReads = l_totalPars * 8;

    for (unsigned int i = 0; i < l_totalReads; ++i) {
#pragma HLS PIPELINE
        l_param = p_paramPtr[i];
        if (i % 8 == 0) {
        } else if (i % 8 < 5) {
            for (unsigned int j = 0; j < t_ParamEntries; ++j) {
                l_compParamVal[(1 - (i % 2)) * t_ParamEntries + j] = l_param[j];
            }
            if (i % 2 == 0) {
                p_compParamStr.write(l_compParamVal);
            }
        } else {
            p_xParamStr.write(l_param);
        }
    }
}

template <unsigned int t_MemBits>
void loadX(ap_uint<t_MemBits>* p_xPtr,
           hls::stream<ap_uint<t_MemBits> >& p_inParamStr,
           hls::stream<ap_uint<t_MemBits> >& p_xStr) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    WideType<uint32_t, t_ParamEntries> l_param;
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
    l_param = p_inParamStr.read();
    uint32_t l_totalPars = l_param[0];

    for (unsigned int i = 0; i < l_totalPars; ++i) {
        l_param = p_inParamStr.read();
        uint32_t l_startAddr = l_param[0];
        uint32_t l_blocks = l_param[1];
        p_xStr.write(l_param);
        p_xStr.write(p_inParamStr.read()); // write offset for all compute modules
        p_xStr.write(p_inParamStr.read()); // write number of blocks for all compute modules
        for (unsigned int j = 0; j < l_blocks; ++j) {
#pragma HLS PIPELINE
            p_xStr.write(p_xPtr[l_startAddr + j]);
        }
    }
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_IndexBits, unsigned int t_MemBits>
void loadCol(ap_uint<t_MemBits>* p_paramPtr,
             ap_uint<t_MemBits>* p_xPtr,
             hls::stream<uint32_t> p_paramStr[t_MemChannels],
             hls::stream<ap_uint<t_MemBits> >& p_xStr) {
    hls::stream<ap_uint<t_MemBits> > l_xParamStr;
#pragma HLS DATAFLOW
    loadColParam<t_IndexType, t_MemChannels, t_IndexBits, t_MemBits>(p_paramPtr, l_xParamStr, p_paramStr);
    loadX<t_MemBits>(p_xPtr, l_xParamStr, p_xStr);
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_IndexBits, unsigned int t_MemBits>
void loadParX(ap_uint<t_MemBits>* p_paramPtr,
              ap_uint<t_MemBits>* p_xPtr,
              hls::stream<ap_uint<32 * t_MemChannels> >& p_paramStr,
              hls::stream<ap_uint<t_MemBits> >& p_xStr) {
    hls::stream<ap_uint<t_MemBits> > l_xParamStr;
#pragma HLS DATAFLOW
    loadParParam<t_IndexType, t_MemChannels, t_IndexBits, t_MemBits>(p_paramPtr, l_xParamStr, p_paramStr);
    loadX<t_MemBits>(p_xPtr, l_xParamStr, p_xStr);
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_MemBits>
void getFwdX(const unsigned int p_chId,
             hls::stream<ap_uint<t_MemBits> >& p_inStr,
             hls::stream<ap_uint<t_MemBits> >& p_fwdStr,
             hls::stream<ap_uint<t_MemBits> >& p_outStr) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    WideType<uint32_t, t_ParamEntries> l_param = p_inStr.read();
    p_fwdStr.write(l_param);
    uint32_t l_size = l_param[1];
    WideType<t_IndexType, t_MemChannels> l_blockParam = p_inStr.read();
    p_fwdStr.write(l_blockParam);
    t_IndexType l_blockAddr = l_blockParam[p_chId];
    l_blockParam = p_inStr.read();
    p_fwdStr.write(l_blockParam);
    t_IndexType l_blockSize = l_blockParam[p_chId];
    t_IndexType l_blocks = 0;
    for (unsigned int i = 0; i < l_size; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_MemBits> l_val = p_inStr.read();
        p_fwdStr.write(l_val);
        if (!(i < l_blockAddr) && (l_blocks < l_blockSize)) {
            p_outStr.write(l_val);
            l_blocks++;
        }
    }
}

template <typename t_IndexType, unsigned int t_MemChannels, unsigned int t_MemBits>
void getX(hls::stream<ap_uint<t_MemBits> >& p_inStr, hls::stream<ap_uint<t_MemBits> >& p_outStr) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    static const unsigned int t_ChId = t_MemChannels - 1;

    WideType<uint32_t, t_ParamEntries> l_param = p_inStr.read();
    uint32_t l_size = l_param[1];
    WideType<t_IndexType, t_MemChannels> l_blockParam = p_inStr.read();
    t_IndexType l_blockAddr = l_blockParam[t_ChId];
    l_blockParam = p_inStr.read();
    t_IndexType l_blockSize = l_blockParam[t_ChId];
    t_IndexType l_blocks = 0;
    for (unsigned int i = 0; i < l_size; ++i) {
#pragma HLS PIPELINE
        ap_uint<t_MemBits> l_val = p_inStr.read();
        if (!(i < l_blockAddr) && (l_blocks < l_blockSize)) {
            p_outStr.write(l_val);
            l_blocks++;
        }
    }
}

template <unsigned int t_MemChannels, unsigned int t_MemBits>
void loadRowParam(ap_uint<t_MemBits>* p_paramPtr,
                  hls::stream<uint32_t>& p_assemParamStr,
                  hls::stream<ap_uint<32 * t_MemChannels> >& p_paramStr) {
    static const unsigned int t_ParamEntries = t_MemBits / 32;
    static const unsigned int t_ParamReads = t_MemChannels / t_ParamEntries;
#ifndef __SYNTHESIS__
    assert(t_MemChannels % t_ParamEntries == 0);
#endif
    WideType<uint32_t, t_ParamEntries> l_param = p_paramPtr[0];
#pragma HLS ARRAY_PARTITION variable = l_param complete dim = 1
    uint32_t l_totalRows = l_param[0];
    uint32_t l_totalRbs = l_param[1];
    p_assemParamStr.write(l_totalRows);
    p_assemParamStr.write(l_totalRbs);
    p_paramStr.write(l_totalRbs);

    WideType<uint32_t, t_MemChannels> l_val;
#pragma HLS ARRAY_PARTITION varible = l_val complete dim = 1

    uint32_t l_totalReads = l_totalRbs * 8;

    for (unsigned int i = 0; i < l_totalReads; ++i) {
#pragma HLS PIPELINE
        l_param = p_paramPtr[i];
        if (i % 8 == 0) {
        } else if (i % 8 == 1) {
            p_assemParamStr.write(l_param[0]); // rbStartId
        } else if (i % 8 == 2) {
            p_assemParamStr.write(l_param[0]); // rbNumRows
            for (unsigned int j = 0; j < t_MemChannels; ++j) {
                l_val[j] = l_param[0];
            }
            p_paramStr.write(l_val);
        } else if (i % 8 < 5) {
            for (unsigned int j = 0; j < t_MemChannels; ++j) {
                ap_uint<32> l_valBit = l_param[j / 2];
                l_val[j] = l_valBit.range(((j % 2) + 1) * 16 - 1, (j % 2) * 16);
            }
            p_paramStr.write(l_val);
        } else if (i % 8 < 7) {
            for (unsigned int j = 0; j < t_ParamEntries; ++j) {
                l_val[(1 - (i % 2)) * t_ParamEntries + j] = l_param[j];
            }
            if (i % 2 == 0) {
                p_paramStr.write(l_val);
            }
        }
    }
}

template <typename t_DataType, unsigned int t_MemChannels, unsigned int t_DataBits>
void accY(hls::stream<ap_uint<t_DataBits> > p_inStr[t_MemChannels], hls::stream<ap_uint<t_DataBits> >& p_outStr) {
    WideType<t_DataType, t_MemChannels> l_val;
#pragma HLS ARRAY_PARTITION variable = l_val complete dim = 1
    for (unsigned int i = 0; i < t_MemChannels; ++i) {
#pragma HLS UNROLL
        l_val[i] = p_inStr[i].read();
    }
    uint32_t l_numRbs = l_val[0];
#ifndef __SYNTHESIS__
    for (unsigned int i = 1; i < t_MemChannels; ++i) {
        assert(l_val[i - 1] == l_val[i]);
    }
#endif
    for (unsigned int r = 0; r < l_numRbs; ++r) {
        for (unsigned int i = 0; i < t_MemChannels; ++i) {
#pragma HLS UNROLL
            l_val[i] = p_inStr[i].read();
        }
        uint32_t l_rows = l_val[0];
#ifndef __SYNTHESIS__
        for (unsigned int i = 1; i < t_MemChannels; ++i) {
            assert(l_val[i - 1] == l_val[i]);
        }
#endif
        for (unsigned int i = 0; i < l_rows; ++i) {
#pragma HLS PIPELINE
            t_DataType l_sum = 0;
            for (unsigned int j = 0; j < t_MemChannels; ++j) {
                ap_uint<t_DataBits> l_valBits = p_inStr[j].read();
                t_DataType l_val = *reinterpret_cast<t_DataType*>(&l_valBits);
                l_sum += l_val;
            }
            ap_uint<t_DataBits> l_sumBits = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_sum);
            p_outStr.write(l_sumBits);
        }
    }
}

template <typename t_DataType, unsigned int t_ParEntries, unsigned int t_DataBits>
void assembleY(hls::stream<uint32_t>& p_paramStr,
               hls::stream<ap_uint<t_DataBits> >& p_inDatStr,
               hls::stream<ap_uint<t_DataBits> >& p_outDatStr) {
    uint32_t l_rowId = 0;
    ap_uint<t_DataBits> l_val;
    t_DataType l_zero = 0;
#pragma HLS ARRAY_PARTITION variable = l_val complete dim = 1
    uint32_t l_totalRows = p_paramStr.read();
    uint32_t l_rowParEntries = (l_totalRows + t_ParEntries - 1) / t_ParEntries;
    uint32_t l_alignedRows = l_rowParEntries * t_ParEntries;
    uint32_t l_totalRbs = p_paramStr.read();
    for (unsigned int i = 0; i < l_totalRbs; ++i) {
        uint32_t l_startId = p_paramStr.read();
        uint32_t l_rowsInRb = p_paramStr.read();
        uint32_t l_rows = l_startId + l_rowsInRb;
        while (l_rowId < l_rows) {
#pragma HLS PIPELINE
            if (!(l_rowId < l_startId)) {
                l_val = p_inDatStr.read();
            } else {
                l_val = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_zero);
            }
            p_outDatStr.write(l_val);
            l_rowId++;
        }
    }
    for (unsigned int i = l_rowId; i < l_alignedRows; ++i) {
#pragma HLS PIPELINE
        l_val = *reinterpret_cast<ap_uint<t_DataBits>*>(&l_zero);
        ;
        p_outDatStr.write(l_val);
    }
}

} // end namespace sparse
} // end namespace xf
#endif
