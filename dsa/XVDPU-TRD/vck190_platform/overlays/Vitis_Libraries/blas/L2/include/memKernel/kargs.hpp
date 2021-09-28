/**********
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
 * **********/
/**
 *  @brief Kernet argrument handling
 *
 *  $DateTime: 2018/02/13 14:44:57 $
 */

#ifndef XF_BLAS_KARGS_HPP
#define XF_BLAS_KARGS_HPP

#include "assert.h"
#include "hls_stream.h"
#include "utils/x_hls_utils.h"
#include "types.hpp"
#include <ap_fixed.h>
#include <stdio.h>
#include <vector>

namespace xf {

namespace blas {

typedef unsigned long int TimeBaseType;

//////////////////////////// TIMESTAMP ////////////////////////////
/*
 * TimeStamp Class provides a function runTs which can be used to
 * calculate the number cycles for which runTs functions is running
 * It is controlled by template parameter that is provided with number
 * of instruction the connected engine will execute.
 * Assumption :: The engine is connected to runTs through a FIFO and
 * reads an element from FIFO per execution of one instruction.
 * */
template <unsigned int t_NumInstr>
class TimeStamp {
   public:
    // typedef enum {OpQuit, OpStamp, OpNoop} OpType;
    typedef TimeBaseType TimeType;
    //#if BLAS_part==vu9pf1
    // static const unsigned int t_FifoDepth = 2;
    //#else
    static const unsigned int t_FifoDepth = 1;
    //#endif
   private:
    TimeType m_Time;

   public:
    void runTs(
        // hls::stream<OpType> &p_Control,
        hls::stream<TimeType>& p_Time) {
#pragma HLS INLINE self off
        // OpType l_op = OpNoop;
        m_Time = 0;
        unsigned int l_count = 0;
        while (l_count < t_NumInstr + t_FifoDepth) {
            if (p_Time.write_nb(m_Time)) {
                l_count++;
            }
            m_Time++;
        }
        assert(l_count == t_NumInstr + t_FifoDepth);
    }
};

//////////////////////////// INSTRUCTION RESULTS ////////////////////////////
/*
 * InstrResArgs class provides a simple function for calculating a time interval
 */
class InstrResArgs {
   public:
    TimeBaseType m_StartTime, m_EndTime;

   public:
    InstrResArgs() {}
    InstrResArgs(TimeBaseType p_StartTime, TimeBaseType p_EndTime) : m_StartTime(p_StartTime), m_EndTime(p_EndTime) {
        assert(m_StartTime <= m_EndTime);
    }
    TimeBaseType getDuration() { return m_EndTime - m_StartTime; }
};

//////////////////////////// CONTROL ////////////////////////////
class ControlArgs {
   public:
    bool m_IsLastOp, m_Noop;

   public:
    ControlArgs() {}
    ControlArgs(bool p_IsLastOp, bool p_Noop) : m_IsLastOp(p_IsLastOp), m_Noop(p_Noop) { assert(m_IsLastOp || m_Noop); }
    bool getIsLastOp() { return m_IsLastOp; }
    bool getNoop() { return m_Noop; }
};

//////////////////////////// GEMV ////////////////////////////
/*
 * Simple container class to hold GEMV parameters :
 *  m_Aoffset, m_Boffset, m_Coffset :: Are matrix address offsets
 *  m_M, m_K, m_Lda : Define the size of matrices
 *  A is a m_M x m_K matrix and B and C are vectors
 *  GEMV operation is defined as  : C = A*B +  C ;
 */
class GemvArgs {
   public:
    unsigned int m_Aoffset, m_Boffset, m_Coffset, m_M, m_K, m_Lda;

   public:
    GemvArgs() {}
    GemvArgs(unsigned int p_Aoffset,
             unsigned int p_Boffset,
             unsigned int p_Coffset,
             unsigned int p_M,
             unsigned int p_K,
             unsigned int p_Lda)
        : m_Aoffset(p_Aoffset), m_Boffset(p_Boffset), m_Coffset(p_Coffset), m_M(p_M), m_K(p_K), m_Lda(p_Lda) {}
};

//////////////////////////// GEMM ////////////////////////////
/*
 * Simple container class to hold GEMM parameters :
 *  m_Aoffset, m_Boffset, m_Coffset :: Are matrix address offsets
 *  m_M, m_K, m_Lda : Define the size of matrices
 *  A  and B are of size a m_M x m_K matrix and X and C are : m_M x m_N
 *  GEMM operation is defined as  : C = A*B+X ;
 */
class GemmArgs {
   public:
    unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N, m_Lda, m_Ldb, m_Ldc, m_Ldx;
    int32_t m_postScale;

   public:
    GemmArgs() {}
    GemmArgs(unsigned int p_Aoffset,
             unsigned int p_Boffset,
             unsigned int p_Coffset,
             unsigned int p_Xoffset,
             unsigned int p_M,
             unsigned int p_K,
             unsigned int p_N,
             unsigned int p_Lda,
             unsigned int p_Ldb,
             unsigned int p_Ldc,
             unsigned int p_Ldx,
             int32_t p_postScale)
        : m_Aoffset(p_Aoffset),
          m_Boffset(p_Boffset),
          m_Coffset(p_Coffset),
          m_Xoffset(p_Xoffset),
          m_M(p_M),
          m_K(p_K),
          m_N(p_N),
          m_Lda(p_Lda),
          m_Ldb(p_Ldb),
          m_Ldc(p_Ldc),
          m_Ldx(p_Ldx),
          m_postScale(p_postScale) {}
    void init(unsigned int p_Aoffset,
              unsigned int p_Boffset,
              unsigned int p_Coffset,
              unsigned int p_Xoffset,
              unsigned int p_M,
              unsigned int p_K,
              unsigned int p_N,
              unsigned int p_Lda,
              unsigned int p_Ldb,
              unsigned int p_Ldc,
              unsigned int p_Ldx,
              int32_t p_postScale) {
        m_Aoffset = p_Aoffset;
        m_Boffset = p_Boffset;
        m_Coffset = p_Coffset;
        m_Xoffset = p_Xoffset;
        m_M = p_M;
        m_K = p_K;
        m_N = p_N;
        m_Lda = p_Lda;
        m_Ldb = p_Ldb;
        m_Ldc = p_Ldc;
        m_Ldx = p_Ldx;
        m_postScale = p_postScale;
    }
};

////////////////////////////FCN////////////////////////////
/*
 * Simple container class to hold FCN parameters :
 *  m_Aoffset, m_Boffset, m_Coffset,m_Xoffset :: Are matrix address offsets
 *  m_M, m_K, m_Lda : Define the size of matrices
 *  A  and B are of size a m_M x m_K matrix and X and C are : m_M x m_N
 *  FCN operation is defined as  : C = A*B+X   plus scaling and ReLU
 */
class FcnArgs {
   public:
    unsigned int m_Aoffset, m_Boffset, m_Coffset, m_Xoffset, m_M, m_K, m_N, m_Lda, m_Ldb, m_Ldc, m_Ldx;
    int32_t m_postScale;
    int16_t m_PReluVal;

   public:
    FcnArgs() {}
    FcnArgs(unsigned int p_Aoffset,
            unsigned int p_Boffset,
            unsigned int p_Coffset,
            unsigned int p_Xoffset,
            unsigned int p_M,
            unsigned int p_K,
            unsigned int p_N,
            unsigned int p_Lda,
            unsigned int p_Ldb,
            unsigned int p_Ldc,
            unsigned int p_Ldx,
            int32_t p_postScale,
            int16_t p_PReluVal)
        : m_Aoffset(p_Aoffset),
          m_Boffset(p_Boffset),
          m_Coffset(p_Coffset),
          m_Xoffset(p_Xoffset),
          m_M(p_M),
          m_K(p_K),
          m_N(p_N),
          m_Lda(p_Lda),
          m_Ldb(p_Ldb),
          m_Ldc(p_Ldc),
          m_Ldx(p_Ldx),
          m_postScale(p_postScale),
          m_PReluVal(p_PReluVal) {}
    void init(unsigned int p_Aoffset,
              unsigned int p_Boffset,
              unsigned int p_Coffset,
              unsigned int p_Xoffset,
              unsigned int p_M,
              unsigned int p_K,
              unsigned int p_N,
              unsigned int p_Lda,
              unsigned int p_Ldb,
              unsigned int p_Ldc,
              unsigned int p_Ldx,
              int32_t p_postScale,
              int16_t p_PReluVal) {
        m_Aoffset = p_Aoffset;
        m_Boffset = p_Boffset;
        m_Coffset = p_Coffset;
        m_Xoffset = p_Xoffset;
        m_M = p_M;
        m_K = p_K;
        m_N = p_N;
        m_Lda = p_Lda;
        m_Ldb = p_Ldb;
        m_Ldc = p_Ldc;
        m_Ldx = p_Ldx;
        m_postScale = p_postScale;
        m_PReluVal = p_PReluVal;
    }
};

//////////////////////////// DDR Transposer ////////////////////////////
class DdrMatrixShape {
   public:
    typedef enum { Unknown, Rm, Cm, GvA, GmA, GmB } FormatType;

   public:
    unsigned int m_Offset, m_Rows, m_Cols, m_Ld, m_GfEdge;
    FormatType m_Format;

   public:
    DdrMatrixShape() {}
    DdrMatrixShape(unsigned int p_Offset,
                   unsigned int p_Rows,
                   unsigned int p_Cols,
                   unsigned int p_Ld,
                   unsigned int p_GfEdge,
                   FormatType p_Format)
        : m_Offset(p_Offset), m_Rows(p_Rows), m_Cols(p_Cols), m_Ld(p_Ld), m_GfEdge(p_GfEdge), m_Format(p_Format) {}
    static FormatType string2format(std::string p_Format) {
        std::vector<std::string> l_formatNames = {"unknown", "rm", "cm", "gva", "gma", "gmb"};
        FormatType l_format = FormatType::Unknown;
        for (unsigned int i = 0; i < l_formatNames.size(); ++i) {
            if (l_formatNames[i] == p_Format) {
                l_format = (FormatType)i;
                break;
            }
        }
        return (l_format);
    }
    void print(std::ostream& os) {
        os << "Rows x Cols  " << m_Rows << "x" << m_Cols << " Ld=" << m_Ld << " Format=" << m_Format
           << " Page=" << m_Offset;
    }
};
inline std::ostream& operator<<(std::ostream& os, DdrMatrixShape& p_Val) {
    p_Val.print(os);
    return (os);
}

////////////////////////////TRANSP////////////////////////////
class TranspArgs {
   public:
    DdrMatrixShape m_Src, m_Dst;

   public:
    TranspArgs() {}
    TranspArgs(DdrMatrixShape p_Src, DdrMatrixShape p_Dst) : m_Src(p_Src), m_Dst(p_Dst) {}
};

////////////////////////////  KARGS module  ////////////////////////////

template <typename t_DataType, // Interface type for client (instead of ap_uint)
          unsigned int t_DdrWidthFloats,
          unsigned int t_InstrWidth, // For narrow DDR, >1 allows to fit the instruction (in DdrWords)
          unsigned int t_ArgPipeline,
          unsigned int t_DataBitWidth = 8 * sizeof(t_DataType)>
class Kargs {
   public:
    typedef typename WideType<t_DataType, t_DdrWidthFloats>::t_TypeInt DdrFloatType;
    static const int t_DdrWidthBits = t_DataBitWidth * t_DdrWidthFloats * t_InstrWidth;
    class DdrInstrType {
       private:
        DdrFloatType v[t_InstrWidth];

       public:
        DdrInstrType() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = v complete dim = 1
        }
        DdrFloatType& operator[](int i) { return (v[i]); }
        void loadFromDdr(DdrFloatType* p_Addr, unsigned int p_Offset) {
            for (unsigned int i = 0; i < t_InstrWidth; ++i) {
                v[i] = p_Addr[p_Offset + i];
            }
        }
        void storeToDdr(DdrFloatType* p_Addr, unsigned int p_Offset) {
            for (unsigned int i = 0; i < t_InstrWidth; ++i) {
                p_Addr[p_Offset + i] = v[i];
            }
        }
    };
    typedef ap_uint<t_DdrWidthBits> DdrBitType;
    typedef enum {
        OpControl = 0x00,
        OpGemv = 0x01,
        OpGemm = 0x02,
        OpTransp = 0x03,
        OpResult = 0x04,
        OpFail = 0x05,
        OpFcn = 0x06
    } OpType;

   private:
    DdrBitType m_Flat;
    unsigned int m_BitPos;
    unsigned int m_WordPos;

   private:
    // Helper functions for serializing and deserializing
    template <typename T>
    void loadVal(T& var) {
        const int l_sizeBits = 8 * sizeof(var);
        unsigned int l_endBit = m_BitPos + l_sizeBits - 1;
        if (l_endBit >= t_DdrWidthBits) {
            m_WordPos++;
            m_BitPos = 0;
            l_endBit = m_BitPos + l_sizeBits - 1;
        }
        assert(m_WordPos <= 0);
        assert(l_endBit <= t_DdrWidthBits);
        ap_uint<l_sizeBits> l_bitVal = m_Flat.range(l_endBit, m_BitPos);
        // printf("  DEBUG loadVal bf=%s\n", l_bitVal.to_string(16).c_str());
        m_BitPos = l_endBit + 1;
        var = l_bitVal;
    }
    template <typename T>
    void storeVal(T& var) {
        const int l_sizeBits = 8 * sizeof(var);
        ap_uint<l_sizeBits> l_bitVal = var;
        // printf("  DEBUG storeVal bf=%s\n", l_bitVal.to_string(16).c_str());
        unsigned int l_endBit = m_BitPos + l_sizeBits - 1;
        if (l_endBit >= t_DdrWidthBits) {
            m_WordPos++;
            m_BitPos = 0;
            l_endBit = m_BitPos + l_sizeBits - 1;
        }
        assert(m_WordPos <= 0);
        assert(l_endBit <= t_DdrWidthBits);
        m_Flat.range(l_endBit, m_BitPos) = l_bitVal;
        m_BitPos = l_endBit + 1;
    }
    template <typename T>
    void storeValConst(const T var) {
        T l_val = var;
        storeVal(l_val);
    }
    void initPos() {
        m_WordPos = 0;
        m_BitPos = 0;
    }

    DdrBitType wideFloatToBits(DdrInstrType p_Val) {
        DdrBitType l_Val;
        for (unsigned int i = 0; i < t_InstrWidth; ++i) {
#pragma HLS unroll
            unsigned int l_instrBits = t_DdrWidthFloats * t_DataBitWidth;
            l_Val.range((i + 1) * l_instrBits - 1, i * l_instrBits) = p_Val[i];
        }
        return (l_Val);
    }

    DdrInstrType wideBitsToFloat(DdrBitType p_Val) {
        DdrInstrType l_Val;
        for (unsigned int i = 0; i < t_InstrWidth; ++i) {
#pragma HLS unroll
            unsigned int l_instrBits = t_DdrWidthFloats * t_DataBitWidth;
            l_Val[i] = p_Val.range((i + 1) * l_instrBits - 1, i * l_instrBits);
        }
        return (l_Val);
    }

   public:
    Kargs() : m_Flat(0) {
#pragma HLS INLINE
    }
    OpType loadFromInstr(DdrInstrType p_Val) {
        m_Flat = wideFloatToBits(p_Val);
        initPos();
        OpType l_op;
        loadVal(reinterpret_cast<unsigned int&>(l_op));
        return (l_op);
    }
    void storeToInstr(DdrInstrType& p_Val) { p_Val = wideBitsToFloat(m_Flat); }

    OpType load(DdrFloatType* p_Addr, unsigned int p_Pc) {
        DdrInstrType l_val;
        l_val.loadFromDdr(p_Addr, p_Pc);
        return (loadFromInstr(l_val));
    }
    void store(DdrFloatType* p_Addr, unsigned int p_Pc) {
        DdrInstrType l_val = wideBitsToFloat(m_Flat);
        l_val.storeToDdr(p_Addr, p_Pc);
    }

    InstrResArgs getInstrResArgs() {
        InstrResArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_StartTime);
        loadVal(l_args.m_EndTime);
        assert(l_args.m_StartTime <= l_args.m_EndTime);
        InstrResArgs l_ret = hlsReg<InstrResArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setInstrResArgs(InstrResArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpResult));
        storeVal(p_args.m_StartTime);
        storeVal(p_args.m_EndTime);
    }

    ControlArgs getControlArgs() {
        ControlArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_IsLastOp);
        loadVal(l_args.m_Noop);
        ControlArgs l_ret = hlsReg<ControlArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setControlArgs(ControlArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpControl));
        storeVal(p_args.m_IsLastOp);
        storeVal(p_args.m_Noop);
    }

    GemvArgs getGemvArgs() {
        GemvArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_Aoffset);
        loadVal(l_args.m_Boffset);
        loadVal(l_args.m_Coffset);
        loadVal(l_args.m_M);
        loadVal(l_args.m_K);
        loadVal(l_args.m_Lda);
        GemvArgs l_ret = hlsReg<GemvArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setGemvArgs(GemvArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpGemv));
        storeVal(p_args.m_Aoffset);
        storeVal(p_args.m_Boffset);
        storeVal(p_args.m_Coffset);
        storeVal(p_args.m_M);
        storeVal(p_args.m_K);
        storeVal(p_args.m_Lda);
    }

    GemmArgs getGemmArgs() {
        GemmArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_Aoffset);
        loadVal(l_args.m_Boffset);
        loadVal(l_args.m_Coffset);
        loadVal(l_args.m_Xoffset);
        loadVal(l_args.m_M);
        loadVal(l_args.m_K);
        loadVal(l_args.m_N);
        loadVal(l_args.m_Lda);
        loadVal(l_args.m_Ldb);
        loadVal(l_args.m_Ldc);
        loadVal(l_args.m_Ldx);
        loadVal(l_args.m_postScale);
        GemmArgs l_ret = hlsReg<GemmArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setGemmArgs(GemmArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpGemm));
        storeVal(p_args.m_Aoffset);
        storeVal(p_args.m_Boffset);
        storeVal(p_args.m_Coffset);
        storeVal(p_args.m_Xoffset);
        storeVal(p_args.m_M);
        storeVal(p_args.m_K);
        storeVal(p_args.m_N);
        storeVal(p_args.m_Lda);
        storeVal(p_args.m_Ldb);
        storeVal(p_args.m_Ldc);
        storeVal(p_args.m_Ldx);
        storeVal(p_args.m_postScale);
    }

    FcnArgs getFcnArgs() {
        FcnArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_Aoffset);
        loadVal(l_args.m_Boffset);
        loadVal(l_args.m_Coffset);
        loadVal(l_args.m_Xoffset);
        loadVal(l_args.m_M);
        loadVal(l_args.m_K);
        loadVal(l_args.m_N);
        loadVal(l_args.m_Lda);
        loadVal(l_args.m_Ldb);
        loadVal(l_args.m_Ldc);
        loadVal(l_args.m_Ldx);
        loadVal(l_args.m_postScale);
        loadVal(l_args.m_PReluVal);
        FcnArgs l_ret = hlsReg<FcnArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setFcnArgs(FcnArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpFcn));
        storeVal(p_args.m_Aoffset);
        storeVal(p_args.m_Boffset);
        storeVal(p_args.m_Coffset);
        storeVal(p_args.m_Xoffset);
        storeVal(p_args.m_M);
        storeVal(p_args.m_K);
        storeVal(p_args.m_N);
        storeVal(p_args.m_Lda);
        storeVal(p_args.m_Ldb);
        storeVal(p_args.m_Ldc);
        storeVal(p_args.m_Ldx);
        storeVal(p_args.m_postScale);
        storeVal(p_args.m_PReluVal);
    }

    TranspArgs getTranspArgs() {
        TranspArgs l_args;
        assert(sizeof(l_args) <= sizeof(m_Flat) - sizeof(OpType));
        loadVal(l_args.m_Src.m_Offset);
        loadVal(l_args.m_Src.m_Rows);
        loadVal(l_args.m_Src.m_Cols);
        loadVal(l_args.m_Src.m_Ld);
        loadVal(l_args.m_Src.m_GfEdge);
        loadVal(reinterpret_cast<unsigned int&>(l_args.m_Src.m_Format));
        loadVal(l_args.m_Dst.m_Offset);
        loadVal(l_args.m_Dst.m_Rows);
        loadVal(l_args.m_Dst.m_Cols);
        loadVal(l_args.m_Dst.m_Ld);
        loadVal(l_args.m_Dst.m_GfEdge);
        loadVal(reinterpret_cast<unsigned int&>(l_args.m_Dst.m_Format));
        TranspArgs l_ret = hlsReg<TranspArgs, t_ArgPipeline>(l_args);
        return l_ret;
    }
    void setTranspArgs(TranspArgs p_args) {
        assert(sizeof(p_args) <= sizeof(m_Flat) - sizeof(OpType));
        initPos();
        storeValConst(int(OpTransp));
        storeVal(p_args.m_Src.m_Offset);
        storeVal(p_args.m_Src.m_Rows);
        storeVal(p_args.m_Src.m_Cols);
        storeVal(p_args.m_Src.m_Ld);
        storeVal(p_args.m_Src.m_GfEdge);
        storeValConst(int(p_args.m_Src.m_Format));
        storeVal(p_args.m_Dst.m_Offset);
        storeVal(p_args.m_Dst.m_Rows);
        storeVal(p_args.m_Dst.m_Cols);
        storeVal(p_args.m_Dst.m_Ld);
        storeVal(p_args.m_Dst.m_GfEdge);
        storeValConst(int(p_args.m_Dst.m_Format));
    }

    static unsigned int getInstrWidth() { return (t_InstrWidth); }
};

} // namespace
}

#endif
