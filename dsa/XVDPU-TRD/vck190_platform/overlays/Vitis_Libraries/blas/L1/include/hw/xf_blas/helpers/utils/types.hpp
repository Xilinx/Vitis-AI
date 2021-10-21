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
 * @file types.hpp
 * @brief common datatypes for L1 modules.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_TYPES_HPP
#define XF_BLAS_TYPES_HPP

#include <stdint.h>
#include <ostream>
#include <iomanip>
#include <iostream>
#include <cassert>
#include "ap_int.h"
#include "ap_shift_reg.h"

#define BLAS_FLOAT_WIDTH 7

namespace xf {

namespace blas {

template <typename T, unsigned int t_Width, unsigned int t_DataWidth = sizeof(T) * 8, typename Enable = void>
class WideType {
   private:
    T m_Val[t_Width];
    static const unsigned int t_4k = 4096;
    static const unsigned int FLOAT_WIDTH = 7;

   public:
    static const unsigned int t_TypeWidth = t_Width * t_DataWidth;
    typedef ap_uint<t_TypeWidth> t_TypeInt;
    typedef T DataType;
    static const unsigned int t_WidthS = t_Width;
    static const unsigned int t_per4k = t_4k / t_DataWidth / t_Width * 8;

   public:
    T& getVal(unsigned int i) {
#ifndef __SYNTHESIS__
        assert(i < t_Width);
#endif
        return (m_Val[i]);
    }
    T& operator[](unsigned int p_Idx) {
#ifndef __SYNTHESIS__
        assert(p_Idx < t_Width);
#endif
        return (m_Val[p_Idx]);
    }
    const T& operator[](unsigned int p_Idx) const {
#ifndef __SYNTHESIS__
        assert(p_Idx < t_Width);
#endif
        return (m_Val[p_Idx]);
    }
    T* getValAddr() { return (&m_Val[0]); }

    WideType() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Val complete dim = 1
    }

    WideType(const WideType& wt) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Val complete dim = 1
        constructor(wt);
    }

    void constructor(const WideType& wt) {
        for (unsigned int i = 0; i < t_Width; i++)
#pragma HLS UNROLL
            m_Val[i] = wt[i];
    }

    WideType(const t_TypeInt& p_val) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Val complete dim = 1
        constructor(p_val);
    }
    void constructor(const t_TypeInt& p_val) {
        for (int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
            ap_uint<t_DataWidth> l_val = p_val.range(t_DataWidth * (1 + i) - 1, t_DataWidth * i);
            m_Val[i] = *reinterpret_cast<T*>(&l_val);
        }
    }

    WideType(const T p_initScalar) {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Val complete dim = 1
        constructor(p_initScalar);
    }
    void constructor(const T p_initScalar) {
        for (int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
            m_Val[i] = p_initScalar;
        }
    }

    operator const t_TypeInt() {
        t_TypeInt l_fVal;
        for (int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
            T l_v = m_Val[i];
            ap_uint<t_DataWidth> l_val = *reinterpret_cast<ap_uint<t_DataWidth>*>(&l_v);
            l_fVal.range(t_DataWidth * (1 + i) - 1, t_DataWidth * i) = l_val;
        }
        return l_fVal;
    }

    bool operator==(const WideType& p_w) const {
        bool l_com = true;
        for (int i = 0; i < t_Width; ++i) {
            l_com = l_com && (m_Val[i] == p_w[i]);
        }
        return l_com;
    }

    T shift(T p_ValIn) {
#pragma HLS INLINE
        T l_valOut = m_Val[t_Width - 1];
    WIDE_TYPE_SHIFT:
        for (int i = t_Width - 1; i > 0; --i) {
#pragma HLS UNROLL
            T l_val = m_Val[i - 1];
            m_Val[i] = l_val;
        }
        m_Val[0] = p_ValIn;
        return (l_valOut);
    }

    T shift() {
#pragma HLS INLINE
        T l_valOut = m_Val[t_Width - 1];
        for (int i = t_Width - 1; i > 0; --i) {
#pragma HLS UNROLL
            T l_val = m_Val[i - 1];
            m_Val[i] = l_val;
        }
        return (l_valOut);
    }

    T unshift() {
#pragma HLS INLINE
        T l_valOut = m_Val[0];
        for (int i = 0; i < t_Width - 1; ++i) {
#pragma HLS UNROLL
            T l_val = m_Val[i + 1];
            m_Val[i] = l_val;
        }
        return (l_valOut);
    }

    T unshift(const T p_val) {
#pragma HLS INLINE
        T l_valOut = m_Val[0];
        for (int i = 0; i < t_Width - 1; ++i) {
#pragma HLS UNROLL
            T l_val = m_Val[i + 1];
            m_Val[i] = l_val;
        }
        m_Val[t_Width - 1] = p_val;
        return (l_valOut);
    }

    static const WideType zero() {
        WideType l_zero;
        for (int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
            l_zero[i] = 0;
        }
        return (l_zero);
    }

    static unsigned int per4k() { return (t_per4k); }
    void print(std::ostream& os) {
        for (int i = 0; i < t_Width; ++i) {
            os << std::setw(FLOAT_WIDTH) << m_Val[i] << " ";
        }
    }

    friend std::ostream& operator<<(std::ostream& os, WideType& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

template <typename T, unsigned int t_DataWidth>
class WideType<T, 1, t_DataWidth, typename std::enable_if<std::is_same<ap_uint<t_DataWidth>, T>::value>::type> {
   private:
    T m_Val;
    static const unsigned int t_4k = 4096;
    static const unsigned int FLOAT_WIDTH = 7;

   public:
    static const unsigned int t_TypeWidth = t_DataWidth;
    typedef ap_uint<t_DataWidth> t_TypeInt;
    typedef T DataType;
    static const unsigned int t_WidthS = 1;
    static const unsigned int t_per4k = t_4k / t_DataWidth * 8;

   public:
    T& operator[](unsigned int p_Idx) {
#ifndef __SYNTHESIS__
        assert(p_Idx == 0);
#endif
        return m_Val;
    }

    const T& operator[](unsigned int p_Idx) const {
#ifndef __SYNTHESIS__
        assert(p_Idx == 0);
#endif
        return m_Val;
    }

    T* getValAddr() { return (&m_Val); }

    WideType() {}

    WideType(const WideType& wt) { m_Val = wt[0]; }

    WideType(const t_TypeInt p_val) { m_Val = p_val; }

    operator const t_TypeInt() { return m_Val; }

    bool operator==(const WideType& p_w) const { return m_Val == p_w[0]; }

    T shift(T p_ValIn) {
        T l_valOut = m_Val;
        m_Val = p_ValIn;
        return l_valOut;
    }
    T shift() { return m_Val; }

    T unshift() { return m_Val; }

    T unshift(T p_ValIn) {
        T l_valOut = m_Val;
        m_Val = p_ValIn;
        return l_valOut;
    }

    static const WideType zero() { return WideType(0); }

    static unsigned int per4k() { return (t_per4k); }
    void print(std::ostream& os) { os << std::setw(FLOAT_WIDTH) << m_Val << " "; }

    friend std::ostream& operator<<(std::ostream& os, WideType& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

template <typename t_FloatType,
          unsigned int t_MemWidth,    // In t_FloatType
          unsigned int t_MemWidthBits // In bits; both must be defined and be consistent
          >
class MemUtil {
   private:
   public:
    typedef WideType<t_FloatType, t_MemWidth> MemWideType;
};
/////////////////////////    Control and helper types    /////////////////////////

template <class T, uint8_t t_NumCycles>
T hlsReg(T p_In) {
#pragma HLS INLINE
#pragma HLS INTERFACE ap_none port = return register
    if (t_NumCycles == 1) {
        return p_In;
    } else {
        return hlsReg<T, uint8_t(t_NumCycles - 1)>(p_In);
    }
}

template <class T>
T hlsReg(T p_In) {
    return hlsReg<T, 1>(p_In);
}

template <unsigned int W>
class BoolArr {
   private:
    bool m_Val[W];

   public:
    BoolArr() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Val COMPLETE
    }
    BoolArr(bool p_Init) {
#pragma HLS inline
        for (unsigned int i = 0; i < W; ++i) {
#pragma HLS UNROLL
            m_Val[i] = p_Init;
        }
    }
    bool& operator[](unsigned int p_Idx) {
#pragma HLS inline
        return m_Val[p_Idx];
    }
    bool And() {
#pragma HLS inline
        bool l_ret = true;
        for (unsigned int i = 0; i < W; ++i) {
#pragma HLS UNROLL
            l_ret = l_ret && m_Val[i];
        }
        return (l_ret);
    }
    bool Or() {
#pragma HLS inline
        bool l_ret = false;
        for (unsigned int i = 0; i < W; ++i) {
#pragma HLS UNROLL
            l_ret = l_ret || m_Val[i];
        }
        return (l_ret);
    }
    void Reset() {
#pragma HLS inline
        for (unsigned int i = 0; i < W; ++i) {
#pragma HLS UNROLL
            m_Val[i] = false;
        }
    }
};

template <class S, int W>
bool streamsAreEmpty(S p_Sin[W]) {
#pragma HLS inline
    bool l_allEmpty = true;
LOOP_S_IDX:
    for (int w = 0; w < W; ++w) {
#pragma HLS UNROLL
        l_allEmpty = l_allEmpty && p_Sin[w].empty();
    }
    return (l_allEmpty);
}

//  Bit converter
template <typename T>
class BitConv {
   public:
    static const unsigned int t_SizeOf = sizeof(T);
    static const unsigned int t_NumBits = 8 * sizeof(T);
    typedef ap_uint<t_NumBits> BitsType;

   public:
    BitsType toBits(T p_Val) { return p_Val; }
    T toType(BitsType p_Val) { return p_Val; }
};

template <>
inline BitConv<float>::BitsType BitConv<float>::toBits(float p_Val) {
    union {
        float f;
        unsigned int i;
    } u;
    u.f = p_Val;
    return (u.i);
}

template <>
inline float BitConv<float>::toType(BitConv<float>::BitsType p_Val) {
    union {
        float f;
        unsigned int i;
    } u;
    u.i = p_Val;
    return (u.f);
}

template <>
inline BitConv<double>::BitsType BitConv<double>::toBits(double p_Val) {
    union {
        double f;
        int64_t i;
    } u;
    u.f = p_Val;
    return (u.i);
}

template <>
inline double BitConv<double>::toType(BitConv<double>::BitsType p_Val) {
    union {
        double f;
        int64_t i;
    } u;
    u.i = p_Val;
    return (u.f);
}

template <unsigned int t_Bits, unsigned int t_Width, typename t_DataType>
ap_uint<t_Bits> convWideVal2Bits(WideType<t_DataType, t_Width> p_val) {
#pragma HLS inline
#ifndef __SYNTHESIS__
    assert((t_Bits > t_Width) && (t_Bits % t_Width == 0));
#endif
    const unsigned int t_DataBits = sizeof(t_DataType) * 8;
    const unsigned int t_ResEntryBits = t_Bits / t_Width;
    ap_uint<t_Bits> l_res;
    for (unsigned int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
        BitConv<t_DataType> l_bitConv;
        ap_uint<t_DataBits> l_datBits = l_bitConv.toBits(p_val[i]);
        ap_uint<t_ResEntryBits> l_resEntry = l_datBits;
        l_res.range((i + 1) * t_ResEntryBits - 1, i * t_ResEntryBits) = l_resEntry;
    }
    return l_res;
}

template <unsigned int t_Bits, unsigned int t_Width, typename t_DataType>
WideType<t_DataType, t_Width> convBits2WideType(ap_uint<t_Bits> p_bits) {
#pragma HLS inline
#ifndef __SYNTHESIS__
    assert((t_Bits > t_Width) && (t_Bits % t_Width == 0));
#endif
    const unsigned int t_DataBits = sizeof(t_DataType) * 8;
    const unsigned int t_InEntryBits = t_Bits / t_Width;
    WideType<t_DataType, t_Width> l_res;
    for (unsigned int i = 0; i < t_Width; ++i) {
#pragma HLS UNROLL
        BitConv<t_DataType> l_bitConv;
        ap_uint<t_InEntryBits> l_inDatBits = p_bits.range((i + 1) * t_InEntryBits - 1, i * t_InEntryBits);
        ap_uint<t_DataBits> l_datBits = l_inDatBits;
        t_DataType l_val = l_bitConv.toType(l_datBits);
        l_res[i] = l_val;
    }
    return l_res;
}
// Type converter - for vectors of different lengths and types
template <typename TS, typename TD>
class WideConv {
   private:
    static const unsigned int t_ws = TS::t_WidthS;
    static const unsigned int t_wd = TD::t_WidthS;
    typedef BitConv<typename TS::DataType> ConvSType;
    typedef BitConv<typename TD::DataType> ConvDType;
    static const unsigned int t_bs = ConvSType::t_NumBits;
    static const unsigned int t_bd = ConvDType::t_NumBits;
    static const unsigned int l_numBits = t_ws * t_bs;

   private:
    ap_uint<l_numBits> l_bits;

   public:
    inline TD convert(TS p_Src) {
        TD l_dst;
        ConvSType l_convS;
#ifndef __SYNTHESIS__
        assert(t_wd * t_bd == l_numBits);
#endif
        for (int ws = 0; ws < t_ws; ++ws) {
            l_bits.range(t_bs * (ws + 1) - 1, t_bs * ws) = l_convS.toBits(p_Src[ws]);
        }
        ConvDType l_convD;
        for (int wd = 0; wd < t_wd; ++wd) {
            l_dst[wd] = l_convD.toType(l_bits.range(t_bd * (wd + 1) - 1, t_bd * wd));
        }

        return (l_dst);
    }
};

//////////////// From GEMX////////////////////////////

template <typename t_FloatType, unsigned int t_DataWidth = sizeof(t_FloatType) * 8>
class TaggedFloat {
   private:
    t_FloatType m_Val;
    bool m_Flush;

   public:
    static const int t_TypeWidth = t_DataWidth + 1;
    typedef ap_uint<t_TypeWidth> t_TypeInt;

    TaggedFloat(const t_TypeInt& p_val) {
        ap_uint<t_DataWidth> l_val = p_val.range(t_TypeWidth - 1, 1);
        m_Val = *reinterpret_cast<t_FloatType*>(&l_val);
        m_Flush = p_val[0];
    }
    operator const t_TypeInt() {
        t_TypeInt l_val;
        l_val[0] = m_Flush;
        l_val.range(t_TypeWidth - 1, 1) = *reinterpret_cast<ap_uint<t_DataWidth>*>(&m_Val);
        return l_val;
    }
    TaggedFloat() {}
    TaggedFloat(t_FloatType p_Val, bool p_Flush) : m_Val(p_Val), m_Flush(p_Flush) {}
    TaggedFloat(t_FloatType p_Val) : m_Val(p_Val), m_Flush(false) {}
    t_FloatType& getVal() { return (m_Val); }
    bool& getFlush() { return (m_Flush); }
    TaggedFloat& operator=(t_FloatType p_Val) {
        m_Val = p_Val;
        m_Flush = false;
        return (*this);
    }
    t_FloatType& operator()() { return (m_Val); }
    void print(std::ostream& os) { os << std::setw(BLAS_FLOAT_WIDTH) << m_Val << "f" << m_Flush; }
    friend std::ostream& operator<<(std::ostream& os, TaggedFloat& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

template <typename T, unsigned int t_Width>
class TaggedWideType {
   private:
    WideType<T, t_Width> m_Val;
    bool m_Flush;
    bool m_Exit;

    static const int t_TypeWidth = WideType<T, t_Width>::t_TypeWidth + 2;

   public:
    typedef ap_uint<t_TypeWidth> t_TypeInt;

    TaggedWideType(const t_TypeInt& p_val) {
        m_Val = (typename WideType<T, t_Width>::t_TypeInt)p_val.range(t_TypeWidth - 1, 2);
        m_Flush = p_val[1];
        m_Exit = p_val[0];
    }
    operator const t_TypeInt() {
        t_TypeInt l_val;
        l_val[0] = m_Exit;
        l_val[1] = m_Flush;
        l_val.range(t_TypeWidth - 1, 2) = (typename WideType<T, t_Width>::t_TypeInt)m_Val;
        return l_val;
    }

    TaggedWideType(WideType<T, t_Width> p_Val, bool p_Flush, bool p_Exit)
        : m_Val(p_Val), m_Flush(p_Flush), m_Exit(p_Exit) {}
    TaggedWideType() {}
    WideType<T, t_Width>& getVal() { return m_Val; }
    T& operator[](unsigned int p_Idx) { return (m_Val[p_Idx]); }

    bool getFlush() { return (m_Flush); }
    bool getExit() { return (m_Exit); }
    WideType<TaggedFloat<T>, t_Width> getVectOfTaggedValues() {
#pragma HLS INLINE
        WideType<TaggedFloat<T>, t_Width> l_vect;
    TWT_FORW:
        for (unsigned int i = 0; i < t_Width; ++i) {
            l_vect.getVal(i) = TaggedFloat<T>(m_Val.getVal(i), m_Flush);
        }
        return (l_vect);
    }
    // void setVal(T p_Val, unsigned int i) {m_Val[i] = p_Val;}
    // void setFlush(bool p_Flush) {m_Flush = p_Flush;}
    // void setExit() {return(m_Exit);}
    void print(std::ostream& os) {
        m_Val.print(os);
        os << " f" << m_Flush << " e" << m_Exit;
    }
    friend std::ostream& operator<<(std::ostream& os, TaggedWideType& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

template <typename T, unsigned int t_Width>
class TriangSrl {
   private:
    ap_shift_reg<T, t_Width> m_Sreg[t_Width];

   public:
    TriangSrl() {
#pragma HLS INLINE
#pragma HLS ARRAY_PARTITION variable = m_Sreg dim = 1 complete
    }
    WideType<T, t_Width> shift(WideType<T, t_Width> p_DiagIn) {
#pragma HLS INLINE
        WideType<T, t_Width> l_edgeOut;
    TRIANGSRL_SHIFT:
        for (unsigned int i = 0; i < t_Width; ++i) {
#pragma HLS unroll
            l_edgeOut[i] = m_Sreg[i].shift(p_DiagIn[i], i);
        }
        return (l_edgeOut);
    }
    void clear() {
#pragma HLS inline
        for (unsigned int row = 0; row < t_Width; ++row) {
            for (unsigned int col = 0; col < t_Width; ++col) {
#pragma HLS PIPELINE
                (void)m_Sreg[row].shift(0, 0);
            }
        }
    }
    void print(std::ostream& os) {
        for (unsigned int row = 0; row < t_Width; ++row) {
            for (unsigned int col = 0; col < t_Width; ++col) {
                if (col > row) {
                    os << std::setw(BLAS_FLOAT_WIDTH + 2) << "- ";
                } else {
                    T l_val = m_Sreg[row].read(col);
                    os << std::setw(BLAS_FLOAT_WIDTH) << l_val;
                }
            }
            os << "\n";
        }
    }
    friend std::ostream& operator<<(std::ostream& os, TriangSrl& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

// Row-major window
template <typename T, unsigned int t_Rows, unsigned int t_Cols>
class WindowRm {
   private:
    WideType<WideType<T, t_Cols>, t_Rows> m_Val;

   public:
    WindowRm() {
#pragma HLS INLINE
    }
    T& getval(unsigned int p_Row, unsigned int p_Col) { return m_Val.getVal(p_Row).getVal(p_Col); }
    WideType<T, t_Cols>& operator[](unsigned int p_Idx) { return (m_Val[p_Idx]); }
    void clear() {
    WINDOWRM_ROW:
        for (unsigned int row = 0; row < t_Rows; ++row) {
#pragma HLS UNROLL
        WINDOWRM_COL:
            for (unsigned int col = 0; col < t_Cols; ++col) {
#pragma HLS UNROLL
                getval(row, col) = -999;
            }
        }
    }
    // DOWN (0th row in, the last row out)
    WideType<T, t_Cols> shift(WideType<T, t_Cols> p_EdgeIn) {
#pragma HLS INLINE
        return (m_Val.shift(p_EdgeIn));
    }
    // DOWN no input
    WideType<T, t_Cols> shift() {
#pragma HLS INLINE
        return (m_Val.shift());
    }
    // UP no input
    WideType<T, t_Cols> unshift() {
#pragma HLS INLINE
        return (m_Val.unshift());
    }
    // RIGHT
    WideType<T, t_Rows> shift_right(WideType<T, t_Rows> p_EdgeIn) {
        WideType<T, t_Cols> l_edgeOut;
    // Shift each row
    WINDOWRM_SHIFT_R1:
        for (unsigned int row = 0; row < t_Rows; ++row) {
            l_edgeOut[row] = m_Val[row].shift(p_EdgeIn[row]);
        }
        return (l_edgeOut);
    }
    void print(std::ostream& os) {
        for (int i = 0; i < t_Rows; ++i) {
            os << m_Val.getVal(i) << "\n";
        }
    }
    friend std::ostream& operator<<(std::ostream& os, WindowRm& p_Val) {
        p_Val.print(os);
        return (os);
    }
};

template <typename t_DataType, unsigned int t_DataWidth = sizeof(t_DataType) * 8>
class DualTaggedType {
   public:
    t_DataType m_val;
    bool m_flush;
    bool m_exit;

    static const unsigned int t_TypeWidth = t_DataWidth + 2;
    typedef ap_uint<t_TypeWidth> t_TypeInt;

    DualTaggedType() {}

    DualTaggedType(const DualTaggedType& p_dt) {
        m_val = p_dt.m_val;
        m_flush = p_dt.m_flush;
        m_exit = p_dt.m_exit;
    }

    DualTaggedType(const t_TypeInt& p_val) {
        ap_uint<t_DataWidth> l_a;
        l_a = p_val.range(t_TypeWidth - 1, t_TypeWidth - t_DataWidth);
        m_val = *reinterpret_cast<t_DataType*>(&l_a);
        m_flush = p_val[1];
        m_exit = p_val[0];
    }

    operator const t_TypeInt() {
        t_TypeInt l_val;
        ap_uint<t_DataWidth> l_a;
        l_a = *reinterpret_cast<ap_uint<t_DataWidth>*>(&m_val);

        l_val.range(t_TypeWidth - 1, t_TypeWidth - t_DataWidth) = l_a;
        l_val[1] = m_flush;
        l_val[0] = m_exit;
        return l_val;
    }
};

} // namespace blas
} // namespace xf
#endif
