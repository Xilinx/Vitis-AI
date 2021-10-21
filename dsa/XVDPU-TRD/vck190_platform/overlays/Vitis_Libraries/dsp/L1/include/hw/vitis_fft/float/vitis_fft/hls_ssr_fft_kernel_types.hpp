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
 * @file hls_ssr_fft_kernel_types.hpp
 * @brief header, describes interface types for kernel
 *
 */

#ifndef __HLS_SSR_FFT_KENEL_TYPES_H__
#define __HLS_SSR_FFT_KENEL_TYPES_H__

#ifndef __SYNTHESIS__
#include <iostream>
#endif

#include "vitis_fft/fft_complex.hpp"
#include "vitis_fft/hls_ssr_fft_super_sample.hpp"

namespace xf {
namespace dsp {
namespace fft {

#if 0
template <typename T_elemType, unsigned int t_width>
class WideInterfaceType {

  public:
    T_elemType  m_Val[t_width];
    T_elemType &getVal(unsigned int i) {
      return(m_Val[i]);
    }
    T_elemType &operator[](unsigned int p_Idx) {
      return(m_Val[p_Idx]);
    }
    T_elemType *getValAddr() {
      return(&m_Val[0]);
    }
    WideInterfaceType() {

    }
    WideInterfaceType(T_elemType p_initScalar) {
        for(int i = 0; i < t_width; ++i) {
          getVal(i) = p_initScalar;
        }
      }
    T_elemType shift(T_elemType p_ValIn) {
#pragma HLS inline self
#pragma HLS data_pack variable = p_ValIn
        T_elemType l_valOut = m_Val[t_width-1];
        WIDE_TYPE_SHIFT:for(int i = t_width - 1; i > 0; --i) {
          T_elemType l_val = m_Val[i - 1];
#pragma HLS data_pack variable = l_val
          m_Val[i] = l_val;
        }
        m_Val[0] = p_ValIn;
        return(l_valOut);
      }
    T_elemType shift() {
#pragma HLS inline self
        T_elemType l_valOut = m_Val[t_width-1];
        WIDE_TYPE_SHIFT:for(int i = t_width - 1; i > 0; --i) {
          T_elemType l_val = m_Val[i - 1];
#pragma HLS data_pack variable = l_val
          m_Val[i] = l_val;
        }
        return(l_valOut);
      }
    T_elemType  unshift() {
#pragma HLS inline self
        T_elemType l_valOut = m_Val[0];
        WIDE_TYPE_SHIFT:for(int i = 0; i < t_width - 1; ++i) {
          T_elemType l_val = m_Val[i + 1];
#pragma HLS data_pack variable = l_val
          m_Val[i] = l_val;
        }
        return(l_valOut);
      }
    static const WideInterfaceType zero() {
        WideInterfaceType l_zero;
#pragma HLS data_pack variable = l_zero
        for(int i = 0; i < t_width; ++i) {
          l_zero[i] = 0;
        }
        return(l_zero);
      }
    void print(std::ostream& os) {
        for(int i = 0; i < t_width; ++i) {
          os << getVal(i) << ", ";
        }
      }
    bool operator==(const WideInterfaceType &rhs_in) const
  {
      for (int i = 0; i < t_width; ++i)
      {
      if(m_Val[i]!=rhs_in.m_Val[i])
        return false;
    }
      return true;
  }

    bool operator!=(const WideInterfaceType &rhs_in) const
  {
      for (int i = 0; i < t_width; ++i)
      {
      if(m_Val[i]!=rhs_in.m_Val[i])
        return true;
    }
      return false;
  }

};

template <typename T1, unsigned int T2>
std::ostream& operator<<(std::ostream& os, WideInterfaceType<T1, T2>& p_Val) {
  p_Val.print(os);
  return(os);
}
template < unsigned int t_width,
           typename T_elemType
           >
struct WideTypeDefs
{
  typedef WideInterfaceType<T_elemType,t_width> WideIFType;
  typedef hls::stream<WideIFType> WideIFStreamType;
};
#endif

template <unsigned int t_width, typename T_elemType>
struct WideTypeDefs {
    typedef SuperSampleContainer<t_width, T_elemType> WideIFType;
    typedef hls::stream<WideIFType> WideIFStreamType;
};

} // end namespace fft
} // end namespace dsp
} // end namespace xf
#endif
