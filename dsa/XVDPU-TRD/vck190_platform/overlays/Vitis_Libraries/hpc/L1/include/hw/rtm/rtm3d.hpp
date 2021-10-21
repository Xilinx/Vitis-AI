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

#ifndef XF_HPC_RTM_RTM3D_HPP
#define XF_HPC_RTM_RTM3D_HPP

/**
 * @file rtm.hpp
 * @brief class RTM3D derived from Stencil3D is defined here, it provides L1 primary
 * streaming modules for 3D-RTM kernels
 */

namespace xf {
namespace hpc {
namespace rtm {

/**
 * @brief RTM3D class defines the basic operations for 3D RTM3D
 *
 * @tparam t_DataType the basic wavefield datatype
 * @tparam t_Order is the spatial discretization order
 * @tparam t_MaxDimZ is the maximum dim along z-axis this kernel can process
 * @tparam t_MaxDimY is the maximum dim along y-axis this kernel can process
 * @tparam t_PEZ is the number of processing elements along z-axis
 * @tparam t_PEX is the number of processing elements along x-axis
 */

template <typename t_Domain,
          typename t_DataType,
          int t_Order,
          int t_MaxDimZ = 128,
          int t_MaxDimY = 128,
          int t_MaxB = 40,
          int t_PEZ = 1,
          int t_PEX = 1>
class RTM3D : public Stencil3D<t_DataType, t_Order, t_MaxDimZ, t_MaxDimY, t_PEZ, t_PEX> {
   public:
    typedef Stencil3D<t_DataType, t_Order, t_MaxDimZ, t_MaxDimY, t_PEZ, t_PEX> t_StencilType;
    using t_StencilType::t_NumData;
    using t_StencilType::t_FifoDepth;
    using t_StencilType::t_HalfOrder;
    typedef typename t_StencilType::t_PairType t_PairType;
    typedef typename t_StencilType::t_DataTypeX t_DataTypeX;
    typedef typename t_StencilType::t_DataTypeZ t_DataTypeZ;
    typedef typename t_StencilType::t_WideType t_WideType;
    typedef typename t_StencilType::t_InType t_InType;
    typedef typename t_StencilType::t_PairInType t_PairInType;

    typedef blas::WideType<t_InType, t_HalfOrder / t_PEZ> t_UpbType;
    typedef typename t_UpbType::t_TypeInt t_UpbInType;

    RTM3D(int p_NZB = t_MaxB, int p_NYB = t_MaxB, int p_NXB = t_MaxB) {
#pragma HLS INLINE
#ifndef __SYNTHESIS__
        assert(t_MaxB >= p_NXB);
        assert(t_MaxB >= p_NYB);
        assert(t_MaxB >= p_NZB);
        assert(p_NZB % t_PEZ == 0);
#endif
        m_NXB = p_NXB;
        m_NYB = p_NYB;
        m_NZB = p_NZB;
    }
    void setDomain(const t_Domain& p_domain) { m_domain = p_domain; }

    void setSrc(int p_srcz, int p_srcy, int p_srcx) {
        m_srcz = p_srcz;
        m_srcy = p_srcy;
        m_srcx = p_srcx;
    }

    void setTaper(const t_DataType* p_taperz, const t_DataType* p_tapery, const t_DataType* p_taperx) {
        for (int i = 0; i < m_NXB / t_PEX; i++) {
            t_DataTypeX l_w;
            for (int pe = 0; pe < t_PEX; pe++) {
#pragma HLS PIPELINE
                l_w.unshift(p_taperx[i * t_PEX + pe]);
            }
            m_taperx[i] = l_w;
        }

        for (int i = 0; i < m_NYB; i++) {
#pragma HLS PIPELINE
            m_tapery[i] = p_tapery[i];
        }

        for (int i = 0; i < m_NZB / t_PEZ; i++) {
            t_DataTypeZ l_w;
            for (int pe = 0; pe < t_PEZ; pe++) {
#pragma HLS PIPELINE
                l_w.unshift(p_taperz[i * t_PEZ + pe]);
            }
            m_taperz[i] = l_w;
        }
    }

   private:
    /**
     * @brief taper function applys to the absorbing boundary
     *
     * @param p_in is the stream of input wavefield
     * @param p_out is the stream of output wavefield
     */
    void taper(hls::stream<t_InType>& p_in, hls::stream<t_InType>& p_out) {
        for (int i = 0, j = 0, k = 0, t = 0; t < this->m_cube; t++) {
#pragma HLS PIPELINE
            t_DataTypeX tx;
            t_DataType ty;
            t_DataTypeZ tz;

            if (k < m_NZB / t_PEZ) {
                tz = m_taperz[k];
                if (i < m_NXB / t_PEX)
                    tx = m_taperx[i];
                else if (i >= this->m_xPE - m_NXB / t_PEX)
                    tx = m_taperx[this->m_xPE - i - 1];
                else
                    tx = 1;

                if (j + m_domain.m_extCoo < m_NYB)
                    ty = m_tapery[j + m_domain.m_extCoo];
                else if (j + m_domain.m_extCoo >= m_domain.m_y - m_NYB)
                    ty = m_tapery[m_domain.m_y - j - 1 - m_domain.m_extCoo];
                else
                    ty = 1;
            } else {
                tz = 1;
                ty = 1;
                tx = 1;
            }

            t_WideType l_w = p_in.read();
            for (int pe = 0; pe < t_PEZ; pe++) {
#pragma HLS UNROLL
                t_DataTypeX l_wx = l_w[pe];
                for (int px = 0; px < t_PEX; px++)
#pragma HLS UNROLL
                    l_wx[px] *= tz[pe] * ty * tx[px];
                l_w[pe] = l_wx;
            }
            p_out.write(l_w);

            if (k == this->m_zPE - 1 && j == this->m_y - 1) {
                k = 0;
                j = 0;
                i++;
            } else if (k == this->m_zPE - 1) {
                k = 0;
                j++;
            } else
                k++;
        }
    }

    void addSrc(const t_DataType p_src, hls::stream<t_InType>& p_p0, hls::stream<t_InType>& p_p1) {
        for (int i = 0, j = 0, k = 0, t = 0; t < this->m_cube; t++) {
#pragma HLS PIPELINE
            t_WideType l_wo = p_p0.read();
            int l_xCoo = m_srcx - i * t_PEX;
            int l_yCoo = j + m_domain.m_extCoo;
            int l_zCoo = m_srcz - k * t_PEZ;
            if (l_xCoo >= 0 && l_xCoo < t_PEX && l_yCoo == m_srcy && l_zCoo >= 0 && l_zCoo < t_PEZ) {
                t_DataTypeX l_wx = l_wo[l_zCoo];
                l_wx[l_xCoo] += p_src;
                l_wo[l_zCoo] = l_wx;
            }

            p_p1.write(l_wo);

            if (k == this->m_zPE - 1 && j == this->m_y - 1) {
                k = 0;
                j = 0;
                i++;
            } else if (k == this->m_zPE - 1) {
                k = 0;
                j++;
            } else
                k++;
        }
    }

    void extractUPB(hls::stream<t_InType>& p_pin, hls::stream<t_InType>& p_pout, hls::stream<t_UpbInType>& p_upb) {
        t_UpbType l_upb;
        for (int i = 0, j = 0, k = 0, t = 0; t < this->m_cube; t++) {
#pragma HLS PIPELINE
            t_WideType l_val = p_pin.read();
            p_pout.write(l_val);
            l_upb.unshift(l_val);

            if (k == m_NZB / t_PEZ - 1 && j >= m_domain.m_dataCoo - m_domain.m_extCoo &&
                j < m_domain.m_dataCoo - m_domain.m_extCoo + m_domain.m_dataDim) {
                p_upb.write(l_upb);
            }

            if (k == this->m_zPE - 1 && j == this->m_y - 1) {
                k = 0;
                j = 0;
                i++;
            } else if (k == this->m_zPE - 1) {
                k = 0;
                j++;
            } else
                k++;
        }
    }

   public:
    /**
     * @brief forward defines the forward streaming module
     *
     * @param p_src is the source wavefield at given time stamp
     * @param p_v2dt2 is the pow(v * dt, 2)
     * @param p_vt is a copy of p_v2dt2
     * @param p_pp0 is the stream of input wavefield p(t - 1)
     * @param p_pp1 is the stream of output wavefield p(t)
     * @param p_p0 is the stream of input wavefield p(t)
     * @param p_p1 is the stream of output wavefield p(t + 1)
     */

    void forward(const t_DataType p_src,
                 hls::stream<t_InType>& p_v2dt2,
                 hls::stream<t_InType>& p_vt,
                 hls::stream<t_InType>& p_pp0,
                 hls::stream<t_InType>& p_p0,
                 hls::stream<t_InType>& p_pp1,
                 hls::stream<t_InType>& p_p1) {
#pragma HLS DATAFLOW
        hls::stream<t_InType> l_p0, l_pp0, l_p1;
        copy(this->m_cube, p_p0, l_p0);
        copy(this->m_cube, p_pp0, l_pp0);
        this->propagate(p_v2dt2, p_vt, l_pp0, l_p0, p_pp1, l_p1);
        addSrc(p_src, l_p1, p_p1);
    }

    /**
     * @brief forward defines the forward streaming module
     *
     * @param p_src is the source wavefield at given time stamp
     * @param p_upb is the upper boundary to be saved
     * @param p_v2dt2 is the pow(v * dt, 2)
     * @param p_vt is a copy of p_v2dt2
     * @param p_pp0 is the stream of input wavefield p(t - 1)
     * @param p_pp1 is the stream of output wavefield p(t)
     * @param p_p0 is the stream of input wavefield p(t)
     * @param p_p1 is the stream of output wavefield p(t + 1)
     */

    void forward(const t_DataType p_src,
                 hls::stream<t_UpbInType>& p_upb,
                 hls::stream<t_InType>& p_v2dt2,
                 hls::stream<t_InType>& p_vt,
                 hls::stream<t_InType>& p_pp0,
                 hls::stream<t_InType>& p_p0,
                 hls::stream<t_InType>& p_pp1,
                 hls::stream<t_InType>& p_p1) {
#pragma HLS DATAFLOW
        hls::stream<t_InType> l_p0, l_pp0, l_p1, l_upb;
        taper(p_p0, l_p0);
        taper(p_pp0, l_pp0);
        this->propagate(p_v2dt2, p_vt, l_pp0, l_p0, p_pp1, l_p1);
        addSrc(p_src, l_p1, l_upb);
        extractUPB(l_upb, p_p1, p_upb);
    }

    template <unsigned int t_NumStream>
    void saveUpb(int p_t, hls::stream<t_UpbInType> p_s[t_NumStream], t_UpbInType* p_mem) {
        int l_pSize = m_domain.m_x * m_domain.m_y / t_PEX;
        int l_n = m_domain.m_x * m_domain.m_dataDim / t_PEX;
        int l_totalSize = l_n * t_NumStream;

        int l_index_x[t_NumStream];
        int l_index_y[t_NumStream];
#pragma HLS ARRAY_PARTITION variable = l_index_x complete dim = 1
#pragma HLS ARRAY_PARTITION variable = l_index_y complete dim = 1

        for (int i = 0; i < t_NumStream; i++) {
#pragma HLS UNROLL
            l_index_x[i] = 0;
            l_index_y[i] = 0;
        }

        while (l_totalSize > 0) {
#pragma HLS PIPELINE
            t_UpbInType l_val;
            for (int i = 0; i < t_NumStream; i++) {
#pragma HLS UNROLL
                if (p_s[i].read_nb(l_val)) {
                    l_totalSize--;
                    p_mem[(p_t * t_NumStream + i) * l_pSize + l_index_x[i] * m_domain.m_y + l_index_y[i] +
                          m_domain.m_dataCoo] = l_val;

                    if (l_index_y[i] == m_domain.m_dataDim - 1) {
                        l_index_x[i]++;
                        l_index_y[i] = 0;
                    } else
                        l_index_y[i]++;

                    break;
                }
            }
        }
    }

   protected:
    int m_NXB, m_NYB, m_NZB;
    int m_srcx, m_srcy, m_srcz;
    int m_recz, m_recy, m_recx;

    t_Domain m_domain;

    t_DataTypeX m_taperx[t_MaxB / t_PEX];
    t_DataType m_tapery[t_MaxB];
    t_DataTypeZ m_taperz[t_MaxB / t_PEZ];
};
}
}
}
#endif
