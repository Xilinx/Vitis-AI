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

#ifndef XF_HPC_RTM_RTM2D_HPP
#define XF_HPC_RTM_RTM2D_HPP

/**
 * @file rtm.hpp
 * @brief class RTM2D derived from Stencil2D is defined here, it provides L1 primary
 * streaming modules for 2D-RTM2D kernels
 */

namespace xf {
namespace hpc {
namespace rtm {

/**
 * @brief RTM2D class defines the basic operations for 2D RTM2D
 *
 * @tparam t_DataType the basic wavefield datatype
 * @tparam t_Order  the spatial discretization order
 * @tparam t_MaxDim  the maximum height this kernel can process
 * @tparam t_PE  the number of processing elements
 */

template <typename t_DataType, int t_Order, int t_MaxDim = 1024, int t_MaxB = 40, int t_PE = t_Order / 2>
class RTM2D : public Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE> {
   public:
    RTM2D(int p_NZB = t_MaxB, int p_NXB = t_MaxB) {
#ifndef __SYNTHESIS__
        assert(t_MaxB >= p_NXB);
        assert(t_MaxB >= p_NZB);
        assert(p_NXB % t_PE == 0);
        assert(p_NZB % t_PE == 0);
#endif
        m_NXB = p_NXB;
        m_NZB = p_NZB;
    }
    using Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE>::t_NumData;
    using Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE>::t_FifoDepth;
    using Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE>::t_HalfOrder;
    typedef typename Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE>::t_PairType t_PairType;
    typedef typename Stencil2D<t_DataType, t_Order, t_MaxDim, t_PE>::t_WideType t_WideType;
    typedef typename t_WideType::t_TypeInt t_InType;
    typedef typename t_PairType::t_TypeInt t_PairInType;

    typedef blas::WideType<t_InType, t_HalfOrder / t_PE> t_UpbType;
    typedef typename t_UpbType::t_TypeInt t_UpbInType;

    void setBoundaryDim(int p_NZB, int p_NXB) {
#ifndef __SYNTHESIS__
        assert(this->m_x >= 2 * p_NXB);
        assert(this->m_z >= 2 * p_NZB);
        assert(t_MaxB >= p_NXB);
        assert(t_MaxB >= p_NZB);
        assert(p_NXB % t_PE == 0);
        assert(p_NZB % t_PE == 0);
#endif
        m_NXB = p_NXB;
        m_NZB = p_NZB;
    }

    inline int getZB() {
#pragma HLS INLINE
        return m_NZB;
    }
    inline int getXB() {
#pragma HLS INLINE
        return m_NXB;
    }
    void setTaper(const t_DataType* p_taperz, const t_DataType* p_taperx) {
        for (int i = 0; i < m_NXB; i++) {
#pragma HLS PIPELINE
            m_taperx[i] = p_taperx[i];
        }
        for (int i = 0; i < m_NZB / t_PE; i++) {
            t_WideType l_w;
            for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS PIPELINE
                l_w.unshift(p_taperz[i * t_PE + pe]);
            }
            m_taperz[i] = l_w;
        }
    }
    void setSrc(int p_srcz, int p_srcx) {
        m_srcz = p_srcz;
        m_srcx = p_srcx;
    }

    void setReceiver(int p_recz) { m_recz = p_recz; }

   private:
    /**
     * @brief taper function applys to the absorbing boundary
     *
     * @param p_in  the stream of input wavefield
     * @param p_out  the stream of output wavefield
     */
    void taper(hls::stream<t_PairInType>& p_in, hls::stream<t_PairInType>& p_out) {
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_PairType l_in = p_in.read();
            t_PairType l_out;
            t_DataType tx;
            t_WideType tz;

            if (j < m_NZB / t_PE) {
                tz = m_taperz[j];
                if (i < m_NXB)
                    tx = m_taperx[i];
                else if (i >= this->m_x - m_NXB)
                    tx = m_taperx[this->m_x - i - 1];
                else
                    tx = 1;
            } else {
                tz = 1;
                tx = 1;
            }

            for (int k = 0; k < t_NumData; k++) {
#pragma HLS UNROLL
                t_WideType l_w = l_in[k];
                t_WideType l_wo;
                for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS UNROLL
                    l_wo[pe] = l_w[pe] * tz[pe] * tx;
                }
                l_out[k] = l_wo;
            }
            p_out.write(l_out);
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

    void extractUPB(hls::stream<t_PairInType>& p_pin,
                    hls::stream<t_PairInType>& p_pout,
                    hls::stream<t_UpbInType>& p_upb) {
        t_UpbType l_upb;
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_PairType l_val = p_pin.read();
            p_pout.write(l_val);
            l_upb.unshift(l_val[1]);

            if (j == m_NZB / t_PE - 1) {
                p_upb.write(l_upb);
            }
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

    void updateByUPB(hls::stream<t_UpbInType>& p_upb,
                     hls::stream<t_PairInType>& p_p0,
                     hls::stream<t_PairInType>& p_p1,
                     hls::stream<t_InType>& p_p) {
        t_UpbType l_upb;
        const unsigned int l_zbPE = m_NZB / t_PE;
        const unsigned int l_hPE = t_HalfOrder / t_PE;
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_PairType l_val = p_p0.read();
            if (j == l_zbPE - l_hPE - 1) l_upb = p_upb.read();

            if (j >= l_zbPE - l_hPE && j < l_zbPE) l_val[1] = l_upb.unshift();

            p_p1.write(l_val);
            p_p.write(l_val[1]);
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

    void addSrc(const t_DataType p_src, hls::stream<t_PairInType>& p_p0, hls::stream<t_PairInType>& p_p1) {
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_PairType l_val = p_p0.read();
            t_PairType l_oVal;
            l_oVal[0] = l_val[0];
            t_WideType l_wo = l_val[1];
            if (i == m_srcx && j == m_srcz / t_PE) {
                l_wo[0] = ((t_WideType)l_val[1])[0] + p_src;
            }

            l_oVal[1] = l_wo;
            p_p1.write(l_oVal);
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

    void addReceiver(hls::stream<t_DataType>& p_rec,
                     hls::stream<t_PairInType>& p_r0,
                     hls::stream<t_PairInType>& p_r1,
                     hls::stream<t_InType>& p_r) {
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_PairType l_val = p_r0.read();
            t_PairType l_oVal;
            l_oVal[0] = l_val[0];
            t_WideType l_wo = l_val[1];
            if (j == m_recz / t_PE && (i >= m_NXB && i < this->m_x - m_NXB)) {
                t_DataType l_rec = p_rec.read();
                l_wo[0] = ((t_WideType)l_val[1])[0] + l_rec;
            }
            l_oVal[1] = l_wo;
            p_r1.write(l_oVal);
            p_r.write(l_wo);
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

   public:
    /**
     * @brief forward defines the forward streaming module
     *
     * @param p_src is the source wavefield at given time stamp
     * @param p_upb is the upper boundary to be saved
     * @param p_v2dt2 is the pow(v * dt, 2)
     * @param p_vt is a copy of p_v2dt2
     * @param p_p0 is the stream of input wavefield p(t-1) and p(t)
     * @param p_p1 is the stream of output wavefield p(t) and p(t+1)
     */

    void forward(const t_DataType p_src,
                 hls::stream<t_UpbInType>& p_upb,
                 hls::stream<t_InType>& p_v2dt2,
                 hls::stream<t_InType>& p_vt,
                 hls::stream<t_PairInType>& p_p0,
                 hls::stream<t_PairInType>& p_p1) {
#pragma HLS DATAFLOW
        hls::stream<t_PairInType> l_p0, l_p1, l_upb;
        taper(p_p0, l_p0);
        this->propagate(p_v2dt2, p_vt, l_p0, l_p1);
        addSrc(p_src, l_p1, l_upb);
        extractUPB(l_upb, p_p1, p_upb);
    }

    /**
     * @brief backwardF defines the backward streaming module for source wavefield
     *
     * @param p_upb  the stream of upper boundary to be restored
     * @param p_v2dt2  the pow(v * dt, 2)
     * @param p_vt  a copy of p_v2dt2
     * @param p_p0  the stream of input wavefield p(t+1) and p(t)
     * @param p_p1  the stream of output wavefield p(t) and p(t-1)
     * @param p_p  the stream of wavefield p(t-1) for cross-correlation
     * @param p_t switch to swap p(t) and p(t-1) memory
     */
    void backwardF(hls::stream<t_UpbInType>& p_upb,
                   hls::stream<t_InType>& p_v2dt2,
                   hls::stream<t_InType>& p_vt,
                   hls::stream<t_PairInType>& p_p0,
                   hls::stream<t_PairInType>& p_p1,
                   hls::stream<t_InType>& p_p,
                   bool p_t = false) {
#pragma HLS DATAFLOW
        hls::stream<t_PairInType> l_p0;
        this->propagate(p_v2dt2, p_vt, p_p0, l_p0, p_t);
        updateByUPB(p_upb, l_p0, p_p1, p_p);
    }

    /**
     * @brief backwardR defines the backward streaming module for receiver wavefield
     *
     * @param p_rec  the stream of the data from receivers
     * @param p_v2dt2  the pow(v * dt, 2)
     * @param p_vt  a copy of p_v2dt2
     * @param p_r0  the stream of input wavefield r(t+1) and r(t)
     * @param p_r1  the stream of output wavefield r(t) and r(t-1)
     * @param p_r  the stream of wavefield r(t-1) for cross-correlation
     */
    void backwardR(hls::stream<t_DataType>& p_rec,
                   hls::stream<t_InType>& p_v2dt2,
                   hls::stream<t_InType>& p_vt,
                   hls::stream<t_PairInType>& p_r0,
                   hls::stream<t_PairInType>& p_r1,
                   hls::stream<t_InType>& p_r) {
#pragma HLS DATAFLOW
        hls::stream<t_PairInType> l_r0, l_r1;
        taper(p_r0, l_r0);
        this->propagate(p_v2dt2, p_vt, l_r0, l_r1);
        addReceiver(p_rec, l_r1, p_r1, p_r);
    }

    /**
     * @brief crossCorrelation constructs the detecting area images
     *
     * @param p_p  the stream of wavefield p(t) for cross-correlation
     * @param p_r  the stream of wavefield r(t) for cross-correlation
     * @param p_imgIn  the stream of input image
     * @param p_imgOut  the stream of output image
     */
    void crossCorrelation(hls::stream<t_InType>& p_p,
                          hls::stream<t_InType>& p_r,
                          hls::stream<t_InType>& p_imgIn,
                          hls::stream<t_InType>& p_imgOut) {
        for (int i = 0, j = 0, t = 0; t < this->m_area; t++) {
#pragma HLS PIPELINE
            t_WideType l_p = p_p.read(), l_r = p_r.read();
            if (i >= m_NXB && i < this->m_x - m_NXB && j >= m_NZB / t_PE && j < this->m_zPE - m_NZB / t_PE) {
                t_WideType l_i = p_imgIn.read(), l_o;
                for (int pe = 0; pe < t_PE; pe++)
#pragma HLS UNROLL
                    l_o[pe] = l_i[pe] + l_p[pe] * l_r[pe];
                p_imgOut.write(l_o);
            }
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
            } else
                j++;
        }
    }

    template <unsigned int t_NumStream, typename T>
    static void loadUpb(unsigned int p_n, int p_t, hls::stream<T> p_s[t_NumStream], const T* p_mem) {
#pragma HLS stream variable = p_s depth = t_Order
        int l_totalSize = p_n * t_NumStream;
        int l_index[t_NumStream];
#pragma HLS ARRAY_PARTITION variable = l_index complete dim = 1
        for (int i = 0; i < t_NumStream; i++)
#pragma HLS UNROLL
            l_index[i] = 0;

        while (l_totalSize > 0) {
#pragma HLS PIPELINE
            unsigned int l_channel = 0;
            bool l_read = false;
            for (int i = 0; i < t_NumStream; i++) {
#pragma HLS UNROLL
                if (!p_s[i].full() && l_index[i] < p_n) {
                    l_channel = i;
                    l_read = true;
                    break;
                }
            }
            if (l_read) {
                T l_val = p_mem[((1 + p_t) * t_NumStream - 1 - l_channel) * p_n + l_index[l_channel]];
                if (p_s[l_channel].write_nb(l_val)) {
                    l_index[l_channel]++;
                    l_totalSize--;
                }
            }
        }
    }

    template <unsigned int t_NumStream, typename T>
    static void saveUpb(unsigned int p_n, int p_t, hls::stream<T> p_s[t_NumStream], T* p_mem) {
        int l_totalSize = p_n * t_NumStream;
        int l_index[t_NumStream];
#pragma HLS ARRAY_PARTITION variable = l_index complete dim = 1
        for (int i = 0; i < t_NumStream; i++)
#pragma HLS UNROLL
            l_index[i] = 0;

        while (l_totalSize > 0) {
#pragma HLS PIPELINE
            T l_val;
            for (int i = 0; i < t_NumStream; i++) {
#pragma HLS UNROLL
                if (p_s[i].read_nb(l_val)) {
                    l_totalSize--;
                    p_mem[p_t * p_n * t_NumStream + i * p_n + l_index[i]] = l_val;
                    l_index[i]++;
                    break;
                }
            }
        }
    }

   protected:
    int m_NXB, m_NZB;
    int m_srcx, m_srcz;
    int m_recz;

    t_DataType m_taperx[t_MaxB];
    t_InType m_taperz[t_MaxB / t_PE];
};
}
}
}
#endif
