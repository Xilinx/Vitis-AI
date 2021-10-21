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

#ifndef XF_HPC_RTM_STENCIL2D_HPP
#define XF_HPC_RTM_STENCIL2D_HPP

/**
 * @file stencil2d.hpp
 * @brief class Stencil2D is defined here, it provides L1 primary methods for
 * 2D-stencil kernels
 */

namespace xf {
namespace hpc {
namespace rtm {

/**
 * @brief Stencil2D class defines isotrophic laplacian operation and time
 * iterations function
 *
 * @tparam t_DataType the basic wavefield datatype
 * @tparam t_Order  the spatial discretization order
 * @tparam t_MaxDim  the maximum height this kernel can process
 * @tparam t_PE  the number of processing elements
 */

template <typename t_DataType, int t_Order, int t_MaxDim = 1024, int t_PE = 1>
class Stencil2D { // column major
   public:
    static const unsigned int t_NumData = 2;
    static const unsigned int t_HalfOrder = t_Order >> 1;
    static const unsigned int t_FifoDepth = t_HalfOrder * t_MaxDim / t_PE + 156;
    typedef blas::WideType<t_DataType, t_PE> t_WideType;
    typedef typename t_WideType::t_TypeInt t_InType;
    typedef blas::WideType<t_InType, t_NumData, t_WideType::t_TypeWidth> t_PairType;
    typedef typename t_PairType::t_TypeInt t_PairInType;

    Stencil2D() {
#pragma HLS ARRAY_PARTITION variable = m_coefx dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = m_coefz dim = 1 complete
    }
    /**
     * @brief setCoef set the coefficients for stencil kernel
     *
     * @param p_coefz  the coefficents along z-direction
     * @param p_coefx  the coefficents along x-direction
     */
    void setCoef(const t_DataType* p_coefz, const t_DataType* p_coefx) {
        for (int i = 0; i < t_HalfOrder + 1; i++) {
#pragma HLS PIPELINE
            m_coefx[i] = p_coefx[i];
        }
        for (int i = 0; i < t_HalfOrder + 1; i++) {
#pragma HLS PIPELINE
            m_coefz[i] = p_coefz[i];
        }
    }
    /**
     * @brief setDim set the dimension of the wavefields
     *
     * @param p_z  the dimension of z-direction
     * @param p_x  the dimension of x-direction
     */
    void setDim(const unsigned int p_z, const unsigned int p_x) {
        m_x = p_x;
        m_z = p_z;
        m_zPE = m_z / t_PE;
        m_area = m_x * m_zPE;
    }

    /**
     * @brief laplacian computes the laplacian of an given wavefield
     *
     * @param p_in  the input wavefield stream
     * @param p_pin  a copy of p_in
     * @param p_out  a stream of laplacian results
     */
    void laplacian(hls::stream<t_InType>& p_in, hls::stream<t_InType>& p_pin, hls::stream<t_InType>& p_out) {
        constexpr int l_orderPE = t_Order / t_PE;

#ifndef __SYNTHESIS__
        assert(m_z % t_PE == 0);
        assert(m_z / t_PE <= t_MaxDim / t_PE);
        assert(t_HalfOrder % t_PE == 0);
#endif

        t_InType l_lineBuffer[t_Order][t_MaxDim / t_PE];
#pragma HLS ARRAY_PARTITION variable = l_lineBuffer dim = 1 complete
#pragma HLS RESOURCE variable = l_lineBuffer core = RAM_2P_URAM

        t_DataType l_crossZ[t_Order + t_PE], l_crossX[t_Order + 1][t_HalfOrder + t_PE];
#pragma HLS ARRAY_PARTITION variable = l_crossZ dim = 1 complete
#pragma HLS ARRAY_PARTITION variable = l_crossX dim = 0 complete
        int i = 0, i_buf = 0, j = 0;
        int l_init = t_HalfOrder / t_PE + t_HalfOrder * m_zPE;
        int l_iter = this->m_area + l_init;
        for (int t = 0; t < l_iter; t++) {
#pragma HLS PIPELINE
#pragma HLS DEPENDENCE variable = l_lineBuffer array inter false
            t_InType l_in = 0;
            if (i < m_x) l_in = p_in.read();

            for (int k = 0; k < t_Order + 1; k++)
#pragma HLS UNROLL
                for (int s = 0; s < t_HalfOrder; s++)
#pragma HLS UNROLL
                    l_crossX[k][s] = l_crossX[k][s + t_PE];

            for (int pe = 0; pe < t_PE; pe++)
                for (int k = 0; k < t_Order; k++)
#pragma HLS UNROLL
                    l_crossX[k][t_HalfOrder + pe] = ((t_WideType)l_lineBuffer[(i_buf + k) % t_Order][j])[pe];

            for (int pe = 0; pe < t_PE; pe++)
#pragma HLS UNROLL
                l_crossX[t_Order][t_HalfOrder + pe] = ((t_WideType)l_in)[pe];

            for (int k = 0; k < t_Order; k++)
#pragma HLS UNROLL
                l_crossZ[k] = l_crossZ[k + t_PE];

            for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS UNROLL
                l_crossZ[t_Order + pe] = ((t_WideType)l_lineBuffer[(i_buf + t_HalfOrder) % t_Order][j])[pe];
            }
            l_lineBuffer[i_buf][j] = l_in;

            t_WideType l_sum;

            for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS UNROLL
                l_sum[pe] = l_crossZ[t_HalfOrder + pe] * m_coefz[t_HalfOrder] +
                            l_crossX[t_HalfOrder][pe] * m_coefx[t_HalfOrder];
                for (int k = 0; k < t_HalfOrder; k++) {
#pragma HLS UNROLL
                    l_sum[pe] += (l_crossZ[k + pe] + l_crossZ[t_Order - k + pe]) * m_coefz[k] +
                                 (l_crossX[k][pe] + l_crossX[t_Order - k][pe]) * m_coefx[k];
                }
            }

            t_WideType l_pin;
            for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS UNROLL
#ifndef __SYNTHESIS__
                if (t >= l_init) {
                    if (l_crossZ[t_HalfOrder + pe] != l_crossX[t_HalfOrder][pe])
                        std::cout << "Z: " << l_crossZ[t_HalfOrder + pe] << '\t' << "X: " << l_crossX[t_HalfOrder][pe]
                                  << std::endl;

                    assert(l_crossZ[t_HalfOrder + pe] == l_crossX[t_HalfOrder][pe]);
                }
#endif
                l_pin[pe] = l_crossZ[t_HalfOrder + pe];
            }

            if (i < t_Order || j < t_Order / t_PE || i >= m_x) l_sum = 0;
            if (t >= l_init) {
                p_pin.write(l_pin);
                p_out.write(l_sum);
            }
            if (j == this->m_zPE - 1) {
                j = 0;
                i++;
                i_buf++;
                i_buf = i_buf % t_Order;
            } else
                j++;
        }
    }

    void laplacian(hls::stream<t_InType>& p_in, hls::stream<t_InType>& p_out) {
#pragma HLS DATAFLOW
        hls::stream<t_InType> l_pin;
        laplacian(p_in, l_pin, p_out);
        dataConsumer(m_area, l_pin);
    }

    void split(hls::stream<t_PairInType>& p_in, hls::stream<t_InType>& p_out0, hls::stream<t_InType>& p_out1) {
        for (int i = 0; i < m_area; i++) {
#pragma HLS PIPELINE
            t_PairType l_in = p_in.read();
            p_out0.write(l_in[0]);
            p_out1.write(l_in[1]);
        }
    }

    void calculate(hls::stream<t_InType>& p_p0,
                   hls::stream<t_InType>& p_p1,
                   hls::stream<t_InType>& p_lap,
                   hls::stream<t_InType>& p_v2dt2,
                   hls::stream<t_InType>& p_cpvt,
                   hls::stream<t_PairInType>& p_out,
                   bool p_sw = false) {
        if (p_sw) {
            for (int i = 0; i < m_area; i++) {
#pragma HLS PIPELINE
                t_PairType l_out;
                t_InType l_vt = p_v2dt2.read();
                t_InType s_p0 = p_p0.read();
                t_InType s_p1 = p_p1.read();
                t_InType s_lap = p_lap.read();
                l_out[0] = s_p1;
                l_out[1] = s_p0;
                p_out.write(l_out);
                p_cpvt.write(l_vt);
            }
        } else {
            for (int i = 0; i < m_area; i++) {
#pragma HLS PIPELINE
                t_PairType l_out;
                t_WideType l_w;
                t_WideType l_vt = p_v2dt2.read();
                t_WideType s_p0 = p_p0.read();
                t_WideType s_p1 = p_p1.read();
                t_WideType s_lap = p_lap.read();
                l_out[0] = s_p1;
                for (int pe = 0; pe < t_PE; pe++) {
#pragma HLS UNROLL
                    l_w[pe] = 2 * s_p1[pe] - s_p0[pe] + l_vt[pe] * s_lap[pe];
                }
                l_out[1] = l_w;
                p_out.write(l_out);
                p_cpvt.write(l_vt);
            }
        }
    }

    /**
     * @brief propagate computes the time iteration for FDTD
     *
     * @param p_v2dt2  the pow(v * dt, 2)
     * @param p_cpvt  a copy of p_v2dt2
     * @param p_in  a stream of input wavefield
     * @param p_out  a stream of output wavefield
     * @param p_sw  a switch to swap outptu streams
     */
    void propagate(hls::stream<t_InType>& p_v2dt2,
                   hls::stream<t_InType>& p_cpvt,
                   hls::stream<t_PairInType>& p_in,
                   hls::stream<t_PairInType>& p_out,
                   bool p_sw = false) {
#pragma HLS DATAFLOW
        hls::stream<t_InType> l_p0, l_p1, l_ps, l_lap;

#pragma HLS stream variable = l_p0 depth = t_FifoDepth

        split(p_in, l_p0, l_p1);
        laplacian(l_p1, l_ps, l_lap);
        calculate(l_p0, l_ps, l_lap, p_v2dt2, p_cpvt, p_out, p_sw);
    }

    inline unsigned int getX() const {
#pragma HLS INLINE
        return m_x;
    }

    inline unsigned int getZ() const {
#pragma HLS INLINE
        return m_z;
    }

    inline unsigned int getArea() const {
#pragma HLS INLINE
        return m_area;
    }

   protected:
    t_DataType m_coefz[t_HalfOrder + 1];
    t_DataType m_coefx[t_HalfOrder + 1];

    unsigned int m_x;
    unsigned int m_z;
    unsigned int m_area;
    unsigned int m_zPE;
};
}
}
}
#endif
