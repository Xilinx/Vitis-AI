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
#ifndef XF_HPC_RTM_DOAMIN3D_HPP
#define XF_HPC_RTM_DOAMIN3D_HPP
namespace xf {
namespace hpc {
namespace rtm {
#define max(x, y) (x) > (y) ? (x) : (y)
#define min(x, y) (x) > (y) ? (y) : (x)

/**
 * @brief Domain3D class  used to partition the pressure field to maximize the
 * kernel performance
 *
 * @tparam t_HalfOrder  the half spatial discretization order
 * @tparam t_MaxDimZ  the maximum dim along z-axis the kernel can process
 * @tparam t_MaxDimY  the maximum dim along y-axis the kernel can process
 * @tparam t_PEZ  the number of processing elements along z-axis
 * @tparam t_PEX  the number of processing elements along x-axis
 * @tparam t_NumSM  the number of streaming modules in the kernel
 */

template <int t_MaxDimZ, int t_MaxDimY, int t_HalfOrder, int t_PEZ, int t_PEX, int t_NumSM>
class Domain3D {
   public:
    static const int s_DimExt = t_NumSM * t_HalfOrder;
    unsigned int m_x, m_y, m_z;

    unsigned int m_dataCoo; // partition coordinates
    unsigned int m_dataDim; // partition dimensions

    unsigned int m_extCoo; // partition valid coordinates
    unsigned int m_extDim; // partition valid coordinates

    Domain3D(int p_x = 0, int p_y = 0, int p_z = 0) {
#ifndef __SYNTHESIS__
        assert(p_z <= t_MaxDimZ);
        assert(p_x % t_PEX == 0);
        assert(p_z % t_PEZ == 0);
#endif
        m_z = p_z;
        m_y = p_y;
        m_x = p_x;
    }

    void extendCoo(const unsigned int& p_coor, unsigned int& p_dim, unsigned int& p_extCo, unsigned int& p_extDim) {
        int l_y = p_coor - s_DimExt;

        p_extCo = max(0, l_y);

        l_y = min(p_extCo + t_MaxDimY, m_y);

        p_extDim = l_y - p_extCo;

        if (l_y == m_y)
            p_dim = l_y - p_coor;
        else
            p_dim = l_y - s_DimExt - p_coor;
    }

    void extendCoo() { extendCoo(m_dataCoo, m_dataDim, m_extCoo, m_extDim); }

    bool reset() {
        m_dataCoo = 0;
        extendCoo();
        return true;
    }

    bool next() {
        m_dataCoo += m_dataDim;
        if (m_dataCoo >= m_y) return false;
        extendCoo();
        return true;
    }

    /**
     * @brief mem2stream reads data memory to a stream
     *
     * @tparam t_InterfaceType  the datatype in memory
     *
     * @param p_mem  the first memory port
     * @param p_str  the output stream
     */
    template <typename t_InterfaceType>
    void mem2stream(const t_InterfaceType* p_mem, hls::stream<t_InterfaceType>& p_str) const {
        for (int i = 0; i < m_x / t_PEX; i++) {
            for (int k = 0; k < m_extDim * m_z / t_PEZ; k++) {
#pragma HLS PIPELINE
                int l_baseAddr = (i * m_y * m_z + m_extCoo * m_z) / t_PEZ;
                t_InterfaceType l_in = p_mem[k + l_baseAddr];
                p_str.write(l_in);
            }
        }
    }

    /**
     * @brief stream2mem reads write alternatively to two memory addresses from a stream
     *
     * @tparam t_InterfaceType  the datatype in memory
     *
     * @param p_mem  the first memory port
     * @param p_str  the input stream
     */
    template <typename t_InterfaceType>
    void stream2mem(hls::stream<t_InterfaceType>& p_str, t_InterfaceType* p_mem) const {
        for (int i = 0; i < m_x / t_PEX; i++) {
            for (int j = 0; j < m_extDim; j++) {
                for (int k = 0; k < m_z / t_PEZ; k++) {
#pragma HLS PIPELINE
                    t_InterfaceType l_data = p_str.read();
                    int l_yCoo = j + m_extCoo;
                    bool l_active = (l_yCoo >= m_dataCoo) && (l_yCoo < m_dataCoo + m_dataDim);
                    int l_baseAddr = (i * m_y * m_z + l_yCoo * m_z) / t_PEZ;
                    if (l_active) {
                        p_mem[k + l_baseAddr] = l_data;
                    }
                }
            }
        }
    }

    /**
     * @brief mem2stream reads data memory to a stream
     *
     * @tparam t_InterfaceType  the datatype in memory
     *
     * @param p_sel  the signal to select read port
     * @param p_mem0  the first memory port
     * @param p_mem1  the second memory port
     * @param p_str  the output stream
     */
    template <typename t_InterfaceType>
    void memSelStream(const bool p_sel,
                      const t_InterfaceType* p_mem0,
                      const t_InterfaceType* p_mem1,
                      hls::stream<t_InterfaceType>& p_str) const {
        if (p_sel)
            for (int i = 0; i < m_x / t_PEX; i++) {
                for (int k = 0; k < m_extDim * m_z / t_PEZ; k++) {
#pragma HLS PIPELINE
                    int l_baseAddr = (i * m_y * m_z + m_extCoo * m_z) / t_PEZ;
                    t_InterfaceType l_in = p_mem0[k + l_baseAddr];
                    //                    std::cout << "Address is " << k+ l_baseAddr << std:: endl;
                    p_str.write(l_in);
                }
            }
        else
            for (int i = 0; i < m_x / t_PEX; i++) {
                for (int k = 0; k < m_extDim * m_z / t_PEZ; k++) {
#pragma HLS PIPELINE
                    int l_baseAddr = (i * m_y * m_z + m_extCoo * m_z) / t_PEZ;
                    t_InterfaceType l_in = p_mem1[k + l_baseAddr];
                    //                   std::cout << "Address is " << k+ l_baseAddr << std:: endl;
                    p_str.write(l_in);
                }
            }
    }

    /**
     * @brief stream2mem reads write alternatively to two memory addresses from a stream
     *
     * @tparam t_InterfaceType  the datatype in memory *
     * @param p_sel  the signal to select read port
     * @param p_mem0  the first memory port
     * @param p_mem1  the second memory port
     * @param p_str  the input stream
     */
    template <typename t_InterfaceType>
    void streamSelMem(const bool p_sel,
                      hls::stream<t_InterfaceType>& p_str,
                      t_InterfaceType* p_mem0,
                      t_InterfaceType* p_mem1) const {
        if (p_sel)
            for (int i = 0; i < m_x / t_PEX; i++) {
                for (int j = 0; j < m_dataCoo - m_extCoo; j++) {
                    for (int k = 0; k < m_z / t_PEZ; k++)
#pragma HLS PIPELINE
                        p_str.read();
                }
                for (int j = 0; j < m_dataDim * m_z / t_PEZ; j++) {
#pragma HLS PIPELINE
                    t_InterfaceType l_data = p_str.read();
                    int l_baseAddr = (i * m_y * m_z + m_dataCoo * m_z) / t_PEZ;
                    p_mem0[j + l_baseAddr] = l_data;
                }
                for (int j = 0; j < m_extDim - m_dataDim - m_dataCoo + m_extCoo; j++) {
                    for (int k = 0; k < m_z / t_PEZ; k++)
#pragma HLS PIPELINE
                        p_str.read();
                }
            }
        else
            for (int i = 0; i < m_x / t_PEX; i++) {
                for (int j = 0; j < m_dataCoo - m_extCoo; j++) {
                    for (int k = 0; k < m_z / t_PEZ; k++)
#pragma HLS PIPELINE
                        p_str.read();
                }
                for (int j = 0; j < m_dataDim * m_z / t_PEZ; j++) {
#pragma HLS PIPELINE
                    t_InterfaceType l_data = p_str.read();
                    int l_baseAddr = (i * m_y * m_z + m_dataCoo * m_z) / t_PEZ;
                    p_mem1[j + l_baseAddr] = l_data;
                    //                    std::cout << "Address is " << j + l_baseAddr << std::endl;
                }
                for (int j = 0; j < m_extDim - m_dataDim - m_dataCoo + m_extCoo; j++) {
                    for (int k = 0; k < m_z / t_PEZ; k++)
#pragma HLS PIPELINE
                        p_str.read();
                }
            }
    }
};
#undef max
#undef min
}
}
}
#endif
