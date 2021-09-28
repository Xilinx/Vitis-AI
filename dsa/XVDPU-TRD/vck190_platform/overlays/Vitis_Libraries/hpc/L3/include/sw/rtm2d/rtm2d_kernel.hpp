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

#include <vector>
#include <chrono>

#include "types.hpp"
#include "fpga_xrt.hpp"

using namespace std;

/**
 * @brief class ForwardKernel is used to manage and run forward kernel on FPGA
 */
template <typename t_DataType, unsigned int t_Order, unsigned int t_PE>
class ForwardKernel : public XKernel {
   public:
    typedef WideData<t_DataType, t_PE> t_WideType;
    typedef WideData<t_WideType, 2> t_PairType;
    static const unsigned int _4k = 4096;

    ForwardKernel(FPGA* fpga,
                  unsigned int p_height,
                  unsigned int p_width,
                  unsigned int p_zb,
                  unsigned int p_xb,
                  unsigned int p_time,
                  unsigned int p_shots)
        : XKernel(fpga) {
        static const string t_KernelName = "rtmforward";
        getCU(t_KernelName);
        m_height = p_height;
        m_width = p_width;
        m_zb = p_zb;
        m_xb = p_xb;
        m_time = p_time;
        m_shots = p_shots;

        m_taperx = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_xb);
        m_taperz = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_zb);
        m_v2dt2 = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_width * m_height);
        m_srcs = new t_DataType*[m_shots];
        for (unsigned int s = 0; s < m_shots; s++)
            m_srcs[s] = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_time);

        m_coefx = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * (t_Order + 1));
        m_coefz = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * (t_Order + 1));
    }

    ~ForwardKernel() {
        free(m_coefx);
        free(m_coefz);
        free(m_taperx);
        free(m_taperz);
        free(m_v2dt2);
        for (unsigned int s = 0; s < m_shots; s++) free(m_srcs[s]);
        delete[] m_srcs;
    }

    /**
     * @brief loadData load parameters for RTM forward kernel from given path
     *
     * @param filePath the path where parameters are stored
     *
     */
    void loadData(const string filePath) {
        readBin(filePath + "v2dt2.bin", sizeof(t_DataType) * m_width * m_height, m_v2dt2);
        for (unsigned int s = 0; s < m_shots; s++)
            readBin(filePath + "src_s" + to_string(s) + ".bin", sizeof(t_DataType) * m_time, m_srcs[s]);
        readBin(filePath + "taperx.bin", sizeof(t_DataType) * m_xb, m_taperx);
        readBin(filePath + "taperz.bin", sizeof(t_DataType) * m_zb, m_taperz);
        readBin(filePath + "coefx.bin", sizeof(t_DataType) * (t_Order + 1), m_coefx);
        readBin(filePath + "coefz.bin", sizeof(t_DataType) * (t_Order + 1), m_coefz);
    }

    /**
     * @brief run launch the RTM forward kernel with given input parameters
     *
     * @param p_shot the shot id
     * @param p_sx the shot coordinate
     * @param p_p seismic snapshot
     * @param p_upb the upper boundary data
     *
     */
    double run(unsigned int p_shot,
               unsigned int p_sx,
               host_buffer_t<t_PairType>& p_p,
               host_buffer_t<t_DataType>& p_upb) {
        p_p.resize(m_width * m_height / t_PE);
        p_upb.resize(m_width * t_Order * m_time / 2);

        xrt::bo d_srcs = xrt::bo(m_fpga->m_device, m_srcs[p_shot], sizeof(t_DataType) * m_time, 32);
        xrt::bo d_coefz = xrt::bo(m_fpga->m_device, m_coefz, sizeof(t_DataType) * (t_Order + 1), 32);
        xrt::bo d_coefx = xrt::bo(m_fpga->m_device, m_coefx, sizeof(t_DataType) * (t_Order + 1), 32);
        xrt::bo d_taperz = xrt::bo(m_fpga->m_device, m_taperz, sizeof(t_DataType) * m_zb, 32);
        xrt::bo d_taperx = xrt::bo(m_fpga->m_device, m_taperx, sizeof(t_DataType) * m_xb, 32);

        xrt::bo d_v2dt2 = xrt::bo(m_fpga->m_device, m_v2dt2, sizeof(t_DataType) * m_width * m_height, 7);

        xrt::bo d_p0 = xrt::bo(m_fpga->m_device, p_p.data(), sizeof(t_PairType) * m_width * m_height / t_PE, 8);

        xrt::bo d_upb =
            xrt::bo(m_fpga->m_device, p_upb.data(), sizeof(t_DataType) * m_width * t_Order * m_time / 2, 32);

        m_fpga->copyToFpga(d_srcs);
        m_fpga->copyToFpga(d_coefz);
        m_fpga->copyToFpga(d_coefx);
        m_fpga->copyToFpga(d_taperz);
        m_fpga->copyToFpga(d_taperx);
        m_fpga->copyToFpga(d_v2dt2);
        m_fpga->copyToFpga(d_p0);
        m_fpga->copyToFpga(d_upb);

        m_run = m_kernel(m_height, m_width, m_time, m_zb, p_sx, d_srcs, d_coefz, d_coefx, d_taperz, d_taperx, d_v2dt2,
                         d_p0, d_p0, d_upb);

        auto start = chrono::high_resolution_clock::now();

        m_run.wait();

        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;

        m_fpga->copyFromFpga(d_p0);
        m_fpga->copyFromFpga(d_upb);

        return elapsed.count();
    }

   private:
    unsigned int m_width, m_height, m_xb, m_zb, m_time, m_shots;
    t_DataType *m_taperx, *m_taperz;
    t_DataType *m_v2dt2, **m_srcs;
    t_DataType *m_coefx, *m_coefz;
};

/**
 * @brief class BackwardKernel is used to manage and run backward kernel on FPGA
 */
template <typename t_DataType, unsigned int t_Order, unsigned int t_PE>
class BackwardKernel : public XKernel {
   public:
    typedef WideData<t_DataType, t_PE> t_WideType;
    typedef WideData<t_WideType, 2> t_PairType;
    static const unsigned int _4k = 4096;

    BackwardKernel(FPGA* fpga,
                   unsigned int p_height,
                   unsigned int p_width,
                   unsigned int p_zb,
                   unsigned int p_xb,
                   unsigned int p_time,
                   unsigned int p_shots)
        : XKernel(fpga) {
        static const string t_KernelName = "rtmbackward";
        getCU(t_KernelName);

        m_height = p_height;
        m_width = p_width;
        m_zb = p_zb;
        m_xb = p_xb;
        m_time = p_time;
        m_shots = p_shots;

        m_imgX = m_width - 2 * m_xb;
        m_imgZ = m_height - 2 * m_zb;

        m_taperx = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_xb);
        m_taperz = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_zb);
        m_v2dt2 = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_width * m_height);
        m_receiver = new t_DataType*[m_shots];
        for (unsigned int i = 0; i < m_shots; i++)
            m_receiver[i] = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * m_imgX * m_time);

        m_coefx = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * (t_Order + 1));
        m_coefz = (t_DataType*)aligned_alloc(_4k, sizeof(t_DataType) * (t_Order + 1));
    }

    ~BackwardKernel() {
        free(m_taperx);
        free(m_taperz);
        free(m_v2dt2);
        for (unsigned int i = 0; i < m_shots; i++) free(m_receiver[i]);
        delete[] m_receiver;
        free(m_coefx);
        free(m_coefz);
    }

    /**
     * @brief loadData load parameters for RTM backward kernel from given path
     *
     * @param filePath the path where parameters are stored
     *
     */
    void loadData(const string filePath) {
        readBin(filePath + "v2dt2.bin", sizeof(t_DataType) * m_width * m_height, m_v2dt2);
        for (unsigned int i = 0; i < m_shots; i++)
            readBin(filePath + "sensor_s" + to_string(i) + ".bin", sizeof(t_DataType) * m_imgX * m_time, m_receiver[i]);
        readBin(filePath + "taperx.bin", sizeof(t_DataType) * m_xb, m_taperx);
        readBin(filePath + "taperz.bin", sizeof(t_DataType) * m_zb, m_taperz);
        readBin(filePath + "coefx.bin", sizeof(t_DataType) * (t_Order + 1), m_coefx);
        readBin(filePath + "coefz.bin", sizeof(t_DataType) * (t_Order + 1), m_coefz);
    }

    /**
     * @brief run launch the RTM backward kernel with given input parameters
     *
     * @param p_shot the shot id
     * @param p_upb the upper boundary data
     * @param p_snaps input seismic snapshot
     * @param p_p output seismic source wavefiled
     * @param p_r output seismic receiver wavefiled
     * @param p_i output seismic image
     *
     */
    double run(unsigned int p_shot,
               host_buffer_t<t_PairType>& p_snaps,
               host_buffer_t<t_DataType>& p_upb,
               host_buffer_t<t_PairType>& p_p,
               host_buffer_t<t_PairType>& p_r,
               host_buffer_t<t_DataType>& p_i) {
        xrt::bo d_coefx = xrt::bo(m_fpga->m_device, m_coefx, sizeof(t_DataType) * (t_Order + 1), 33);
        xrt::bo d_coefz = xrt::bo(m_fpga->m_device, m_coefz, sizeof(t_DataType) * (t_Order + 1), 33);
        xrt::bo d_taperx = xrt::bo(m_fpga->m_device, m_taperx, sizeof(t_DataType) * m_xb, 33);
        xrt::bo d_taperz = xrt::bo(m_fpga->m_device, m_taperz, sizeof(t_DataType) * m_zb, 33);
        xrt::bo d_v2dt2 = xrt::bo(m_fpga->m_device, m_v2dt2, sizeof(t_DataType) * m_width * m_height, 3);

        p_p.resize(m_width * m_height / t_PE);
        p_r.resize(m_width * m_height / t_PE);
        if (p_i.size() == 0) p_i.resize(m_imgX * m_imgZ);

        for (unsigned int i = 0; i < m_width * m_height / t_PE; i++) {
            p_p[i][0] = p_snaps[i][1];
            p_p[i][1] = p_snaps[i][0];
        }

        xrt::bo d_receiver = xrt::bo(m_fpga->m_device, m_receiver[p_shot], sizeof(t_DataType) * m_time * m_imgX, 33);
        xrt::bo d_upb =
            xrt::bo(m_fpga->m_device, p_upb.data(), sizeof(t_DataType) * m_width * t_Order * m_time / 2, 33);
        xrt::bo d_p0 = xrt::bo(m_fpga->m_device, p_p.data(), sizeof(t_PairType) * m_width * m_height / t_PE, 0);
        xrt::bo d_r0 = xrt::bo(m_fpga->m_device, p_r.data(), sizeof(t_PairType) * m_width * m_height / t_PE, 1);
        xrt::bo d_i0 = xrt::bo(m_fpga->m_device, p_i.data(), sizeof(RTM_dataType) * m_imgX * m_imgZ, 2);

        m_fpga->copyToFpga(d_coefx);
        m_fpga->copyToFpga(d_coefz);
        m_fpga->copyToFpga(d_taperx);
        m_fpga->copyToFpga(d_taperz);
        m_fpga->copyToFpga(d_v2dt2);
        m_fpga->copyToFpga(d_p0);
        m_fpga->copyToFpga(d_upb);
        m_fpga->copyToFpga(d_receiver);
        m_fpga->copyToFpga(d_r0);
        m_fpga->copyToFpga(d_i0);

        m_run = m_kernel(m_height, m_width, m_time, m_zb, d_receiver, d_coefz, d_coefx, d_taperz, d_taperx, d_v2dt2,
                         d_p0, d_p0, d_r0, d_r0, d_i0, d_i0, d_upb);

        auto start = chrono::high_resolution_clock::now();

        m_run.wait();
        auto finish = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = finish - start;

        m_fpga->copyFromFpga(d_i0);
        m_fpga->copyFromFpga(d_p0);
        m_fpga->copyFromFpga(d_r0);

        return elapsed.count();
    }

   private:
    unsigned int m_width, m_height, m_xb, m_zb, m_time, m_shots, m_imgX, m_imgZ;

    t_DataType *m_taperx, *m_taperz;
    t_DataType *m_v2dt2, **m_receiver;
    t_DataType *m_coefx, *m_coefz;
};
