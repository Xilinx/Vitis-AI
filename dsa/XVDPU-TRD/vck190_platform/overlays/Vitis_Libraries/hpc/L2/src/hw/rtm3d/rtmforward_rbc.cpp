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
 * @file rtmforward.cpp
 * @brief It defines the forward kernel function
 */

#include "rtmforward_rbc.hpp"

extern "C" void rtmforward(const unsigned int p_z,
                           const unsigned int p_y,
                           const unsigned int p_x,
                           const unsigned int p_t,
                           const unsigned int p_srcz,
                           const unsigned int p_srcy,
                           const unsigned int p_srcx,
                           const RTM_dataType* p_src,
                           const RTM_dataType* p_coefz,
                           const RTM_dataType* p_coefy,
                           const RTM_dataType* p_coefx,
                           const RTM_type* p_v2dt2,
                           RTM_type* p_pi0,
                           RTM_type* p_pi1,
                           RTM_type* p_po0,
                           RTM_type* p_po1,
                           RTM_type* p_ppi0,
                           RTM_type* p_ppi1,
                           RTM_type* p_ppo0,
                           RTM_type* p_ppo1) {
    SCALAR(p_x)
    SCALAR(p_y)
    SCALAR(p_z)
    SCALAR(p_t)
    SCALAR(p_srcx)
    SCALAR(p_srcy)
    SCALAR(p_srcz)
    SCALAR(return )

    POINTER(p_coefx, gmemParam)
    POINTER(p_coefy, gmemParam)
    POINTER(p_coefz, gmemParam)
    POINTER(p_src, gmemParam)

    POINTER(p_pi0, gmem_pi0)
    POINTER(p_pi1, gmem_pi1)
    POINTER(p_po0, gmem_po0)
    POINTER(p_po1, gmem_po1)

    POINTER(p_ppi0, gmem_ppi0)
    POINTER(p_ppi1, gmem_ppi1)
    POINTER(p_ppo0, gmem_ppo0)
    POINTER(p_ppo1, gmem_ppo1)
    POINTER(p_v2dt2, gmem_v2dt2)

    RTM_TYPE l_s[RTM_numFSMs];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    RTM_dataType l_src[RTM_numFSMs];
#pragma HLS ARRAY_PARTITION variable = l_src complete dim = 1

    DOMAIN_TYPE l_domain(p_x, p_y, p_z);

    for (int i = 0; i < RTM_numFSMs; i++) {
        l_s[i].setCoef(p_coefz, p_coefy, p_coefx);
        l_s[i].setSrc(p_srcz, p_srcy, p_srcx);
    }

    for (int t = 0; t < p_t / RTM_numFSMs; t++) {
        bool cont = l_domain.reset();
        while (cont) {
            for (int i = 0; i < RTM_numFSMs; i++) {
                l_s[i].setDomain(l_domain);
                l_s[i].setDim(l_domain.m_z, l_domain.m_extDim, l_domain.m_x);
                l_src[i] = p_src[t * RTM_numFSMs + i];
            }
            forward<RTM_numFSMs, RTM_dataType, RTM_type>(t & 0x01, l_domain, l_s, l_src, p_v2dt2, p_pi0, p_pi1, p_po0,
                                                         p_po1, p_ppi0, p_ppi1, p_ppo0, p_ppo1);
            cont = l_domain.next();
        }
    }
}
