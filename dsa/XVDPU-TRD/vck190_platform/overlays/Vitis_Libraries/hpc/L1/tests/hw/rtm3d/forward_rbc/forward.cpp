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

#include "forward.hpp"
#include "rtmforward.hpp"

extern "C" void top(const bool p_sel,
                    const unsigned int p_z,
                    const unsigned int p_y,
                    const unsigned int p_x,
                    const unsigned int p_t,
                    const unsigned int p_srcz,
                    const unsigned int p_srcy,
                    const unsigned int p_srcx,
                    const DATATYPE p_src[NTime],
                    const DATATYPE p_coefz[ORDER + 1],
                    const DATATYPE p_coefy[ORDER + 1],
                    const DATATYPE p_coefx[ORDER + 1],
                    const IN_TYPE p_v2dt2[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_p0[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_p1[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_po0[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_po1[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_pp0[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_pp1[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_ppo0[RTM_x * RTM_y * RTM_z / nPE / nPE],
                    IN_TYPE p_ppo1[RTM_x * RTM_y * RTM_z / nPE / nPE]) {
    RTM_TYPE l_s[NUM_INST];
#pragma HLS ARRAY_PARTITION variable = l_s complete dim = 1
    DATATYPE l_src[NUM_INST];
#pragma HLS ARRAY_PARTITION variable = l_src complete dim = 1

    DOMAIN_TYPE l_domain(p_x, p_y, p_z);
    for (int i = 0; i < NUM_INST; i++) {
        l_s[i].setCoef(p_coefz, p_coefy, p_coefx);
        l_s[i].setSrc(p_srcz, p_srcy, p_srcx);
    }
    bool cont = l_domain.reset();
    while (cont) {
        for (int i = 0; i < NUM_INST; i++) {
            l_s[i].setDomain(l_domain);
            l_s[i].setDim(l_domain.m_z, l_domain.m_extDim, l_domain.m_x);
            l_src[i] = p_src[p_t * NUM_INST + i];
        }
        forward<NUM_INST, DATATYPE, IN_TYPE>(p_sel, l_domain, l_s, l_src, p_v2dt2, p_p0, p_p1, p_po0, p_po1, p_pp0,
                                             p_pp1, p_ppo0, p_ppo1);
        cont = l_domain.next();
    }
}
