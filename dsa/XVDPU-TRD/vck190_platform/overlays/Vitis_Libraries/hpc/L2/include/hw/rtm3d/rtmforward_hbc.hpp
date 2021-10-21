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

#ifndef RTM_FORWARD_DOMAIN_RBC_HPP
#define RTM_FORWARD_DOMAIN_RBC_HPP

#include "interface.hpp"
#include "rtmforward.hpp"

using namespace xf::hpc::rtm;

typedef Domain3D<RTM_maxZ, RTM_maxY, RTM_order / 2, RTM_nPEZ, RTM_nPEX, RTM_numFSMs> DOMAIN_TYPE;
typedef RTM3D<DOMAIN_TYPE, RTM_dataType, RTM_order, RTM_maxZ, RTM_maxY, RTM_MaxB, RTM_nPEZ, RTM_nPEX> RTM_TYPE;

typedef typename RTM_TYPE::t_InType RTM_type;

typedef RTM_TYPE::t_UpbInType RTM_upbType;

/**
 * @brief rfmforward kernel function
 *
 * @param p_z is the number of grids along z
 * @param p_y is the number of grids along y
 * @param p_x is the number of grids along x
 * @param p_t is the number of detecting time parititons
 *
 * @param p_srcz is the source z coordinate
 * @param p_srcy is the source y coordinate
 * @param p_srcx is the source x coordinate
 * @param p_src is the source wavefiled
 *
 * @param p_coefz is the laplacian z-direction coefficients
 * @param p_coefy is the laplacian y-direction coefficients
 * @param p_coefx is the laplacian x-direction coefficients
 *
 * @param p_taperz is the absorbing factor along z
 * @param p_tapery is the absorbing factor along y
 * @param p_taperx is the absorbing factor along x
 *
 * @param p_v2dt2 is the velocity model v^2 * dt^2
 *
 * @param p_pi0  the first input memory of pressure wavefield at t-1
 * @param p_pi1  the second input memory of pressure wavefield at t-1
 *
 * @param p_po0  the first output memory of pressure wavefield at t-1
 * @param p_po1  the second output memory of pressure wavefield at t-1
 *
 * @param p_ppi0  the first input memory of pressure wavefield at t-2
 * @param p_ppi1  the second input memory of pressure wavefield at t-2
 *
 * @param p_ppo0  the first output memory of pressure wavefield at t-2
 * @param p_ppo1  the second output memory of pressure wavefield at t-2
 *
 * @param p_upb is the uppper bounday wavefiled
 */
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
                           const RTM_dataType* p_taperz,
                           const RTM_dataType* p_tapery,
                           const RTM_dataType* p_taperx,
                           const RTM_type* p_v2dt2,
                           RTM_type* p_pi0,
                           RTM_type* p_pi1,
                           RTM_type* p_po0,
                           RTM_type* p_po1,
                           RTM_type* p_ppi0,
                           RTM_type* p_ppi1,
                           RTM_type* p_ppo0,
                           RTM_type* p_ppo1,
                           RTM_upbType* p_upb);

#endif
