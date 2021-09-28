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

#ifndef _XF_FINTECH_API_H_
#define _XF_FINTECH_API_H_

#include <CL/cl_ext_xilinx.h>

#include "xcl2.hpp"

#include "xf_fintech_device_manager.hpp"
#include "xf_fintech_device.hpp"
#include "xf_fintech_error_codes.hpp"
#include "xf_fintech_timestamp.hpp"
#include "xf_fintech_trace.hpp"
#include "xf_fintech_types.hpp"

#include "models/xf_fintech_cf_black_scholes.hpp"
#include "models/xf_fintech_cf_black_scholes_merton.hpp"
#include "models/xf_fintech_cf_garman_kohlhagen.hpp"
#include "models/xf_fintech_cf_b76.hpp"
#include "models/xf_fintech_po.hpp"
#include "models/xf_fintech_quanto.hpp"
#include "models/xf_fintech_fd_heston.hpp"
#include "models/xf_fintech_mc_european.hpp"
#include "models/xf_fintech_mc_european_dje.hpp"
#include "models/xf_fintech_mc_american.hpp"
#include "models/xf_fintech_binomialtree.hpp"
#include "models/xf_fintech_hcf.hpp"
#include "models/xf_fintech_m76.hpp"
#include "models/xf_fintech_pop_mcmc.hpp"
#include "models/xf_fintech_hjm.hpp"
#include "models/xf_fintech_lmm.hpp"
#include "models/xf_fintech_fdbslv.hpp"
#include "models/xf_fintech_hullwhite.hpp"
#include "models/xf_fintech_cds.hpp"

#endif //_XF_FINTECH_API_H_
