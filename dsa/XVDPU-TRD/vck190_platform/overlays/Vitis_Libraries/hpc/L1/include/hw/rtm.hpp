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
#ifndef XF_HPC_RTM_RTM_HPP
#define XF_HPC_RTM_RTM_HPP

#ifndef __SYNTHESIS__
#include <cassert>
#include <cstring>
#endif

#include "ap_int.h"
#include "hls_stream.h"

#include "xf_blas.hpp"
#include "streamOps.hpp"
#include "rtm/dataMover.hpp"

// RTM 2D
#include "rtm/stencil2d.hpp"
#include "rtm/rtm2d.hpp"

// RTM 3D
#include "rtm/domain3d.hpp"
#include "rtm/stencil3d.hpp"
#include "rtm/rtm3d.hpp"

#endif
