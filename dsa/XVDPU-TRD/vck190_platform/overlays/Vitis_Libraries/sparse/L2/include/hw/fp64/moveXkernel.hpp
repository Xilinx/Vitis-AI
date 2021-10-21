
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
 * @file moveXkernel.hpp
 * @brief select and multiply input vector X.
 *
 * This file is part of Vitis SPARSE Library.
 */

#ifndef XF_SPARSE_MOVEXKERNEL_HPP
#define XF_SPARSE_MOVEXKERNEL_HPP

#include "kernel.hpp"

/**
 * @brief moveXkernel is used to dispatch X entries to multiple computation paths
 * @param p_inStr input axis stream of X entries
 * @param p_outStr output axis streams of X entries
 */
extern "C" void moveXkernel(HBM_StrTyp& p_inStr,
                            HBM_StrTyp& p_outStr0,
                            HBM_StrTyp& p_outStr1,
                            HBM_StrTyp& p_outStr2,
                            HBM_StrTyp& p_outStr3,
                            HBM_StrTyp& p_outStr4,
                            HBM_StrTyp& p_outStr5,
                            HBM_StrTyp& p_outStr6,
                            HBM_StrTyp& p_outStr7,
                            HBM_StrTyp& p_outStr8,
                            HBM_StrTyp& p_outStr9,
                            HBM_StrTyp& p_outStr10,
                            HBM_StrTyp& p_outStr11,
                            HBM_StrTyp& p_outStr12,
                            HBM_StrTyp& p_outStr13,
                            HBM_StrTyp& p_outStr14,
                            HBM_StrTyp& p_outStr15);

#endif
