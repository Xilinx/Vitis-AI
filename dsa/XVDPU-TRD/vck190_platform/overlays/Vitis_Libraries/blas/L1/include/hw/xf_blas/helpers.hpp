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
 * @file helpers.hpp
 * @brief common datatypes for L1 modules.
 *
 * This file is part of Vitis BLAS Library.
 */

#ifndef XF_BLAS_HPPELPERS_HPP
#define XF_BLAS_HPPELPERS_HPP

/*        UTILITY FUNCTIONS             */
#include "helpers/utils/types.hpp"
#include "helpers/utils/utils.hpp"

/*        DATA MOVER            */
#include "helpers/dataMover/vecMoverB1.hpp"
#include "helpers/dataMover/matMoverB2.hpp"
#include "helpers/dataMover/bandedMatMoverB2.hpp"
#include "helpers/dataMover/transpMatB2.hpp"
#include "helpers/dataMover/symMatMoverB2.hpp"
#include "helpers/dataMover/trmMatMoverB2.hpp"
#include "helpers/dataMover/gemmMatMover.hpp"

/*        HELPER FUNCTIONS             */
#include "helpers/funcs/padding.hpp"
#include "helpers/funcs/maxmin.hpp"
#include "helpers/funcs/abs.hpp"
#include "helpers/funcs/sum.hpp"
#include "helpers/funcs/mul.hpp"
#include "helpers/funcs/dotHelper.hpp"

#endif
