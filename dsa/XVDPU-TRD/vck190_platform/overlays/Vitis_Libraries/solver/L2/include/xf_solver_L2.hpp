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
 * @file xf_solver_L2.h
 * @brief Top-levle header for XF Solver Libaray level-2.
 */

#ifndef _XF_SOLVER_L2_HPP_
#define _XF_SOLVER_L2_HPP_

// Matrix decomposition
#include "hw/MatrixDecomposition/potrf.hpp"
#include "hw/MatrixDecomposition/getrf_nopivot.hpp"
#include "hw/MatrixDecomposition/getrf.hpp"

#include "hw/MatrixDecomposition/geqrf.hpp"
#include "hw/MatrixDecomposition/gesvdj.hpp"
#include "hw/MatrixDecomposition/gesvj.hpp"

// Linear solver
#include "hw/LinearSolver/pomatrixinverse.hpp"
#include "hw/LinearSolver/gematrixinverse.hpp"
#include "hw/LinearSolver/trtrs.hpp"
#include "hw/LinearSolver/polinearsolver.hpp"
#include "hw/LinearSolver/gelinearsolver.hpp"
#include "hw/LinearSolver/gtsv_pcr.hpp"

// Eigen value solver
#include "hw/EigenSolver/syevj.hpp"

#endif
