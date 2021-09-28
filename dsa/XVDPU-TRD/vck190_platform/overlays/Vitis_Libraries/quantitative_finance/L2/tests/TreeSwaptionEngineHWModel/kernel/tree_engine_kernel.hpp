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

#ifndef _TREE_KERNEL_H_
#define _TREE_KERNEL_H_

#include "xf_fintech/tree_engine.hpp"
#include "xf_fintech/hw_model.hpp"
#include "xf_fintech/trinomial_tree.hpp"
#include "xf_fintech/ornstein_uhlenbeck_process.hpp"
using namespace xf::fintech;

#define N 1
#define DIM 1
#define LEN 1024
#define LEN2 2048
#define ExerciseLen 5
#define FloatingLen 10
#define FixedLen 5

typedef double DT;
typedef OrnsteinUhlenbeckProcess<DT> Process;
typedef TrinomialTree<DT, Process, LEN> Tree;
typedef HWModel<DT, Tree, LEN2> Model;

extern "C" void TREE_k0(int type,
                        DT fixedRate,
                        int timestep,
                        DT initTime[LEN],
                        int initSize,
                        int exerciseCnt[ExerciseLen],
                        int floatingResetCnt[FloatingLen],
                        int fixedResetCnt[FixedLen],
                        DT NPV[N]);

#endif
