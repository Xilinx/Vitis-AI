/**
 * Copyright (C) 2020 Xilinx, Inc
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may
 * not use this file except in compliance with the License. A copy of the
 * License is located at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 */

#ifndef _LOUVAINPHASE_H_
#define _LOUVAINPHASE_H_
#include "xilinxlouvain.hpp"
#include "partition/ParLV.h"  //tmp include
#include "partition/ctrlLV.h" //tmp include
enum MOKD_PHASEFLOW { MD_CLASSIC = 0, MD_NORMAL, MD_FAST };
// MD_CLASSIC
void runLouvainWithFPGA_demo(graphNew* G,
                             long* C_orig,
                             char* opts_xclbinPath,
                             bool opts_coloring,
                             long opts_minGraphSize,
                             double opts_threshold,
                             double opts_C_thresh,
                             int numThreads);

// MD_NORMAL
void runLouvainWithFPGA_demo_par_core(bool hasGhost,
                                      int id_dev,
                                      GLV* pglv_orig,
                                      GLV* pglv_iter,
                                      char* opts_xclbinPath,
                                      bool opts_coloring,
                                      long opts_minGraphSize,
                                      double opts_threshold,
                                      double opts_C_thresh,
                                      int numThreads);
// MD_FAST
void runLouvainWithFPGA_demo_par_core_prune(bool hasGhost,
                                            int id_dev,
                                            GLV* pglv_orig,
                                            GLV* pglv_iter,
                                            char* opts_xclbinPath,
                                            bool opts_coloring,
                                            long opts_minGraphSize,
                                            double opts_threshold,
                                            double opts_C_thresh,
                                            int numThreads);

GLV* LouvainGLV_general(bool hasGhost,
                        int flowMode,
                        int id_dev,
                        GLV* glv_src,
                        char* xclbinPath,
                        int numThreads,
                        int& id_glv,
                        long minGraphSize,
                        double threshold,
                        double C_threshold,
                        bool isParallel,
                        int numPhase);

void LouvainGLV_general_batch_thread(bool hasGhost,
                                     int flowMode,
                                     int id_dev,
                                     int id_glv,
                                     int num_dev,
                                     int num_par,
                                     double* timeLv,
                                     GLV* par_src[],
                                     GLV* par_lved[],
                                     char* xclbinPath,
                                     int numThreads,
                                     long minGraphSize,
                                     double threshold,
                                     double C_threshold,
                                     bool isParallel,
                                     int numPhase);

#endif
