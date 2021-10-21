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

#include <omp.h>
#include "xilinxlouvain.hpp"
#include "ParLV.h"  //tmp include
#include "ctrlLV.h" //tmp include
#include "partitionLouvain.hpp"
#include "louvainPhase.h"
#include "defs.h"
#include "xcl2.hpp"
#include "string.h"
#include "ap_int.h"
#include "utils.hpp"
#include "xf_utils_sw/logger.hpp"
///////////////////////////////////////////////////////////////////////////////////////////////////////////
void PrintReport_MultiPhase(bool opts_coloring,
                            long opts_minGraphSize,
                            double opts_threshold,
                            double opts_C_thresh,
                            int numThreads,
                            long numClusters,
                            double totTimeClustering,
                            double totTimeBuildingPhase,
                            double totTimeColoring,
                            double prevMod,
                            int phase,
                            double totTimeE2E_2,
                            int num_runsFPGA,
                            int totItr,
                            double eachTimeE2E_2[MAX_NUM_PHASE],
                            double eachMod[MAX_NUM_PHASE],
                            int eachItrs[MAX_NUM_PHASE],
                            long eachClusters[MAX_NUM_PHASE]) {
    printf("********************************************\n");
    printf("*********    Compact Summary   *************\n");
    printf("********************************************\n");
    printf("Number of threads              : %-8d \t m=%d \t thhd=%lf \t thhd_c=%lf\n", numThreads, opts_minGraphSize,
           opts_threshold, opts_C_thresh);
    printf("Total number of phases         : %-8d\n", phase);
    printf("Total number of iterations     : %-8d = \t", totItr);
    for (int i = 0; i < phase; i++) printf(" + %8d  ", eachItrs[i]);
    printf("\n");
    printf("Final number of clusters       : %-8d : \t", numClusters);
    for (int i = 0; i < phase; i++) printf("   %8d  ", eachClusters[i]);
    printf("\n");
    printf("Final modularity               : %lf : \t ", prevMod);
    for (int i = 0; i < phase; i++) printf("  %2.6f  ", eachMod[i]);
    printf("\n");
    printf("Total time for clustering      : %lf\n", totTimeClustering);
    printf("Total time for building phases : %lf\n", totTimeBuildingPhase);
    printf("Total E2E time(s)              : %lf = \t", totTimeE2E_2);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeE2E_2[i]);
    printf("\n");
    if (opts_coloring) {
        printf("Total time for coloring        : %lf\n", totTimeColoring);
    }
    printf("********************************************\n");
    printf("TOTAL TIME                     : %lf\n",
           totTimeE2E_2 + totTimeClustering + totTimeBuildingPhase + totTimeColoring);
    printf("********************************************\n");
}

void PrintReport_MultiPhase_2(int phase,
                              double totTimeE2E_2,
                              double totTimeAll,
                              double totTimeInitBuff,
                              double totTimeReadBuff,
                              double totTimeReGraph,
                              double totTimeFeature,
                              double eachTimeInitBuff[MAX_NUM_PHASE],
                              double eachTimeReadBuff[MAX_NUM_PHASE],
                              double eachTimeReGraph[MAX_NUM_PHASE],
                              double eachTimeE2E_2[MAX_NUM_PHASE],
                              double eachTimePhase[MAX_NUM_PHASE],
                              double eachTimeFeature[MAX_NUM_PHASE],
                              double eachNum[MAX_NUM_PHASE],
                              double eachC[MAX_NUM_PHASE],
                              double eachM[MAX_NUM_PHASE],
                              double eachBuild[MAX_NUM_PHASE],
                              double eachSet[MAX_NUM_PHASE]) {
    printf("Total time for Init buff_host  : %lf = \t", totTimeInitBuff);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeInitBuff[i]);
    printf("\n");
    printf("Total time for Read buff_host  : %lf = \t", totTimeReadBuff);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeReadBuff[i]);
    printf("\n");
    printf("Total time for totTimeE2E_2    : %lf = \t", totTimeE2E_2);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeE2E_2[i]);
    printf("\n");

    printf("Total time for totTimeReGraph  : %lf = \t", totTimeReGraph);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeReGraph[i]);
    printf("\n");

    double ToTeachNum = 0;   //        [MAX_NUM_PHASE];
    double ToTeachC = 0;     //        [MAX_NUM_PHASE];
    double ToTeachM = 0;     //        [MAX_NUM_PHASE];
    double ToTeachBuild = 0; //        [MAX_NUM_PHASE];
    double ToTeachSet = 0;   //         [MAX_NUM_PHASE];

    for (int i = 0; i < phase; i++) {
        ToTeachNum += eachNum[i];     //        [MAX_NUM_PHASE];
        ToTeachC += eachC[i];         //        [MAX_NUM_PHASE];
        ToTeachM += eachM[i];         //        [MAX_NUM_PHASE];
        ToTeachBuild += eachBuild[i]; //        [MAX_NUM_PHASE];
        ToTeachSet += eachSet[i];
    }
    printf("----- time for ToTeachNum      : %lf = \t", ToTeachNum);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachNum[i]);
    printf("\n");
    printf("----- time for ToTeachC        : %lf = \t", ToTeachC);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachC[i]);
    printf("\n");
    printf("----- time for ToTeachM        : %lf = \t", ToTeachM);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachM[i]);
    printf("\n");
    printf("----- time for ToTeachBuild    : %lf = \t", ToTeachBuild);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachBuild[i]);
    printf("\n");
    printf("----- time for ToTeachSet      : %lf = \t", ToTeachSet);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachSet[i]);
    printf("\n");

    printf("Total time for totTimeFeature  : %lf = \t", totTimeFeature);
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimeFeature[i]);
    printf("\n");
    printf("********************************************\n");
    printf("TOTAL TIME2                    : %lf = \t", totTimeAll); // eachTimePhase
    for (int i = 0; i < phase; i++) printf("+ %2.6f  ", eachTimePhase[i]);
    printf("\n");
    printf("********************************************\n");
}

int Phaseloop_UsingFPGA_InitColorBuff(graphNew* G, int* colors, int numThreads, double& totTimeColoring) {
#pragma omp parallel for
    for (long i = 0; i < G->numVertices; i++) {
        colors[i] = -1;
    }
    double tmpTime;
    int numColors = algoDistanceOneVertexColoringOpt(G, colors, numThreads, &tmpTime) + 1;
    totTimeColoring += tmpTime;
    return numColors;
}

void PhaseLoop_UpdatingC_org(int phase, long NV, long NV_G, long* C, long* C_orig) {
    if (phase == 1) {
#pragma omp parallel for
        for (long i = 0; i < NV; i++) {
            C_orig[i] = C[i]; // After the first phase
        }
    } else {
#pragma omp parallel for
        for (long i = 0; i < NV; i++) {
            assert(C_orig[i] < NV_G);
            if (C_orig[i] >= 0) C_orig[i] = C[C_orig[i]]; // Each cluster in a previous phase becomes a vertex
        }
    }
}

void inline PhaseLoop_Kernel_Enable(cl::Kernel& kernel_louvain,
                                    cl::CommandQueue& q,
                                    std::vector<cl::Memory>& ob_in,
                                    std::vector<cl::Memory>& ob_out,
                                    std::vector<std::vector<cl::Event> >& kernel_evt0,
                                    std::vector<std::vector<cl::Event> >& kernel_evt1) {
    kernel_evt0[0].resize(1);
    kernel_evt1[0].resize(1);

    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, kernel_evt0[0].data()); // 0 : migrate from host to dev
    q.enqueueTask(kernel_louvain, &kernel_evt0[0], kernel_evt1[0].data());
    q.enqueueMigrateMemObjects(ob_out, 1, &kernel_evt1[0], nullptr); // 1 : migrate from dev to host
    q.finish();
}
long* CreateM(long NV_new, long NV_orig, long* C_orig, long* M_orig) {
    long* M = (long*)malloc(NV_new * sizeof(long));
    assert(M);
    memset(M, 0, NV_new * sizeof(long));
    for (int i = 0; i < NV_orig; i++) {
        if (M_orig[i] < 0) M[C_orig[i]] = M_orig[i];
    }
    return M;
}

double PhaseLoop_CommPostProcessing_par(GLV* pglv_orig,
                                        GLV* pglv_iter,
                                        int numThreads,
                                        double opts_threshold,
                                        bool opts_coloring,
                                        // modified:
                                        bool& nonColor,
                                        int& phase,
                                        int& totItr,
                                        long& numClusters,
                                        double& totTimeBuildingPhase) {
    double time1 = 0;
    time1 = omp_get_wtime();
    graphNew* Gnew;
    numClusters = renumberClustersContiguously_ghost(pglv_iter->C, pglv_iter->G->numVertices, pglv_iter->NVl);
    printf("Number of unique clusters: %ld\n", numClusters);
    PhaseLoop_UpdatingC_org(phase, pglv_orig->NV, pglv_iter->NV, pglv_iter->C, pglv_orig->C);

    long* M_new = CreateM(numClusters, pglv_orig->NV, pglv_orig->C, pglv_orig->M);

    Gnew = (graphNew*)malloc(sizeof(graphNew));
    assert(Gnew != 0);
    double tmpTime = buildNextLevelGraphOpt(pglv_iter->G, Gnew, pglv_iter->C, numClusters, numThreads);
    totTimeBuildingPhase += tmpTime;
    pglv_iter->SetByOhterG(Gnew, M_new);
    time1 = omp_get_wtime() - time1;
    return time1;
}

double PhaseLoop_CommPostProcessing_par(GLV* pglv_orig,
                                        GLV* pglv_iter,
                                        int numThreads,
                                        double opts_threshold,
                                        bool opts_coloring,
                                        // modified:
                                        bool& nonColor,
                                        int& phase,
                                        int& totItr,
                                        long& numClusters,
                                        double& totTimeBuildingPhase,
                                        double& time_renum,
                                        double& time_C,
                                        double& time_M,
                                        double& time_buid,
                                        double& time_set) {
    double time1 = 0;
    time1 = omp_get_wtime();
    time_renum = time1;
    graphNew* Gnew;
    // numClusters = renumberClustersContiguously_ghost(pglv_iter->C, pglv_iter->G->numVertices, pglv_iter->NVl);
    printf("Number of unique clusters ghost: %ld\n", numClusters);
    time_renum = omp_get_wtime() - time_renum;

    time_C = omp_get_wtime();
    PhaseLoop_UpdatingC_org(phase, pglv_orig->NV, pglv_iter->NV, pglv_iter->C, pglv_orig->C);
    time_C = omp_get_wtime() - time_C;

    time_M = omp_get_wtime();
    long* M_new = CreateM(numClusters, pglv_orig->NV, pglv_orig->C, pglv_orig->M);
    time_M = omp_get_wtime() - time_M;

    Gnew = (graphNew*)malloc(sizeof(graphNew));
    assert(Gnew != 0);
    double tmpTime = buildNextLevelGraphOpt(pglv_iter->G, Gnew, pglv_iter->C, numClusters, numThreads);
    totTimeBuildingPhase += tmpTime;
    time_buid = tmpTime;

    time_set = omp_get_wtime();
    pglv_iter->SetByOhterG(Gnew, M_new);
    time_set = omp_get_wtime() - time_set;

    time1 = omp_get_wtime() - time1;

    return time1;
}
double PhaseLoop_CommPostProcessing(GLV* pglv_orig,
                                    GLV* pglv_iter,
                                    int numThreads,
                                    double opts_threshold,
                                    bool opts_coloring,
                                    // modified:
                                    bool& nonColor,
                                    int& phase,
                                    int& totItr,
                                    long& numClusters,
                                    double& totTimeBuildingPhase,
                                    double& time_renum,
                                    double& time_C,
                                    double& time_M,
                                    double& time_buid,
                                    double& time_set) {
    double time1 = 0;
    time1 = omp_get_wtime();
    time_renum = time1;
    graphNew* Gnew;
    // numClusters = renumberClustersContiguously(pglv_iter->C, pglv_iter->G->numVertices);
    printf("Number of unique clusters: %ld\n", numClusters);
    time_renum = omp_get_wtime() - time_renum;

    time_C = omp_get_wtime();
    PhaseLoop_UpdatingC_org(phase, pglv_orig->NV, pglv_iter->NV, pglv_iter->C, pglv_orig->C);
    time_C = omp_get_wtime() - time_C;

    time_M = 0; // omp_get_wtime();
                // long* M_new = CreateM( numClusters, pglv_orig->NV, pglv_orig->C, pglv_orig->M );
    // time_M  = omp_get_wtime() - time_M;

    Gnew = (graphNew*)malloc(sizeof(graphNew));
    assert(Gnew != 0);
    double tmpTime = buildNextLevelGraphOpt(pglv_iter->G, Gnew, pglv_iter->C, numClusters, numThreads);
    totTimeBuildingPhase += tmpTime;
    time_buid = tmpTime;

    time_set = omp_get_wtime();
    pglv_iter->SetByOhterG(Gnew); //, M_new);
    time_set = omp_get_wtime() - time_set;

    time1 = omp_get_wtime() - time1;

    return time1;
}
bool PhaseLoop_CommPostProcessing(long NV,
                                  int numThreads,
                                  double opts_threshold,
                                  bool opts_coloring,
                                  double prevMod,
                                  double currMod,
                                  // modified:
                                  graphNew*& G,
                                  long*& C,
                                  long*& C_orig,
                                  bool& nonColor,
                                  int& phase,
                                  int& totItr,
                                  long& numClusters,
                                  double& totTimeBuildingPhase) {
    double time1 = 0;
    time1 = omp_get_wtime();
    graphNew* Gnew;
    numClusters = renumberClustersContiguously(C, G->numVertices);
    printf("Number of unique clusters: %ld\n", numClusters);
    PhaseLoop_UpdatingC_org(phase, NV, G->numVertices, C, C_orig);

    if ((phase > MAX_NUM_PHASE) || (totItr > MAX_NUM_TOTITR)) {
        return 1; // Break if too many phases or iterations
    } else {
        if ((currMod - prevMod) > opts_threshold) {
            Gnew = (graphNew*)malloc(sizeof(graphNew));
            assert(Gnew != 0);
            double tmpTime = buildNextLevelGraphOpt(G, Gnew, C, numClusters, numThreads);
            // totTimeBuildingPhase += tmpTime;
            // Free up the previous graphNew
            free(G->edgeListPtrs);
            free(G->edgeList);
            free(G);
            G = Gnew; // Swap the pointers
            G->edgeListPtrs = Gnew->edgeListPtrs;
            G->edgeList = Gnew->edgeList;
            // Free up the previous cluster & create new one of a different size
            free(C);
            C = (long*)malloc(numClusters * sizeof(long));
            assert(C != 0);
#pragma omp parallel for
            for (long i = 0; i < numClusters; i++) {
                C[i] = -1;
            }
            phase++; // Increment phase number
        } else {     // when !((opts_coloring == 1) && (G->numVertices > opts_minGraphSize) && (nonColor == false))
            if ((opts_coloring) && (nonColor == false)) {
                nonColor = true; // Run at least one loop of non-coloring routine
            } else {
                return true; // Modularity gain is not enough. Exit.
            }
        }
    }
    time1 = omp_get_wtime() - time1;
    totTimeBuildingPhase += time1;
    return false;
}

void inline PhaseLoop_MapHostBuff(long NV,
                                  long NE_mem_1,
                                  long NE_mem_2,
                                  std::vector<cl_mem_ext_ptr_t>& mext_in,
                                  cl::Context& context,
                                  KMemorys_host& buff_host) {
    buff_host.config0 = aligned_alloc<int64_t>(4);
    buff_host.config1 = aligned_alloc<DWEIGHT>(4);
    buff_host.offsets = aligned_alloc<int>(NV + 1);
    buff_host.indices = aligned_alloc<int>(NE_mem_1);
    buff_host.weights = aligned_alloc<float>(NE_mem_1);
    if (NE_mem_2 > 0) buff_host.indices2 = aligned_alloc<int>(NE_mem_2);
    if (NE_mem_2 > 0) buff_host.weights2 = aligned_alloc<float>(NE_mem_2);
    buff_host.cidPrev = aligned_alloc<int>(NV);
    buff_host.cidCurr = aligned_alloc<int>(NV);
    buff_host.cidSizePrev = aligned_alloc<int>(NV);
    buff_host.totPrev = aligned_alloc<float>(NV);
    buff_host.cidSizeCurr = aligned_alloc<int>(NV);
    buff_host.totCurr = aligned_alloc<float>(NV);
    buff_host.cidSizeUpdate = aligned_alloc<int>(NV);
    buff_host.totUpdate = aligned_alloc<float>(NV);
    buff_host.cWeight = aligned_alloc<float>(NV);
    buff_host.colorAxi = aligned_alloc<int>(NV);
    buff_host.colorInx = aligned_alloc<int>(NV);

    ap_uint<CSRWIDTHS>* axi_offsets = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.offsets);
    ap_uint<CSRWIDTHS>* axi_indices = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indices);
    ap_uint<CSRWIDTHS>* axi_weights = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.weights);
    ap_uint<CSRWIDTHS>* axi_indices2;
    ap_uint<CSRWIDTHS>* axi_weights2;
    if (NE_mem_2 > 0) {
        axi_indices2 = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indices2);
        axi_weights2 = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.weights2);
    }
    ap_uint<DWIDTHS>* axi_cidPrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidPrev);
    ap_uint<DWIDTHS>* axi_cidCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidCurr);
    ap_uint<DWIDTHS>* axi_cidSizePrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizePrev);
    ap_uint<DWIDTHS>* axi_totPrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totPrev);
    ap_uint<DWIDTHS>* axi_cidSizeCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizeCurr);
    ap_uint<DWIDTHS>* axi_totCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totCurr);
    ap_uint<DWIDTHS>* axi_cidSizeUpdate = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizeUpdate);
    ap_uint<DWIDTHS>* axi_totUpdate = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totUpdate);
    ap_uint<DWIDTHS>* axi_cWeight = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cWeight);
    ap_uint<COLORWIDTHS>* axi_colorAxi = reinterpret_cast<ap_uint<COLORWIDTHS>*>(buff_host.colorAxi);
    ap_uint<COLORWIDTHS>* axi_colorInx = reinterpret_cast<ap_uint<COLORWIDTHS>*>(buff_host.colorInx);

    // DDR Settings
    mext_in[0] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, buff_host.config0, 0};
    mext_in[1] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, buff_host.config1, 0};
    mext_in[2] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, axi_offsets, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, axi_indices, 0};
    mext_in[4] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, axi_weights, 0};
    if (NE_mem_2 > 0) {
        mext_in[3 + 13] = {(unsigned int)(1) | XCL_MEM_TOPOLOGY, axi_indices2, 0};
        mext_in[4 + 13] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, axi_weights2, 0};
    }
    mext_in[5] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, axi_colorAxi, 0};
    mext_in[6] = {(unsigned int)(8) | XCL_MEM_TOPOLOGY, axi_colorInx, 0};
    mext_in[7] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, axi_cidPrev, 0};
    mext_in[8] = {(unsigned int)(12) | XCL_MEM_TOPOLOGY, axi_cidSizePrev, 0};
    mext_in[9] = {(unsigned int)(14) | XCL_MEM_TOPOLOGY, axi_totPrev, 0};
    mext_in[10] = {(unsigned int)(16) | XCL_MEM_TOPOLOGY, axi_cidCurr, 0};
    mext_in[11] = {(unsigned int)(18) | XCL_MEM_TOPOLOGY, axi_cidSizeCurr, 0};
    mext_in[12] = {(unsigned int)(20) | XCL_MEM_TOPOLOGY, axi_totCurr, 0};
    mext_in[13] = {(unsigned int)(22) | XCL_MEM_TOPOLOGY, axi_cidSizeUpdate, 0};
    mext_in[14] = {(unsigned int)(24) | XCL_MEM_TOPOLOGY, axi_totUpdate, 0};
    mext_in[15] = {(unsigned int)(26) | XCL_MEM_TOPOLOGY, axi_cWeight, 0};
}
void inline PhaseLoop_MapClBuff(long NV,
                                long NE_mem_1,
                                long NE_mem_2,
                                std::vector<cl_mem_ext_ptr_t>& mext_in,
                                cl::Context& context,
                                KMemorys_clBuff& buff_cl) {
    // Create device buffer and map dev buf to host buf
    int flag_RW = CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE;
    int flag_RD = CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY;

    buff_cl.db_config0 = cl::Buffer(context, flag_RW, sizeof(int64_t) * (4), &mext_in[0]);
    buff_cl.db_config1 = cl::Buffer(context, flag_RW, sizeof(DWEIGHT) * (4), &mext_in[1]);

    buff_cl.db_offsets = cl::Buffer(context, flag_RD, sizeof(int) * (NV + 1), &mext_in[2]);
    buff_cl.db_indices = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_1, &mext_in[3]);
    buff_cl.db_weights = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_1, &mext_in[4]);
    if (NE_mem_2 > 0) {
        buff_cl.db_indices2 = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_2, &mext_in[3 + 13]);
        buff_cl.db_weights2 = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_2, &mext_in[4 + 13]);
    }
    buff_cl.db_colorAxi = cl::Buffer(context, flag_RD, sizeof(int) * (NV), &mext_in[5]);
    buff_cl.db_colorInx = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[6]);
    buff_cl.db_cidPrev = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[7]);
    buff_cl.db_cidSizePrev = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[8]);
    buff_cl.db_totPrev = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[9]);
    buff_cl.db_cidCurr = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[10]);
    buff_cl.db_cidSizeCurr = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[11]);
    buff_cl.db_totCurr = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[12]);
    buff_cl.db_cidSizeUpdate = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[13]);
    buff_cl.db_totUpdate = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[14]);
    buff_cl.db_cWeight = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[15]);
}
void UsingFPGA_MapHostClBuff(
    long NV, long NE_mem_1, long NE_mem_2, cl::Context& context, KMemorys_host& buff_host, KMemorys_clBuff& buff_cl) {
    std::vector<cl_mem_ext_ptr_t> mext_in(NUM_PORT_KERNEL + 2);
    PhaseLoop_MapHostBuff(NV, NE_mem_1, NE_mem_2, mext_in, context, buff_host);
    PhaseLoop_MapClBuff(NV, NE_mem_1, NE_mem_2, mext_in, context, buff_cl);
}

int PhaseLoop_UsingCPU(double opts_threshold,
                       int numThreads,
                       double& currMod,
                       graphNew*& G,
                       long*& C,
                       long*& C_orig,
                       int& totItr,
                       bool& nonColor,
                       double& totTimeClustering) {
    double tmpTime;
    int tmpItr = 0;
    currMod = parallelLouvianMethod(G, C, numThreads, currMod, opts_threshold, &tmpTime, &tmpItr);
    totTimeClustering += tmpTime;
    totItr += tmpItr;
    nonColor = true;
    return 0;
}
void PhaseLoop_UsingFPGA_1_KernelSetup(bool isLargeEdge,
                                       cl::Kernel& kernel_louvain,
                                       std::vector<cl::Memory>& ob_in,
                                       std::vector<cl::Memory>& ob_out,
                                       KMemorys_clBuff& buff_cl) {
    // Data transfer from host buffer to device buffer
    ob_in.push_back(buff_cl.db_config0);
    ob_in.push_back(buff_cl.db_config1);
    ob_in.push_back(buff_cl.db_offsets);
    ob_in.push_back(buff_cl.db_indices);
    if (isLargeEdge) ob_in.push_back(buff_cl.db_indices2);
    ob_in.push_back(buff_cl.db_weights);
    if (isLargeEdge) ob_in.push_back(buff_cl.db_weights2);
    ob_in.push_back(buff_cl.db_cidCurr);
    ob_in.push_back(buff_cl.db_cidSizePrev);
    ob_in.push_back(buff_cl.db_cidSizeUpdate);
    ob_in.push_back(buff_cl.db_cidSizeCurr);
    ob_in.push_back(buff_cl.db_totPrev);
    ob_in.push_back(buff_cl.db_totUpdate);
    ob_in.push_back(buff_cl.db_totCurr);
    ob_in.push_back(buff_cl.db_cWeight);
    ob_in.push_back(buff_cl.db_colorAxi);
    ob_in.push_back(buff_cl.db_colorInx);
    ob_out.push_back(buff_cl.db_config0);
    ob_out.push_back(buff_cl.db_config1);
    ob_out.push_back(buff_cl.db_cidPrev);

    kernel_louvain.setArg(0, buff_cl.db_config0);        // config0
    kernel_louvain.setArg(1, buff_cl.db_config1);        // config1
    kernel_louvain.setArg(2, buff_cl.db_offsets);        // offsets
    kernel_louvain.setArg(3, buff_cl.db_indices);        // indices
    kernel_louvain.setArg(4, buff_cl.db_weights);        // weights
    kernel_louvain.setArg(5, buff_cl.db_colorAxi);       // colorAxi
    kernel_louvain.setArg(6, buff_cl.db_colorInx);       // colorInx
    kernel_louvain.setArg(7, buff_cl.db_cidPrev);        // cidPrev
    kernel_louvain.setArg(8, buff_cl.db_cidSizePrev);    // cidSizePrev
    kernel_louvain.setArg(9, buff_cl.db_totPrev);        // totPrev
    kernel_louvain.setArg(10, buff_cl.db_cidCurr);       // cidCurr
    kernel_louvain.setArg(11, buff_cl.db_cidSizeCurr);   // cidSizeCurr
    kernel_louvain.setArg(12, buff_cl.db_totCurr);       // totCurr
    kernel_louvain.setArg(13, buff_cl.db_cidSizeUpdate); // cUpdate
    kernel_louvain.setArg(14, buff_cl.db_totUpdate);     // totCurr
    kernel_louvain.setArg(15, buff_cl.db_cWeight);       // cWeight
    std::cout << "INFO: Finish kernel setup" << std::endl;
}

void PhaseLoop_UsingFPGA_2_DataWriteTo(cl::CommandQueue& q,
                                       std::vector<std::vector<cl::Event> >& kernel_evt0,
                                       std::vector<cl::Memory>& ob_in) { /* 0 : migrate from host to dev */
    q.enqueueMigrateMemObjects(ob_in, 0, nullptr, kernel_evt0[0].data());
}

void PhaseLoop_UsingFPGA_3_KernelRun(cl::CommandQueue& q,
                                     std::vector<std::vector<cl::Event> >& kernel_evt0,
                                     std::vector<std::vector<cl::Event> >& kernel_evt1,
                                     cl::Kernel& kernel_louvain) {
    q.enqueueTask(kernel_louvain, &kernel_evt0[0], kernel_evt1[0].data());
}

void PhaseLoop_UsingFPGA_4_DataReadBack(cl::CommandQueue& q,
                                        std::vector<std::vector<cl::Event> >& kernel_evt1,
                                        std::vector<cl::Memory>& ob_out) { /* kernel_evt1 : migrate from dev to host*/
    q.enqueueMigrateMemObjects(ob_out, 1, &kernel_evt1[0], nullptr);
}

void PhaseLoop_UsingFPGA_5_KernelFinish(cl::CommandQueue& q) {
    q.finish();
}
void PhaseLoop_UsingFPGA_Post(long vertexNum,
                              int num_runsFPGA,
                              KMemorys_host& buff_host,
                              struct timeval& tstartE2E,
                              struct timeval& tendE2E,
                              std::vector<std::vector<cl::Event> >& kernel_evt1,
                              // output
                              long* C,
                              int& totItr,
                              double& currMod,
                              int& totTimeE2E) {
    // updating
    totItr += buff_host.config0[2];
    currMod = buff_host.config1[1];
    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    unsigned long timeStart, timeEnd;
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    int exec_timeE2E = diff(&tendE2E, &tstartE2E);
    totTimeE2E += exec_timeE2E;
    // showing
    unsigned long exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Average execution per run: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;
    // std::cout << "INFO: FPGA execution time of " << num_runsFPGA << " runs:" << exec_timeE2E << " us\n"
    printf("INFO: FPGA execution time of %1d runs:%9d us    Iteration times = %2d currMod=%f\n", num_runsFPGA,
           exec_timeE2E, buff_host.config0[2], currMod);
    std::cout << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * num_runsFPGA + exec_time0 << " us\n"
              << "INFO: The iterations is: " << buff_host.config0[2] << "\n";
    std::cout << "-------------------------------------------------------" << std::endl;
}

void PhaseLoop_UsingFPGA_Post_par(long vertexNum,
                                  int num_runsFPGA,
                                  KMemorys_host& buff_host,
                                  struct timeval& tstartE2E,
                                  struct timeval& tendE2E,
                                  std::vector<std::vector<cl::Event> >& kernel_evt1,
                                  // output
                                  long* C,
                                  int& totItr,
                                  double& currMod,
                                  int& totTimeE2E) {
    // updating
    totItr += buff_host.config0[2];
    currMod = buff_host.config1[1];
    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    unsigned long timeStart, timeEnd;
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    int exec_timeE2E = diff(&tendE2E, &tstartE2E);
    totTimeE2E += exec_timeE2E;
    // showing
    unsigned long exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Average execution per run: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;
    // std::cout << "INFO: FPGA execution time of " << num_runsFPGA << " runs:" << exec_timeE2E << " us\n"
    printf("INFO: FPGA execution time of %1d runs:%9d us    Iteration times = %2d currMod=%f\n", num_runsFPGA,
           exec_timeE2E, buff_host.config0[2], currMod);
    std::cout << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * num_runsFPGA + exec_time0 << " us\n"
              << "INFO: The iterations is: " << buff_host.config0[2] << "\n";
    std::cout << "-------------------------------------------------------" << std::endl;
}

void PhaseLoop_UsingFPGA_Prep(graphNew* G,
                              double opts_C_thresh,
                              double currMod,
                              int numThreads,
                              // Updated variables
                              double& totTimeColoring,
                              int* colors,
                              KMemorys_host& buff_host) {
    int edgeNum;
    int numColors = Phaseloop_UsingFPGA_InitColorBuff(G, colors, numThreads, totTimeColoring);
    double time1 = 0;
    time1 = omp_get_wtime();
    assert(numColors < COLORS);
    long vertexNum = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    long NE = G->numEdges;
    long NEx2 = NE << 1;
    long NE1 = NEx2 < (1 << 26) ? NEx2 : (1 << 26); // 256MB/sizeof(int/float)=64M

    long cnt_e = 0;
    for (int i = 0; i < vertexNum + 1; i++) {
        buff_host.offsets[i] = (int)vtxPtr[i];
    }
    edgeNum = buff_host.offsets[vertexNum];
    for (int i = 0; i < vertexNum; i++) {
        int adj1 = vtxPtr[i];
        int adj2 = vtxPtr[i + 1];
        for (int j = adj1; j < adj2; j++) {
            if (cnt_e < NE1) {
                buff_host.indices[j] = (int)vtxInd[j].tail;
                buff_host.weights[j] = vtxInd[j].weight;
            } else {
                buff_host.indices2[j - NE1] = (int)vtxInd[j].tail;
                buff_host.weights2[j - NE1] = vtxInd[j].weight;
            }
            cnt_e++;
        }
    }
    for (int i = 0; i < vertexNum; i++) {
        buff_host.colorAxi[i] = colors[i];
    }
    buff_host.config0[0] = vertexNum;
    buff_host.config0[1] = numColors;
    buff_host.config0[2] = 0;
    buff_host.config0[3] = edgeNum;
    buff_host.config1[0] = opts_C_thresh;
    buff_host.config1[1] = currMod;
    time1 = omp_get_wtime() - time1;
}

void PhaseLoop_UsingFPGA(long NV,
                         double opts_C_thresh,
                         bool opts_coloring,
                         double opts_threshold,
                         int numThreads,
                         double& currMod,
                         graphNew*& G,
                         long*& C,
                         long*& C_orig,
                         int& totItr,
                         KMemorys_host& buff_host,
                         KMemorys_clBuff& buff_cl,
                         cl::Kernel& kernel_louvain,
                         cl::CommandQueue& q,
                         int& num_runsFPGA,
                         int*& colors,
                         int& totTimeE2E,
                         double& totTimeColoring) {
    long vertexNum = G->numVertices;
    bool isLargeEdge = G->numEdges > (1 << 25);
    num_runsFPGA++;
    struct timeval tstartE2E, tendE2E;
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<std::vector<cl::Event> > kernel_evt0(1);
    std::vector<std::vector<cl::Event> > kernel_evt1(1);
    kernel_evt0[0].resize(1);
    kernel_evt1[0].resize(1);

    PhaseLoop_UsingFPGA_Prep(G, opts_C_thresh, currMod, numThreads, totTimeColoring, colors, buff_host);
    gettimeofday(&tstartE2E, 0);
    PhaseLoop_UsingFPGA_1_KernelSetup(isLargeEdge, kernel_louvain, ob_in, ob_out, buff_cl);
    PhaseLoop_UsingFPGA_2_DataWriteTo(q, kernel_evt0, ob_in);
    PhaseLoop_UsingFPGA_3_KernelRun(q, kernel_evt0, kernel_evt1, kernel_louvain);
    PhaseLoop_UsingFPGA_4_DataReadBack(q, kernel_evt1, ob_out);
    PhaseLoop_UsingFPGA_5_KernelFinish(q);
    gettimeofday(&tendE2E, 0);
    PhaseLoop_UsingFPGA_Post(vertexNum, num_runsFPGA, buff_host, tstartE2E, tendE2E, kernel_evt1, C, totItr, currMod,
                             totTimeE2E);
}

void inline PhaseLoop_MapHostBuff_prune(long NV,
                                        long NE_mem_1,
                                        long NE_mem_2,
                                        std::vector<cl_mem_ext_ptr_t>& mext_in,
                                        cl::Context& context,
                                        KMemorys_host_prune& buff_host) {
    buff_host.config0 = aligned_alloc<int64_t>(6); // zyl
    buff_host.config1 = aligned_alloc<DWEIGHT>(4);
    buff_host.offsets = aligned_alloc<int>(NV + 1);
    buff_host.indices = aligned_alloc<int>(NE_mem_1);
    buff_host.offsetsdup = aligned_alloc<int>(NV + 1);   //
    buff_host.indicesdup = aligned_alloc<int>(NE_mem_1); //
    buff_host.weights = aligned_alloc<float>(NE_mem_1);
    buff_host.flag = aligned_alloc<ap_uint<8> >(NV);
    buff_host.flagUpdate = aligned_alloc<ap_uint<8> >(NV);
    if (NE_mem_2 > 0) {
        buff_host.indices2 = aligned_alloc<int>(NE_mem_2);
        buff_host.indicesdup2 = aligned_alloc<int>(NE_mem_2);
    }
    if (NE_mem_2 > 0) {
        buff_host.weights2 = aligned_alloc<float>(NE_mem_2);
    }
    buff_host.cidPrev = aligned_alloc<int>(NV);
    buff_host.cidCurr = aligned_alloc<int>(NV);
    buff_host.cidSizePrev = aligned_alloc<int>(NV);
    buff_host.totPrev = aligned_alloc<float>(NV);
    buff_host.cidSizeCurr = aligned_alloc<int>(NV);
    buff_host.totCurr = aligned_alloc<float>(NV);
    buff_host.cidSizeUpdate = aligned_alloc<int>(NV);
    buff_host.totUpdate = aligned_alloc<float>(NV);
    buff_host.cWeight = aligned_alloc<float>(NV);
    buff_host.colorAxi = aligned_alloc<int>(NV);
    buff_host.colorInx = aligned_alloc<int>(NV);

    ap_uint<CSRWIDTHS>* axi_offsets = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.offsets);
    ap_uint<CSRWIDTHS>* axi_indices = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indices);
    ap_uint<CSRWIDTHS>* axi_offsetsdup = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.offsetsdup);
    ap_uint<CSRWIDTHS>* axi_indicesdup = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indicesdup);
    ap_uint<CSRWIDTHS>* axi_weights = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.weights);
    ap_uint<CSRWIDTHS>* axi_indices2;
    ap_uint<CSRWIDTHS>* axi_indicesdup2;
    ap_uint<CSRWIDTHS>* axi_weights2;
    if (NE_mem_2 > 0) {
        axi_indices2 = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indices2);
        axi_indicesdup2 = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.indicesdup2);
        axi_weights2 = reinterpret_cast<ap_uint<CSRWIDTHS>*>(buff_host.weights2);
    }
    ap_uint<DWIDTHS>* axi_cidPrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidPrev);
    ap_uint<DWIDTHS>* axi_cidCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidCurr);
    ap_uint<DWIDTHS>* axi_cidSizePrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizePrev);
    ap_uint<DWIDTHS>* axi_totPrev = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totPrev);
    ap_uint<DWIDTHS>* axi_cidSizeCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizeCurr);
    ap_uint<DWIDTHS>* axi_totCurr = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totCurr);
    ap_uint<DWIDTHS>* axi_cidSizeUpdate = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cidSizeUpdate);
    ap_uint<DWIDTHS>* axi_totUpdate = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.totUpdate);
    ap_uint<DWIDTHS>* axi_cWeight = reinterpret_cast<ap_uint<DWIDTHS>*>(buff_host.cWeight);
    ap_uint<COLORWIDTHS>* axi_colorAxi = reinterpret_cast<ap_uint<COLORWIDTHS>*>(buff_host.colorAxi);
    ap_uint<COLORWIDTHS>* axi_colorInx = reinterpret_cast<ap_uint<COLORWIDTHS>*>(buff_host.colorInx);
    ap_uint<8>* axi_flag = reinterpret_cast<ap_uint<8>*>(buff_host.flag);
    ap_uint<8>* axi_flagUpdate = reinterpret_cast<ap_uint<8>*>(buff_host.flagUpdate);

    // DDR Settings
    mext_in[0] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, buff_host.config0, 0};
    mext_in[1] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, buff_host.config1, 0};
    mext_in[2] = {(unsigned int)(4) | XCL_MEM_TOPOLOGY, axi_offsets, 0};
    mext_in[3] = {(unsigned int)(0) | XCL_MEM_TOPOLOGY, axi_indices, 0};
    mext_in[4] = {(unsigned int)(2) | XCL_MEM_TOPOLOGY, axi_weights, 0};
    if (NE_mem_2 > 0) {
        mext_in[3 + 18] = {(unsigned int)(1) | XCL_MEM_TOPOLOGY, axi_indices2, 0};
        mext_in[4 + 18] = {(unsigned int)(3) | XCL_MEM_TOPOLOGY, axi_weights2, 0};
    }
    mext_in[5] = {(unsigned int)(6) | XCL_MEM_TOPOLOGY, axi_colorAxi, 0};
    mext_in[6] = {(unsigned int)(8) | XCL_MEM_TOPOLOGY, axi_colorInx, 0};
    mext_in[7] = {(unsigned int)(10) | XCL_MEM_TOPOLOGY, axi_cidPrev, 0};
    mext_in[8] = {(unsigned int)(12) | XCL_MEM_TOPOLOGY, axi_cidSizePrev, 0};
    mext_in[9] = {(unsigned int)(14) | XCL_MEM_TOPOLOGY, axi_totPrev, 0};
    mext_in[10] = {(unsigned int)(16) | XCL_MEM_TOPOLOGY, axi_cidCurr, 0};
    mext_in[11] = {(unsigned int)(18) | XCL_MEM_TOPOLOGY, axi_cidSizeCurr, 0};
    mext_in[12] = {(unsigned int)(20) | XCL_MEM_TOPOLOGY, axi_totCurr, 0};
    mext_in[13] = {(unsigned int)(22) | XCL_MEM_TOPOLOGY, axi_cidSizeUpdate, 0};
    mext_in[14] = {(unsigned int)(24) | XCL_MEM_TOPOLOGY, axi_totUpdate, 0};
    mext_in[15] = {(unsigned int)(26) | XCL_MEM_TOPOLOGY, axi_cWeight, 0};

    mext_in[16] = {(unsigned int)(27) | XCL_MEM_TOPOLOGY, axi_offsetsdup, 0};
    // mext_in[17] = {(unsigned int)(28) | XCL_MEM_TOPOLOGY, axi_indicesdup, 0};
    mext_in[17] = {(unsigned int)(28) | XCL_MEM_TOPOLOGY, axi_indicesdup, 0}; //| (unsigned int)(29)
    if (NE_mem_2 > 0) {
        mext_in[20] = {(unsigned int)(29) | XCL_MEM_TOPOLOGY, axi_indicesdup2, 0};
    }

    mext_in[18] = {(unsigned int)(7) | XCL_MEM_TOPOLOGY, axi_flag, 0};
    mext_in[19] = {(unsigned int)(9) | XCL_MEM_TOPOLOGY, axi_flagUpdate, 0};
}
void inline PhaseLoop_MapClBuff_prune(long NV,
                                      long NE_mem_1,
                                      long NE_mem_2,
                                      std::vector<cl_mem_ext_ptr_t>& mext_in,
                                      cl::Context& context,
                                      KMemorys_clBuff_prune& buff_cl) {
    // Create device buffer and map dev buf to host buf
    int flag_RW = CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_WRITE;
    int flag_RD = CL_MEM_EXT_PTR_XILINX | CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY;

    buff_cl.db_config0 = cl::Buffer(context, flag_RW, sizeof(int64_t) * (6), &mext_in[0]); // zyl
    buff_cl.db_config1 = cl::Buffer(context, flag_RW, sizeof(DWEIGHT) * (4), &mext_in[1]);

    buff_cl.db_offsets = cl::Buffer(context, flag_RD, sizeof(int) * (NV + 1), &mext_in[2]);
    buff_cl.db_indices = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_1, &mext_in[3]);
    buff_cl.db_weights = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_1, &mext_in[4]);
    if (NE_mem_2 > 0) {
        buff_cl.db_indices2 = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_2, &mext_in[3 + 18]);
        buff_cl.db_weights2 = cl::Buffer(context, flag_RD, sizeof(int) * NE_mem_2, &mext_in[4 + 18]);
    }
    buff_cl.db_colorAxi = cl::Buffer(context, flag_RD, sizeof(int) * (NV), &mext_in[5]);
    buff_cl.db_colorInx = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[6]);
    buff_cl.db_cidPrev = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[7]);
    buff_cl.db_cidSizePrev = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[8]);
    buff_cl.db_totPrev = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[9]);
    buff_cl.db_cidCurr = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[10]);
    buff_cl.db_cidSizeCurr = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[11]);
    buff_cl.db_totCurr = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[12]);
    buff_cl.db_cidSizeUpdate = cl::Buffer(context, flag_RW, sizeof(int) * (NV), &mext_in[13]);
    buff_cl.db_totUpdate = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[14]);
    buff_cl.db_cWeight = cl::Buffer(context, flag_RW, sizeof(float) * (NV), &mext_in[15]);

    buff_cl.db_offsetsdup = cl::Buffer(context, flag_RD, sizeof(int) * (NV + 1), &mext_in[16]);
    buff_cl.db_indicesdup = cl::Buffer(context, flag_RD, sizeof(int) * (NE_mem_1), &mext_in[17]);

    buff_cl.db_flag = cl::Buffer(context, flag_RW, sizeof(ap_uint<8>) * (NV), &mext_in[18]);
    buff_cl.db_flagUpdate = cl::Buffer(context, flag_RW, sizeof(ap_uint<8>) * (NV), &mext_in[19]);
    // printf("INFO:  sizeof(ap_uint<8>) = %d",  sizeof(ap_uint<8>) );

    if (NE_mem_2 != 0) buff_cl.db_indicesdup2 = cl::Buffer(context, flag_RD, sizeof(int) * (NE_mem_2), &mext_in[20]);
}
void UsingFPGA_MapHostClBuff_prune(long NV,
                                   long NE_mem_1,
                                   long NE_mem_2,
                                   cl::Context& context,
                                   KMemorys_host_prune& buff_host,
                                   KMemorys_clBuff_prune& buff_cl) {
    std::vector<cl_mem_ext_ptr_t> mext_in(NUM_PORT_KERNEL + 7);
    PhaseLoop_MapHostBuff_prune(NV, NE_mem_1, NE_mem_2, mext_in, context, buff_host);
    PhaseLoop_MapClBuff_prune(NV, NE_mem_1, NE_mem_2, mext_in, context, buff_cl);
}
void PhaseLoop_UsingFPGA_1_KernelSetup_prune(bool isLargeEdge,
                                             cl::Kernel& kernel_louvain,
                                             std::vector<cl::Memory>& ob_in,
                                             std::vector<cl::Memory>& ob_out,
                                             KMemorys_clBuff_prune& buff_cl) {
    // Data transfer from host buffer to device buffer
    ob_in.push_back(buff_cl.db_config0);
    ob_in.push_back(buff_cl.db_config1);
    ob_in.push_back(buff_cl.db_offsets);
    ob_in.push_back(buff_cl.db_indices);
    if (isLargeEdge) ob_in.push_back(buff_cl.db_indices2);
    ob_in.push_back(buff_cl.db_weights);
    if (isLargeEdge) ob_in.push_back(buff_cl.db_weights2);
    ob_in.push_back(buff_cl.db_cidCurr);
    ob_in.push_back(buff_cl.db_cidSizePrev);
    ob_in.push_back(buff_cl.db_cidSizeUpdate);
    ob_in.push_back(buff_cl.db_cidSizeCurr);
    ob_in.push_back(buff_cl.db_totPrev);
    ob_in.push_back(buff_cl.db_totUpdate);
    ob_in.push_back(buff_cl.db_totCurr);
    ob_in.push_back(buff_cl.db_cWeight);
    ob_in.push_back(buff_cl.db_colorAxi);
    ob_in.push_back(buff_cl.db_colorInx);
    ob_in.push_back(buff_cl.db_cidPrev);
    ob_in.push_back(buff_cl.db_offsetsdup);
    ob_in.push_back(buff_cl.db_indicesdup);
    if (isLargeEdge) ob_in.push_back(buff_cl.db_indicesdup2);
    ob_in.push_back(buff_cl.db_flag);
    ob_in.push_back(buff_cl.db_flagUpdate);

    ob_out.push_back(buff_cl.db_config0);
    ob_out.push_back(buff_cl.db_config1);
    ob_out.push_back(buff_cl.db_cidPrev);

    kernel_louvain.setArg(0, buff_cl.db_config0);        // config0
    kernel_louvain.setArg(1, buff_cl.db_config1);        // config1
    kernel_louvain.setArg(2, buff_cl.db_offsets);        // offsets
    kernel_louvain.setArg(3, buff_cl.db_indices);        // indices
    kernel_louvain.setArg(4, buff_cl.db_weights);        // weights
    kernel_louvain.setArg(5, buff_cl.db_colorAxi);       // colorAxi
    kernel_louvain.setArg(6, buff_cl.db_colorInx);       // colorInx
    kernel_louvain.setArg(7, buff_cl.db_cidPrev);        // cidPrev
    kernel_louvain.setArg(8, buff_cl.db_cidSizePrev);    // cidSizePrev
    kernel_louvain.setArg(9, buff_cl.db_totPrev);        // totPrev
    kernel_louvain.setArg(10, buff_cl.db_cidCurr);       // cidCurr
    kernel_louvain.setArg(11, buff_cl.db_cidSizeCurr);   // cidSizeCurr
    kernel_louvain.setArg(12, buff_cl.db_totCurr);       // totCurr
    kernel_louvain.setArg(13, buff_cl.db_cidSizeUpdate); // cUpdate
    kernel_louvain.setArg(14, buff_cl.db_totUpdate);     // totCurr
    kernel_louvain.setArg(15, buff_cl.db_cWeight);       // cWeight
    kernel_louvain.setArg(16, buff_cl.db_offsetsdup);    // offsets
    kernel_louvain.setArg(17, buff_cl.db_indicesdup);    // indices
    kernel_louvain.setArg(18, buff_cl.db_flag);          // offsets
    kernel_louvain.setArg(19, buff_cl.db_flagUpdate);    // indices
    std::cout << "INFO: Finish kernel setup" << std::endl;
}

void PhaseLoop_UsingFPGA_Post_par_prune(long vertexNum,
                                        int num_runsFPGA,
                                        KMemorys_host_prune& buff_host,
                                        struct timeval& tstartE2E,
                                        struct timeval& tendE2E,
                                        std::vector<std::vector<cl::Event> >& kernel_evt1,
                                        // output
                                        long* C,
                                        int& totItr,
                                        double& currMod,
                                        int& totTimeE2E) {
    // updating
    totItr += buff_host.config0[2];
    currMod = buff_host.config1[1];
    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    unsigned long timeStart, timeEnd;
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    int exec_timeE2E = diff(&tendE2E, &tstartE2E);
    totTimeE2E += exec_timeE2E;
    // showing
    unsigned long exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish kernel execution" << std::endl;
    std::cout << "INFO: Average execution per run: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;
    std::cout << "INFO: FPGA execution time of " << num_runsFPGA << " runs:" << exec_timeE2E << " us\n"
              << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * num_runsFPGA + exec_time0 << " us\n"
              << "INFO: The iterations is: " << buff_host.config0[2] << "\n";
    std::cout << "-------------------------------------------------------" << std::endl;
}

double PhaseLoop_UsingFPGA_Prep_Init_buff_host_prune(int numColors,
                                                     long NVl,
                                                     graphNew* G,
                                                     long* M,
                                                     double opts_C_thresh,
                                                     double currMod,
                                                     // Updated variables
                                                     int* colors,
                                                     KMemorys_host_prune& buff_host) {
    int edgeNum;
    double time1 = omp_get_wtime();
    assert(numColors < COLORS);
    long vertexNum = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    long NE = G->numEdges;
    long NEx2 = NE << 1;
    long NE1 = NEx2 < (1 << 26) ? NEx2 : (1 << 26); // 256MB/sizeof(int/float)=64M

    long cnt_e = 0;

    for (int i = 0; i < vertexNum + 1; i++) {
        buff_host.offsets[i] = (int)vtxPtr[i];
        if (i != vertexNum) {
            if (M[i] < 0) {
                buff_host.offsets[i] = (int)(0x80000000 | (unsigned int)vtxPtr[i]);
            }
        } else {
            buff_host.offsets[i] = (int)(vtxPtr[i]);
        }
        buff_host.offsetsdup[i] = buff_host.offsets[i]; // zyl
    }
    edgeNum = buff_host.offsets[vertexNum];

    for (int i = 0; i < vertexNum; i++) {
        int adj1 = vtxPtr[i];
        int adj2 = vtxPtr[i + 1];
        buff_host.flag[i] = 0;       // zyl
        buff_host.flagUpdate[i] = 0; // zyl
        for (int j = adj1; j < adj2; j++) {
            if (cnt_e < NE1) {
                buff_host.indices[j] = (int)vtxInd[j].tail;
                buff_host.indicesdup[j] = (int)vtxInd[j].tail;
                buff_host.weights[j] = vtxInd[j].weight;
            } else {
                buff_host.indices2[j - NE1] = (int)vtxInd[j].tail;
                buff_host.indicesdup2[j - NE1] = (int)vtxInd[j].tail;
                buff_host.weights2[j - NE1] = vtxInd[j].weight;
            }
            cnt_e++;
        }
    }
    for (int i = 0; i < vertexNum; i++) {
        buff_host.colorAxi[i] = colors[i];
    }
    buff_host.config0[0] = vertexNum;
    buff_host.config0[1] = numColors;
    buff_host.config0[2] = 0;
    buff_host.config0[3] = edgeNum;
    buff_host.config0[4] = 0;   // renumber numClusters
    buff_host.config0[5] = NVl; // ghost number

    buff_host.config1[0] = opts_C_thresh;
    buff_host.config1[1] = currMod;
    time1 = omp_get_wtime() - time1;
    return time1;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////

void runLouvainWithFPGA(graphNew* G,  // Input graphNew, undirectioned
                        long* C_orig, // Output
                        char* opts_xclbinPath,
                        bool opts_coloring,
                        long opts_minGraphSize,
                        double opts_threshold,
                        double opts_C_thresh,
                        int numThreads) {
    long NV = G->numVertices;
    long NE_org = G->numEdges;
    long numClusters;
    /* Check graphNew size, limited by hardware features */
    assert(NV < MAXNV);
    assert(NE_org < MAXNE);

    /* For coloring */
    int* colors;
    if (opts_coloring) {
        colors = (int*)malloc(G->numVertices * sizeof(int));
        assert(colors != 0);
    }

    /* To build new hierarchical graphNews*/
    long* C = (long*)malloc(NV * sizeof(long));
    assert(C != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) {
        C[i] = -1;
    }

    /*******************************/
    /* FPGA-related data structures*/
    /*******************************/
    /* platform related operations */
    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    cl::Program::Binaries xclBins = xcl::import_binary_file(opts_xclbinPath);
    devices.resize(1);
    cl::Program program(context, devices, xclBins);
    cl::Kernel kernel_louvain(program, "kernel_louvain");
    printf("INFO: kernel has been created\n");
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    /* Memories mapping */
    KMemorys_host buff_host;
    KMemorys_clBuff buff_cl;
    long NE_mem = NE_org * 2; // number for real edge to be stored in memory
    long NE_mem_1 = NE_mem < (1 << 26) ? NE_mem : (1 << 26);
    long NE_mem_2 = NE_mem - NE_mem_1;
    UsingFPGA_MapHostClBuff(NV, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);

    /*******************************/
    /* Loop for phase(s)           */
    /*******************************/
    double totTimeClustering = 0;    // time accumulator for clustering by CPU
    double totTimeBuildingPhase = 0; // time accumulator for Building new graphNew for next phase by CPU
    double totTimeColoring = 0;      // time accumulator for coloring
    double prevMod = -1;             // Last-phase modularity
    double currMod = -1;             // Current modularity
    int phase = 1;                   // Total phase counter
    int totTimeE2E = 0;              // FPGA E2E time accumulator
    int num_runsFPGA = 0;            // FPGA calling times counter
    int totItr = 0;                  // Total iteration counter
    bool nonColor = false;           // Make sure that at least one phase with lower opts_threshold runs
    bool isItrStop = false;
    while (!isItrStop) {
        printf("===============================\n");
        printf("Phase %d\n", phase);
        printf("===============================\n");
        prevMod = currMod;
        bool isUsingFPGA = ((opts_coloring) && (G->numVertices > opts_minGraphSize) && (nonColor == false));

        if (isUsingFPGA) {
            PhaseLoop_UsingFPGA(NV, opts_C_thresh, opts_coloring, opts_threshold, numThreads, currMod, G, C, C_orig,
                                totItr, buff_host, buff_cl, kernel_louvain, q, /*kernel's parameter*/
                                num_runsFPGA, colors, totTimeE2E, totTimeColoring);
        } else {
            PhaseLoop_UsingCPU(opts_threshold, numThreads, currMod, G, C, C_orig, totItr, nonColor, totTimeClustering);
        }
        /* General post-processing for both FPGA and CPU */
        isItrStop = PhaseLoop_CommPostProcessing(NV, numThreads, opts_threshold, opts_coloring, prevMod, currMod, G, C,
                                                 C_orig, nonColor, phase, totItr, numClusters, totTimeBuildingPhase);
    } // End of while(1) = End of Louvain

    /* Print information*/
    printf("********************************************\n");
    printf("*********    Compact Summary   *************\n");
    printf("********************************************\n");
    printf("Number of threads              : %d\n", numThreads);
    printf("Total number of phases         : %d\n", phase);
    printf("Total number of iterations     : %d\n", totItr);
    printf("Final number of clusters       : %ld\n", numClusters);
    printf("Final modularity               : %lf\n", prevMod);
    printf("Total time for clustering      : %lf\n", totTimeClustering);
    printf("Total time for building phases : %lf\n", totTimeBuildingPhase);
    printf("Total E2E time(s)              : %lf\n", (1.0 * totTimeE2E * 1e-6));
    if (opts_coloring) {
        printf("Total time for coloring        : %lf\n", totTimeColoring);
    }
    printf("********************************************\n");
    printf("TOTAL TIME                     : %lf\n",
           ((1.0 * totTimeE2E * 1e-6) + totTimeClustering + totTimeBuildingPhase + totTimeColoring));
    printf("********************************************\n");

    /* Clean up memories */
    free(C);
    if (G != 0) {
        free(G->edgeListPtrs);
        free(G->edgeList);
        free(G);
    }
    if (opts_coloring) {
        if (colors != 0) free(colors);
    }
    buff_host.freeMem();
} // End of runMultiPhaseLouvainAlgorithm()

void runLouvainWithFPGA_demo(graphNew* G,
                             long* C_orig,
                             char* opts_xclbinPath,
                             bool opts_coloring,
                             long opts_minGraphSize,
                             double opts_threshold,
                             double opts_C_thresh,
                             int numThreads) {
    long NV = G->numVertices;
    long NE_org = G->numEdges;
    long numClusters;
    assert(NV < MAXNV);
    assert(NE_org < MAXNE);

    int* colors;
    if (opts_coloring) {
        colors = (int*)malloc(G->numVertices * sizeof(int));
        assert(colors != 0);
    }
    long* C = (long*)malloc(NV * sizeof(long));
    assert(C != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) {
        C[i] = -1;
    }

    std::vector<cl::Device> devices = xcl::get_xil_devices();
    cl::Device device = devices[0];
    cl::Context context(device);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    cl::Program::Binaries xclBins = xcl::import_binary_file(opts_xclbinPath);
    devices.resize(1);
    cl::Program program(context, devices, xclBins);
    cl::Kernel kernel_louvain(program, "kernel_louvain");
    printf("INFO: kernel has been created\n");
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);

    KMemorys_host buff_host;
    KMemorys_clBuff buff_cl;
    long NE_mem = NE_org * 2; // number for real edge to be stored in memory
    long NE_mem_1 = NE_mem < (MAXNV) ? NE_mem : (MAXNV);
    long NE_mem_2 = NE_mem - NE_mem_1;
    UsingFPGA_MapHostClBuff(NV, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);

    double totTimeClustering = 0;    // time accumulator for clustering by CPU
    double totTimeBuildingPhase = 0; // time accumulator for Building new graphNew for next phase by CPU
    double totTimeColoring = 0;      // time accumulator for coloring
    double prevMod = -1;             // Last-phase modularity
    double currMod = -1;             // Current modularity
    int phase = 1;                   // Total phase counter
    int totTimeE2E = 0;              // FPGA E2E time accumulator
    int num_runsFPGA = 0;            // FPGA calling times counter
    int totItr = 0;                  // Total iteration counter
    bool nonColor = false;           // Make sure that at least one phase with lower opts_threshold runs
    bool isItrStop = false;
    while (!isItrStop) {
        printf("===============================\n");
        printf("Phase %d\n", phase);
        printf("===============================\n");
        prevMod = currMod;
        bool isUsingFPGA = ((opts_coloring) && (G->numVertices > opts_minGraphSize) && (nonColor == false));

        if (isUsingFPGA) {
            long vertexNum = G->numVertices;
            bool isLargeEdge = G->numEdges > (MAXNV / 2);
            num_runsFPGA++;
            struct timeval tstartE2E, tendE2E;
            std::vector<cl::Memory> ob_in;
            std::vector<cl::Memory> ob_out;
            std::vector<std::vector<cl::Event> > kernel_evt0(1);
            std::vector<std::vector<cl::Event> > kernel_evt1(1);
            kernel_evt0[0].resize(1);
            kernel_evt1[0].resize(1);

            PhaseLoop_UsingFPGA_Prep(G, opts_C_thresh, currMod, numThreads, totTimeColoring, colors, buff_host);

            gettimeofday(&tstartE2E, 0);
            { /* 5-Step flow for using FPGA */
                PhaseLoop_UsingFPGA_1_KernelSetup(isLargeEdge, kernel_louvain, ob_in, ob_out, buff_cl);

                PhaseLoop_UsingFPGA_2_DataWriteTo(q, kernel_evt0, ob_in);

                PhaseLoop_UsingFPGA_3_KernelRun(q, kernel_evt0, kernel_evt1, kernel_louvain);

                PhaseLoop_UsingFPGA_4_DataReadBack(q, kernel_evt1, ob_out);

                PhaseLoop_UsingFPGA_5_KernelFinish(q);
            }
            gettimeofday(&tendE2E, 0);

            PhaseLoop_UsingFPGA_Post(vertexNum, num_runsFPGA, buff_host, tstartE2E, tendE2E, kernel_evt1, C, totItr,
                                     currMod, totTimeE2E);
        } else {
            PhaseLoop_UsingCPU(opts_threshold, numThreads, currMod, G, C, C_orig, totItr, nonColor, totTimeClustering);
        }
        isItrStop = PhaseLoop_CommPostProcessing(NV, numThreads, opts_threshold, opts_coloring, prevMod, currMod, G, C,
                                                 C_orig, nonColor, phase, totItr, numClusters, totTimeBuildingPhase);
    }

    printf("********************************************\n");
    printf("*********    Compact Summary   *************\n");
    printf("********************************************\n");
    printf("Number of threads              : %d\tm=%d \t thhd=%lf \t thhd_c=%lf\n", numThreads, opts_minGraphSize,
           opts_threshold, opts_C_thresh);
    printf("Total number of phases         : %d\n", phase);
    printf("Total number of iterations     : %d\n", totItr);
    printf("Final number of clusters       : %ld\n", numClusters);
    printf("Final modularity               : %lf\n", prevMod);
    printf("Total time for clustering      : %lf\n", totTimeClustering);
    printf("Total time for building phases : %lf\n", totTimeBuildingPhase);
    printf("Total E2E time(s)              : %lf\n", (1.0 * totTimeE2E * 1e-6));
    if (opts_coloring) {
        printf("Total time for coloring        : %lf\n", totTimeColoring);
    }
    printf("********************************************\n");
    printf("TOTAL TIME                     : %lf\n",
           ((1.0 * totTimeE2E * 1e-6) + totTimeClustering + totTimeBuildingPhase + totTimeColoring));
    printf("********************************************\n");

    /* Clean up memories */
    free(C);
    if (G != 0) {
        free(G->edgeListPtrs);
        free(G->edgeList);
        free(G);
    }
    if (opts_coloring) {
        if (colors != 0) free(colors);
    }
    buff_host.freeMem();
} // End of runMultiPhaseLouvainAlgorithm()

double PhaseLoop_UsingFPGA_Prep_Init_buff_host(int numColors,
                                               graphNew* G,
                                               long* M,
                                               double opts_C_thresh,
                                               double currMod,
                                               // Updated variables
                                               int* colors,
                                               KMemorys_host& buff_host) {
    int edgeNum;
    double time1 = omp_get_wtime();
    assert(numColors < COLORS);
    long vertexNum = G->numVertices;
    long* vtxPtr = G->edgeListPtrs;
    edge* vtxInd = G->edgeList;
    long NE = G->numEdges;
    long NEx2 = NE << 1;
    long NE1 = NEx2 < (1 << 26) ? NEx2 : (1 << 26); // 256MB/sizeof(int/float)=64M

    long cnt_e = 0;
    for (int i = 0; i < vertexNum + 1; i++) {
        buff_host.offsets[i] = (int)vtxPtr[i];
        if (i != vertexNum) {
            if (M[i] < 0) buff_host.offsets[i] = (int)(0x80000000 | (unsigned int)vtxPtr[i]);
        } else
            buff_host.offsets[i] = (int)(vtxPtr[i]);
    }
    edgeNum = buff_host.offsets[vertexNum];
    for (int i = 0; i < vertexNum; i++) {
        int adj1 = vtxPtr[i];
        int adj2 = vtxPtr[i + 1];
        for (int j = adj1; j < adj2; j++) {
            if (cnt_e < NE1) {
                buff_host.indices[j] = (int)vtxInd[j].tail;
                buff_host.weights[j] = vtxInd[j].weight;
            } else {
                buff_host.indices2[j - NE1] = (int)vtxInd[j].tail;
                buff_host.weights2[j - NE1] = vtxInd[j].weight;
            }
            cnt_e++;
        }
    }
    for (int i = 0; i < vertexNum; i++) {
        buff_host.colorAxi[i] = colors[i];
    }
    buff_host.config0[0] = vertexNum;
    buff_host.config0[1] = numColors;
    buff_host.config0[2] = 0;
    buff_host.config0[3] = edgeNum;
    buff_host.config1[0] = opts_C_thresh;
    buff_host.config1[1] = currMod;
    time1 = omp_get_wtime() - time1;
    return time1;
}
double PhaseLoop_UsingFPGA_Prep_Read_buff_host(long vertexNum,
                                               KMemorys_host& buff_host,
                                               int phase,
                                               int eachItrs[MAX_NUM_PHASE],
                                               // output
                                               long* C,
                                               int& totItr,
                                               double& currMod) {
    double time1 = omp_get_wtime();
    // updating
    eachItrs[phase - 1] = buff_host.config0[2];
    totItr += buff_host.config0[2];
    currMod = buff_host.config1[1];
    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    time1 = omp_get_wtime() - time1;
    return time1;
}
double PhaseLoop_UsingFPGA_Prep_Read_buff_host(long vertexNum,
                                               KMemorys_host& buff_host,
                                               int& eachItrs,
                                               // output
                                               long* C,
                                               int& eachItr,
                                               double& currMod) {
    double time1 = omp_get_wtime();
    // updating
    eachItrs = buff_host.config0[2];
    eachItr = buff_host.config0[2];
    currMod = buff_host.config1[1];
    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    time1 = omp_get_wtime() - time1;
    return time1;
}

double PhaseLoop_UsingFPGA_Prep_Read_buff_host_prune(long vertexNum,
                                                     KMemorys_host_prune& buff_host,
                                                     int& eachItrs,
                                                     // output
                                                     long* C,
                                                     int& eachItr,
                                                     double& currMod,
                                                     long& numClusters) {
    double time1 = omp_get_wtime();
    // updating
    eachItrs = buff_host.config0[2];
    eachItr = buff_host.config0[2];
    currMod = buff_host.config1[1];
    numClusters = buff_host.config0[4];

    for (int i = 0; i < vertexNum; i++) {
        C[i] = (long)buff_host.cidPrev[i];
    }
    time1 = omp_get_wtime() - time1;
    return time1;
}
unsigned long diff2(const struct timeval* newTime, const struct timeval* oldTime) {
    return (newTime->tv_sec - oldTime->tv_sec) * 1000000 + (newTime->tv_usec - oldTime->tv_usec);
}
void PhaseLoop_UsingFPGA_Post_par_noRead(double* p_eachTimeE2E,
                                         int num_runsFPGA,
                                         int num_iter,
                                         struct timeval& tstartE2E,
                                         struct timeval& tendE2E,
                                         // std::vector<std::vector<cl::Event> > &kernel_evt1,
                                         double currMod,
                                         int& totTimeE2E) {
    // unsigned long timeStart, timeEnd;
    // kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_START, &timeStart);
    // kernel_evt1[0][0].getProfilingInfo(CL_PROFILING_COMMAND_END, &timeEnd);
    int exec_timeE2E = diff2(&tendE2E, &tstartE2E);
    totTimeE2E += exec_timeE2E;
    // showing
    // unsigned long exec_time0 = (timeEnd - timeStart) / 1000.0;
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish kernel execution" << std::endl;
    // std::cout << "INFO: Average execution per run: " << exec_time0 << " us\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    std::cout << "INFO: Finish E2E execution" << std::endl;
    // std::cout << "INFO: FPGA execution time of " << num_runsFPGA << " runs:" << exec_timeE2E << " us\n"
    printf("INFO: FPGA execution time of %1d runs:%9d us    Iteration times = %2d currMod=%f\n", num_runsFPGA,
           exec_timeE2E, num_iter, currMod);
    // std::cout << "INFO: Average execution per run: " << exec_timeE2E - exec_time0 * num_runsFPGA + exec_time0
    //         << " us\n"
    std::cout << "INFO: The iterations is: " << num_iter << "\n";
    std::cout << "-------------------------------------------------------" << std::endl;
    *p_eachTimeE2E = (1.0 * exec_timeE2E * 1e-6);
}
double PhaseLoop_CommPostProcessing_par(GLV* pglv_orig,
                                        GLV* pglv_iter,
                                        int numThreads,
                                        double opts_threshold,
                                        bool opts_coloring,
                                        // modified:
                                        bool& nonColor,
                                        int& phase,
                                        int& totItr,
                                        long& numClusters,
                                        double& totTimeBuildingPhase,
                                        double& time_renum,
                                        double& time_C,
                                        double& time_M,
                                        double& time_buid,
                                        double& time_set);
double PhaseLoop_CommPostProcessing(GLV* pglv_orig,
                                    GLV* pglv_iter,
                                    int numThreads,
                                    double opts_threshold,
                                    bool opts_coloring,
                                    // modified:
                                    bool& nonColor,
                                    int& phase,
                                    int& totItr,
                                    long& numClusters,
                                    double& totTimeBuildingPhase,
                                    double& time_renum,
                                    double& time_C,
                                    double& time_M,
                                    double& time_buid,
                                    double& time_set);

void ConsumingOnePhase(GLV* pglv_iter,
                       double opts_C_thresh,
                       KMemorys_clBuff& buff_cl,
                       KMemorys_host& buff_host,
                       cl::Kernel& kernel_louvain,
                       cl::CommandQueue& q,
                       int& eachItrs,
                       double& currMod,
                       double& eachTimeInitBuff,
                       double& eachTimeReadBuff) {
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<std::vector<cl::Event> > kernel_evt0(1);
    std::vector<std::vector<cl::Event> > kernel_evt1(1);
    kernel_evt0[0].resize(1);
    kernel_evt1[0].resize(1);
    bool isLargeEdge = pglv_iter->G->numEdges > (MAXNV / 2);

    eachTimeInitBuff = PhaseLoop_UsingFPGA_Prep_Init_buff_host(pglv_iter->numColors, pglv_iter->G, pglv_iter->M,
                                                               opts_C_thresh, currMod, pglv_iter->colors, buff_host);

    PhaseLoop_UsingFPGA_1_KernelSetup(isLargeEdge, kernel_louvain, ob_in, ob_out, buff_cl);
    std::cout << "\t\PhaseLoop_UsingFPGA_1_KernelSetup Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_2_DataWriteTo(q, kernel_evt0, ob_in);
    std::cout << "\t\PhaseLoop_UsingFPGA_2_DataWriteTo Device Available: "
              << std::endl; //  << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_3_KernelRun(q, kernel_evt0, kernel_evt1, kernel_louvain);
    std::cout << "\t\PhaseLoop_UsingFPGA_3_KernelRun Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_4_DataReadBack(q, kernel_evt1, ob_out);
    std::cout << "\t\PhaseLoop_UsingFPGA_4_DataReadBack Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_5_KernelFinish(q);
    std::cout << "\t\PhaseLoop_UsingFPGA_5_KernelFinish Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    eachTimeReadBuff =
        PhaseLoop_UsingFPGA_Prep_Read_buff_host(pglv_iter->NV, buff_host, eachItrs, pglv_iter->C, eachItrs, currMod);
}
void runLouvainWithFPGA_demo_par_core(bool hasGhost,
                                      int id_dev,
                                      GLV* pglv_orig,
                                      GLV* pglv_iter,
                                      char* opts_xclbinPath,
                                      bool opts_coloring,
                                      long opts_minGraphSize,
                                      double opts_threshold,
                                      double opts_C_thresh,
                                      int numThreads) {
    double timePrePre;
    double timePrePre_dev;
    double timePrePre_xclbin;
    double timePrePre_buff;

    timePrePre = omp_get_wtime();
    long NV_orig = pglv_orig->G->numVertices;
    long NE_orig = pglv_orig->G->numEdges;
    long NE_max = NE_orig; // hasGhost?(1.4 * NE_orig):NE_orig;//1.4 is Experience value, make clbuffer enough space
    long numClusters;

    assert(NV_orig < MAXNV);
    assert(NE_orig < MAXNE);

    timePrePre_dev = omp_get_wtime();
    std::vector<cl::Device> devices = xcl::get_xil_devices();

    int d_num = devices.size();
    if (id_dev >= d_num) {
        printf("\033[1;31;40mERROR\033[0m: id_dev(%d) >= d_num(%d)\n", id_dev, d_num);
        return;
    }
    cl::Device device = devices[id_dev];
    cl::Context context(device);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    timePrePre_dev = omp_get_wtime() - timePrePre_dev;

    timePrePre_xclbin = omp_get_wtime();
    cl::Program::Binaries xclBins = xcl::import_binary_file(opts_xclbinPath);

    std::vector<cl::Device> devices2;
    devices2.push_back(device);
    cl::Program program(context, devices2, xclBins);
    cl::Kernel kernel_louvain(program, "kernel_louvain");
    printf("INFO: kernel has been created\n");
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE);
    timePrePre_xclbin = omp_get_wtime() - timePrePre_xclbin;
    /* Memories mapping */
    KMemorys_host buff_host;
    KMemorys_clBuff buff_cl;
    long NE_mem = NE_max * 2; // number for real edge to be stored in memory
    long NE_mem_1 = NE_mem < (MAXNV) ? NE_mem : (MAXNV);
    long NE_mem_2 = NE_mem - NE_mem_1;
    timePrePre_buff = omp_get_wtime();
    UsingFPGA_MapHostClBuff(NV_orig, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);
    timePrePre_buff = omp_get_wtime() - timePrePre_buff;

    std::cout << "\t\t UsingFPGA_MapHostClBuff Device Available: " << device.getInfo<CL_DEVICE_AVAILABLE>()
              << std::endl;

    double totTimeClustering = 0;    // time accumulator for clustering by CPU
    double totTimeBuildingPhase = 0; // time accumulator for Building new graphNew for next phase by CPU
    double totTimeColoring = 0;      // time accumulator for coloring
    double prevMod = -1;             // Last-phase modularity
    double currMod = -1;             // Current modularity

    int phase = 1;        // Total phase counter
    int totTimeE2E = 0;   // FPGA E2E time accumulator
    int num_runsFPGA = 0; // FPGA calling times counter
    int totItr = 0;       // Total iteration counter

    bool nonColor = false; // Make sure that at least one phase with lower opts_threshold runs
    bool isItrStop = false;

    double totTimeInitBuff = 0;
    double totTimeReadBuff = 0;
    double totTimeReGraph = 0;
    double totTimeE2E_2 = 0;
    double totTimeFeature = 0;
    double eachTimeInitBuff[MAX_NUM_PHASE];
    double eachTimeReadBuff[MAX_NUM_PHASE];
    double eachTimeReGraph[MAX_NUM_PHASE];
    double eachTimeE2E_2[MAX_NUM_PHASE];
    double eachTimePhase[MAX_NUM_PHASE];
    double eachTimeFeature[MAX_NUM_PHASE];
    double eachMod[MAX_NUM_PHASE];
    int eachItrs[MAX_NUM_PHASE];
    long eachClusters[MAX_NUM_PHASE];

    double eachNum[MAX_NUM_PHASE];
    double eachC[MAX_NUM_PHASE];
    double eachM[MAX_NUM_PHASE];
    double eachBuild[MAX_NUM_PHASE];
    double eachSet[MAX_NUM_PHASE];

    timePrePre = omp_get_wtime() - timePrePre;
    double totTimeAll = omp_get_wtime();

    double eachTimeE2E[MAX_NUM_PHASE];

    while (!isItrStop) {
        eachTimePhase[phase - 1] = omp_get_wtime();
        printf("===============================\n");
        printf("Phase %d\n", phase);
        printf("===============================\n");
        {
            eachTimeE2E_2[phase - 1] = omp_get_wtime();
            ConsumingOnePhase(pglv_iter, opts_C_thresh, buff_cl, buff_host, kernel_louvain, q, eachItrs[phase - 1],
                              currMod, eachTimeInitBuff[phase - 1], eachTimeReadBuff[phase - 1]
                              //,ob_in, ob_out,
                              );
            eachTimeE2E_2[phase - 1] = omp_get_wtime() - eachTimeE2E_2[phase - 1];
        }
        totTimeInitBuff += eachTimeInitBuff[phase - 1];
        totTimeReadBuff += eachTimeReadBuff[phase - 1];
        totTimeE2E_2 += eachTimeE2E_2[phase - 1];
        totItr += eachItrs[phase - 1];

        if (hasGhost)
            eachTimeReGraph[phase - 1] = PhaseLoop_CommPostProcessing_par(
                pglv_orig, pglv_iter, numThreads, opts_threshold, opts_coloring, nonColor, phase, totItr, numClusters,
                totTimeBuildingPhase, eachNum[phase - 1], eachC[phase - 1], eachM[phase - 1], eachBuild[phase - 1],
                eachSet[phase - 1]);
        else
            eachTimeReGraph[phase - 1] = PhaseLoop_CommPostProcessing(
                pglv_orig, pglv_iter, numThreads, opts_threshold, opts_coloring, nonColor, phase, totItr, numClusters,
                totTimeBuildingPhase, eachNum[phase - 1], eachC[phase - 1], eachM[phase - 1], eachBuild[phase - 1],
                eachSet[phase - 1]);
        eachClusters[phase - 1] = numClusters;
        eachMod[phase - 1] = currMod;
        totTimeReGraph += eachTimeReGraph[phase - 1];
        pglv_orig->NC = numClusters;
        pglv_iter->NC = numClusters;
        // pglv_orig->Q = currMod;
        // pglv_iter->Q = currMod;
        eachTimeFeature[phase - 1] = omp_get_wtime();
        eachTimeFeature[phase - 1] = omp_get_wtime() - eachTimeFeature[phase - 1];
        totTimeFeature += eachTimeFeature[phase - 1];
        if ((phase > MAX_NUM_PHASE) || (totItr > MAX_NUM_TOTITR)) {
            isItrStop = true; // Break if too many phases or iterations
        } else if ((currMod - prevMod) <= opts_threshold) {
            isItrStop = true;
        } else if (pglv_iter->NV <= opts_minGraphSize) {
            isItrStop = true;
        } else {
            isItrStop = false;
            phase++;
        }
        prevMod = currMod;
        if (isItrStop == false)
            eachTimePhase[phase - 2] = omp_get_wtime() - eachTimePhase[phase - 2];
        else
            eachTimePhase[phase - 1] = omp_get_wtime() - eachTimePhase[phase - 1];
        if (NE_max < pglv_iter->G->numEdges) {
            printf("WARNING: ReMapBuff as %d < %d \n", NE_max, pglv_iter->G->numEdges);
            NE_max = pglv_iter->G->numEdges;
            long NE_mem = NE_max * 2; // number for real edge to be stored in memory
            long NE_mem_1 = NE_mem < (MAXNV) ? NE_mem : (MAXNV);
            long NE_mem_2 = NE_mem - NE_mem_1;
            double tm = omp_get_wtime();
            buff_host.freeMem();
            UsingFPGA_MapHostClBuff(pglv_iter->G->numVertices, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);
            timePrePre_buff += omp_get_wtime() - tm;
        }

    } // End of while(1) = End of Louvain
    totTimeAll = omp_get_wtime() - totTimeAll;

    double timePostPost = omp_get_wtime();
    /* Print information*/
    PrintReport_MultiPhase(opts_coloring, opts_minGraphSize, opts_threshold, opts_C_thresh, numThreads, numClusters,
                           totTimeClustering, totTimeBuildingPhase, totTimeColoring, prevMod, phase, totTimeE2E_2,
                           num_runsFPGA, totItr, eachTimeE2E_2, eachMod, eachItrs, eachClusters);
    /* Print additional information*/
    PrintReport_MultiPhase_2(phase, totTimeE2E_2, totTimeAll, totTimeInitBuff, totTimeReadBuff, totTimeReGraph,
                             totTimeFeature, eachTimeInitBuff, eachTimeReadBuff, eachTimeReGraph, eachTimeE2E_2,
                             eachTimePhase, eachTimeFeature, eachNum, eachC, eachM, eachBuild, eachSet);
    buff_host.freeMem();

    double timePostPost_feature = omp_get_wtime();
    pglv_orig->PushFeature(phase, totItr, totTimeE2E_2, true); // isUsingFPGA);
    pglv_iter->PushFeature(phase, totItr, totTimeE2E_2, true); // isUsingFPGA);
    timePostPost_feature = omp_get_wtime() - timePostPost_feature;

    timePostPost = omp_get_wtime() - timePostPost;
    printf("TOTAL PrePre                   : %lf = %f(dev) + %lf(bin) + %lf(buff) +%lf\n", timePrePre, timePrePre_dev,
           timePrePre_xclbin, timePrePre_buff,
           timePrePre - timePrePre_dev - timePrePre_xclbin - timePrePre_buff); // eachTimePhase
    printf("TOTAL PostPost                 : %lf = %lf + %lf\n", timePostPost, timePostPost_feature,
           timePostPost - timePostPost_feature); // eachTimePhase
} // End of runM

void ConsumingOnePhase_prune(GLV* pglv_iter,
                             double opts_C_thresh,
                             KMemorys_clBuff_prune& buff_cl,
                             KMemorys_host_prune& buff_host,
                             cl::Kernel& kernel_louvain,
                             cl::CommandQueue& q,
                             int& eachItrs,
                             double& currMod,
                             long& numClusters,
                             double& eachTimeInitBuff,
                             double& eachTimeReadBuff) {
    std::vector<cl::Memory> ob_in;
    std::vector<cl::Memory> ob_out;
    std::vector<std::vector<cl::Event> > kernel_evt0(1);
    std::vector<std::vector<cl::Event> > kernel_evt1(1);
    kernel_evt0[0].resize(1);
    kernel_evt1[0].resize(1);

    bool isLargeEdge = pglv_iter->G->numEdges > (MAXNV / 2);
    eachTimeInitBuff =
        PhaseLoop_UsingFPGA_Prep_Init_buff_host_prune(pglv_iter->numColors, pglv_iter->NVl, pglv_iter->G, pglv_iter->M,
                                                      opts_C_thresh, currMod, pglv_iter->colors, buff_host);

    PhaseLoop_UsingFPGA_1_KernelSetup_prune(isLargeEdge, kernel_louvain, ob_in, ob_out, buff_cl);
    std::cout << "\t\PhaseLoop_UsingFPGA_1_KernelSetup Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_2_DataWriteTo(q, kernel_evt0, ob_in);
    std::cout << "\t\PhaseLoop_UsingFPGA_2_DataWriteTo Device Available: "
              << std::endl; //  << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_3_KernelRun(q, kernel_evt0, kernel_evt1, kernel_louvain);
    std::cout << "\t\PhaseLoop_UsingFPGA_3_KernelRun Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_4_DataReadBack(q, kernel_evt1, ob_out);
    std::cout << "\t\PhaseLoop_UsingFPGA_4_DataReadBack Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    PhaseLoop_UsingFPGA_5_KernelFinish(q);
    std::cout << "\t\PhaseLoop_UsingFPGA_5_KernelFinish Device Available: "
              << std::endl; // << device.getInfo<CL_DEVICE_AVAILABLE>() << std::endl;

    eachTimeReadBuff = PhaseLoop_UsingFPGA_Prep_Read_buff_host_prune(pglv_iter->NV, buff_host, eachItrs, pglv_iter->C,
                                                                     eachItrs, currMod, numClusters);
}
void runLouvainWithFPGA_demo_par_core_prune(bool hasGhost,
                                            int id_dev,
                                            GLV* pglv_orig,
                                            GLV* pglv_iter,
                                            char* opts_xclbinPath,
                                            bool opts_coloring,
                                            long opts_minGraphSize,
                                            double opts_threshold,
                                            double opts_C_thresh,
                                            int numThreads) {
    double timePrePre;
    double timePrePre_dev;
    double timePrePre_xclbin;
    double timePrePre_buff;

    timePrePre = omp_get_wtime();
    // double time2  = omp_get_wtime();
    long NV_orig = pglv_orig->G->numVertices;
    long NE_orig = pglv_orig->G->numEdges;
    long numClusters;

    assert(NV_orig < MAXNV);
    assert(NE_orig < MAXNE);

    timePrePre_dev = omp_get_wtime();
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    cl_int err;
    std::vector<cl::Device> devices = xcl::get_xil_devices();

    int d_num = devices.size();
    if (id_dev >= d_num) {
        printf("\033[1;31;40mERROR\033[0m: id_dev(%d) >= d_num(%d)\n", id_dev, d_num);
        return;
    }
    cl::Device device = devices[id_dev];
    cl::Context context(device, NULL, NULL, NULL, &err);
    logger.logCreateContext(err);
    std::string devName = device.getInfo<CL_DEVICE_NAME>();
    printf("INFO: Found Device=%s\n", devName.c_str());
    timePrePre_dev = omp_get_wtime() - timePrePre_dev;

    timePrePre_xclbin = omp_get_wtime();
    cl::Program::Binaries xclBins = xcl::import_binary_file(opts_xclbinPath);

    std::vector<cl::Device> devices2;
    devices2.push_back(device);
    devices2.resize(1);

    cl::Program program(context, devices2, xclBins, NULL, &err);
    logger.logCreateProgram(err);
    cl::Kernel kernel_louvain(program, "kernel_louvain", &err);
    logger.logCreateKernel(err);
    cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE | CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE, &err);
    logger.logCreateCommandQueue(err);
    timePrePre_xclbin = omp_get_wtime() - timePrePre_xclbin;
    /* Memories mapping */
    KMemorys_host_prune buff_host;
    KMemorys_clBuff_prune buff_cl;
    long NE_max = NE_orig;    // hasGhost?(1.4 * NE_orig):NE_orig;//Experience value, make clbuffer enough space
    long NE_mem = NE_max * 2; // number for real edge to be stored in memory
    long NE_mem_1 = NE_mem < (MAXNV) ? NE_mem : (MAXNV);
    long NE_mem_2 = NE_mem - NE_mem_1;

    timePrePre_buff = omp_get_wtime();
    UsingFPGA_MapHostClBuff_prune(NV_orig, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);
    timePrePre_buff = omp_get_wtime() - timePrePre_buff;

    std::cout << "\t\t UsingFPGA_MapHostClBuff Device Available: " << device.getInfo<CL_DEVICE_AVAILABLE>()
              << std::endl;

    double totTimeClustering = 0;    // time accumulator for clustering by CPU
    double totTimeBuildingPhase = 0; // time accumulator for Building new graphNew for next phase by CPU
    double totTimeColoring = 0;      // time accumulator for coloring
    double prevMod = -1;             // Last-phase modularity
    double currMod = -1;             // Current modularity

    int phase = 1;        // Total phase counter
    int totTimeE2E = 0;   // FPGA E2E time accumulator
    int num_runsFPGA = 0; // FPGA calling times counter
    int totItr = 0;       // Total iteration counter

    bool nonColor = false; // Make sure that at least one phase with lower opts_threshold runs
    bool isItrStop = false;

    double totTimeInitBuff = 0;
    double totTimeReadBuff = 0;
    double totTimeReGraph = 0;
    double totTimeE2E_2 = 0;
    double totTimeFeature = 0;
    double eachTimeInitBuff[MAX_NUM_PHASE];
    double eachTimeReadBuff[MAX_NUM_PHASE];
    double eachTimeReGraph[MAX_NUM_PHASE];
    double eachTimeE2E_2[MAX_NUM_PHASE];
    double eachTimePhase[MAX_NUM_PHASE];
    double eachTimeFeature[MAX_NUM_PHASE];
    double eachMod[MAX_NUM_PHASE];
    int eachItrs[MAX_NUM_PHASE];
    long eachClusters[MAX_NUM_PHASE];

    double eachNum[MAX_NUM_PHASE];
    double eachC[MAX_NUM_PHASE];
    double eachM[MAX_NUM_PHASE];
    double eachBuild[MAX_NUM_PHASE];
    double eachSet[MAX_NUM_PHASE];

    timePrePre = omp_get_wtime() - timePrePre;
    double totTimeAll = omp_get_wtime();

    double eachTimeE2E[MAX_NUM_PHASE];

    while (!isItrStop) {
        eachTimePhase[phase - 1] = omp_get_wtime();
        printf("===============================\n");
        printf("Phase %d\n", phase);
        printf("===============================\n");
        {
            eachTimeE2E_2[phase - 1] = omp_get_wtime();
            ConsumingOnePhase_prune(pglv_iter, opts_C_thresh, buff_cl, buff_host, kernel_louvain, q,
                                    eachItrs[phase - 1], currMod, numClusters, eachTimeInitBuff[phase - 1],
                                    eachTimeReadBuff[phase - 1]);
            eachTimeE2E_2[phase - 1] = omp_get_wtime() - eachTimeE2E_2[phase - 1];
        }
        totTimeInitBuff += eachTimeInitBuff[phase - 1];
        totTimeReadBuff += eachTimeReadBuff[phase - 1];
        totTimeE2E_2 += eachTimeE2E_2[phase - 1];
        totItr += eachItrs[phase - 1];
        if (hasGhost)
            eachTimeReGraph[phase - 1] = PhaseLoop_CommPostProcessing_par(
                pglv_orig, pglv_iter, numThreads, opts_threshold, opts_coloring, nonColor, phase, totItr, numClusters,
                totTimeBuildingPhase, eachNum[phase - 1], eachC[phase - 1], eachM[phase - 1], eachBuild[phase - 1],
                eachSet[phase - 1]);
        else
            eachTimeReGraph[phase - 1] = PhaseLoop_CommPostProcessing(
                pglv_orig, pglv_iter, numThreads, opts_threshold, opts_coloring, nonColor, phase, totItr, numClusters,
                totTimeBuildingPhase, eachNum[phase - 1], eachC[phase - 1], eachM[phase - 1], eachBuild[phase - 1],
                eachSet[phase - 1]);
        eachClusters[phase - 1] = numClusters;
        eachMod[phase - 1] = currMod;
        totTimeReGraph += eachTimeReGraph[phase - 1];
        pglv_orig->NC = numClusters;
        pglv_iter->NC = numClusters;
        // pglv_orig->Q = currMod;
        // pglv_iter->Q = currMod;
        eachTimeFeature[phase - 1] = omp_get_wtime();
        eachTimeFeature[phase - 1] = omp_get_wtime() - eachTimeFeature[phase - 1];
        totTimeFeature += eachTimeFeature[phase - 1];
        if ((phase > MAX_NUM_PHASE) || (totItr > MAX_NUM_TOTITR)) {
            isItrStop = true; // Break if too many phases or iterations
        } else if ((currMod - prevMod) <= opts_threshold) {
            isItrStop = true;
        } else if (pglv_iter->NV <= opts_minGraphSize) {
            isItrStop = true;
        } else {
            isItrStop = false;
            phase++;
        }
        prevMod = currMod;
        if (isItrStop == false)
            eachTimePhase[phase - 2] = omp_get_wtime() - eachTimePhase[phase - 2];
        else
            eachTimePhase[phase - 1] = omp_get_wtime() - eachTimePhase[phase - 1];
        if (NE_max < pglv_iter->G->numEdges) {
            printf("WARNING: ReMapBuff as %d < %d \n", NE_max, pglv_iter->G->numEdges);
            NE_max = pglv_iter->G->numEdges;
            long NE_mem = NE_max * 2; // number for real edge to be stored in memory
            long NE_mem_1 = NE_mem < (MAXNV) ? NE_mem : (MAXNV);
            long NE_mem_2 = NE_mem - NE_mem_1;
            double tm = omp_get_wtime();
            buff_host.freeMem();
            UsingFPGA_MapHostClBuff_prune(pglv_iter->G->numVertices, NE_mem_1, NE_mem_2, context, buff_host, buff_cl);
            timePrePre_buff += omp_get_wtime() - tm;
        }
    } // End of while(1) = End of Louvain
    totTimeAll = omp_get_wtime() - totTimeAll;

    double timePostPost = omp_get_wtime();
    /* Print information*/
    PrintReport_MultiPhase(opts_coloring, opts_minGraphSize, opts_threshold, opts_C_thresh, numThreads, numClusters,
                           totTimeClustering, totTimeBuildingPhase, totTimeColoring, prevMod, phase, totTimeE2E_2,
                           num_runsFPGA, totItr, eachTimeE2E_2, eachMod, eachItrs, eachClusters);
    /* Print additional information*/
    PrintReport_MultiPhase_2(phase, totTimeE2E_2, totTimeAll, totTimeInitBuff, totTimeReadBuff, totTimeReGraph,
                             totTimeFeature, eachTimeInitBuff, eachTimeReadBuff, eachTimeReGraph, eachTimeE2E_2,
                             eachTimePhase, eachTimeFeature, eachNum, eachC, eachM, eachBuild, eachSet);
    buff_host.freeMem();

    double timePostPost_feature = omp_get_wtime();
    pglv_orig->PushFeature(phase, totItr, totTimeE2E_2, true); // isUsingFPGA);
    pglv_iter->PushFeature(phase, totItr, totTimeE2E_2, true); // isUsingFPGA);
    timePostPost_feature = omp_get_wtime() - timePostPost_feature;

    timePostPost = omp_get_wtime() - timePostPost;

    printf("TOTAL PrePre                   : %lf = %f(dev) + %lf(bin) + %lf(buff) +%lf\n", timePrePre, timePrePre_dev,
           timePrePre_xclbin, timePrePre_buff,
           timePrePre - timePrePre_dev - timePrePre_xclbin - timePrePre_buff); // eachTimePhase
    printf("TOTAL PostPost                 : %lf = %lf + %lf\n", timePostPost, timePostPost_feature,
           timePostPost - timePostPost_feature); // eachTimePhase
} // End of runM

GLV* LouvainGLV_general(bool hasGhost,
                        int mode_flow,
                        int id_dev,
                        GLV* glv_src,
                        char* xclbinPath,
                        int numThreads,
                        int& id_glv,
                        long minGraphSize,
                        double threshold,
                        double C_threshold,
                        bool isParallel,
                        int numPhase) {
    double time1 = omp_get_wtime();
    assert(glv_src);

    GLV* glv = glv_src->CloneSelf(id_glv);
    assert(glv);
    glv->SetName_lv(glv->ID, glv_src->ID);
    if (mode_flow == MD_NORMAL) {
        printf("\033[1;37;40mINFO\033[0m: START Kernel with partition ! \n");
        runLouvainWithFPGA_demo_par_core(hasGhost, id_dev, glv_src, glv, xclbinPath, true, minGraphSize, threshold,
                                         C_threshold, numThreads);
    } else if (mode_flow == MD_FAST) {
        printf("\033[1;37;40mINFO\033[0m: START PRUNE ! \n");
        runLouvainWithFPGA_demo_par_core_prune(hasGhost, id_dev, glv_src, glv, xclbinPath, true, minGraphSize,
                                               threshold, C_threshold, numThreads);
    }

    time1 = omp_get_wtime() - time1;
    return glv;
}

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
                                     int numPhase) {
    GLV* glv_t;
    // int id_glv_dev= id_glv+(id_dev)*num_dev;
    for (int p = id_dev; p < num_par; p += num_dev) {
        double time1 = omp_get_wtime();
        int id_glv_dev = id_glv + p;
        glv_t = LouvainGLV_general(hasGhost, flowMode, id_dev, par_src[p], xclbinPath, numThreads, id_glv_dev,
                                   minGraphSize, threshold, C_threshold, isParallel, numPhase);
        par_lved[p] = glv_t;
        // pushList(glv_t);
        timeLv[p] = omp_get_wtime() - time1;
    }
}
