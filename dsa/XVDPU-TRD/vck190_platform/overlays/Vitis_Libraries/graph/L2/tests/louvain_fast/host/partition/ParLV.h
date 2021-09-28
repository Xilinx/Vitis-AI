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

#ifndef _PARLV_H_
#define _PARLV_H_
#include "string.h"
#include "stdio.h"
#include "stdlib.h"
#include "partitionLouvain.hpp"
#define MAX_PARTITION (512)
#define MAX_DEVICE (64)

struct TimePartition {
    double time_star;
    double time_done_par;
    double time_done_lv;
    double time_done_pre;
    double time_done_mg;
    double time_done_fnl;
    //
    double timePar[MAX_PARTITION];
    double timePar_all;
    double timeLv[MAX_PARTITION];
    double timeLv_dev[MAX_DEVICE];
    double timeLv_all;
    double timePre;
    double timeMerge;
    double timeFinal;
    double timeAll;
};

class ParLV {
   public:
    bool st_Partitioned;
    bool st_ParLved;
    bool st_PreMerged;
    bool st_Merged;
    bool st_FinalLved;
    bool st_Merged_ll;
    bool st_Merged_gh;
    bool isMergeGhost;
    bool isOnlyGL;
    GLV* plv_src;
    GLV* par_src[MAX_PARTITION];
    GLV* par_lved[MAX_PARTITION];
    GLV* plv_merged;
    GLV* plv_final;
    int flowMode;
    int num_par;
    int num_dev;
    long NV;
    long NVl;
    long NV_gh;
    long NE;
    long NEll;
    long NElg;
    long NEgl;
    long NEgg;
    long NEself;
    SttGPar stt[MAX_PARTITION];
    long off_src[MAX_PARTITION];
    long off_lved[MAX_PARTITION];
    long* p_v_new[MAX_PARTITION];
    /*	double timePar[MAX_PARTITION];
            double timePar_all;
            double timeLv[MAX_PARTITION];
            double timeLv_dev[MAX_DEVICE];
            double timeLv_all;
            double timePre;
            double timeMerge;
            double timeFinal;
            double timeAll;*/
    TimePartition timesPar;
    map<long, long> m_v_gh;
    // list<GLV*> lv_list;
    long max_NV;
    long max_NE;
    long max_NVl;
    long max_NElg;
    int scl_NV;
    int scl_NE;
    int scl_NVl;
    int scl_NElg;
    edge* elist;
    long* M_v;
    long NE_list_all;
    long NE_list_ll;
    long NE_list_gl;
    long NE_list_gg;
    long NV_list_all;
    long NV_list_l;
    long NV_list_g;
    bool isPrun;
    int th_prun;
    void Init(int mode);
    void Init(int mode, GLV* src, int numpar, int numdev);
    void Init(int mode, GLV* src, int num_p, int num_d, bool isPrun, int th_prun);
    ParLV();
    ~ParLV();
    void PrintSelf();
    double UpdateTimeAll(); //{timeAll = timeFinal+timeMerge+timePre+timePar_all+timeLv_all;};
    void CleanList(GLV* glv_curr, GLV* glv_temp);
    GLV* MergingPar2(int&);
    GLV* FinalLouvain(
        char*, int, int&, long minGraphSize, double threshold, double C_threshold, bool isParallel, int numPhase);

    long MergingPar2_ll();
    long MergingPar2_gh();
    long CheckGhost();
    int partition(GLV* glv_src, int& id_glv, int num, long th_size, int th_maxGhost);
    void PreMerge();
    void Addedge(edge* edges, long head, long tail, double weight, long* M_g);
    long FindGhostInLocalC(long m);
    int FindParIdx(long e_org);
    int FindParIdxByID(int id);
    pair<long, long> FindCM_1hop(int idx, long e_org);
    pair<long, long> FindCM_1hop(long e_org);
    long FindC_nhop(long m_gh);
    int AddGLV(GLV* plv);
    void PrintTime();
    void PrintTime2();
    void CleanTmpGlv();
    double TimeStar();
    double TimeDonePar();
    double TimeDoneLv();
    double TimeDonePre();
    double TimeDoneMerge();
    double TimeDoneFinal();
    double TimeAll_Done();
};
GLV* par_general(GLV* src, SttGPar* pstt, int& id_glv, long start, long end, bool isPrun, int th_prun);
GLV* par_general(GLV* src, int& id_glv, long start, long end, bool isPrun, int th_prun);

GLV* LouvainGLV_general_par(int flowMode,
                            ParLV& parlv,
                            //
                            char* xclbinPath,
                            int numThreads,
                            int& id_glv,
                            long minGraphSize,
                            double threshold,
                            double C_threshold,
                            bool isParallel,
                            int numPhase);
GLV* LouvainGLV_general_par_OneDev(int flowMode,
                                   ParLV& parlv,
                                   //
                                   char* xclbinPath,
                                   int numThreads,
                                   int& id_glv,
                                   long minGraphSize,
                                   double threshold,
                                   double C_threshold,
                                   bool isParallel,
                                   int numPhase);
GLV* LouvainGLV_general_par(int flowMode,
                            GLV* glv_orig,
                            int num_par,
                            int num_dev,
                            int isPrun,
                            int th_prun,
                            //
                            char* xclbinPath,
                            int numThreads,
                            int& id_glv,
                            long minGraphSize,
                            double threshold,
                            double C_threshold,
                            bool isParallel,
                            int numPhase);
void ParLV_general_batch_thread(int flowMode,
                                GLV* plv_orig,
                                int id_dev,
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

;
#endif
