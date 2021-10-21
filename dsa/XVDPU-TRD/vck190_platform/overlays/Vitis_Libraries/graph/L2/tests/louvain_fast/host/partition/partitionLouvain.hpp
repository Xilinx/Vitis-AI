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

#ifndef _PARTITIONLOUVAIN_H_
#define _PARTITIONLOUVAIN_H_
#include "stdlib.h"
#include <list>
using namespace std;

#define MAXGHOST_PRUN 128
// save the info of smallest ghost
struct VGMinDgr {
    long tail[MAXGHOST_PRUN];
    long dgrs[MAXGHOST_PRUN];
    double wght[MAXGHOST_PRUN];
};

void FreeG(graphNew*& G);
void printG(graphNew* G);
void printG(graphNew* G, long* C);
void printG(graphNew* G, long* C, long star, long end);
void printG(graphNew* G, long* C, long* M);
void printG(graphNew* G, long* C, long* M, long star, long end);
void printG(graphNew* G, long* C, long* M, long star, long end, bool isCid, bool isDir);
void printG(char* name, graphNew* G, long* C, long* M, long star, long end);

void InitC(long* C, long NV);
void CreateSubG(graphNew* G_src, long start, long size, graphNew* G_sub, long* M_sub2src, long* M_sub2ghost);
void CreatSubG(long head, long end_line, graphNew* G_scr, graphNew* G_des);
void CopyG(graphNew* G_scr, graphNew* G_des);
graphNew* CloneG(graphNew* G_scr);

int Phaseloop_UsingFPGA_InitColorBuff(graphNew* G, int* colors, int numThreads, double& totTimeColoring);

struct TimePar {
    double total;
    double par;
    double lv;
    double pre;
    double merge;
    double final;
};

class SttGPar {
   public:
    long num_e;
    long num_e_dir;
    long num_v;
    long start;
    long end;
    long num_v_l;
    long num_v_g;
    long num_e_ll;
    long num_e_ll_dir;
    long num_e_lg;
    long num_e_gl;
    long num_e_gg;
    map<long, long> map_v_l;
    map<long, long> map_v_g;
    map<long, long> map_v;

   public:
    SttGPar();
    SttGPar(long s, long e);
    void PrintStt();
    // void AddEdge(edge* edges, long head, long tail, long* M_g);
    void AddEdge(edge* edges, long head, long tail, double weight, long* M_g);
    void AddEdge2(edge* edges, long head, long tail, long* M_g);
    void AddEdge_org(edge* edges, long head, long tail, long* M_g);
    void EdgePruning(edge* edges,
                     long head,
                     long tail,
                     double weight,
                     long* M_g,
                     VGMinDgr& min,
                     long& num_vg,
                     long& e_dgr,
                     int th_maxGhost);
    // void CountV(graphNew* G, long star, long end);
    void CountV(graphNew* G, edge* elist, long* M_g);
    void CountV(graphNew* G, long star, long end, edge* elist, long* M_g);
    void CountV(graphNew* G, long star, long end);
    void CountVPruning(graphNew* G, long star, long end, int th_maxGhost);
    void CountV(graphNew* G);
    bool InRange(long v);
    GLV* ParNewGlv(graphNew* G, long star, long end, int& id_glv);
    GLV* ParNewGlv_Prun(graphNew* G, long star, long end, int& id_glv, int th_maxGhost);
    long findAt(VGMinDgr& gMinDgr, long tail, long dgr, long num_vg, int th_maxGhost);
};

long CountV(graphNew* G, long star, long end);
long CountV(graphNew* G);
long CountVGh(graphNew* G, long star, long end);
long CountVGh(graphNew* G);
void GetGFromEdge(graphNew* G, edge* edgeListTmp, long num_v, long num_e_dir);
long GetGFromEdge_selfloop(graphNew* G, edge* edgeListTmp, long num_v, long num_e_dir);
GLV* CloneGlv(GLV* glv_src, int& id_glv);
void cmd_runMultiPhaseLouvainAlgorithm(
    GLV* pglv, GLV* pglv_iter, long minGraphSize, double threshold, double C_threshold, int numPhase);
void cmd_runMultiPhaseLouvainAlgorithm(GLV* pglv,
                                       GLV* pglv_iter,
                                       long minGraphSize,
                                       double threshold,
                                       double C_threshold,
                                       bool isNoParallelLouvain,
                                       int numPhase);
GLV* cmd_runMultiPhaseLouvainAlgorithm(GLV* pglv,
                                       int& id_glv,
                                       long minGraphSize,
                                       double threshold,
                                       double C_threshold,
                                       bool isNoParallelLouvain,
                                       int numPhase);
GLV* cmd_runMultiPhaseLouvainAlgorithm(
    GLV* pglv, int& id_glv, long minGraphSize, double threshold, double C_threshold, int numPhase);
double cmd_lv_parallelLouvianMethod(
    graphNew* G, long* C, long* M, int nThreads, double Lower, double thresh, double* totTime, int* numItr);

;
#endif
