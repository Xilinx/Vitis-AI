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

#ifndef _XILINXLOUVAIN_H_
#define _XILINXLOUVAIN_H_
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include <map>
#include <vector>
#include <unistd.h>
#include <ap_int.h>
#include "xcl2.hpp"
#include <list>
using namespace std;

#define DWIDTHS (256)
#define CSRWIDTHS (256)
#define COLORWIDTHS (32)
#define NUM (DWIDTHS / 32)
#define MAXNV (1 << 26)
#define MAXNE (1 << 27)
#define VERTEXS (MAXNV / NUM)
#define EDGES (MAXNE / NUM)
#define DEGREES (1 << 17)
#define COLORS (4096)
typedef double DWEIGHT;

#ifndef MULTITHREAD /* Enable multi-thread mode for using OpenMP for host code */
#define NUMTHREAD (1)
#else
#define NUMTHREAD (16) /* For number of thread, usually it is not the more the best, 16 is based on expericece*/
#endif
// For enlarge size of HBM space
#define WKARND_HBM
#define NUM_PORT_KERNEL (16)
#define MAX_NUM_PHASE (200)
#define MAX_NUM_TOTITR (10000)
#define MAX_NUM_DEV (64)

/* Start Kumar Interface */

/* Data type of the inputs */
#define SPATIAL_dataType int

/* Data type of the result */
#define SPATIAL_resDataType double

/* Data type of the id used for identifying each data. Example: Graph Node VID can be used as id whose type is
 * uint64_t*/
#define SPATIAL_idType uint64_t

/* Number of FPGA devices. Example: Each U50 card has one device. So, 3 U50 cards will have 3 FPGA devices */
#define SPATIAL_numDevices 3

/* Number of compute units (kernel instances) on each FPGA device */
#define SPATIAL_numComputUnits 2

/* Number of parallel channels on each compute unit */
#define SPATIAL_numChannels 16

typedef struct /* the edge data structure */
{
    long head;
    long tail;
    double weight;
} edge;

class graphNew {
   public:
    long numVertices;   /* Number of columns                                */
    long sVertices;     /* Number of rows: Bipartite graph: number of S vertices; T = N - S */
    long numEdges;      /* Each edge stored twice, but counted once        */
    long* edgeListPtrs; /* start vertex of edge, sorted, primary key        */
    edge* edgeList;     /* end   vertex of edge, sorted, secondary key      */
};

struct TimeLv {
    int parNo;
    int phase;
    int totItr;
    unsigned int deviceID[MAX_NUM_PHASE];
    unsigned int cuID[MAX_NUM_PHASE];
    unsigned int channelID[MAX_NUM_PHASE];

    double totTimeAll;
    double totTimeInitBuff;             // = 0;
    double totTimeReadBuff;             // = 0;
    double totTimeReGraph;              // = 0;
    double totTimeE2E_2;                // = 0;
    double totTimeE2E;                  // = 0;
    double totTimeE2E_DEV[MAX_NUM_DEV]; // = 0;
    double totTimeBuildingPhase;
    double totTimeClustering;
    double totTimeColoring;
    double totTimeFeature; // = 0;

    double timePrePre;
    double timePrePre_dev;
    double timePrePre_xclbin;
    double timePrePre_buff;

    double eachTimeInitBuff[MAX_NUM_PHASE];
    double eachTimeReadBuff[MAX_NUM_PHASE];
    double eachTimeReGraph[MAX_NUM_PHASE];
    double eachTimeE2E_2[MAX_NUM_PHASE];
    double eachTimeE2E[MAX_NUM_PHASE];
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
};

struct FeatureLV;
class GLV {
   public:
    graphNew* G;
    long NV;
    long NVl;
    long NE;
    long NElg;
    long* C;
    long NC;
    long* M;
    int* colors;
    int numColors;
    int numThreads;
    char name[256];
    int ID;
    list<FeatureLV> com_list;
    TimeLv times;

   public:
    GLV(int& id);
    ~GLV();
    void InitVar();
    void FreeMem();
    void CleanCurrentG();
    void InitByFile(char*);
    void InitByOhterG(graphNew*);
    void SetByOhterG(graphNew*);
    void SetByOhterG(graphNew* G_src, long* M_src);
    void InitG(graphNew* G_src);
    void InitC(long* C_src);
    void InitM(long* M_src);
    void InitC();
    void InitM();
    void InitColor();
    void SetM(long* M_src);
    void SetM();
    void SetC(long* C_src);
    void SetC();
    void ResetColor();
    void ResetC();
    void SyncWithG();
    void print();
    void printSimple();
    void SetName(char* nm); //{strcpy(name, nm);};
    void SetName_par(int, int, long, long, int);
    void SetName_loadg(int ID_curr, char* path);
    void SetName_ParLvMrg(int num_par, int ID_src);
    void SetName_lv(int, int);
    void SetName_cat(int ID_src1, int ID_src2);
    void PushFeature(int ph, int iter, double time, bool FPGA);
    void printFeature();
    GLV* CloneSelf(int& id_glv);
    GLV* RstColorWithGhost();
    void RstNVlByM();
    void RstNVElg();
};

struct FeatureLV {
    double totalTot;
    double totalIn;
    double m;
    double Q;
    long NV;
    long NE;
    long NC; // number of community/number of clusters
    int No_phase;
    int Num_iter;
    double time;
    bool isFPGA;
    void init();
    FeatureLV();
    FeatureLV(GLV* glv);
    double ComputeQ(GLV* glv);
    double ComputeQ2(GLV* glv);
    void PrintFeature();
};

extern "C" {
/* A pointer to any object that needs to be passed around. Object can be a C++
   object. The host code library can cast it to right object based upon context
   and have access to its fields. The client can just hold on to it and passes it
   back to the host library for providing the right data to operate on.
   As per need, client can include a separate header .hpp file provided by the
   host library to gain access to specific fields.
 */
typedef void* xaiHandle;

typedef enum e_xai_algorithm {
    xai_algo_cosinesim_ss,
    xai_algo_jackardsim_ss,
    xai_algo_knearestneighbor
} t_xai_algorithm;

/* Struct type to hold id and result */
typedef struct s_xai_id_value_pair {
    SPATIAL_idType id;
    SPATIAL_resDataType value;
} t_xai_id_value_pair, *p_xai_id_value_pair;

/* Struct type to hold all the necessary state of one invocation instance */

typedef struct s_xai_context {
    const char* xclbin_filename; // Name of the xilinx xclbin file name
    unsigned int num_devices;    // Use specified number of devices if available
    unsigned int vector_length;  // Number of elements in each vector/record: l_n
    unsigned int num_result;     // Number of result elements: l_k
    unsigned int num_CUs;        // Number of CUs in xclbin
    unsigned int num_Channels;   // Number of channels in xclbin
    unsigned int start_index;    // Starting index of compute elements in the vector/record
    unsigned int element_size;   // Number of bytes needed to store one element
} t_xai_context, *p_xai_context;

/* Function pointer types for the interface functions */
typedef xaiHandle (*t_fp_xai_open)(p_xai_context);
typedef int (*t_fp_xai_cache_reset)(xaiHandle);
typedef int (*t_fp_xai_cache)(xaiHandle, SPATIAL_dataType*, unsigned int);
typedef int (*t_fp_xai_cache_flush)(xaiHandle);
typedef xaiHandle (*t_fp_xai_execute)(xaiHandle, t_xai_algorithm, SPATIAL_dataType*, p_xai_id_value_pair);
typedef int (*t_fp_xai_close)(xaiHandle xaiInstance);
typedef int (*t_fp_xai_louvain_main)(int argc, char** argv);

/* Open Device, Allocate Buffer, Doanload FPGA bit stream and return an object
   as xaihandle that captures full state of this particualar instance of
   opening */
xaiHandle xai_open(p_xai_context context);

/* Reset writing to beginning of the arg_Y buffer */
int xai_cache_reset(xaiHandle xaiInstance);

/* Incrementally buffer for arg_Y in the x86 attached memory */
int xai_cache(xaiHandle xaiInstance, SPATIAL_dataType* populationVec, unsigned int numElements);

/* Write input arg arg_Y to alveo attached memory. */
int xai_cache_flush(xaiHandle xaiInstance);

/* Run algorith "algo" on data already stored on alveo memory using xai_write and data passed
   using arg_X and return result as an array of s_xai_id_value_pair
*/
int xai_execute(xaiHandle xaiInstance, t_xai_algorithm algo, SPATIAL_dataType* arg_X, p_xai_id_value_pair result);

/* Free up memory and close device */
int xai_close(xaiHandle xaiInstance);

/* Run louvain_main executable */
int xai_louvain_main(int argc, char** argv);
}
/* End Kumar Interface */

struct KMemorys_host {
    int64_t* config0;
    DWEIGHT* config1;
    // Graph data
    int* offsets;
    int* indices;
    float* weights;
    int* indices2;
    float* weights2;
    int* colorAxi;
    int* colorInx;
    // Updated Community info
    int* cidPrev;
    int* cidCurr;
    // Iterated size of communities
    int* cidSizePrev;
    int* cidSizeUpdate;
    int* cidSizeCurr;
    // Iterated tot of communities
    float* totPrev;
    float* totUpdate;
    float* totCurr;
    //
    float* cWeight;
    KMemorys_host() { memset((void*)this, 0, sizeof(KMemorys_host)); }
    void freeMem() {
        free(config0);
        free(config1);
        //
        free(offsets);
        free(indices);
        free(weights);
        if (indices2) free(indices2);
        if (weights2) free(weights2);
        free(colorAxi);
        free(colorInx);
        //
        free(cidPrev);
        free(cidCurr);
        //
        free(totPrev);
        free(totUpdate);
        free(totCurr);
        //
        free(cidSizeCurr);
        free(cidSizeUpdate);
        free(cidSizePrev);
        //
        free(cWeight);
    }
};

struct KMemorys_clBuff {
    cl::Buffer db_config0;
    cl::Buffer db_config1;
    //
    cl::Buffer db_offsets;
    cl::Buffer db_indices;
    cl::Buffer db_weights;
    cl::Buffer db_indices2;
    cl::Buffer db_weights2;
    cl::Buffer db_colorAxi;
    cl::Buffer db_colorInx;
    //
    cl::Buffer db_cidPrev;
    cl::Buffer db_cidCurr;
    //
    cl::Buffer db_cidSizePrev;
    cl::Buffer db_cidSizeUpdate;
    cl::Buffer db_cidSizeCurr;
    //
    cl::Buffer db_totPrev;
    cl::Buffer db_totUpdate;
    cl::Buffer db_totCurr;
    //
    cl::Buffer db_cWeight;
};

struct KMemorys_host_prune {
    int64_t* config0;
    DWEIGHT* config1;
    // Graph data
    int* offsets;
    int* indices;
    float* weights;
    int* indices2;
    float* weights2;
    int* colorAxi;
    int* colorInx;
    // Updated Community info
    int* cidPrev;
    int* cidCurr;
    // Iterated size of communities
    int* cidSizePrev;
    int* cidSizeUpdate;
    int* cidSizeCurr;
    // Iterated tot of communities
    float* totPrev;
    float* totUpdate;
    float* totCurr;
    //
    float* cWeight;
    //
    int* offsetsdup;
    int* indicesdup;
    int* indicesdup2;
    ap_uint<8>* flag;
    ap_uint<8>* flagUpdate;
    KMemorys_host_prune() { memset((void*)this, 0, sizeof(KMemorys_host_prune)); }
    void freeMem() {
        free(config0);
        free(config1);
        //
        free(offsets);
        free(indices);
        free(weights);
        if (indices2) free(indices2);
        if (weights2) free(weights2);
        free(colorAxi);
        free(colorInx);
        //
        free(cidPrev);
        free(cidCurr);
        //
        free(totPrev);
        free(totUpdate);
        free(totCurr);
        //
        free(cidSizeCurr);
        free(cidSizeUpdate);
        free(cidSizePrev);
        //
        free(cWeight);
        // free
        free(offsetsdup);
        free(indicesdup);
        if (indicesdup2) free(indicesdup2);
        free(flag);
        free(flagUpdate);
    }
};

struct KMemorys_clBuff_prune {
    cl::Buffer db_config0;
    cl::Buffer db_config1;
    //
    cl::Buffer db_offsets;
    cl::Buffer db_indices;
    cl::Buffer db_weights;
    cl::Buffer db_indices2;
    cl::Buffer db_weights2;
    cl::Buffer db_colorAxi;
    cl::Buffer db_colorInx;
    //
    cl::Buffer db_cidPrev;
    cl::Buffer db_cidCurr;
    //
    cl::Buffer db_cidSizePrev;
    cl::Buffer db_cidSizeUpdate;
    cl::Buffer db_cidSizeCurr;
    //
    cl::Buffer db_totPrev;
    cl::Buffer db_totUpdate;
    cl::Buffer db_totCurr;
    //
    cl::Buffer db_cWeight;
    //
    cl::Buffer db_offsetsdup;
    cl::Buffer db_indicesdup;
    cl::Buffer db_indicesdup2;
    cl::Buffer db_flag;
    cl::Buffer db_flagUpdate;
};

int host_ParserParameters(int argc,
                          char** argv,
                          double& opts_C_thresh,   //; //Threshold with coloring on
                          long& opts_minGraphSize, //; //Min |V| to enable coloring
                          double& opts_threshold,  //; //Value of threshold
                          int& opts_ftype,         //; //File type
                          char* opts_inFile,       //;
                          bool& opts_coloring,     //
                          bool& opts_output,       //;
                          bool& opts_VF,           //;
                          char* opts_xclbinPath);

graphNew* host_PrepareGraph(int opts_ftype, char* opts_inFile, bool opts_VF);

int host_writeOut(char* opts_inFile, long NV_begin, long* C_orig);

void runLouvainWithFPGA(graphNew* G,  // Input graph, undirectioned
                        long* C_orig, // Output
                        char* opts_xclbinPath,
                        bool opts_coloring,
                        long opts_minGraphSize,
                        double opts_threshold,
                        double opts_C_thresh,
                        int numThreads);

long renumberClustersContiguously_ghost(long* C, long size, long NV_l);

// Only used by L3

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
                                  double& totTimeBuildingPhase);

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

double PhaseLoop_UsingFPGA_Prep_Init_buff_host(int numColors,
                                               graphNew* G,
                                               long* M,
                                               double opts_C_thresh,
                                               double* currMod,
                                               // Updated variables
                                               int* colors,
                                               KMemorys_host* buff_host);
double PhaseLoop_UsingFPGA_Prep_Read_buff_host(long vertexNum,
                                               KMemorys_host* buff_host,
                                               int* eachItrs,
                                               // output
                                               long* C,
                                               int* eachItr,
                                               double* currMod);

double PhaseLoop_UsingFPGA_Prep_Init_buff_host_prune(int numColors,
                                                     graphNew* G,
                                                     long* M,
                                                     double opts_C_thresh,
                                                     double* currMod,
                                                     // Updated variables
                                                     int* colors,
                                                     KMemorys_host_prune* buff_host_prune);
double PhaseLoop_UsingFPGA_Prep_Read_buff_host_prune(long vertexNum,
                                                     KMemorys_host_prune* buff_host_prune,
                                                     int* eachItrs,
                                                     // output
                                                     long* C,
                                                     int* eachItr,
                                                     double* currMod);

#endif
