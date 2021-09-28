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
#include "partition/ParLV.h"  //tmp include
#include "partition/ctrlLV.h" //tmp include
#include "partition/partitionLouvain.hpp"
#include "partition/louvainPhase.h" //tmp include
#include "string.h"
#include "xf_utils_sw/logger.hpp"

int general_findPara(int argc, char** argv, char* para) {
    for (int i = 1; i < argc; i++) {
        if (0 == strcmp(argv[i], para)) return i;
    }
    return -1;
}
int host_ParserParameters(int argc,
                          char** argv,
                          double& opts_C_thresh,   //; //Threshold with coloring on
                          long& opts_minGraphSize, //; //Min |V| to enable coloring
                          double& opts_threshold,  //; //Value of threshold
                          int& opts_ftype,         //; //File type
                          char opts_inFile[4096],  //;
                          bool& opts_coloring,     //
                          bool& opts_output,       //;
                          bool& opts_VF,           //;
                          char opts_xclbinPath[4096],
                          int& numThread,
                          int& num_par,
                          int& gh_par,
                          bool& flow_prune) {
    const int max_parameter = 100;
    bool rec[max_parameter];
    for (int i = 1; i < argc; i++) rec[i] = false;
    int has_opts_C_thresh = general_findPara(argc, argv, "-d");
    int has_opts_minGraphSize = general_findPara(argc, argv, "-m");
    int has_opts_threshold = general_findPara(argc, argv, "-t");
    int has_opts_ftype = general_findPara(argc, argv, "-f");
    int has_opts_inFile; //= general_findPara(argc, argv, "-thread");
    int has_opts_coloring = general_findPara(argc, argv, "-c");
    int has_opts_output = general_findPara(argc, argv, "-o");
    int has_opts_VF = general_findPara(argc, argv, "-v");
    int has_opts_xclbinPath = general_findPara(argc, argv, "-x");
    int has_numThread = general_findPara(argc, argv, "-thread");
    int has_num_par = general_findPara(argc, argv, "-num_par");
    int has_gh_par = general_findPara(argc, argv, "-gh_par");
    int has_flow_prune = general_findPara(argc, argv, "-prun");

    if (has_opts_C_thresh != -1) {
        rec[has_opts_C_thresh] = true;
        rec[has_opts_C_thresh + 1] = true;
        opts_C_thresh = atof(argv[has_opts_C_thresh + 1]);
    } else
        opts_C_thresh = 0.0002;
    printf("PARAMETER  opts_C_thresh = %f\n", opts_C_thresh);

    if (has_opts_minGraphSize != -1) {
        rec[has_opts_minGraphSize] = true;
        rec[has_opts_minGraphSize + 1] = true;
        opts_minGraphSize = atoi(argv[has_opts_minGraphSize + 1]);
    } else
        opts_minGraphSize = 10;
    printf("PARAMETER  has_opts_minGraphSize= %d\n", opts_minGraphSize);

    if (has_opts_threshold != -1) {
        rec[has_opts_threshold] = true;
        rec[has_opts_threshold + 1] = true;
        opts_threshold = atof(argv[has_opts_threshold + 1]);
    } else
        opts_threshold = 0.000001;
    printf("PARAMETER  opts_C_thresh= %f\n", opts_threshold);

    if (has_opts_ftype != -1) {
        rec[has_opts_ftype] = true;
        rec[has_opts_ftype + 1] = true;
        opts_ftype = atof(argv[has_opts_ftype + 1]);
    } else
        opts_ftype = 3;
    printf("PARAMETER  opts_ftype = %i\n", opts_ftype);

    if (has_opts_coloring != -1) {
        rec[has_opts_coloring] = true;
        opts_coloring = true;
    }
    printf("PARAMETER  opts_coloring = %d\n", opts_coloring);
    if (has_opts_output != -1) {
        rec[has_opts_output] = true;
        opts_output = true;
    }
    printf("PARAMETER  opts_output = %d\n", opts_output);
    if (has_opts_VF != -1) {
        rec[has_opts_VF] = true;
        opts_VF = true;
    }
    printf("PARAMETER  opts_VF = %d\n", opts_VF);
    if (has_opts_xclbinPath != -1) {
        rec[has_opts_xclbinPath] = true;
        rec[has_opts_xclbinPath + 1] = true;
        strcpy(opts_xclbinPath, argv[has_opts_xclbinPath + 1]);
        printf("PARAMETER  opts_xclbinPath = %s\n", opts_xclbinPath);
    }

    if (has_numThread != -1) {
        rec[has_numThread] = true;
        rec[has_numThread + 1] = true;
        numThread = atoi(argv[has_numThread + 1]);
    } else
        numThread = 16;
    printf("PARAMETER  numThread = %i\n", numThread);

    if (has_num_par != -1) {
        rec[has_num_par] = true;
        rec[has_num_par + 1] = true;
        num_par = atoi(argv[has_num_par + 1]);
    } else
        num_par = 1;
    printf("PARAMETER  num_par = %i\n", num_par);

    if (has_gh_par != -1) {
        rec[has_gh_par] = true;
        rec[has_gh_par + 1] = true;
        gh_par = atoi(argv[has_gh_par + 1]);
    } else
        gh_par = 1;
    printf("PARAMETER  gh_par = %i\n", gh_par);

    if (has_flow_prune != -1) {
        rec[has_flow_prune] = true;
        flow_prune = true;
    }
    printf("PARAMETER  flow_prune = %d\n", flow_prune);

    for (int i = 1; i < argc; i++) {
        // printf("i= %d rec[i]=%d\n", i , rec[i]);
        if (rec[i] == false) {
            has_opts_inFile = i;
            strcpy(opts_inFile, argv[has_opts_inFile]);
            printf("PARAMETER opts_inFile = %s\n", opts_inFile);
            break;
        }
    }
    return 0;
}
int GetNumThreadsForOpMP(int& argc, char** argv) {
    int numThreads;
    int idx_th = general_findPara(argc, argv, "-thread");
    if (idx_th != -1) {
        if (argc != (idx_th + 2)) {
            printf("ERROR for using \"-thread <num_thread> \" which should be the last 2 parameters now!!!\n");
            return -1;
        }
        numThreads = atoi(argv[argc - 1]);
        if (numThreads < 0) {
            printf("ERROR for using \"-thread  <num_thread> \" <num_thread>(%d) should be a positive number!!!\n",
                   numThreads);
            return -1;
        }
        argc -= 2;
        return numThreads;
    }
    return 16; // no thread parameters
}

int LvTest_main_par_cmd(int argc, char** argv) {
    CtrlLouvain lvc;
    int numThreads = GetNumThreadsForOpMP(argc, argv); //(general_findPara(argc, argv, "-1th")!=-1) ? 1: NUMTHREAD; //
                                                       // using fixed number of thread instead of omp_get_num_threads();
    lvc.numThreads = numThreads;
    lvc.C_threshold = 0.0001;
    int idx_KnlPath = general_findPara(argc, argv, "-x");
    if (idx_KnlPath != -1 && argc <= idx_KnlPath) {
        printf("\033[1;31;40mERROR\033[0m: Wrong number of parameters for command-line mode\n");
        return -1;
    } else {
        strcpy(lvc.xclbinPath, argv[idx_KnlPath + 1]);
        lvc.useKernel = true;
        lvc.exe_LV_DEV();
    }

    if (general_findPara(argc, argv, "-prun") != -1 || general_findPara(argc, argv, "-lvprun") != -1 ||
        general_findPara(argc, argv, "-fast") != -1)
        lvc.flowMode = MD_FAST;

    int idx_m = general_findPara(argc, argv, "-m");
    if (idx_m != -1 && argc <= idx_m) {
        printf("\033[1;31;40mERROR\033[0m: Wrong number of parameters for command-line mode\n");
        return -1;
    } else if (idx_m != -1)
        lvc.minGraphSize = atoi(argv[idx_m + 1]);

    int idx_d = general_findPara(argc, argv, "-d");
    if (idx_d != -1 && argc <= idx_d) {
        printf("\033[1;31;40mERROR\033[0m: Wrong number of parameters for command-line mode\n");
        return -1;
    } else if (idx_d != -1)
        lvc.C_threshold = atof(argv[idx_d + 1]);

    int idx = general_findPara(argc, argv, "-cmdbat");

    if (idx > 0 && argc > idx)
        return lvc.exe_batch(argv[idx + 1]);
    else
        lvc.Run();

    return 0;
}

int LvTest_main_par(int argc, char** argv) {
    // Parse Input parameters:
    double opts_C_thresh;   // Threshold with coloring on
    long opts_minGraphSize; // Min |V| to enable coloring
    double opts_threshold;  // Value of threshold
    int opts_ftype;         // File type
    char opts_inFile[4096];
    bool opts_coloring;
    bool opts_output;
    bool opts_VF;
    char opts_xclbinPath[4096];
    int num_par;
    int gh_par;
    bool usingPrune = false;
    int numThreads; /*= GetNumThreadsForOpMP(argc, argv);


   if( general_findPara(argc, argv, "-prun")!=-1 ||  general_findPara(argc, argv, "-lvprun")!=-1||
   general_findPara(argc, argv, "-fast")!=-1)
   {
           usingPrune=true;
           argc--;
   }
   host_ParserParameters(
      argc,
      argv,
      opts_C_thresh,    //double opts_C_thresh; //Threshold with coloring on
      opts_minGraphSize,//long   opts_minGraphSize; //Min |V| to enable coloring
      opts_threshold,   //double opts_threshold; //Value of threshold
      opts_ftype,       //int    opts_ftype; //File type
      opts_inFile,      //char   opts_inFile[4096];
      opts_coloring,    //bool   opts_coloring;
      opts_output,      //bool   opts_output;
      opts_VF,          //bool   opts_VF;
      opts_xclbinPath);

 */
    host_ParserParameters(argc, argv,
                          opts_C_thresh,     // double opts_C_thresh; //Threshold with coloring on
                          opts_minGraphSize, // long   opts_minGraphSize; //Min |V| to enable coloring
                          opts_threshold,    // double opts_threshold; //Value of threshold
                          opts_ftype,        // int    opts_ftype; //File type
                          opts_inFile,       // char   opts_inFile[4096];
                          opts_coloring,     // bool   opts_coloring;
                          opts_output,       // bool   opts_output;
                          opts_VF, opts_xclbinPath, num_par, gh_par, numThreads, usingPrune);

    graphNew* G = host_PrepareGraph(opts_ftype, opts_inFile, opts_VF);
    int id_glv = 0;
    GLV* pglv_src = new GLV(id_glv);
    pglv_src->SetByOhterG(G);
    long NV_begin = pglv_src->G->numVertices;
    long* C_orig = pglv_src->C; //= (long *) malloc (NV_begin * sizeof(long)); assert(C_orig != 0);

    /*  #pragma omp parallel for
      for (long i=0; i<NV_begin; i++) {
           C_orig[i] = -1;
      }
     */

    if (!usingPrune) {
        runLouvainWithFPGA_demo(G, C_orig, opts_xclbinPath, opts_coloring, opts_minGraphSize, opts_threshold,
                                opts_C_thresh, numThreads);
    } else {
        int id_dev = 0;
        int numPhase = 100;
        GLV* pglv_iter = LouvainGLV_general(true, MD_FAST, id_dev, pglv_src, opts_xclbinPath, numThreads, id_glv,
                                            opts_minGraphSize, opts_threshold, opts_C_thresh, false, numPhase);
        delete (pglv_iter);
    }

    // Check if cluster ids need to be written to a file:

    if (opts_output) host_writeOut(opts_inFile, NV_begin, C_orig);

    // Cleanup:
    // if( C_orig != 0 ) free(C_orig);
    // delete(pglv_src);
    int errs = 0;
    xf::common::utils_sw::Logger logger(std::cout, std::cerr);
    errs ? logger.error(xf::common::utils_sw::Logger::Message::TEST_FAIL)
         : logger.info(xf::common::utils_sw::Logger::Message::TEST_PASS);
    return errs;
} // End of main()

int main(int argc, char** argv) {
    if (general_findPara(argc, argv, "-cmd") != -1)
        return LvTest_main_par_cmd(argc, argv);
    else
        return LvTest_main_par(argc, argv);
}
