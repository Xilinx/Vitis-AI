// ***********************************************************************
//
//            Grappolo: A C++ library for graph clustering
//               Mahantesh Halappanavar (hala@pnnl.gov)
//               Pacific Northwest National Laboratory
//
// ***********************************************************************
//
//       Copyright (2014) Battelle Memorial Institute
//                      All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions
// are met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
// FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
// COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
// INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
// BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
// LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
// ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// ************************************************************************

#include "defs.h"
#include "xcl2.hpp"
//#include "utils.hpp"
//#include "kernel_louvain.hpp"
#include "partitionLouvain.hpp"
#define NUM_PORT_KERNEL (16)
#define WKARND_HBM         /* Enable 2-HBM for storing large graphNew, to be as a default configuration */
//#define MAX_NUM_PHASE   (200)
//#define MAX_NUM_TOTITR  (10000)
#define ITR_STOP        (1)
#define ITR_CONTINUE    (0)


using namespace std;
// WARNING: This will overwrite the original graphNew data structure to
//         minimize memory footprint
// Return: C_orig will hold the cluster ids for vertices in the original graphNew
//         Assume C_orig is initialized appropriately
// WARNING: Graph G will be destroyed at the end of this routine


void runMultiPhaseLouvainAlgorithm(
    graphNew* G, long* C_orig, int coloring, long opts_minGraphSize, double opts_threshold, double opts_C_thresh, int numThreads) {
    double totTimeClustering = 0, totTimeBuildingPhase = 0, totTimeColoring = 0, tmpTime;
    int tmpItr = 0, totItr = 0;
    long NV = G->numVertices;

    // long opts_minGraphSize = 100000; //Need at least 100,000 vertices to turn coloring on

    int* colors;
    int numColors = 0;
    if (coloring == 1) {
        colors = (int*)malloc(G->numVertices * sizeof(int));
        assert(colors != 0);
#pragma omp parallel for
        for (long i = 0; i < G->numVertices; i++) {
            colors[i] = -1;
        }
        numColors = algoDistanceOneVertexColoringOpt(G, colors, numThreads, &tmpTime) + 1;
        totTimeColoring += tmpTime;
        // printf("Number of colors used: %d\n", numColors);
    }

    /* Step 3: Find communities */
    double prevMod = -1;
    double currMod = -1;
    long phase = 1;

    graphNew* Gnew; // To build new hierarchical graphNews
    long numClusters;
    long* C = (long*)malloc(NV * sizeof(long));
    assert(C != 0);
#pragma omp parallel for
    for (long i = 0; i < NV; i++) {
        C[i] = -1;
    }
    bool nonColor = false; // Make sure that at least one phase with lower opts_threshold runs
    while (1) {
        printf("===============================\n");
        printf("Phase %ld\n", phase);
        printf("===============================\n");
        prevMod = currMod;
        // Compute clusters
        if ((coloring == 1) && (G->numVertices > opts_minGraphSize) && (nonColor == false)) {
            // Use higher modularity for the first few iterations when graphNew is big enough
            currMod = algoLouvainWithDistOneColoring(G, C, numThreads, colors, numColors, currMod, opts_C_thresh,
                                                     &tmpTime, &tmpItr);
            totTimeClustering += tmpTime;
            totItr += tmpItr;
        } else {
            currMod = parallelLouvianMethod(G, C, numThreads, currMod, opts_threshold, &tmpTime, &tmpItr);
            totTimeClustering += tmpTime;
            totItr += tmpItr;
            nonColor = true;
        }
        // Renumber the clusters contiguiously
        numClusters = renumberClustersContiguously(C, G->numVertices);
        printf("Number of unique clusters: %ld\n", numClusters);
        // printf("About to update C_orig\n");
        // Keep track of clusters in C_orig
        if (phase == 1) {
#pragma omp parallel for
            for (long i = 0; i < NV; i++) {
                C_orig[i] = C[i]; // After the first phase
            }
        } else {
#pragma omp parallel for
            for (long i = 0; i < NV; i++) {
                assert(C_orig[i] < G->numVertices);
                if (C_orig[i] >= 0) C_orig[i] = C[C_orig[i]]; // Each cluster in a previous phase becomes a vertex
            }
        }
        printf("Done updating C_orig\n");
        // Break if too many phases or iterations
        if ((phase > 200) || (totItr > 10000)) {
            break;
        }
        // Check for modularity gain and build the graphNew for next phase
        // In case coloring is used, make sure the non-coloring routine is run at least once
        if ((currMod - prevMod) > opts_threshold) {
            Gnew = (graphNew*)malloc(sizeof(graphNew));
            assert(Gnew != 0);
            tmpTime = buildNextLevelGraphOpt(G, Gnew, C, numClusters, numThreads);
            totTimeBuildingPhase += tmpTime;
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
            // If coloring is enabled & graphNew is of minimum size, recolor the new graphNew
            if ((coloring == 1) && (G->numVertices > opts_minGraphSize) && (nonColor == false)) {
#pragma omp parallel for
                for (long i = 0; i < G->numVertices; i++) {
                    colors[i] = -1;
                }
                numColors = algoDistanceOneVertexColoringOpt(G, colors, numThreads, &tmpTime) + 1;
                printf("numColors=%d\n", numColors);
                totTimeColoring += tmpTime;
            }
        } else {
            if ((coloring == 1) && (nonColor == false)) {
                nonColor = true; // Run at least one loop of non-coloring routine
            } else {
                break; // Modularity gain is not enough. Exit.
            }
        }
    } // End of while(1)

    printf("********************************************\n");
    printf("*********    Compact Summary   *************\n");
    printf("********************************************\n");
    printf("Number of threads              : %d\n",  numThreads);
    printf("Total number of phases         : %ld\n", phase);
    printf("Total number of iterations     : %d\n",  totItr);
    printf("Final number of clusters       : %ld\n", numClusters);
    printf("Final modularity               : %lf\n", prevMod);
    printf("Total time for clustering      : %lf\n", totTimeClustering);
    printf("Total time for building phases : %lf\n", totTimeBuildingPhase);
    if (coloring == 1) {
        printf("Total time for coloring        : %lf\n", totTimeColoring);
    }
    printf("********************************************\n");
    printf("TOTAL TIME                     : %lf\n", (totTimeClustering + totTimeBuildingPhase + totTimeColoring));
    printf("********************************************\n");

    // Clean up:
    free(C);
    if (G != 0) {
        free(G->edgeListPtrs);
        free(G->edgeList);
        free(G);
    }

    if (coloring == 1) {
        if (colors != 0) free(colors);
    }

} // End of runMultiPhaseLouvainAlgorithm()

