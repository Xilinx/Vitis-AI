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

#ifndef _DEFS_H
#define _DEFS_H

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
#include <unistd.h> //For getopts()
#include "xilinxlouvain.hpp"

#define MilanRealMax HUGE_VAL       // +INFINITY
#define MilanRealMin -MilanRealMax  // -INFINITY

#define PRINT_DETAILED_STATS_

typedef struct comm
{
  long size;
  long degree;
}Comm;

struct clustering_parameters 
{
  const char *inFile; //Input file
  int ftype;  //File type

  const char *xclbin; // xclbin file

  bool strongScaling; //Enable strong scaling
  bool output; //Printout the clustering data
  bool VF; //Vertex following turned on
  bool coloring; //If coloring is turned on 

  double C_thresh; //Threshold with coloring on
  long minGraphSize; //Min |V| to enable coloring
  double threshold; //Value of threshold
       
  clustering_parameters();
  void usage();    
  bool parse(int argc, char *argv[]);
};


/////////////////// FUNCTION CALLS ////////////////////
void displayGraphCharacteristics(graphNew *G);
void displayGraph(graphNew *G);
void displayGraphEdgeList(graphNew *G);
void displayGraphEdgeList(graphNew *G, FILE* out);
//Graph Clustering (Community detection)
double parallelLouvianMethod(graphNew *G, long *C, int nThreads, double Lower, 
				double thresh, double *totTime, int *numItr);
double algoLouvainWithDistOneColoring(graphNew* G, long *C, int nThreads, int* color, 
			int numColor, double Lower, double thresh, double *totTime, int *numItr);
void runMultiPhaseLouvainAlgorithm(graphNew *G, long *C_orig, int coloring, long minGraphSize, 
			double threshold, double C_threshold, int numThreads);

void runLouvainWithFPGA(graphNew *G, long *C_orig, char *xclbinName, bool coloring, long minGraphSize,
			double threshold, double C_threshold, int numThreads);


//***  Clustering Utility Functions ***//
//Distance-1 Coloring
int algoDistanceOneVertexColoring(graphNew *G, int *vtxColor, int nThreads, double *totTime);
int algoDistanceOneVertexColoringOpt(graphNew *G, int *vtxColor, int nThreads, double *totTime);

//Other 
inline void Visit(long v, long myCommunity, short *Visited, long *Volts, 
				  long* vtxPtr, edge* vtxInd, long *C);
long buildCommunityBasedOnVoltages(graphNew *G, long *Volts, long *C, long *Cvolts);
void buildNextLevelGraph(graphNew *Gin, graphNew *Gout, long *C, long numUniqueClusters);
long renumberClustersContiguously(long *C, long size);
//long renumberClustersContiguously_ghost(long *C, long size, long NV_l);
double buildNextLevelGraphOpt(graphNew *Gin, graphNew *Gout, long *C, long numUniqueClusters, int nThreads);
//Vertex following functions:
long vertexFollowing(graphNew *G, long *C);
double buildNewGraphVF(graphNew *Gin, graphNew *Gout, long *C, long numUniqueClusters);

//***  Utility Functions ***//
void duplicateGivenGraph(graphNew *Gin, graphNew *Gout);
void writeEdgeListToFile(graphNew *G, FILE* out);

//Random Number Generation:
void generateRandomNumbers(double *RandVec, long size);

void displayGraph(graphNew *G);
void displayGraphCharacteristics(graphNew *G);
graphNew * convertDirected2Undirected(graphNew *G);

void segregateEdgesBasedOnVoltages(graphNew *G, long *Volts);
void writeGraphPajekFormat(graphNew* G, char * filename);
void writeGraphPajekFormatWithNodeVolts(graphNew* G, long *Cvolts, char * filename);
void writeGraphBinaryFormat(graphNew* G, char * filename); //Binary (each edge once)
void writeGraphMetisSimpleFormat(graphNew* G, char *filename); //Metis format; no weights

//File parsers:
void parse_Dimacs9FormatDirectedNewD(graphNew * G, char *fileName);
long removeEdges(long NV, long NE, edge *edgeList);
void SortNodeEdgesByIndex2(long NV, edge *list1, edge *list2, long *ptrs);
void SortEdgesUndirected2(long NV, long NE, edge *list1, edge *list2, long *ptrs);

void loadMetisFileFormat(graphNew *G, const char* filename); //Metis (DIMACS#10)
void parse_MatrixMarket(graphNew * G, char *fileName);       //Matrix-Market
void parse_MatrixMarket_Sym_AsGraph(graphNew * G, char *fileName);

void parse_Dimacs1Format(graphNew * G, char *fileName);      //DIMACS#1 Challenge format
void parse_Dimacs9FormatDirectedNewD(graphNew * G, char *fileName); //DIMACS#9 Challenge format
void parse_PajekFormat(graphNew * G, char *fileName);        //Pajek format (each edge stored only once
void parse_PajekFormatUndirected(graphNew * G, char *fileName);
void parse_DoulbedEdgeList(graphNew * G, char *fileName);

void parse_EdgeListBinary(graphNew * G, char *fileName); //Binary: Each edge stored only once 
void parse_SNAP(graphNew * G, char *fileName);

//For reading power grid data
long* parse_MultiKvPowerGridGraph(graphNew * G, char *fileName); //Four-column format


//Graph partitioning with Metis:
void MetisGraphPartitioner( graphNew *G, long *VertexPartitioning, int numParts );

#endif
