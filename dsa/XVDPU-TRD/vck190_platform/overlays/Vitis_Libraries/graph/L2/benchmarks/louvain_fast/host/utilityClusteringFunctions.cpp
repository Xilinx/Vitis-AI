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

#include "utilityClusteringFunctions.h"

using namespace std;

void sumVertexDegree(edge* vtxInd, long* vtxPtr, long* vDegree, long NV, Comm* cInfo) {
#pragma omp parallel for
  for (long i=0; i<NV; i++) {
    long adj1 = vtxPtr[i];	//Begining
    long adj2 = vtxPtr[i+1];	//End
    long totalWt = 0;
    for(long j=adj1; j<adj2; j++) {
      totalWt += (long)vtxInd[j].weight;
    }
    vDegree[i] = totalWt;	//Degree of each node
    cInfo[i].degree = totalWt;	//Initialize the community
    cInfo[i].size = 1;
  }
}//End of sumVertexDegree()

double calConstantForSecondTerm(long* vDegree, long NV) {
  long totalEdgeWeightTwice = 0;
#pragma omp parallel
  {
    long localWeight = 0;
#pragma omp for
    for (long i=0; i<NV; i++) {
      localWeight += vDegree[i];
    }
#pragma omp critical
    {
      totalEdgeWeightTwice += localWeight; //Update the global weight
    }
  }//End of parallel region

  return 1/(double)totalEdgeWeightTwice;
}//End of calConstantForSecondTerm()

void initCommAss(long* pastCommAss, long* currCommAss, long NV) {
#pragma omp parallel for
  for (long i=0; i<NV; i++) {
    pastCommAss[i] = i; //Initialize each vertex to its cluster
    currCommAss[i] = i;
  }
}//End of initCommAss()

long buildLocalMapCounter(long adj1, long adj2, map<long, long> &clusterLocalMap, 
			 vector<double> &Counter, edge* vtxInd, long* currCommAss, long me) {
  
  map<long, long>::iterator storedAlready;
  long numUniqueClusters = 1;
  long selfLoop = 0;
  for(long j=adj1; j<adj2; j++) {
    if(vtxInd[j].tail == me) {	// SelfLoop need to be recorded
      selfLoop += (long)vtxInd[j].weight;
    }
    
    storedAlready = clusterLocalMap.find(currCommAss[vtxInd[j].tail]); //Check if it already exists
    if( storedAlready != clusterLocalMap.end() ) {	//Already exists
      Counter[storedAlready->second]+= vtxInd[j].weight; //Increment the counter with weight
    } else {
      clusterLocalMap[currCommAss[vtxInd[j].tail]] = numUniqueClusters; //Does not exist, add to the map
      Counter.push_back(vtxInd[j].weight); //Initialize the count
      numUniqueClusters++;
    }
  }//End of for(j)

  return selfLoop;
}//End of buildLocalMapCounter()

long max(map<long, long> &clusterLocalMap, vector<double> &Counter, 
	 long selfLoop, Comm* cInfo, long degree, long sc, double constant ) {

  map<long, long>::iterator storedAlready;
  long maxIndex = sc;	//Assign the initial value as self community
  double curGain = 0;
  double maxGain = 0;
  double eix = Counter[0] - selfLoop;
  double ax = cInfo[sc].degree - degree;
  double eiy = 0;
  double ay = 0;
  
  storedAlready = clusterLocalMap.begin();
  do {
    if(sc != storedAlready->first) {
      ay = cInfo[storedAlready->first].degree; // degree of cluster y
      eiy = Counter[storedAlready->second]; 	//Total edges incident on cluster y
      curGain = 2*(eiy - eix) - 2*degree*(ay - ax)*constant;
 
      if( (curGain > maxGain) || 
          ((curGain==maxGain) && (curGain != 0) && (storedAlready->first < maxIndex)) ) {
	maxGain = curGain;
	maxIndex = storedAlready->first;
      }
    }
    storedAlready++; //Go to the next cluster
  } while ( storedAlready != clusterLocalMap.end() );
  
  if(cInfo[maxIndex].size == 1 && cInfo[sc].size ==1 && maxIndex > sc) { //Swap protection
    maxIndex = sc;
  }

  return maxIndex;		
}//End max()
  
