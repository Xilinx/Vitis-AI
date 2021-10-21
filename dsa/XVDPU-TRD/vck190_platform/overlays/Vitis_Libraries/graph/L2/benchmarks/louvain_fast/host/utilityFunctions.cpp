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
#include "RngStream.h"

using namespace std;

void generateRandomNumbers(double *RandVec, long size) {
	int nT;
#pragma omp parallel
	{
		nT = omp_get_num_threads();
	}
#ifdef PRINT_DETAILED_STATS_
	printf("Within generateRandomNumbers() -- Number of threads: %d\n", nT);
#endif	
    //Initialize parallel pseudo-random number generator
	unsigned long seed[6] = {1, 2, 3, 4, 5, 6};
	RngStream::SetPackageSeed(seed);
	RngStream RngArray[nT]; //array of RngStream Objects
	
	long block = size / nT;
#ifdef PRINT_DETAILED_STATS_
	cout<<"Each thread will add "<<block<<" edges\n";
#endif
	//Each thread will generate m/nT edges each
	//double start = omp_get_wtime();// unused variable ‘start’
#pragma omp parallel
	{
		int myRank = omp_get_thread_num();
#pragma omp for schedule(static)
		for (long i=0; i<size; i++) {
			RandVec[i] =  RngArray[myRank].RandU01();
		}
	}//End of parallel region	
} //End of generateRandomNumbers()

void displayGraph(graphNew *G) {
  long    NV        = G->numVertices;  
  long    NE        = G->numEdges;
  long    *vtxPtr   = G->edgeListPtrs;
  edge    *vtxInd   = G->edgeList;  
  printf("***********************************");
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  printf("***********************************");
  for (long i = 0; i < NV; i++) {
    long adj1 = vtxPtr[i];
    long adj2 = vtxPtr[i+1];
    printf("\nVtx: %ld [%ld]: ",i+1,adj2-adj1);
    for(long j=adj1; j<adj2; j++) {      
      printf("%ld (%g), ", vtxInd[j].tail+1, vtxInd[j].weight);
    }
  }
  printf("\n***********************************\n");
}

void duplicateGivenGraph(graphNew *Gin, graphNew *Gout) {
	long    NV        = Gin->numVertices;  
	long    NE        = Gin->numEdges;
	long    *vtxPtr   = Gin->edgeListPtrs;
	edge    *vtxInd   = Gin->edgeList;  
#ifdef PRINT_DETAILED_STATS_
	printf("|V|= %ld, |E|= %ld \n", NV, NE);
#endif		
	double time1 = omp_get_wtime();
	long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
	assert(edgeListPtr != NULL);		
#pragma omp parallel for
	for (long i=0; i<=NV; i++) {
		edgeListPtr[i] = vtxPtr[i]; //Prefix Sum
	}
	
	//WARNING: There is a bug in edge counting when self-loops exist
	edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); 
	assert( edgeList != NULL);	
#pragma omp parallel for
	for (long i=0; i<NV; i++) {
		for (long j=vtxPtr[i]; j<vtxPtr[i+1]; j++) {
			edgeList[j].head = vtxInd[j].head;
			edgeList[j].tail = vtxInd[j].tail;
			edgeList[j].weight = vtxInd[j].weight;
		}		
	}

	//The last element of Cumulative will hold the total number of characters
	double time2 = omp_get_wtime();
#ifdef PRINT_DETAILED_STATS_
	printf("Done duplicating the graphNew:  %9.6lf sec.\n", time2 - time1);
#endif	
	Gout->sVertices    = NV;
	Gout->numVertices  = NV;
	Gout->numEdges     = NE;
	Gout->edgeListPtrs = edgeListPtr;
	Gout->edgeList     = edgeList;	
} //End of duplicateGivenGraph()


void displayGraphEdgeList(graphNew *G) {
	long    NV        = G->numVertices;  
	long    NE        = G->numEdges;
	long    *vtxPtr   = G->edgeListPtrs;
	edge    *vtxInd   = G->edgeList;  
	printf("***********************************");
	printf("|V|= %ld, |E|= %ld \n", NV, NE);
	for (long i = 0; i < NV; i++) {
		long adj1 = vtxPtr[i];
		long adj2 = vtxPtr[i+1];
		for(long j=adj1; j<adj2; j++) {      
			printf("%ld %ld %g\n", i+1, vtxInd[j].tail+1, vtxInd[j].weight);
		}
	}
	printf("\n***********************************\n");
}


void displayGraphEdgeList(graphNew *G, FILE* out) {
	long    NV        = G->numVertices;  
	long    NE        = G->numEdges;
	long    *vtxPtr   = G->edgeListPtrs;
	edge    *vtxInd   = G->edgeList;  
	printf("********PRINT OUTPUT********************");
	fprintf(out,"p sp %ld %ld \n", NV, NE/2);
	for (long i = 0; i < NV; i++) {
		long adj1 = vtxPtr[i];
		long adj2 = vtxPtr[i+1];
		for(long j=adj1; j<adj2; j++) {    
			if( i+1 < vtxInd[j].tail+1)
			{
				fprintf(out,"a %ld %ld %g\n", i+1, vtxInd[j].tail+1, vtxInd[j].weight);
			}
		}
	}
}

void writeEdgeListToFile(graphNew *G, FILE* out) {
	long    NV        = G->numVertices;
	long    *vtxPtr   = G->edgeListPtrs;
	edge    *vtxInd   = G->edgeList;
	for (long i = 0; i < NV; i++) {
		long adj1 = vtxPtr[i];
		long adj2 = vtxPtr[i+1];
		for(long j=adj1; j<adj2; j++) {    
			if( i < vtxInd[j].tail) {
				fprintf(out,"%ld %ld\n", i, vtxInd[j].tail);
			}
		}
	}
}

void displayGraphCharacteristics(graphNew *G) {
  printf("Within displayGraphCharacteristics()\n");        
  long    sum = 0, sum_sq = 0;
  double  average, avg_sq, variance, std_dev;
  long    maxDegree = 0;
  long    isolated  = 0;
  long    degreeOne = 0;
  long    NS        = G->sVertices;    
  long    NV        = G->numVertices;
  long    NT        = NV - NS;
  long    NE        = G->numEdges;
  long    *vtxPtr   = G->edgeListPtrs;
  long    tNV       = NV; //Number of vertices    
     
  if ( (NS == 0)||(NS == NV) ) {  //Nonbiparite graphNew	
    for (long i = 0; i < NV; i++) {
      long degree = vtxPtr[i+1] - vtxPtr[i];
      sum_sq += degree*degree;
      sum    += degree;
      if (degree > maxDegree)
	maxDegree = degree;
      if ( degree == 0 )
	isolated++;
      if ( degree == 1 )
	degreeOne++;
    }	
    average  = (double) sum / tNV;
    avg_sq   = (double) sum_sq / tNV;
    variance = avg_sq - (average*average);
    std_dev  = sqrt(variance);
    
    printf("*******************************************\n");
    printf("General Graph: Characteristics :\n");
    printf("*******************************************\n");
    printf("Number of vertices   :  %ld\n", NV);
    printf("Number of edges      :  %ld\n", NE);
    printf("Maximum out-degree is:  %ld\n", maxDegree);
    printf("Average out-degree is:  %lf\n",average);
    printf("Expected value of X^2:  %lf\n",avg_sq);
    printf("Variance is          :  %lf\n",variance);
    printf("Standard deviation   :  %lf\n",std_dev);
    printf("Isolated vertices    :  %ld (%3.2lf%%)\n", isolated, ((double)isolated/tNV)*100);
    printf("Degree-one vertices  :  %ld (%3.2lf%%)\n", degreeOne, ((double)degreeOne/tNV)*100);
    printf("Density              :  %lf%%\n",((double)NE/(NV*NV))*100);
    printf("*******************************************\n");
    
  }//End of nonbipartite graphNew
  else { //Bipartite graphNew
    
    //Compute characterisitcs from S side:
    for (long i = 0; i < NS; i++) {
      long degree = vtxPtr[i+1] - vtxPtr[i];
      sum_sq += degree*degree;
      sum    += degree;
      if (degree > maxDegree)
	maxDegree = degree;
      if ( degree == 0 )
	isolated++;
      if ( degree == 1 )
	degreeOne++;
    }    
    average  = (double) sum / NS;
    avg_sq   = (double) sum_sq / NS;
    variance = avg_sq - (average*average);
    std_dev  = sqrt(variance);
    
    printf("*******************************************\n");
    printf("Bipartite Graph: Characteristics of S:\n");
    printf("*******************************************\n");
    printf("Number of S vertices :  %ld\n", NS);
    printf("Number of T vertices :  %ld\n", NT);
    printf("Number of edges      :  %ld\n", NE);
    printf("Maximum out-degree is:  %ld\n", maxDegree);
    printf("Average out-degree is:  %lf\n",average);
    printf("Expected value of X^2:  %lf\n",avg_sq);
    printf("Variance is          :  %lf\n",variance);
    printf("Standard deviation   :  %lf\n",std_dev);
    printf("Isolated (S)vertices :  %ld (%3.2lf%%)\n", isolated, ((double)isolated/NS)*100);
    printf("Degree-one vertices  :  %ld (%3.2lf%%)\n", degreeOne, ((double)degreeOne/tNV)*100);
    printf("Density              :  %lf%%\n",((double)NE/(NS*NS))*100);
    printf("*******************************************\n");
    
    sum = 0; 
    sum_sq = 0;
    maxDegree = 0;
    isolated  = 0;
    //Compute characterisitcs from T side:
    for (long i = NS; i < NV; i++) {
      long degree = vtxPtr[i+1] - vtxPtr[i];
      sum_sq += degree*degree;
      sum    += degree;
      if (degree > maxDegree)
	maxDegree = degree;
      if ( degree == 0 )
	isolated++;
      if ( degree == 1 )
	degreeOne++;
    }
    
    average  = (double) sum / NT;
    avg_sq   = (double) sum_sq / NT;
    variance = avg_sq - (average*average);
    std_dev  = sqrt(variance);
    
    printf("Bipartite Graph: Characteristics of T:\n");	
    printf("*******************************************\n");
    printf("Number of T vertices :  %ld\n", NT);
    printf("Number of S vertices :  %ld\n", NS);
    printf("Number of edges      :  %ld\n", NE);
    printf("Maximum out-degree is:  %ld\n", maxDegree);
    printf("Average out-degree is:  %lf\n",average);
    printf("Expected value of X^2:  %lf\n",avg_sq);
    printf("Variance is          :  %lf\n",variance);
    printf("Standard deviation   :  %lf\n",std_dev);
    printf("Isolated (T)vertices :  %ld (%3.2lf%%)\n", isolated, ((double)isolated/NT)*100);
    printf("Degree-one vertices  :  %ld (%3.2lf%%)\n", degreeOne, ((double)degreeOne/tNV)*100);
    printf("Density              :  %lf%%\n",((double)NE/(NT*NT))*100);
    printf("*******************************************\n");    
  }//End of bipartite graphNew    
}


//Convert a directed graphNew into an undirected graphNew:
//Parse through the directed graphNew and add edges in both directions
graphNew * convertDirected2Undirected(graphNew *G) {
  printf("Within convertDirected2Undirected()\n");
  //-Wunused-but-set-variable int nthreads;//-Wunused-but-set-variable
#pragma omp parallel
  {
   //-Wunused-but-set-variable nthreads = omp_get_num_threads();
  }
    
  //double time1=0, time2=0, totalTime=0;//-Wunused-variable
  //Get the iterators for the graphNew:
  long NVer     = G->numVertices;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd  = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("N= %ld  NE=%ld\n", NVer, NEdge);
  
  long *degrees = (long *) malloc ((NVer+1) * sizeof(long));
  assert(degrees != NULL);
  
  //Count the edges from source --> sink (for sink > source)
#pragma omp parallel for
  for (long v=0; v < NVer; v++ ) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    for(long k = adj1; k < adj2; k++ ) {
      long w = verInd[k].tail;
      if (w < v)
	continue;
      __sync_fetch_and_add(&degrees[v+1], 1); //Increment by one to make space for zero
      __sync_fetch_and_add(&degrees[w+1], 1);
    }//End of for(k)
  }//End of for(i)
  
  //Build the pointer array:
  long m=0;
  for (long i=1; i<=NVer; i++) {
    m += degrees[i]; //Accumulate the number of edges
    degrees[i] += degrees[i-1];
  }
  //Sanity check:
  if(degrees[NVer] != 2*m) {
    printf("Number of edges added is not correct (%ld, %ld)\n", degrees[NVer], 2*m);
    exit(1);
  }
  printf("Done building pointer array\n");
  
  //Build CSR for Undirected graphNew:
  long* counter = (long *) malloc (NVer * sizeof(long));
  assert(counter != NULL);
#pragma omp parallel for
  for (long i=0; i<NVer; i++)
    counter[i] = 0;
  
  //Allocate memory for Edge list:
  edge *eList = (edge *) malloc ((2*m) * sizeof (edge));
  assert(eList != NULL);
  
#pragma omp parallel for
  for (long v=0; v < NVer; v++ ) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    for(long k = adj1; k < adj2; k++ ) {
      long w = verInd[k].tail;
      double weight = verInd[k].weight;
      if (w < v)
	continue;
      //Add edge v --> w
      long location = degrees[v] + __sync_fetch_and_add(&counter[v], 1);
      if (location >= 2*m) {
	printf("location is out of bound: %ld \n", location);
	exit(1);
      }
      eList[location].head   = v;
      eList[location].tail   = w;
      eList[location].weight = weight;
      
      //Add edge w --> v
      location = degrees[w] + __sync_fetch_and_add(&counter[w], 1);
      if (location >= 2*m) {
	printf("location is out of bound: %ld \n", location);
	exit(1);
      }
      eList[location].head   = w;
      eList[location].tail   = v;
      eList[location].weight = weight;      
    }//End of for(k)
  }//End of for(v)
  
  //Clean up:
  free(counter);

  //Build and return a graphNew data structure
  graphNew * Gnew = (graphNew *) malloc (sizeof(graphNew));
  Gnew->numVertices  = NVer;
  Gnew->sVertices    = NVer;
  Gnew->numEdges     = m;
  Gnew->edgeListPtrs = degrees;
  Gnew->edgeList     = eList;
  
  return Gnew;
  
}//End of convertDirected2Undirected()

