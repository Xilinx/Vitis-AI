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
#include "utilityStringTokenizer.hpp"

/* Remove self- and duplicate edges.                                */
/* For each node, we store its non-duplicate edges as a linked list */
long removeEdges(long NV, long NE, edge *edgeList) {
  printf("Within removeEdges()\n");
  long NGE = 0;
  long *head = (long *) malloc(NV * sizeof(long));     /* head of linked list points to an edge */   
  long *next = (long *) malloc(NE * sizeof(long));     /* ptr to next edge in linked list       */
  
/* Initialize linked lists */
  for (long i = 0; i < NV; i++) head[i] = -1;
  for (long i = 0; i < NE; i++) next[i] = -2;
  
  for (long i = 0; i < NE; i++) {
    long sv  = edgeList[i].head;
    long ev  = edgeList[i].tail;
    if (sv == ev) continue;    /* self edge */    
    long * ptr = head + sv;     /* start at head of list for this key */    
    while (1) {
      long edgeId = *ptr;      
      if (edgeId == -1) {         /* at the end of the list */
	edgeId = *ptr;             /* lock ptr               */	
	if (edgeId == -1) {       /* if still end of list   */	  
	  //long newId = NGE;//unused variable
	  NGE++;     /* increment number of good edges */
	  //edgeList[i].id = newId;                 /* set id of edge                 */	  	  
	  next[i] = -1;                           /* insert edge in linked list     */
	  *ptr = i;
	  break;
	}	
	*ptr = edgeId;	
      } else 
	if (edgeList[edgeId].tail == ev) break;     /* duplicate edge */
	else  ptr = next + edgeId;
    }
  }  
  /* Move good edges to front of edgeList                    */
  /* While edge i is a bad edge, swap with last edge in list */
  for (long i = 0; i < NGE; i++) {
    while (next[i] == -2) {
      long k = NE - 1;
      NE--;
      edgeList[i] = edgeList[k];
      next[i] = next[k];
    }
  }
  printf("About to free memory\n");
  free(head);
  free(next);
  printf("Exiting removeEdges()\n");
  return NGE;
}//End of removeEdges()

/* Since graphNew is undirected, sort each edge head --> tail AND tail --> head */
void SortEdgesUndirected(long NV, long NE, edge *list1, edge *list2, long *ptrs) {
  for (long i = 0; i < NV + 2; i++) 
    ptrs[i] = 0;
  ptrs += 2;

  /* Histogram key values */
  for (long i = 0; i < NE; i++) {
    ptrs[list1[i].head]++;
    ptrs[list1[i].tail]++;
  }
  /* Compute start index of each bucket */
  for (long i = 1; i < NV; i++) 
    ptrs[i] += ptrs[i-1];
  ptrs--;

  /* Move edges into its bucket's segment */
  for (long i = 0; i < NE; i++) {
    long head   = list1[i].head;
    long index          = ptrs[head]++;
    //list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].head;
    list2[index].tail   = list1[i].tail;
    list2[index].weight = list1[i].weight;

    long tail   = list1[i].tail;
    index               = ptrs[tail]++;
    //list2[index].id     = list1[i].id;
    list2[index].head   = list1[i].tail;
    list2[index].tail   = list1[i].head;
    list2[index].weight = list1[i].weight;
  } 
}//End of SortEdgesUndirected2()

/* Sort each node's neighbors by tail from smallest to largest. */
void SortNodeEdgesByIndex(long NV, edge *list1, edge *list2, long *ptrs) {  
  for (long i = 0; i < NV; i++) {
    edge *edges1 = list1 + ptrs[i];
    edge *edges2 = list2 + ptrs[i];
    long size    = ptrs[i+1] - ptrs[i];

    /* Merge Sort */
    for (long skip = 2; skip < 2 * size; skip *= 2) {
      for (long sect = 0; sect < size; sect += skip)  {
	long j = sect;
	long l = sect;
	long half_skip = skip / 2;
	long k = sect + half_skip;
	
	long j_limit = (j + half_skip < size) ? j + half_skip : size;
	long k_limit = (k + half_skip < size) ? k + half_skip : size;
	
	while ((j < j_limit) && (k < k_limit)) {
	  if   (edges1[j].tail < edges1[k].tail) {edges2[l] = edges1[j]; j++; l++;}
	  else                                   {edges2[l] = edges1[k]; k++; l++;}
	}	
	while (j < j_limit) {edges2[l] = edges1[j]; j++; l++;}
	while (k < k_limit) {edges2[l] = edges1[k]; k++; l++;}
      }
      edge *tmp = edges1;
      edges1 = edges2;
      edges2 = tmp;
    }
    // result is in list2, so move to list1
    if (edges1 == list2 + ptrs[i])
      for (long j = ptrs[i]; j < ptrs[i+1]; j++) list1[j] = list2[j];
  } 
}//End of SortNodeEdgesByIndex2()

//Parse files in Metis format:
void loadMetisFileFormat(graphNew *G, const char* filename) {
  long i, j, value, neighbor,  mNVer=0,  mNEdge=0;
  double edgeWeight;//, vertexWeight;//-Wunused-but-set-variable
  string oneLine, myDelimiter(" "); //Delimiter is a blank space
  ifstream fin;
  char comment;

  fin.open(filename);
  if(!fin) {
    cerr<<"Within Function: loadMetisFileFormat() \n";
    cerr<<"Could not open the file.. \n";
    exit(1);
  }
  do { //Ignore the comment lines
    getline(fin, oneLine);
    comment = oneLine[0];
  } while ( comment == '%');
  
  StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
  if ( ST->HasMoreTokens() )
    mNVer  = atol( ST->GetNextToken().c_str() ); //Number of Vertices
  if ( ST->HasMoreTokens() )
    mNEdge  = atol( ST->GetNextToken().c_str() ); //Number of Edges
  if ( ST->HasMoreTokens() )
    value = atol( ST->GetNextToken().c_str() ); //Indication of the weights
  else
    value = 0;
  delete ST;
#ifdef PRINT_CF_DEBUG_INFO_
  cout<<"N Ver: "<<mNVer<<" N Edge: "<<mNEdge<<" value: "<<value<<" \n";
#endif
  
  long *mVerPtr   = (long *) malloc ((mNVer+1)  * sizeof(long)); //The Pointer
  edge *mEdgeList = (edge *) malloc ((mNEdge*2) * sizeof(edge)); //The Indices
  assert(mVerPtr != 0); assert(mEdgeList != 0); 
// printf("hi\n"); 
#pragma omp parallel for
  for (long i=0; i<=mNVer; i++) {
    mVerPtr[i] = 0;
  }
#pragma omp parallel for
  for (long i=0; i<(2*mNEdge); i++) {
    mEdgeList[i].tail   = -1;
    mEdgeList[i].weight = 0;
  }

  //Read the rest of the file:
  long PtrPos = 0, IndPos = 0, cumulative = 0;
  mVerPtr[PtrPos] = cumulative; PtrPos++;		
  //printf("hi %d\n",value);
  switch ( value ) {
  case 0:  //No Weights
#ifdef PRINT_CF_DEBUG_INFO_
    cout<<"Graph Type: No Weights. \n";
#endif			
    
    for ( i=0; i < mNVer; i++) {
      if ( fin.eof() )	    {
	cout<<" Error reading the Metis input File \n";
	cout<<" Reached Abrupt End \n";
	exit(1);
      }
      j=0;
      getline (fin, oneLine);
      StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
#ifdef PRINT_CF_DEBUG_INFO_
      cout<<"\n"<<i<<": ";
#endif
      while( ST->HasMoreTokens() )	{
	neighbor = (atol( ST->GetNextToken().c_str() ) - 1); //Zero-based index
#ifdef PRINT_CF_DEBUG_INFO_
	cout<<neighbor<<" ";
#endif
	j++;
	mEdgeList[IndPos].head = i;
        mEdgeList[IndPos].tail = neighbor; //IndPos++;
	mEdgeList[IndPos].weight = 1;
        IndPos++;
      }
      delete ST; //Clear the buffer
      cumulative += j;
      mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector			  
    }
    break;
  case 1: //Only Edge weights
#ifdef PRINT_CF_DEBUG_INFO_
    cout<<"Graph Type: Only Edge Weights. \n";
#endif
        
    for ( i=0; i < mNVer; i++) 		{
      if ( fin.eof() ) 			{
	cerr<<"Within Function: loadMetisFileFormat() \n";
	cerr<<" Error reading the Metis input File \n";
	cerr<<" Reached Abrupt End \n";
	exit(1);
      }
      j=0;
      getline (fin, oneLine);
      StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
      while( ST->HasMoreTokens() )	{
	neighbor = (atol( ST->GetNextToken().c_str() )- 1); //Zero-based index
	mEdgeList[IndPos].tail = neighbor; IndPos++;
	edgeWeight = atof( ST->GetNextToken().c_str() );
	mEdgeList[IndPos].weight = (long)edgeWeight;  //Type casting
	j++;				
      }
      delete ST; //Clear the buffer
      cumulative += j;
      mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector				
    }
    break;
  case 10: //Only Vertex Weights
#ifdef PRINT_CF_DEBUG_INFO_
    cout<<"Graph Type: Only Vertex Weights. \n";
    cout<<"Will ignore vertex weights.\n";
#endif
   
    for ( i=0; i < mNVer; i++)      {
      if ( fin.eof() )	{
	cout<<" Error reading the Metis input File \n";
	cout<<" Reached Abrupt End \n";
	exit(1);
      }
      j=0;
      getline (fin, oneLine);
      StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
      //vertexWeight = atof( ST->GetNextToken().c_str() );	//-Wunused-but-set-variable
      while( ST->HasMoreTokens() )	{
	neighbor = (atol( ST->GetNextToken().c_str() ) - 1); //Zero-based index
#ifdef PRINT_CF_DEBUG_INFO_
	cout<<"Neighbors: "<<neighbor<<" \t";
#endif
	j++;
	mEdgeList[IndPos].tail = neighbor; IndPos++;
	mEdgeList[IndPos].weight = 1;
      }
      delete ST; //Clear the buffer
      cumulative += j;
      mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector	      
    }
    break;
  case 11: //Both Edge and Vertex Weights:
#ifdef PRINT_CF_DEBUG_INFO_
    cout<<"Graph Type: Both Edge and Vertex Weights. \n";
#endif
    cout<<"Will ignore vertex weights.\n";
    
    for ( i=0; i < mNVer; i++) 		{
      if ( fin.eof() ) 	{
	cerr<<"Within Function: loadMetisFileFormat() \n";
	cerr<<" Error reading the Metis input File \n";
	cerr<<" Reached Abrupt End \n";
	exit(1);
      }
      j=0;
      getline (fin, oneLine);
      StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
      //vertexWeight = atof( ST->GetNextToken().c_str() ); //Vertex weight////-Wunused-but-set-variable
      while( ST->HasMoreTokens() )			{
	neighbor = (atol( ST->GetNextToken().c_str() )- 1); //Zero-based index
	mEdgeList[IndPos].tail = neighbor; IndPos++;
	edgeWeight = atof( ST->GetNextToken().c_str() );
	mEdgeList[IndPos].weight = (long)edgeWeight;  //Type casting
	j++;				
      }
      delete ST; //Clear the buffer
      cumulative += j;
      mVerPtr[PtrPos] = cumulative; PtrPos++; //Add to the Pointer Vector		
    }
    break;
  } //End of switch(value)
  fin.close();
  G->numVertices  = mNVer;
  G->sVertices    = mNVer;
  G->numEdges     = mNEdge;  //This is what the code expects
  G->edgeListPtrs = mVerPtr;  //Vertex Pointer
  G->edgeList     = mEdgeList;

} //End of loadMetisFileFormat()

/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and builds the graphNew
 *-------------------------------------------------------*/
void parse_MatrixMarket(graphNew * G, char *fileName) {
  printf("Parsing a Matrix Market File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("parse_MatrixMarket: Number of threads: %d\n", nthreads);

  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  //int isComplex = 0;    //-Wunused-but-set-variable
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    //isComplex = 1;//-Wunused-but-set-variable
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0, isGeneral = 0;
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    isGeneral = 1;
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }	
  if ( (isGeneral==0) && (isSymmetric==0) ) 	  {
    printf("Warning: Matrix type should be General or Symmetric. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS + NT;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  long newNNZ = 0;    //New edges because of symmetric matrices
  long Si, Ti;
  double weight = 1;
  if( isSymmetric == 1 ) {
    printf("Matrix is of type: Symmetric Real or Complex\n");
    printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      if (isPattern == 1)
	fscanf(file, "%ld %ld", &Si, &Ti);
      else
	fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      Si--; Ti--;            // One-based indexing
      assert((Si >= 0)&&(Si < NV));
      assert((Ti >= 0)&&(Ti < NV));
      weight = fabs(weight); //Make it positive  : Leave it as is
      if ( Si == Ti ) {
	edgeListTmp[i].head = Si;       //The S index
	edgeListTmp[i].tail = NS+Ti;    //The T index 
	edgeListTmp[i].weight = weight; //The value
	edgeListPtr[Si+1]++;
	edgeListPtr[NS+Ti+1]++;
      }
      else { //an off diagonal element: Also store the upper part
	//LOWER PART:
	edgeListTmp[i].head = Si;       //The S index 
	edgeListTmp[i].tail = NS+Ti;    //The T index 
	edgeListTmp[i].weight = weight; //The value
	edgeListPtr[Si+1]++;
	edgeListPtr[NS+Ti+1]++;
	//UPPER PART:
	edgeListTmp[NE+newNNZ].head = Ti;       //The S index
	edgeListTmp[NE+newNNZ].tail = NS+Si;    //The T index
	edgeListTmp[NE+newNNZ].weight = weight; //The value
	newNNZ++; //Increment the number of edges
	edgeListPtr[Ti+1]++;
	edgeListPtr[NS+Si+1]++;
      }
    }
  } //End of Symmetric
  /////// General Real or Complex ///////
  else {
    printf("Matrix is of type: Unsymmetric Real or Complex\n");
    printf("Weights will be converted to positive numbers.\n");
    edgeListTmp = (edge *) malloc( NE * sizeof(edge));
    for (long i = 0; i < NE; i++) {
      if (isPattern == 1)
	fscanf(file, "%ld %ld", &Si, &Ti);
      else
	fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
      //printf("(%d, %d) %lf\n",Si, Ti, weight);
      Si--; Ti--;            // One-based indexing
      assert((Si >= 0)&&(Si < NV));
      assert((Ti >= 0)&&(Ti < NV));
      weight = fabs(weight); //Make it positive    : Leave it as is
      edgeListTmp[i].head = Si;       //The S index
      edgeListTmp[i].tail = NS+Ti;    //The T index
      edgeListTmp[i].weight = weight; //The value
      edgeListPtr[Si+1]++;
      edgeListPtr[NS+Ti+1]++;
    }
  } //End of Real or Complex
  fclose(file); //Close the file
  printf("Done reading from file.\n");

  if( isSymmetric ) {
    printf("Modified the number of edges from %ld ",NE);
    NE += newNNZ; //#NNZ might change
    printf("to %ld \n",NE);
  }

  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = omp_get_wtime();
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *added    = (long *)  malloc( NV  * sizeof(long)); assert(added != 0);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  time2 = omp_get_wtime();
  printf("Time for allocating memory for marks and edgeList = %lf\n", time2 - time1);
  
  time1 = omp_get_wtime();

  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NS;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}


/*-------------------------------------------------------*
 * This function reads a MATRIX MARKET file and build the graphNew
 * graphNew is nonbipartite: each diagonal entry is a vertex, and 
 * each non-diagonal entry becomes an edge. Assume structural and 
 * numerical symmetry.
 *-------------------------------------------------------*/
void parse_MatrixMarket_Sym_AsGraph(graphNew * G, char *fileName) {
  printf("Parsing a Matrix Market File as a general graphNew...\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_MatrixMarket: Number of threads: %d\n ", nthreads);

  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);
  char  LS1[25], LS2[25], LS3[25], LS4[25], LS5[25];    
  if (sscanf(line, "%s %s %s %s %s", LS1, LS2, LS3, LS4, LS5) != 5) {
    printf("parse_MatrixMarket(): bad file format - 01");
    exit(1);
  }
  printf("%s %s %s %s %s\n", LS1, LS2, LS3, LS4, LS5);
  if ( strcmp(LS1,"%%MatrixMarket") != 0 ) {
    printf("Error: The first line should start with %%MatrixMarket word \n");
    exit(1);
  }
  if ( !( strcmp(LS2,"matrix")==0 || strcmp(LS2,"Matrix")==0 || strcmp(LS2,"MATRIX")==0 ) ) {
    printf("Error: The Object should be matrix or Matrix or MATRIX \n");
    exit(1);
  }
  if ( !( strcmp(LS3,"coordinate")==0 || strcmp(LS3,"Coordinate")==0 || strcmp(LS3,"COORDINATE")==0) ) {
    printf("Error: The Object should be coordinate or Coordinate or COORDINATE \n");
    exit(1);
  }
  //int isComplex = 0;//-Wunused-but-set-variable
  if ( strcmp(LS4,"complex")==0 || strcmp(LS4,"Complex")==0 || strcmp(LS4,"COMPLEX")==0 ) {
    //isComplex = 1;//-Wunused-but-set-variable
    printf("Warning: Will only read the real part. \n");
  }
  int isPattern = 0;
  if ( strcmp(LS4,"pattern")==0 || strcmp(LS4,"Pattern")==0 || strcmp(LS4,"PATTERN")==0 ) {
    isPattern = 1;
    printf("Note: Matrix type is Pattern. Will set all weights to 1.\n");
    //exit(1);
  }
  int isSymmetric = 0;//, isGeneral = 0;//-Wunused-but-set-variable
  if ( strcmp(LS5,"general")==0 || strcmp(LS5,"General")==0 || strcmp(LS5,"GENERAL")==0 )
    ;//isGeneral = 1;//-Wunused-but-set-variable
  else {
    if ( strcmp(LS5,"symmetric")==0 || strcmp(LS5,"Symmetric")==0 || strcmp(LS5,"SYMMETRIC")==0 ) {
      isSymmetric = 1;
      printf("Note: Matrix type is Symmetric: Converting it into General type. \n");
    }
  }	
  if ( isSymmetric==0 ) 	  {
    printf("Warning: Matrix type should be Symmetric for this routine. \n");
    exit(1);
  }
  
  /* Parse all comments starting with '%' symbol */
  do {
    fgets(line, 1024, file);
  } while ( line[0] == '%' );
  
  /* Read the matrix parameters */
  long NS=0, NT=0, NV = 0;
  long NE=0;
  if (sscanf(line, "%ld %ld %ld",&NS, &NT, &NE ) != 3) {
    printf("parse_MatrixMarket(): bad file format - 02");
    exit(1);
  }
  NV = NS;
  printf("|S|= %ld, |T|= %ld, |E|= %ld \n", NS, NT, NE);

  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  edge *edgeListTmp; //Read the data in a temporary list
  long newNNZ = 0;    //New edges because of symmetric matrices
  long Si, Ti;
  double weight = 1;
  printf("Matrix is of type: Symmetric Real or Complex\n");
  printf("Weights will be converted to positive numbers.\n");
  edgeListTmp = (edge *) malloc(2 * NE * sizeof(edge));
  for (long i = 0; i < NE; i++) {
    if (isPattern == 1)
      fscanf(file, "%ld %ld", &Si, &Ti);
    else
      fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight);
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    weight = fabs(weight); //Make it positive  : Leave it as is
    if ( Si == Ti ) {
      //Do nothing...
    }
    else { //an off diagonal element: store the edge
      //LOWER PART:
      edgeListTmp[newNNZ].head = Si;       //The S index 
      edgeListTmp[newNNZ].tail = Ti;       //The T index 
      edgeListTmp[newNNZ].weight = weight; //The value
      edgeListPtr[Si+1]++;
      edgeListPtr[Ti+1]++;
      newNNZ++;
    }//End of Else
  }//End of for loop
  fclose(file); //Close the file
  //newNNZ = newNNZ / 2;
  printf("Done reading from file.\n");
  printf("Modified the number of edges from %ld ", NE);
  NE = newNNZ; //#NNZ might change
  printf("to %ld \n", NE);

  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/
  time1 = omp_get_wtime();
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *Counter = (long *) malloc (NV  * sizeof(long)); assert(Counter != 0);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) {
    Counter[i] = 0;
  }
  time2 = omp_get_wtime();
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
  printf("About to build edgeList...\n");

  time1 = omp_get_wtime();
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head     = edgeListTmp[i].head;
    long tail     = edgeListTmp[i].tail;
    double weight = edgeListTmp[i].weight;

    long Where    = edgeListPtr[head] + __sync_fetch_and_add(&Counter[head], 1);
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the edge the other way:
    Where                  = edgeListPtr[tail] + __sync_fetch_and_add(&Counter[tail], 1);
    edgeList[Where].head   = tail;
    edgeList[Where].tail   = head;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;

  free(edgeListTmp);
  free(Counter);

}//End of parse_MatrixMarket_Sym_AsGraph()

/* 
-------------------------------------------------------------------------
INPUT FORMAT FOR WMATCH:
-------------------------------------------------------------------------
   Graph I/O is performed by a generic graphNew library package, 
   so some of the fields are ignored by the "wmatch" code (but 
   you must include dummy fields in the input). 

   There are three types of lines: the first line, vertex lines, 
   and edge lines. The fields in each line type are as follows. 

   First line-> size edges U
      size: integer giving number of vertices
      edges: integer giving number of edges 
      U: character ``U'' or ``u'' specifying an undirected graphNew

   Vertex lines->  degree vlabel xcoord ycoord
      degree: edge degree of the vertex
      vlabel: vertex label (ignored--vertices are referred to by index)
      xcoord: integer x-coordinate location (ignored)
      ycoord: integer y-coordinate location (ignored) 

      *****Each vertex line is followed immediately by the lines 
      for all its adjacent edges (thus each edge appears twice, 
      once for each vertex).******

   Edge lines-> adjacent  weight
      adjacent: index (not vlabel) of the adjacent vertex
      weight: integer edge weight 
-------------------------------------------------------------------------
*/
void parse_Dimacs1Format(graphNew * G, char *fileName) {
  printf("Parsing a DIMACS-1 formatted file as a general graphNew...\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs1Format: Number of threads: %d\n ", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  /* -----      Read File in Matrix Market Format     ------ */  
  /* Read the matrix parameters */
  long NV = 0, NE=0;
  char line[1024], LS1[25];
  fgets(line, 1024, file);
  //Parse the first line:
  if (sscanf(line, "%ld %ld %s",&NV, &NE, LS1) != 3) {
    printf("parse_Dimacs1(): bad file format - 01");
    exit(1);
  }
  assert((LS1[0] == 'U')||(LS1[0] == 'u')); //graphNew is undirected
  //NE = NE / 2; //Each edge is stored twice, but counted only once, in the file
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* S vertices: 0 to NS-1                                               */
  /* T vertices: NS to NS+NT-1                                           */
  /*---------------------------------------------------------------------*/
  //Allocate for Edge Pointer and keep track of degree for each vertex
  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graphNew = %lf\n", time2 - time1);
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  long Degree, Ti;
  //double weight = 1;//-Wunused-variable
  int Twt=0, label, xCoord, yCoord;
  long nE = 0;
  printf("Weights will be converted to positive integers.\n");
  
  time1 = omp_get_wtime();
  for (long i = 0; i < NV; i++) {
    //Vertex lines:  degree vlabel xcoord ycoord
    fscanf(file, "%ld %d %d %d", &Degree, &label, &xCoord, &yCoord);
    edgeListPtr[i+1] = Degree;
    for (long j=0; j<Degree; j++) {
      fscanf(file, "%ld %d", &Ti, &Twt);
      assert((Ti > 0)&&(Ti < NV));
      edgeList[nE].head   = i;       //The S index 
      edgeList[nE].tail   = Ti-1;    //The T index: One-based indexing
      edgeList[nE].weight = fabs((double)Twt); //Make it positive and cast to Double      
      nE++;
    }//End of inner for loop
  }//End of outer for loop
  fclose(file); //Close the file
  time2 = omp_get_wtime(); 
  printf("Done reading from file: nE= %ld. Time= %lf\n", nE, time2-time1);
  assert(NE == nE/2);
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;

}//End of parse_Dimacs1Format()


/* ****************************************** */
//Loading Functions:
/* NOTE: Indices are ZERO-based, i.e. G(0,0) is the first element, 
         while the indices stored in the file are ONE-based. 
Details:
* A graphNew contains n nodes and m arcs
* Nodes are identified by integers 1...* Graphs can be interpreted as directed or undirected, depending on the problem being studied
* Graphs can have parallel arcs and self-loops
* Arc weights are signed integers

** c : This is a comment
** p sp n m : Unique problem line: sp is for shortest path - no meaning for us
** a U V W  : a=arc, U, V, W are the I-J-V format for an edge

* Assumption: Graphs are DIRECTED ==> Each edge is stored ONLY ONCE.
* (If Undirected, then edge is stored TWICE)
*/
void parse_Dimacs9FormatDirectedNewD(graphNew * G, char *fileName) {
  printf("Parsing a DIMACS-9 formatted file as a general graphNew...\n");
  printf("WARNING: Assumes that the graphNew is directed -- an edge is stored only once.\n");
  printf("       : Graph will be stored as undirected, each edge appears twice.\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs9FormatDirectedNewD: Number of threads: %d\n ", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }  
  char line[1024], LS1[25], LS2[25];
  //Ignore the Comment lines starting with "c"
  do {
    fgets(line, 1024, file);
  } while ( line[0] == 'c');  
  //Expecting a problem line here:  p sp n m  
  long NV = 0, NE=0;  
  //Parse the problem line:
  if(line[0] == 'p') {
    if (sscanf(line, "%s %s %ld %ld", LS1, LS2, &NV, &NE) != 4) {
      printf("parse_Dimacs9(): bad file format - 01");
      exit(1);   
    }
  } else {
    printf("parse_Dimacs9(): bad file format, expecting a line starting with p\n");
    printf("%s\n", line);
    exit(1);
  }
  printf("|V|= %ld, |E|= %ld \n", NV, NE);  
  printf("Weights will be converted to positive numbers.\n");
  /*---------------------------------------------------------------------*/
  /* Read edge list: a U V W                                             */
  /*---------------------------------------------------------------------*/  
  edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
  assert( tmpEdgeList != NULL);
  long Si, Ti;
  double Twt;
  time1 = omp_get_wtime();
  for (long i = 0; i < NE; i++) {
    //Vertex lines:  degree vlabel xcoord ycoord
    fscanf(file, "%s %ld %ld %lf", LS1, &Si, &Ti, &Twt);
    assert((Si > 0)&&(Si <= NV));
    assert((Ti > 0)&&(Ti <= NV));
    tmpEdgeList[i].head   = Si-1;       //The S index 
    tmpEdgeList[i].tail   = Ti-1;    //The T index: One-based indexing
    tmpEdgeList[i].weight = fabs((double)Twt); //Make it positive and cast to Double
  }//End of outer for loop
  fclose(file); //Close the file
  time2 = omp_get_wtime(); 
  printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
  
  //Remove duplicate entries:
 /* long NewEdges = removeEdges(NV, NE, tmpEdgeList);
  if (NewEdges < NE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    NE = NewEdges; //Only look at clean edges
  } else {
    printf("No dubplicates found.\n");
  }*/
  ///////////
  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graphNew = %lf\n", time2 - time1);
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].tail+1], 1);
  }
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
  printf("*********** (%ld)\n", NV);
  
  printf("About to build edgeList...\n");
  time1 = omp_get_wtime();
  //Keep track of how many edges have been added for a vertex:
  long  *added  = (long *)  malloc( NV  * sizeof(long)); assert( added != NULL);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;

  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = tmpEdgeList[i].head;
    long tail      = tmpEdgeList[i].tail;
    double weight  = tmpEdgeList[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;    
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  //Clean up
  free(tmpEdgeList);
  free(added);

}//End of parse_Dimacs9FormatDirectedNewD()


/*-------------------------------------------------------*
 * This function reads a Pajek file and builds the graphNew
 *-------------------------------------------------------*/
void parse_PajekFormat(graphNew * G, char *fileName) {
  printf("Parsing a Pajek File...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = NUMTHREAD; // omp_get_num_threads();    
  }
  printf("parse_Pajek: Number of threads: %d\n", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);  
  char  LS1[25], LS2[25];
  long NV = 0, ED=0, NE=0;
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 01");
    exit(1);
  }  
  //printf("(%s) --- (%s) \n", LS1, LS2);
  if ( strcmp(LS1,"*Vertices")!= 0 ) {
    printf("Error: The first line should start with *Vertices word \n");
    exit(1);
  }
  NV = atol(LS2);
  printf("|V|= %ld \n", NV);
  /* Ignore all the vertex lines */
  //for (long i=0; i <= NV; i++) {
    fgets(line, 1024, file);
  //}
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 02");
    exit(1);
  }
  if ( strcmp(LS1,"*Edges")!= 0 ) {
    printf("Error: The next line should start with *Edges word \n");
    exit(1);
  }

  ED = atol(LS2);
  printf("|E|= %ld \n", ED);
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* (i , j, value ) 1-based index                                       */
  /*---------------------------------------------------------------------*/  
  edge *edgeListTmp; //Read the data in a temporary list
  long Si, Ti;
  double weight = 1;
  long edgeEstimate = ED; // NV * 2; //25% density -- not valid for large graphNews
  edgeListTmp = (edge *) malloc( edgeEstimate * sizeof(edge));
  assert(edgeListTmp != 0);

  printf("Parsing edges -- with weights\n");
  
  while (fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight) != EOF) {
  //while (fscanf(file, "%ld %ld", &Si, &Ti) != EOF) {
    if(NE >= edgeEstimate) {
      printf("Temporary buffer is not enough. \n");
      exit(1);
    }
    //printf("%ld -- %ld -- %lf\n", Si, Ti, weight);
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    //if(Si%99999 == 0)
    //  printf("%ld -- %ld\n", Si, Ti);
    if (Si == Ti) //Ignore self-loops
      continue; 
    //weight = fabs(weight); //Make it positive    : Leave it as is
    //weight = 1.0; //Make it positive    : Leave it as is
    edgeListTmp[NE].head = Si;       //The S index
    edgeListTmp[NE].tail = Ti;    //The T index
    edgeListTmp[NE].weight = weight; //The value
    NE++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  //Remove duplicate entries:
  long NewEdges = removeEdges(NV, NE, edgeListTmp);
  if (NewEdges < NE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    NE = NewEdges; //Only look at clean edges
  }
  
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes

#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); //Plus one to take care of the zeroth location
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1);
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  printf("About to allocate memory for graphNew data structures\n");
  time1 = omp_get_wtime();
  edge *edgeList = (edge *) malloc ((2*NE) * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *added = (long *)  malloc (NV * sizeof(long)); assert (added != 0);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  time2 = omp_get_wtime();  
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
  
  time1 = omp_get_wtime();
  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head; 
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //added[head]++;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
    //added[tail]++;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}

/*-------------------------------------------------------*
 * This function reads a Pajek file and builds the graphNew
 *-------------------------------------------------------*/
//Assume every edge is stored twice:
void parse_PajekFormatUndirected(graphNew * G, char *fileName) {
  printf("Parsing a Pajek File *** Undirected ***...\n");
  int nthreads;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();    
  }
  printf("parse_Pajek_undirected: Number of threads: %d\n", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);  
  char  LS1[25], LS2[25];
  long NV = 0, NE=0;
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 01");
    exit(1);
  }  
  //printf("(%s) --- (%s) \n", LS1, LS2);
  if ( strcmp(LS1,"*Vertices")!= 0 ) {
    printf("Error: The first line should start with *Vertices word \n");
    exit(1);
  }
  NV = atol(LS2);
  printf("|V|= %ld \n", NV);
  /* Ignore all the vertex lines */
  for (long i=0; i <= NV; i++) {
    fgets(line, 1024, file);
  }
  printf("Done parsing through vertex lines\n");
  if (sscanf(line, "%s", LS1) != 1) {
    printf("parse_Pajek(): bad file format - 02");
    exit(1);
  }
  if ( strcmp(LS1,"*Edges")!= 0 ) {
    printf("Error: The next line should start with *Edges word \n");
    exit(1);
  }
  printf("About to read edges -- no weights\n");
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* (i , j, value ) 1-based index                                       */
  /*---------------------------------------------------------------------*/
  
  edge *edgeListTmp; //Read the data in a temporary list
  long Si, Ti;
  double weight = 1;
  long edgeEstimate = NV * NV / 8; //12.5% density -- not valid for dense graphNews
  edgeListTmp = (edge *) malloc( edgeEstimate * sizeof(edge));

  //while (fscanf(file, "%ld %ld %lf", &Si, &Ti, &weight) != EOF) {
  while (fscanf(file, "%ld %ld ", &Si, &Ti) != EOF) {
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    if (Si == Ti) //Ignore self-loops 
      continue; 
    //weight = fabs(weight); //Make it positive    : Leave it as is
    weight = 1.0; //Make it positive    : Leave it as is
    edgeListTmp[NE].head = Si;       //The S index
    edgeListTmp[NE].tail = Ti;    //The T index
    edgeListTmp[NE].weight = weight; //The value
    NE++;
  }
  fclose(file); //Close the file
  printf("Done reading from file.\n");
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  
  //Remove duplicate entries:
  /*
  long NewEdges = removeEdges(NV, NE, edgeListTmp);
  if (NewEdges < NE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    NE = NewEdges; //Only look at clean edges
  } else
    printf("No duplicates were found\n");
  */

  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *) malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != 0);

#pragma omp parallel for
  for (long i=0; i <= NV; i++) {
    edgeListPtr[i] = 0; //For first touch purposes
  }
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].head + 1], 1); //Plus one to take care of the zeroth location
    //__sync_fetch_and_add(&edgeListPtr[edgeListTmp[i].tail + 1], 1); //No need
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", NE, edgeListPtr[NV]);
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  time1 = omp_get_wtime();
  printf("Size of edge: %ld  and size of NE*edge= %ld\n",  sizeof(edge), NE*sizeof(edge) );
  edge *edgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);
  //Keep track of how many edges have been added for a vertex:
  long  *added    = (long *)  malloc( NV  * sizeof(long)); assert (added != 0);
 #pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  
  time2 = omp_get_wtime();
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);
  
  time1 = omp_get_wtime();

  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head; 
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE/2; //Each edge had been presented twice
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
}


/*-------------------------------------------------------*
 * This function reads a Ppower grid file and builds the graphNew
 * First line: <#nodes> <#edges>
 * <From> <To> <Voltage-Source> <Voltage-Sink>
 * Indices are one-based numbers
 *-------------------------------------------------------*/
long* parse_MultiKvPowerGridGraph(graphNew * G, char *fileName) {
  printf("Parsing a multi-KV power grid graphNew...\n");
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }
  //Parse the first line:
  char line[1024];
  fgets(line, 1024, file);  
  char  LS1[25], LS2[25];
  long NV = 0, NE=0;
  if (sscanf(line, "%s %s", LS1, LS2) != 2) {
    printf("parse_Pajek(): bad file format - 01");
    exit(1);
  }  
  NV = atol(LS1);
  NE = atol(LS2); 
  NE = NE; //Each edge is stored only once
  printf("|V|= %ld, |E|= %ld \n", NV, NE);
  /*---------------------------------------------------------------------*/
  /* Read edge list                                                      */
  /* (i , j, value, value ) 1-based index                                       */
  /*---------------------------------------------------------------------*/
  long Si, Ti, SiV, TiV;
  edge *edgeListTmp = (edge *) malloc( NE * sizeof(edge) );
  assert(edgeListTmp != 0);
  long *Volts = (long *) malloc (NV * sizeof(long));
  assert(Volts != 0);
  for (long i=0; i<NV; i++)
    Volts[i] = -1;
  long tmpNE = 0;  
  
  double tSiV, tTiV;
  while (fscanf(file, "%ld %ld %lf %lf", &Si, &Ti, &tSiV, &tTiV) != EOF) {
    //printf("(%ld, %ld) (%ld %ld)\n",Si, Ti, SiV, TiV);
    Si--; Ti--;            // One-based indexing
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    if (Si == Ti) //Ignore self-loops
      continue;
	//if (Ti < Si) //Each edge is stored twice; so ignore the flip part.
	//	continue;
	SiV = (long)tSiV; //assert(SiV > 0);
	TiV = (long)tTiV; //assert(TiV > 0);
    edgeListTmp[tmpNE].head = Si;    //The S index
    edgeListTmp[tmpNE].tail = Ti;    //The T index
    
    if (Volts[Si] == -1) Volts[Si] = SiV; //Volts will get rewritten; correctness assumed.
    if (Volts[Ti] == -1) Volts[Ti] = TiV;
    if (SiV == TiV)
      edgeListTmp[tmpNE].weight = 1; //The value
    else
      edgeListTmp[tmpNE].weight = 0; //The value
    tmpNE++;
  }
  fclose(file); //Close the file
  printf("Done reading from file. (Edge-lines read = %ld)\n", tmpNE);
	
  //Volts not set correctly -- could be isolated vertices
  for (long i=0; i<NV; i++) {
      if (Volts[i] <= 0) {
	 printf("*** Voltage[%ld] = %ld\n", i, Volts[i]);
	 Volts[i] = 4000; //Some meaningless number
      }
  }//End of for(i)
  
  //Remove duplicate entries:
  /*
  long NewEdges = removeEdges(NV, tmpNE, edgeListTmp);
  if (NewEdges < tmpNE) {
    printf("Number of duplicate entries detected: %ld\n", NE-NewEdges);
    tmpNE = NewEdges; //Only look at clean edges
  }
  NE = tmpNE;
  */
  
  //Allocate for Edge Pointer and keep track of degree for each vertex
  long  *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));

  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  for(long i=0; i<NE; i++) {
    edgeListPtr[edgeListTmp[i].head + 1]++; //Plus one to take care of the zeroth location
    edgeListPtr[edgeListTmp[i].tail + 1]++;
  }
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
  assert(NE*2 == edgeListPtr[NV]);
  
  /*---------------------------------------------------------------------*/
  /* Allocate memory for G & Build it                                    */
  /*---------------------------------------------------------------------*/    
  printf("About to allocate memory for graphNew data structures\n");
  time1 = omp_get_wtime();
  edge *edgeList = (edge *) malloc ((2*NE) * sizeof(edge)); //Every edge stored twice
  assert(edgeList != 0);  
  //Keep track of how many edges have been added for a vertex:
  long  *added = (long *)  malloc (NV * sizeof(long)); assert (added != 0);  
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  time2 = omp_get_wtime();  
  printf("Time for allocating memory for edgeList = %lf\n", time2 - time1);

  //Build the edgeList from edgeListTmp:
  printf("About to build edgeList...\n");
  time1 = omp_get_wtime();  
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head  = edgeListTmp[i].head;
    long tail  = edgeListTmp[i].tail;
    double weight      = edgeListTmp[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head; 
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the same edge from the other direction:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);
  
  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  free(edgeListTmp);
  free(added);
	
  return Volts;
}

void parse_DoulbedEdgeList(graphNew * G, char *fileName) {
  printf("Parsing a DoulbedEdgeList formatted file as a general graphNew...\n");
  printf("WARNING: Assumes that the graphNew is undirected -- an edge is stored twince.\n");
  int nthreads = 0;

#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs9FormatDirectedNewD: Number of threads: %d\n ", nthreads);
  
  double time1, time2;
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }  
  
  long NV=2361670, NE=191402826;  
  
  printf("|V|= %ld, |E|= %ld \n", NV, NE);  
  printf("Weights will be converted to positive numbers.\n");
  /*---------------------------------------------------------------------*/
  /* Read edge list: a U V W                                             */
  /*---------------------------------------------------------------------*/  
  edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
  assert( tmpEdgeList != NULL);
  long Si, Ti;
  //double Twt;//-Wunused-variable
  time1 = omp_get_wtime();
  for (long i = 0; i < NE; i++) {
    fscanf(file, "%ld %ld", &Si, &Ti);
    assert((Si >= 0)&&(Si < NV));
    assert((Ti >= 0)&&(Ti < NV));
    tmpEdgeList[i].head   = Si;       //The S index 
    tmpEdgeList[i].tail   = Ti;    //The T index: Zero-based indexing
    tmpEdgeList[i].weight = 1; //Make it positive and cast to Double
  }//End of outer for loop
  fclose(file); //Close the file
  time2 = omp_get_wtime(); 
  printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
  
  ///////////
  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graphNew = %lf\n", time2 - time1);

#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
  }
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", NE, edgeListPtr[NV]);
  printf("*********** (%ld)\n", NV);
  
  //time1 = omp_get_wtime();
  //Keep track of how many edges have been added for a vertex:
  printf("About to allocate for added vector: %ld\n", NV);
  long  *added  = (long *)  malloc( NV  * sizeof(long));
  printf("Done allocating memory fors added vector\n");
  assert( added != NULL);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = tmpEdgeList[i].head;
    long tail      = tmpEdgeList[i].tail;
    double weight  = tmpEdgeList[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
  }
  //time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE/2;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  //Clean up
  free(tmpEdgeList);
  free(added);

}//End of parse_Dimacs9FormatDirectedNewD()

//Binary File Parser:
//Format: Zero-based indices; Every edge stored "TWICE"
//Line 1: #Vertices #Edges
//Line 2: <vertex-1> <vertex-2>
void parse_EdgeListBinary(graphNew * G, char *fileName) {
  printf("Parsing a file in binary format...\n");
  printf("WARNING: Assumes that the graphNew is undirected -- every edge is stored twice.\n");
  int nthreads = 0;

#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs9FormatDirectedNewD: Number of threads: %d\n ", nthreads);
  
  double time1, time2;

  std::ifstream ifs;  
  ifs.open(fileName, std::ifstream::in | std::ifstream::binary);
  if (!ifs) {
    std::cerr << "Error opening binary format file: " << fileName << std::endl;
    exit(EXIT_FAILURE);
  }

  long NV, NE;
  //Parse line-1: #Vertices #Edges
  ifs.read(reinterpret_cast<char*>(&NV), sizeof(NV));
  ifs.read(reinterpret_cast<char*>(&NE), sizeof(NE));
  
  std::cout << "Loading " << fileName << ", |V|: " <<
    NV << ", |E|: " << NE << std::endl;
  printf("Weights not stored in the file. All edge weights are set to one.\n");
  /*---------------------------------------------------------------------*/
  /* Read edge list: U V                                                 */
  /*---------------------------------------------------------------------*/  
  edge *tmpEdgeList = (edge *) malloc(2 * NE * sizeof(edge)); //Every edge stored TWICE
  assert( tmpEdgeList != NULL);

  //Read the entire edge list in one shot:
  time1 = omp_get_wtime();
  //int64_t *edgeListRaw = new int64_t[2*NE]; assert(edgeListRaw != 0);
  long *edgeListRaw = new long[4*NE]; assert(edgeListRaw != 0);
  //ifs.read(reinterpret_cast<char*>(edgeListRaw), sizeof(int64_t) * (2*NE));
  ifs.read(reinterpret_cast<char*>(edgeListRaw), sizeof(long) * (4*NE));
  //Need 2*NE because each edge has two numbers (vertex-1, vertex-2)
  ifs.close(); //Close the file
  time2 = omp_get_wtime(); 
  printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);

  //Now parse through the list for edges:
  printf("Parsing edges: \n");
  time1 = omp_get_wtime();
  for (long i = 0; i < 2*NE; i++) {
     tmpEdgeList[i].head = edgeListRaw[2*i]; //each edge has two numbers
     tmpEdgeList[i].tail = edgeListRaw[(2*i) + 1];
     //printf("(%ld, %ld)", tmpEdgeList[i].head, tmpEdgeList[i].tail);
     assert(tmpEdgeList[i].head >=0 && tmpEdgeList[i].head < NV);
     assert(tmpEdgeList[i].tail >=0 && tmpEdgeList[i].tail < NV);
     tmpEdgeList[i].weight = 1; //Default value of one
  }
  delete [] edgeListRaw;
  time2 = omp_get_wtime();
  printf("Time for parse through edgelist = %lf\n", time2 - time1);
 
  ///////////
  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long)); assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc(2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  long  *added  = (long *)  malloc(NV * sizeof(long)); assert( added != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graphNew = %lf\n", time2 - time1);

#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
#pragma omp parallel for
  for(long i=0; i<2*NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
  }
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: |E| = %ld, edgeListPtr[NV]= %ld\n", 2*NE, edgeListPtr[NV]);
  printf("*********** (%ld)\n", NV);
  
  time1 = omp_get_wtime();
  //Keep track of how many edges have been added for a vertex:
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;
  
  printf("About to build edgeList...\n");
  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<2*NE; i++) {
    long head      = tmpEdgeList[i].head;
    long tail      = tmpEdgeList[i].tail;
    double weight  = tmpEdgeList[i].weight;
    //Add edge head --> tail
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  //Clean up
  free(tmpEdgeList);
  free(added);

  //displayGraph(G);
}//End of parse_Dimacs9FormatDirectedNewD()


/* ****************************************** */
//Loading Functions:
/* NOTE: Indices are ZERO-based, i.e. G(0,0) is the first element, 
         while the indices stored in the file are ONE-based. 
Details:
* A graphNew contains n nodes and m arcs
* Nodes are identified by integers 1...* Graphs can be interpreted as directed or undirected, depending on the problem being studied
* Graphs can have parallel arcs and self-loops
* Arc weights are signed integers

** #... : This is a comment
** # Nodes: 65608366 Edges: 1806067135
** U V   : U = from; V = to -- is an edge

* Assumption: Each edge is stored ONLY ONCE.
*/
void parse_SNAP(graphNew * G, char *fileName) {
  printf("Parsing a SNAP formatted file as a general graphNew...\n");
  printf("WARNING: Assumes that the graphNew is directed -- an edge is stored only once.\n");
  printf("       : Graph will be stored as undirected, each edge appears twice.\n");
  int nthreads = 0;
#pragma omp parallel
  {
    nthreads = omp_get_num_threads();
  }
  printf("parse_Dimacs9FormatDirectedNewD: Number of threads: %d\n ", nthreads);
  
  long   NV=0,  NE=0;
  string oneLine, myDelimiter(" "), myDelimiter2("\t"), oneWord; //Delimiter is a blank space
  //ifstream fin;
  char comment;

  double time1, time2;
  /*
  FILE *file = fopen(fileName, "r");
  if (file == NULL) {
    printf("Cannot open the input file: %s\n",fileName);
    exit(1);
  }  
  */
  ifstream fin;
  fin.open(fileName);
  if(!fin) {
    cerr<<"Within Function: loadMetisFileFormat() \n";
    cerr<<"Could not open the file.. \n";
    exit(1);
  }
  
  do { //Parse the comment lines for problem size
    getline(fin, oneLine);
    cout<<"Read line: "<<oneLine<<endl;
    comment = oneLine[0];
    if (comment == '#') { //Check if this line has problem sizes
      StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter);
      if ( ST->HasMoreTokens() )
	oneWord = ST->GetNextToken(); //Ignore #
      if ( ST->HasMoreTokens() )
	oneWord = ST->GetNextToken(); //Ignore #
      if(oneWord == "Nodes:") {	
	//# Nodes: 65608366 Edges: 1806067135	
	NV  = atol( ST->GetNextToken().c_str() ); //Number of Vertices
	oneWord = ST->GetNextToken(); //Ignore Edges:
	NE  = atol( ST->GetNextToken().c_str() ); //Number of Edges   
      }
      delete ST;
    }
  } while ( comment == '#');
      
  printf("|V|= %ld, |E|= %ld \n", NV, NE);  
  printf("Weight of 1 will be assigned to each edge.\n");
  cout << oneLine <<endl;
  /*---------------------------------------------------------------------*/
  /* Read edge list: a U V W                                             */
  /*---------------------------------------------------------------------*/  
  edge *tmpEdgeList = (edge *) malloc( NE * sizeof(edge)); //Every edge stored ONCE
  assert( tmpEdgeList != NULL);
  long Si, Ti;

  map<long, long> clusterLocalMap; //Map each neighbor's cluster to a local number
  map<long, long>::iterator storedAlready;
  long numUniqueVertices = 0;
  
  
  //Parse the first edge already read from the file and stored in oneLine
  
  long i=0;
  do {
    StringTokenizer* ST = new StringTokenizer(oneLine, myDelimiter2);
    if ( ST->HasMoreTokens() )
      Si  = atol( ST->GetNextToken().c_str() );   
    if ( ST->HasMoreTokens() )
      Ti  = atol( ST->GetNextToken().c_str() ); 
    delete ST;  
    
    storedAlready = clusterLocalMap.find(Si); //Check if it already exists
    if( storedAlready != clusterLocalMap.end() ) {	//Already exists
      Si = storedAlready->second; //Renumber the cluster id
    } else {
      clusterLocalMap[Si] = numUniqueVertices; //Does not exist, add to the map
      Si = numUniqueVertices; //Renumber the vertex id
      numUniqueVertices++; //Increment the number
    }
    
    storedAlready = clusterLocalMap.find(Ti); //Check if it already exists
    if( storedAlready != clusterLocalMap.end() ) {	//Already exists
      Ti = storedAlready->second; //Renumber the cluster id
    } else {
      clusterLocalMap[Ti] = numUniqueVertices; //Does not exist, add to the map
      Ti = numUniqueVertices; //Renumber the vertex id
      numUniqueVertices++; //Increment the number
    }
    tmpEdgeList[i].head   = Si;  //The S index 
    tmpEdgeList[i].tail   = Ti;  //The T index: One-based indexing
    tmpEdgeList[i].weight = 1;     //default weight of one 
    //cout<<" Adding edge ("<<Si<<", "<<Ti<<")\n";
    i++;
    //Read-in the next line
    getline(fin, oneLine);
    if ((i % 999999) == 1) {
      cout <<"Reading Line: "<<i<<endl;
    }
  } while ( !fin.eof() );//End of while

  fin.close(); //Close the file
  time2 = omp_get_wtime(); 
  printf("Done reading from file: NE= %ld. Time= %lf\n", NE, time2-time1);
  
  
  ///////////
  time1 = omp_get_wtime();
  long *edgeListPtr = (long *)  malloc((NV+1) * sizeof(long));
  assert(edgeListPtr != NULL);
  edge *edgeList = (edge *) malloc( 2*NE * sizeof(edge)); //Every edge stored twice
  assert( edgeList != NULL);
  time2 = omp_get_wtime();
  printf("Time for allocating memory for storing graphNew = %lf\n", time2 - time1);
#pragma omp parallel for
  for (long i=0; i <= NV; i++)
    edgeListPtr[i] = 0; //For first touch purposes
  
  //////Build the EdgeListPtr Array: Cumulative addition 
  time1 = omp_get_wtime();
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].head+1], 1); //Leave 0th position intact
    __sync_fetch_and_add(&edgeListPtr[tmpEdgeList[i].tail+1], 1);
  }
  for (long i=0; i<NV; i++) {
    edgeListPtr[i+1] += edgeListPtr[i]; //Prefix Sum:
  }
  //The last element of Cumulative will hold the total number of characters
  time2 = omp_get_wtime();
  printf("Done cumulative addition for edgeListPtrs:  %9.6lf sec.\n", time2 - time1);
  printf("Sanity Check: 2|E| = %ld, edgeListPtr[NV]= %ld\n", NE*2, edgeListPtr[NV]);
  printf("*********** (%ld)\n", NV);
  
  printf("About to build edgeList...\n");
  time1 = omp_get_wtime();
  //Keep track of how many edges have been added for a vertex:
  long  *added  = (long *)  malloc( NV  * sizeof(long)); assert( added != NULL);
#pragma omp parallel for
  for (long i = 0; i < NV; i++) 
    added[i] = 0;

  //Build the edgeList from edgeListTmp:
#pragma omp parallel for
  for(long i=0; i<NE; i++) {
    long head      = tmpEdgeList[i].head;
    long tail      = tmpEdgeList[i].tail;
    double weight  = tmpEdgeList[i].weight;
    
    long Where = edgeListPtr[head] + __sync_fetch_and_add(&added[head], 1);   
    edgeList[Where].head = head;
    edgeList[Where].tail = tail;
    edgeList[Where].weight = weight;
    //Now add the counter-edge:
    Where = edgeListPtr[tail] + __sync_fetch_and_add(&added[tail], 1);
    edgeList[Where].head = tail;
    edgeList[Where].tail = head;
    edgeList[Where].weight = weight;    
  }
  time2 = omp_get_wtime();
  printf("Time for building edgeList = %lf\n", time2 - time1);

  G->sVertices    = NV;
  G->numVertices  = NV;
  G->numEdges     = NE;
  G->edgeListPtrs = edgeListPtr;
  G->edgeList     = edgeList;
  
  //Clean up
  free(tmpEdgeList);
  free(added);

}//End of parse_Dimacs9FormatDirectedNewD()
