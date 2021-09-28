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

using namespace std;

//Write file in binary format: 64-bit format 
//NOTE: Each edge is written only once.
//NOTE: Zero-based indexing
void writeGraphBinaryFormat(graphNew* G, char *filename) {
  //Get the iterators for the graphNew:
  long NVer    = G->numVertices;
  long NEdge   = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);
  printf("Writing graphNew in a simple edgelist format - each edge stored TWICE.\n");
 
  ofstream ofs;
  ofs.open(filename, std::ofstream::out | std::ofstream::binary | std::ofstream::trunc);
  if (!ofs) {
    std::cerr << "Error opening output file: " << filename << std::endl;
    exit(EXIT_FAILURE);
  }

  //First Line: #Vertices #Edges
  ofs.write(reinterpret_cast<char*>(&NVer), sizeof(NVer));
  ofs.write(reinterpret_cast<char*>(&NEdge), sizeof(NEdge));
  //Write all the edges (each edge stored twice):
  for (long v=0; v<NVer; v++) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    //Edge lines: <vertex-1> <vertex-2>
    for(long k = adj1; k < adj2; k++ ) {
        long tail = verInd[k].tail;	
         ofs.write(reinterpret_cast<char*>(&v), sizeof(long));
	 ofs.write(reinterpret_cast<char*>(&tail), sizeof(long));
    }
  }
  ofs.close();
  printf("Graph has been stored in file: %s\n",filename);
}//End of writeGraphBinaryFormatTwice()

//Output the graphNew in Pajek format:
//Each edge is represented twice
void writeGraphPajekFormat(graphNew* G, char *filename) {
  //Get the iterators for the graphNew:
  long NVer     = G->numVertices;
  //long NS       = G->sVertices;//-Wunused-variable
  //long NT       = NVer - NS;//-Wunused-variable
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);

  printf("Writing graphNew in Pajek format - Undirected graphNew - each edge represented ONLY ONCE!\n");
  printf("Graph will be stored in file: %s\n", filename);
  
   
  FILE *fout;
  fout = fopen(filename, "w");
  if (!fout) {
     printf("Could not open the file \n");
     exit(1);
  }
  //First Line: Vertices
  fprintf(fout, "*Vertices %ld\n", NVer);
  for (long i=0; i<NVer; i++) {
	  fprintf(fout, "%ld\n", i+1);
  }
  
  //Write the edges:
  fprintf(fout, "*Edges %ld\n", NEdge);
  for (long v=0; v<NVer; v++) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    //Edge lines: <adjacent> <weight>
    for(long k = adj1; k < adj2; k++ ) {
		if (v <= verInd[k].tail) { //Print only once
	     fprintf(fout, "%ld %ld %g\n", v+1, (verInd[k].tail+1), (verInd[k].weight) );
		}
	}
  }
  fclose(fout);
  printf("Graph has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()

//Output the graphNew in Pajek format:
//Each edge is represented twice
void writeGraphMetisSimpleFormat(graphNew* G, char *filename) {
  //Get the iterators for the graphNew:
  long NVer     = G->numVertices;
  long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
  long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
  edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
  printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);

  printf("Writing graphNew in Metis format - each edge represented twice -- no weights; 1-based indices\n");
  printf("Graph will be stored in file: %s\n", filename);
  
   
  FILE *fout;
  fout = fopen(filename, "w");
  if (!fout) {
    printf("Could not open the file \n");
    exit(1);
  }
  //First Line: #Vertices #Edges
  fprintf(fout, "%ld %ld\n", NVer, NEdge);
  //Write the edges:
  for (long v=0; v<NVer; v++) {
    long adj1 = verPtr[v];
    long adj2 = verPtr[v+1];
    for(long k = adj1; k < adj2; k++ ) {
      fprintf(fout, "%ld ", (verInd[k].tail+1) );
    }
    fprintf(fout, "\n");
  }
  fclose(fout);
  printf("Graph has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()

void writeGraphPajekFormatWithNodeVolts(graphNew* G, long *Cvolts, char * filename) {
	//Get the iterators for the graphNew:
	long NVer     = G->numVertices;
	//long NS       = G->sVertices;//-Wunused-variable
	//long NT       = NVer - NS;//-Wunused-variable
	long NEdge    = G->numEdges;       //Returns the correct number of edges (not twice)
	long *verPtr  = G->edgeListPtrs;   //Vertex Pointer: pointers to endV
	edge *verInd = G->edgeList;       //Vertex Index: destination id of an edge (src -> dest)
	printf("NVer= %ld --  NE=%ld\n", NVer, NEdge);
	
	printf("Writing graphNew in Pajek format - Undirected graphNew - each edge represented ONLY ONCE!\n");
	printf("Graph will be stored in file: %s\n", filename);
		
	FILE *fout;
	fout = fopen(filename, "w");
	if (!fout) {
		printf("Could not open the file \n");
		exit(1);
	}
	//First Line: Vertices
	fprintf(fout, "*Vertices %ld\n", NVer);
	for (long i=0; i<NVer; i++) {
		fprintf(fout, "%ld  %ld\n", i+1, Cvolts[i]);
	}
	
	//Write the edges:
	fprintf(fout, "*Edges %ld\n", NEdge);
	for (long v=0; v<NVer; v++) {
		long adj1 = verPtr[v];
		long adj2 = verPtr[v+1];
		//Edge lines: <adjacent> <weight>
		for(long k = adj1; k < adj2; k++ ) {			
			if (v <= verInd[k].tail) { //Print only once; keep self-loops
				fprintf(fout, "%ld %ld %g\n", v+1, (verInd[k].tail+1), (verInd[k].weight) );
			}
		}
	}
	fclose(fout);
	printf("Graph has been stored in file: %s\n",filename);
}//End of writeGraphPajekFormat()
