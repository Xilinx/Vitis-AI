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

clustering_parameters::clustering_parameters() 
: ftype(1), strongScaling(false), output(false), VF(false), coloring(false),
C_thresh(0.0001), minGraphSize(100000), threshold(0.000001)
{}
    
void clustering_parameters::usage() {
  cout << "***************************************************************************************"<< endl;
  cout << "Basic usage: Driver <Options> FileName\n";
  cout << "***************************************************************************************"<< endl;
  cout << "Input Options: \n";
  cout << "***************************************************************************************"<< endl;
  cout << "File-type  : -f <1-8>   -- default=7" << endl;
  cout << "File-Type  : (1) Matrix-Market  (2) DIMACS#9 (3) Pajek (each edge once) (4) Pajek (twice) \n";
  cout << "           : (5) Metis (DIMACS#10) (6) Simple edge list twice (7) Binary format (8) SNAP\n";
  cout << "--------------------------------------------------------------------------------------" << endl;
  cout << "Strong scaling : -s         -- default=false" << endl; 
  cout << "VF             : -v         -- default=false" << endl;
  cout << "Output         : -o         -- default=false" << endl;
  cout << "Coloring       : -c         -- default=false" << endl; 
  cout << "--------------------------------------------------------------------------------------" << endl;
  cout << "Min-size       : -m <value> -- default=100000" << endl;  
  cout << "C-threshold    : -d <value> -- default=0.01" << endl;
  cout << "Threshold      : -t <value> -- default=0.000001" << endl;
  cout << "***************************************************************************************"<< endl;
}//end of usage()

bool clustering_parameters::parse(int argc, char *argv[]) {
  static const char *opt_string = "x:csvof:t:d:m:";
  int opt = getopt(argc, argv, opt_string);
  while (opt != -1) {
    switch (opt) {

   case 'x': xclbin = optarg; break; 
   case 'c': coloring = true; break;
   case 's': strongScaling = true; break;
   case 'v': VF = true; break;
   case 'o': output = true; break;    
  
   case 'f': ftype = atoi(optarg);
      if((ftype >8)||(ftype<0)) {
	cout << "ftype must be an integer between 1 to 6" << endl;
	return false;
      }
      break;

    case 't': threshold = atof(optarg);
      if (threshold < 0.0) {
	cout << "Threshold must be non-negative" << endl;
	return false;
      }
      break;
  
     case 'd': C_thresh = atof(optarg);
      if (C_thresh < 0.0) {
	cout << "Threshold must be non-negative" << endl;
	return false;
      }
      break;

    case 'm': minGraphSize = atol(optarg);
      if(minGraphSize <0) {
	cout << "minGraphSize must be non-negative" << endl;
	return false; 
      }
      break;    
 
    default:
      cerr << "unknown argument" << endl;
      return false;                
    }
    opt = getopt(argc, argv, opt_string);
  }
 
   if (argc - optind != 1) {
    cout << "Problem name not specified.  Exiting." << endl;
    usage();
    return false;
  } else {
    inFile = argv[optind];
  }
 
#ifdef PRINT_DETAILED_STATS_
  cout << "********************************************"<< endl;
  cout << "Input Parameters: \n";
  cout << "********************************************"<< endl;    
#ifndef HLS_TEST
  cout << "Xclbin File: " << xclbin << endl;
#endif
  cout << "Input File: " << inFile << endl;
  cout << "File type : " << ftype  << endl;
  cout << "Threshold  : " << threshold << endl;
  cout << "C-threshold: " << C_thresh << endl;
  cout << "Min-size   : " << minGraphSize << endl;
  cout << "--------------------------------------------" << endl;
  if (coloring) 
    cout << "Coloring   : TRUE" << endl;
  else
    cout << "Coloring   : FALSE" << endl;
  if (strongScaling)
    cout << "Strong scaling : TRUE" << endl; 
  else
    cout << "Strong scaling : FALSE" << endl; 
  if(VF)
    cout << "VF         : TRUE" << endl;
  else
    cout << "VF         : FLASE" << endl;
  if(output)
    cout << "Output     : TRUE"  << endl;
  else
    cout << "Output     : FALSE"  << endl;    
  cout << "********************************************"<< endl;    
#endif

  return true;
}
